from __future__ import annotations

import base64
import hashlib
import json
import secrets
import urllib.error
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event

from agentm.ai.types import (
    OAuthAuthInfo,
    OAuthCredentials,
    OAuthLoginCallbacks,
    OAuthPrompt,
    OAuthProviderInterface,
)

_OAUTH_REGISTRY: dict[str, OAuthProviderInterface] = {}
_BUILTINS: dict[str, OAuthProviderInterface] = {}


def _decode_client_id(value: str) -> str:
    return base64.b64decode(value).decode("utf-8")


class AnthropicOAuthProvider:
    id = "anthropic"
    name = "Anthropic (Claude Pro/Max)"
    uses_callback_server = True

    _client_id = _decode_client_id("OWQxYzI1MGEtZTYxYi00NGQ5LTg4ZWQtNTk0NGQxOTYyZjVl")
    _authorize_url = "https://claude.ai/oauth/authorize"
    _token_url = "https://platform.claude.com/v1/oauth/token"
    _callback_host = "127.0.0.1"
    _callback_port = 53692
    _callback_path = "/callback"
    _scopes = (
        "org:create_api_key user:profile user:inference "
        "user:sessions:claude_code user:mcp_servers user:file_upload"
    )

    async def login(self, callbacks: OAuthLoginCallbacks) -> OAuthCredentials:
        verifier, challenge = _generate_pkce()
        state = verifier
        redirect_uri = (
            f"http://localhost:{self._callback_port}{self._callback_path}"
        )
        receiver = _CallbackReceiver(
            host=self._callback_host,
            port=self._callback_port,
            path=self._callback_path,
            expected_state=state,
        )
        receiver.start()
        try:
            params = urllib.parse.urlencode(
                {
                    "code": "true",
                    "client_id": self._client_id,
                    "response_type": "code",
                    "redirect_uri": redirect_uri,
                    "scope": self._scopes,
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                    "state": state,
                }
            )
            callbacks.on_auth(
                OAuthAuthInfo(
                    url=f"{self._authorize_url}?{params}",
                    instructions=(
                        "Complete login in your browser. If the browser is on another "
                        "machine, paste the final redirect URL or auth code here."
                    ),
                )
            )

            code: str | None = None
            callback_state: str | None = None
            manual_input: str | None = None
            if self.uses_callback_server:
                try:
                    manual_input = await callbacks.on_manual_code_input()
                except Exception:
                    manual_input = None
                if manual_input:
                    parsed_code, parsed_state = _parse_authorization_input(manual_input)
                    code = parsed_code
                    callback_state = parsed_state or state
                else:
                    result = receiver.wait()
                    if result is not None:
                        code, callback_state = result

            if not code:
                entered = await callbacks.on_prompt(
                    OAuthPrompt(
                        message="Paste the authorization code or full redirect URL:",
                        placeholder=redirect_uri,
                    )
                )
                code, callback_state = _parse_authorization_input(entered)
                callback_state = callback_state or state

            if not code or callback_state != state:
                raise RuntimeError("OAuth state mismatch or missing authorization code")

            callbacks.on_progress("Exchanging authorization code for tokens...")
            return self._exchange_authorization_code(
                code=code,
                state=callback_state,
                verifier=verifier,
                redirect_uri=redirect_uri,
            )
        finally:
            receiver.close()

    async def refresh_token(self, credentials: OAuthCredentials) -> OAuthCredentials:
        return self._post_for_tokens(
            {
                "grant_type": "refresh_token",
                "client_id": self._client_id,
                "refresh_token": credentials["refresh"],
            }
        )

    def get_api_key(self, credentials: OAuthCredentials) -> str:
        return credentials["access"]

    def _exchange_authorization_code(
        self,
        *,
        code: str,
        state: str,
        verifier: str,
        redirect_uri: str,
    ) -> OAuthCredentials:
        return self._post_for_tokens(
            {
                "grant_type": "authorization_code",
                "client_id": self._client_id,
                "code": code,
                "state": state,
                "redirect_uri": redirect_uri,
                "code_verifier": verifier,
            }
        )

    def _post_for_tokens(self, payload: dict[str, str]) -> OAuthCredentials:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._token_url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Anthropic OAuth request failed with HTTP {exc.code}: {detail}"
            ) from exc
        data = json.loads(raw)
        expires_in = int(data["expires_in"])
        return {
            "refresh": data["refresh_token"],
            "access": data["access_token"],
            "expires": _now_ms() + expires_in * 1000 - 5 * 60 * 1000,
        }


class _CallbackReceiver:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        path: str,
        expected_state: str,
    ) -> None:
        self._host = host
        self._port = port
        self._path = path
        self._expected_state = expected_state
        self._server: ThreadingHTTPServer | None = None
        self._done = Event()
        self._result: tuple[str, str] | None = None

    def start(self) -> None:
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return None

            def do_GET(self) -> None:  # noqa: N802
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path != parent._path:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                params = urllib.parse.parse_qs(parsed.query)
                code = params.get("code", [None])[0]
                state = params.get("state", [None])[0]
                error = params.get("error", [None])[0]
                if error is not None:
                    self.send_response(HTTPStatus.BAD_REQUEST)
                    self.end_headers()
                    self.wfile.write(b"OAuth failed")
                    parent._done.set()
                    return
                if not code or not state or state != parent._expected_state:
                    self.send_response(HTTPStatus.BAD_REQUEST)
                    self.end_headers()
                    self.wfile.write(b"Invalid OAuth callback")
                    parent._done.set()
                    return
                parent._result = (code, state)
                self.send_response(HTTPStatus.OK)
                self.end_headers()
                self.wfile.write(b"Anthropic authentication completed. You can close this window.")
                parent._done.set()

        self._server = ThreadingHTTPServer((self._host, self._port), Handler)
        self._server.timeout = 0.25

        import threading

        thread = threading.Thread(target=self._serve, daemon=True)
        thread.start()

    def _serve(self) -> None:
        assert self._server is not None
        while not self._done.is_set():
            self._server.handle_request()

    def wait(self, timeout: float = 120.0) -> tuple[str, str] | None:
        self._done.wait(timeout)
        return self._result

    def close(self) -> None:
        self._done.set()
        if self._server is not None:
            self._server.server_close()


def _parse_authorization_input(value: str) -> tuple[str | None, str | None]:
    stripped = value.strip()
    if not stripped:
        return None, None
    try:
        parsed = urllib.parse.urlparse(stripped)
        if parsed.scheme and parsed.netloc:
            params = urllib.parse.parse_qs(parsed.query)
            return params.get("code", [None])[0], params.get("state", [None])[0]
    except ValueError:
        pass
    if "#" in stripped:
        code, state = stripped.split("#", 1)
        return code or None, state or None
    if "code=" in stripped:
        params = urllib.parse.parse_qs(stripped)
        return params.get("code", [None])[0], params.get("state", [None])[0]
    return stripped, None


def _generate_pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)


def get_oauth_provider(provider_id: str) -> OAuthProviderInterface | None:
    return _OAUTH_REGISTRY.get(provider_id)


def get_oauth_providers() -> list[OAuthProviderInterface]:
    return list(_OAUTH_REGISTRY.values())


def register_oauth_provider(provider: OAuthProviderInterface) -> None:
    _OAUTH_REGISTRY[provider.id] = provider
    _BUILTINS.setdefault(provider.id, provider)


def unregister_oauth_provider(provider_id: str) -> None:
    built_in = _BUILTINS.get(provider_id)
    if built_in is None:
        _OAUTH_REGISTRY.pop(provider_id, None)
    else:
        _OAUTH_REGISTRY[provider_id] = built_in


def reset_oauth_providers() -> None:
    _OAUTH_REGISTRY.clear()
    _OAUTH_REGISTRY.update(_BUILTINS)


register_oauth_provider(AnthropicOAuthProvider())
