# code-health: ignore-file[AM025] -- normalizes untyped vendor SDK payloads at the provider boundary
"""Provider-neutral scaffolding shared by the bundled LLM provider atoms.

The wire formats of the OpenAI and Anthropic adapters have nothing in common
and deliberately stay in their own modules. Everything *around* the wire format
does: reading fields off untyped vendor SDK objects, deciding whether a
``base_url`` points at the vendor's own endpoint or at a compatible gateway,
and turning install-time config into a registry name.

Lives in ``core.lib`` because atoms may import it (``core.abi``, ``core.lib``,
``agentm.extensions``) while they may not import each other.

Every message raised from here names the provider it came from, so an
Anthropic failure never reports an OpenAI field.
"""

from __future__ import annotations

from dataclasses import dataclass

_SDK_MISSING = object()


@dataclass(frozen=True, slots=True)
class SdkFieldReader:
    """Reads attributes off vendor SDK response objects with typed coercion.

    ``label`` is the vendor name as it should appear in error messages
    (``"OpenAI"``, ``"Anthropic"``).
    """

    label: str

    def optional_attr(self, value: object, name: str) -> object | None:
        item = getattr(  # code-health: ignore[AM021] -- vendor SDK model boundary
            value,
            name,
            _SDK_MISSING,
        )
        return None if item is _SDK_MISSING else item

    def required_attr(self, value: object, name: str) -> object:
        item = self.optional_attr(value, name)
        if item is None:
            raise ValueError(f"{self.label} SDK field {name!r} is required")
        return item

    def optional_string(self, value: object, name: str) -> str | None:
        item = self.optional_attr(value, name)
        if item is None:
            return None
        if not isinstance(item, str):
            raise TypeError(f"{self.label} SDK field {name!r} must be a string or None")
        return item

    def required_string(
        self,
        value: object,
        name: str,
        *,
        allow_empty: bool = True,
    ) -> str:
        item = self.required_attr(value, name)
        if not isinstance(item, str) or (not allow_empty and not item):
            raise TypeError(f"{self.label} SDK field {name!r} must be a string")
        return item

    def nonnegative_int(
        self,
        value: object,
        name: str,
        *,
        default: int | None = None,
    ) -> int:
        item = self.optional_attr(value, name)
        if item is None:
            if default is None:
                raise ValueError(f"{self.label} SDK field {name!r} is required")
            return default
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise TypeError(
                f"{self.label} SDK field {name!r} must be a non-negative integer"
            )
        return item

    def sequence(
        self,
        value: object,
        name: str,
        *,
        optional: bool = False,
    ) -> tuple[object, ...]:
        item = self.optional_attr(value, name)
        if item is None and optional:
            return ()
        if not isinstance(item, (list, tuple)):
            raise TypeError(f"{self.label} SDK field {name!r} must be a list")
        return tuple(item)


class DuplicateProviderError(ValueError):
    """Raised when a provider install would shadow an existing registration.

    Two situations trigger this:

    * ``config.base_url`` points at a non-canonical (custom) endpoint that
      speaks the vendor's protocol and ``config.name`` was not supplied — the
      install would otherwise default to the bare vendor registry key and
      silently collide with another custom endpoint registered in the same
      session.
    * The session already has a provider registered under the requested name.
    """


@dataclass(frozen=True, slots=True)
class ProviderInstallSpec:
    """The per-provider strings that shape install-time error messages.

    ``atom`` is the dotted module name of the provider atom, ``label`` the
    vendor name, ``default_name`` the registry key used when ``config.name`` is
    omitted, and the two example tuples supply the hints appended to errors.
    """

    atom: str
    label: str
    default_name: str
    canonical_base_urls: frozenset[str]
    name_examples: tuple[str, ...]
    model_examples: tuple[str, ...]


def is_non_canonical_base_url(
    base_url: object,
    canonical_urls: frozenset[str],
) -> bool:
    """True when ``base_url`` names a compatible gateway rather than the vendor."""

    if base_url is None:
        return False
    if not isinstance(base_url, str) or not base_url.strip():
        return False
    return base_url.rstrip("/") not in {url.rstrip("/") for url in canonical_urls}


def resolve_model_id(model: object, *, spec: ProviderInstallSpec) -> str:
    """Validate the configured model id, or explain what a valid one looks like."""

    if not model or not isinstance(model, str):
        raise ValueError(
            f"{spec.atom}.install: config.model is required and must "
            f"be a non-empty string ({_examples(spec.model_examples, ' or ')})."
        )
    return model


def resolve_provider_name(
    raw_name: object,
    base_url: object,
    *,
    spec: ProviderInstallSpec,
) -> str:
    """Pick the registry name, refusing defaults that would shadow a sibling.

    Omitting ``config.name`` for a custom endpoint is the dangerous case: every
    such install lands on the bare default key and the last one wins.
    """

    if raw_name is None:
        if is_non_canonical_base_url(base_url, spec.canonical_base_urls):
            raise DuplicateProviderError(
                f"{spec.atom}.install: config.name is required when "
                f"base_url={base_url!r} is set to a non-canonical "
                f"{spec.label}-compatible endpoint. Multiple custom endpoints "
                f"default to the bare {spec.default_name!r} registry name and "
                "would silently overwrite each other. Pass an explicit "
                f"config.name ({_examples(spec.name_examples, ', ')})."
            )
        name: object = spec.default_name
    else:
        name = raw_name
    if not isinstance(name, str) or not name:
        raise ValueError(f"{spec.atom}.install: config.name must be a non-empty string")
    return name


def _examples(values: tuple[str, ...], separator: str) -> str:
    return "e.g. " + separator.join(repr(value) for value in values)


__all__ = (
    "DuplicateProviderError",
    "ProviderInstallSpec",
    "SdkFieldReader",
    "is_non_canonical_base_url",
    "resolve_model_id",
    "resolve_provider_name",
)
