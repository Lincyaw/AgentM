"""Skill management tools -- factory creating closures over a MarkdownVault.

Skills are vault notes with ``type: skill`` frontmatter. These tools let the
orchestrator discover, load, and search skills at runtime.  Loaded skill
content is returned in the tool response for the LLM to use.
"""

from __future__ import annotations

import json
from typing import Any

from agentm.tools.vault.store import MarkdownVault


def _ok(**kwargs: Any) -> str:
    return json.dumps({"status": "ok", **kwargs}, ensure_ascii=False)


def _err(msg: str) -> str:
    return json.dumps({"error": msg}, ensure_ascii=False)


def create_skill_tools(vault: MarkdownVault) -> dict[str, Any]:
    """Create skill management tool functions with closure over vault instance.

    Returns a dict mapping tool name to callable.  Every callable returns
    a JSON string and never raises -- errors are encoded as ``{"error": ...}``.
    """

    # ------------------------------------------------------------------
    # 1. skill_list
    # ------------------------------------------------------------------

    def skill_list(request: str) -> str:
        """List all available skills with descriptions and trigger patterns.

        Discovers skills stored in the knowledge vault.  Use this to find
        what specialized capabilities are available before loading one.

        Args:
            request: Any value (e.g. "list").
        """
        try:
            notes = vault.list_notes("", depth=3, type_filter="skill")
            results: list[dict[str, Any]] = []
            for note in notes:
                full = vault.read(note["path"])
                if full is None:
                    continue
                fm = full["frontmatter"]
                results.append({
                    "path": note["path"],
                    "name": fm.get("name", note["path"]),
                    "description": fm.get("description", ""),
                    "trigger_patterns": fm.get("trigger_patterns", []),
                    "tags": fm.get("tags", []),
                })
            return json.dumps({"skills": results}, ensure_ascii=False)
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # 2. skill_load
    # ------------------------------------------------------------------

    def skill_load(name: str) -> str:
        """Load a skill's full instructions into the active context.

        After loading, the skill's instructions will be included in the
        system prompt for subsequent rounds.  Use ``skill_search`` if you
        are not sure which skill to load.

        Args:
            name: The skill vault path (e.g. "skill/data-analysis") or name.
        """
        try:
            result = vault.read(name)
            if result is None:
                # Fallback: search by name or title
                from agentm.tools.vault.search import keyword_search

                conn = vault._get_conn()
                hits = keyword_search(conn, name, {"type": "skill"}, 1)
                if hits:
                    result = vault.read(hits[0].path)

            if result is None:
                return _err(f"Skill not found: {name}")

            fm = result["frontmatter"]
            return json.dumps({
                "status": "loaded",
                "path": result["path"],
                "name": fm.get("name", result["path"]),
                "description": fm.get("description", ""),
                "body": result["body"],
            }, ensure_ascii=False)
        except Exception as exc:
            return _err(str(exc))

    # ------------------------------------------------------------------
    # 3. skill_search
    # ------------------------------------------------------------------

    def skill_search(query: str, limit: int = 5) -> str:
        """Search for skills by semantic similarity or keyword.

        Use this when you are not sure which skill to use.  Returns
        matching skills ranked by relevance.

        Args:
            query: Natural language description of what you need.
            limit: Maximum number of results (default 5).
        """
        try:
            from agentm.tools.vault.search import hybrid_search, keyword_search

            conn = vault._get_conn()

            if vault._embedding_model is not None:
                from agentm.tools.vault.tools import _get_embed_fn

                hits = hybrid_search(
                    conn, query, _get_embed_fn(vault), {"type": "skill"}, limit
                )
            else:
                hits = keyword_search(conn, query, {"type": "skill"}, limit)

            results: list[dict[str, Any]] = []
            for hit in hits:
                results.append({
                    "path": hit.path,
                    "title": hit.title,
                    "score": hit.score,
                    "snippet": hit.snippet,
                })
            return json.dumps({"results": results}, ensure_ascii=False)
        except Exception as exc:
            return _err(str(exc))

    return {
        "skill_list": skill_list,
        "skill_load": skill_load,
        "skill_search": skill_search,
    }
