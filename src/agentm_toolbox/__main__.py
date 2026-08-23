"""Command-line entry points executed inside remote AgentM sandboxes."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from agentm_toolbox._repository_index import (
    RepositoryIndexWorkerError,
    load_repository_documents,
    update_repository_index,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentm-toolbox")
    commands = parser.add_subparsers(dest="command", required=True)
    repository = commands.add_parser(
        "repository-index",
        help="Build or refresh a sandbox-local ast-grep repository index.",
    )
    repository.add_argument("--root", required=True)
    repository.add_argument("--target", required=True)
    repository.add_argument("--db", required=True)
    repository.add_argument("--replace", action="store_true")
    repository.add_argument("--include-documents", action="store_true")
    repository.add_argument("--load-only", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.load_only:
            result = load_repository_documents(
                root=args.root,
                target=args.target,
                db_path=args.db,
            )
        else:
            result = update_repository_index(
                root=args.root,
                target=args.target,
                db_path=args.db,
                replace=args.replace,
                include_documents=args.include_documents,
            )
    except RepositoryIndexWorkerError as exc:
        print(json.dumps({"version": 1, "ok": False, "error": str(exc)}))
        return 1
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
