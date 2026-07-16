#!/usr/bin/env python3
"""Serve the trajectory index viewer with an index.json file.

Usage:
    python serve.py /path/to/index.json [--port 8765]
"""

import argparse
import http.server
import os
import shutil
import sys
import tempfile
import webbrowser
from pathlib import Path

VIEWER_DIR = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve trajectory index viewer")
    parser.add_argument("index_json", help="Path to index.json")
    parser.add_argument("--port", type=int, default=8736)
    parser.add_argument("--no-open", action="store_true", help="Don't open browser")
    args = parser.parse_args()

    src = Path(args.index_json).resolve()
    if not src.exists():
        print(f"File not found: {src}", file=sys.stderr)
        sys.exit(1)

    tmpdir = tempfile.mkdtemp(prefix="traj-viewer-")
    try:
        shutil.copy2(VIEWER_DIR / "index.html", tmpdir)
        shutil.copy2(src, Path(tmpdir) / "data.json")

        os.chdir(tmpdir)
        url = f"http://localhost:{args.port}/index.html?file=data.json"
        print(f"Serving {src.name} at {url}")

        if not args.no_open:
            webbrowser.open(url)

        handler = http.server.SimpleHTTPRequestHandler
        handler.log_message = lambda *a: None
        with http.server.HTTPServer(("", args.port), handler) as httpd:
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nStopped.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
