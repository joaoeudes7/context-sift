"""Zero-configuration ContextSift command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from context_sift.msc import CompactorService


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact text before LLM prefill.")
    parser.add_argument("file", nargs="?", type=Path, help="UTF-8 input; defaults to stdin")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="run a warm JSON-lines daemon on stdin/stdout (one request per line)",
    )
    parser.add_argument(
        "--serve-socket",
        type=Path,
        help="run a shared JSON-lines daemon on this Unix socket until idle",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=60.0,
        help="seconds without a client before --serve-socket exits",
    )
    args = parser.parse_args()
    if args.serve:
        from context_sift.serve import serve

        serve(sys.stdin, sys.stdout, CompactorService())
        return
    if args.serve_socket:
        from context_sift.serve import acquire_singleton, serve_socket, socket_is_live

        if socket_is_live(args.serve_socket):
            return  # a warm daemon already serves; never load a second model
        lock = acquire_singleton(args.serve_socket)
        if lock is None:
            # Another process is starting the daemon; wait for its socket.
            for _ in range(300):
                if socket_is_live(args.serve_socket):
                    return
                time.sleep(0.05)
            return
        with CompactorService() as sift:
            asyncio.run(serve_socket(args.serve_socket, sift, args.idle_timeout))
        return
    text = args.file.read_text(encoding="utf-8") if args.file else sys.stdin.read()
    with CompactorService() as sift:
        sys.stdout.write(sift(text))


if __name__ == "__main__":
    main()
