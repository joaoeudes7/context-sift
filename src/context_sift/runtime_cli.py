"""Zero-configuration ContextSift command-line interface."""

from __future__ import annotations

import argparse
import sys
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
    args = parser.parse_args()
    if args.serve:
        from context_sift.serve import serve

        serve(sys.stdin, sys.stdout, CompactorService())
        return
    text = args.file.read_text(encoding="utf-8") if args.file else sys.stdin.read()
    with CompactorService() as sift:
        sys.stdout.write(sift(text))


if __name__ == "__main__":
    main()
