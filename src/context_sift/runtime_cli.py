"""Zero-configuration ContextSift command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from context_sift.msc import CompactorService


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact text before LLM prefill.")
    parser.add_argument("file", nargs="?", type=Path, help="UTF-8 input; defaults to stdin")
    args = parser.parse_args()
    text = args.file.read_text(encoding="utf-8") if args.file else sys.stdin.read()
    with CompactorService() as sift:
        sys.stdout.write(sift(text))


if __name__ == "__main__":
    main()
