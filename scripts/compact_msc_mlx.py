#!/usr/bin/env python3
"""Read text from stdin; emit only Minimum Sufficient Context text."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from context_sift import Compactor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/context-sift"))
    parser.add_argument("--threshold", type=float, default=0.15)
    args = parser.parse_args()
    print(Compactor(args.model, threshold=args.threshold)(sys.stdin.read()))


if __name__ == "__main__":
    main()
