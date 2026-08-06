"""CLI entry point for the context-sift proxy."""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible proxy that compacts requests before forwarding.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port (default: 8000)"
    )
    parser.add_argument(
        "--upstream",
        default=os.environ.get("CONTEXT_SIFT_UPSTREAM_URL", "https://api.openai.com/v1"),
        help="Upstream API base URL (env: CONTEXT_SIFT_UPSTREAM_URL)",
    )
    parser.add_argument(
        "--key",
        default=os.environ.get("CONTEXT_SIFT_UPSTREAM_KEY"),
        help="Upstream API key (env: CONTEXT_SIFT_UPSTREAM_KEY)",
    )
    parser.add_argument(
        "--backend", default=None,
        help="Backend: auto, mlx, torch, cpu (default: auto)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Compaction threshold (default: model config)",
    )
    args = parser.parse_args()

    if args.upstream:
        os.environ["CONTEXT_SIFT_UPSTREAM_URL"] = args.upstream
    if args.key:
        os.environ["CONTEXT_SIFT_UPSTREAM_KEY"] = args.key

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: pip install 'context-sift[proxy]'", file=sys.stderr)
        sys.exit(1)

    from context_sift.msc import CompactorService
    from context_sift.proxy import create_app

    compactor_kwargs: dict = {}
    if args.backend:
        compactor_kwargs["backend"] = args.backend
    if args.threshold is not None:
        compactor_kwargs["threshold"] = args.threshold

    app = create_app(compactor=CompactorService(**compactor_kwargs))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
