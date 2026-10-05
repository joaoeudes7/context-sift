"""Warm JSON-lines compaction daemon: load the model once, serve many requests."""

from __future__ import annotations

import json
import time
from typing import IO, Any

from context_sift.msc import CompactorService


def _respond(out: IO[str], payload: dict[str, Any]) -> None:
    out.write(json.dumps(payload, ensure_ascii=False) + "\n")
    out.flush()


def serve(inp: IO[str], out: IO[str], service: CompactorService) -> None:
    """Read one JSON request per line, write one JSON response per line.

    Protocol::

        in : {"id": <any>, "text": "<str>"}
        out: {"id": <any>, "text": "<str>", "ratio": <out/in chars>, "ms": <float>}
             {"id": <any>, "error": "<msg>"}

    A ``{"ready": true}`` line is written once, after the model is loaded, so a
    supervising client can wait for readiness before sending work.
    """
    _respond(out, {"ready": True})
    for raw in inp:
        line = raw.strip()
        if not line:
            continue
        identifier: Any = None
        try:
            request = json.loads(line)
            identifier = request.get("id")
            text = request["text"]
            if not isinstance(text, str):
                raise TypeError("text must be a string")
            started = time.perf_counter()
            result = service(text)
            _respond(
                out,
                {
                    "id": identifier,
                    "text": result,
                    "ratio": round(len(result) / len(text), 4) if text else 1.0,
                    "ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
        except Exception as error:  # a bad line must never kill the daemon
            _respond(out, {"id": identifier, "error": f"{type(error).__name__}: {error}"})
