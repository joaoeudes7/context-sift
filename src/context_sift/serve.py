"""Warm JSON-lines compaction daemon: load the model once, serve many requests.

Two transports:

* :func:`serve` — stdin/stdout, one JSON request per line. Used by ``--serve``.
* :func:`serve_socket` — a Unix-socket daemon shared by every client on the
  machine. Used by ``--serve-socket``; exits after ``idle_timeout`` seconds
  with no connected client.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import IO, Any, Callable

from context_sift.gain import record
from context_sift.msc import CompactorService


def _process(
    line: str,
    service: CompactorService,
    sink: Callable[[dict[str, Any]], None] | None = record,
) -> dict[str, Any]:
    identifier: Any = None
    try:
        request = json.loads(line)
        identifier = request.get("id")
        text = request["text"]
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        meta = request.get("meta") if isinstance(request.get("meta"), dict) else {}
        started = time.perf_counter()
        result = service(text)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        if sink is not None:
            original = meta.get("in_chars")
            if not isinstance(original, (int, float)) or original <= 0:
                original = len(text)
            # ponytail: open+append per request in the event loop; a persistent
            # handle/batching only if write latency ever shows up.
            sink(
                {
                    "source": str(meta.get("source") or "cli"),
                    "cwd": str(meta.get("cwd") or ""),
                    "model": str(meta.get("model") or ""),
                    "session": str(meta.get("session") or ""),
                    "in": int(original),
                    "out": len(result),
                    "ms": elapsed_ms,
                }
            )

        return {
            "id": identifier,
            "text": result,
            "ratio": round(len(result) / len(text), 4) if text else 1.0,
            "ms": elapsed_ms,
        }
    except Exception as error:  # a bad line must never kill the daemon
        return {"id": identifier, "error": f"{type(error).__name__}: {error}"}


def _respond(out: IO[str], payload: dict[str, Any]) -> None:
    out.write(json.dumps(payload, ensure_ascii=False) + "\n")
    out.flush()


def serve(inp: IO[str], out: IO[str], service: CompactorService) -> None:
    """Read one JSON request per line, write one JSON response per line.

    Protocol::

        in : {"id": <any>, "text": "<str>"}
        out: {"id": <any>, "text": "<str>", "ratio": <out/in chars>, "ms": <float>}
             {"id": <any>, "error": "<msg>"}

    A ``{"ready": true}`` line is written once, after the model is loaded.
    """
    _respond(out, {"ready": True})
    for raw in inp:
        line = raw.strip()
        if line:
            _respond(out, _process(line, service))


async def _handle(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    service: CompactorService,
    touch: Callable[[], None],
) -> None:
    try:
        while True:
            raw = await reader.readline()
            if not raw:
                break
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            touch()
            response = _process(line, service)
            writer.write((json.dumps(response, ensure_ascii=False) + "\n").encode())
            await writer.drain()
    finally:
        writer.close()


async def _is_live(socket_path: Path) -> bool:
    """True when a daemon is already listening on ``socket_path``."""
    try:
        reader, writer = await asyncio.open_unix_connection(str(socket_path))
    except OSError:
        return False
    writer.close()
    return True


def socket_is_live(socket_path: Path | str) -> bool:
    """Blocking liveness probe: can we connect to a daemon right now?"""
    import socket as _socket

    client = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    try:
        client.settimeout(0.3)
        client.connect(str(socket_path))
        return True
    except OSError:
        return False
    finally:
        client.close()


def acquire_singleton(socket_path: Path | str) -> int | None:
    """Take the daemon's start lock, or return ``None`` if another process holds it.

    Held for the process lifetime; the OS releases it on exit, so a crash never
    leaves a stuck lock. Only the holder loads the model and serves the socket.
    """
    import fcntl
    import os

    lock_path = str(socket_path) + ".lock"
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(handle)
        return None
    return handle


async def _wait_for_live(socket_path: Path, attempts: int = 20, delay: float = 0.1) -> bool:
    """Give a racing winner time to finish bind+listen before declaring a socket stale."""
    for _ in range(attempts):
        if await _is_live(socket_path):
            return True
        await asyncio.sleep(delay)
    return False


async def serve_socket(
    path: Path | str,
    service: CompactorService,
    idle_timeout: float = 60.0,
) -> None:
    """Serve JSON-lines over a Unix socket until idle for ``idle_timeout`` seconds.

    Lifecycle: exactly one daemon owns the socket (the model holder). Clients
    connect and disconnect freely; the daemon stays warm while at least one
    client is connected and for ``idle_timeout`` seconds after the last one
    leaves, then exits and unlinks the socket. A losing racer that finds a live
    daemon simply returns, so concurrent spawns collapse to one process.
    """
    loop = asyncio.get_running_loop()
    socket_path = Path(path)
    socket_path.parent.mkdir(parents=True, exist_ok=True)

    state = {"last": loop.time(), "active": 0}

    def touch() -> None:
        state["last"] = loop.time()

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        state["active"] += 1
        touch()
        try:
            await _handle(reader, writer, service, touch)
        finally:
            state["active"] -= 1
            touch()

    try:
        server = await asyncio.start_unix_server(handler, path=str(socket_path))
    except OSError:
        # Either a live daemon already owns the path, or the socket is stale.
        # Wait out a racing winner's bind+listen window first; only then steal.
        if await _wait_for_live(socket_path):
            return
        socket_path.unlink(missing_ok=True)
        server = await asyncio.start_unix_server(handler, path=str(socket_path))

    async def reap_when_idle() -> None:
        interval = max(0.05, min(1.0, idle_timeout / 4))
        while True:
            await asyncio.sleep(interval)
            if state["active"] == 0 and loop.time() - state["last"] > idle_timeout:
                server.close()
                return

    reaper = asyncio.create_task(reap_when_idle())
    try:
        await server.serve_forever()
    except asyncio.CancelledError:
        pass  # 3.14: Server.close() cancels serve_forever; that is our idle exit
    finally:
        reaper.cancel()
        server.close()
        socket_path.unlink(missing_ok=True)
