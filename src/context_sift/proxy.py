"""OpenAI-compatible proxy that compacts requests before forwarding."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from context_sift.gain import record
from context_sift.msc import CompactorService

log = logging.getLogger(__name__)

_UPSTREAM_DEFAULT = "https://api.openai.com/v1"


def _sift(sift: CompactorService, text: str) -> str:
    result = sift(text)
    record({"source": "proxy", "cwd": os.getcwd(), "in": len(text), "out": len(result)})
    return result


def _compact_content(sift: CompactorService, content: Any) -> Any:
    """Compact string content; leave non-string (multimodal) untouched."""
    if isinstance(content, str):
        return _sift(sift, content)
    if isinstance(content, list):
        parts: list[Any] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                item = dict(item)
                item["text"] = _sift(sift, item["text"])
            parts.append(item)
        return parts
    return content


def _build_headers(request: Request) -> dict[str, str]:
    """Copy request headers, apply key override, drop hop-by-hop."""
    headers = dict(request.headers)
    override_key = os.environ.get("CONTEXT_SIFT_UPSTREAM_KEY")
    if override_key:
        headers["authorization"] = f"Bearer {override_key}"
    headers.pop("host", None)
    return headers


def create_app(
    compactor: CompactorService | None = None,
    http_client_factory: Callable[..., httpx.AsyncClient] | None = None,
) -> FastAPI:
    """Build the proxy FastAPI app.

    *compactor*: injected instance; if ``None``, a shared lazy instance is
    created on first request (keeps startup fast in tests).

    *http_client_factory*: callable that returns an ``httpx.AsyncClient``;
    injected by tests to fake upstream responses.
    """
    app = FastAPI(title="context-sift-proxy", version="1.0.0")
    state = app.state

    state.compactor = compactor
    state._http_client_factory = http_client_factory or httpx.AsyncClient

    async def _get_compactor() -> CompactorService:
        if state.compactor is None:
            state.compactor = CompactorService()
        return state.compactor

    async def _get_client() -> httpx.AsyncClient:
        base = os.environ.get("CONTEXT_SIFT_UPSTREAM_URL", _UPSTREAM_DEFAULT)
        timeout = httpx.Timeout(300.0, connect=20.0)
        return state._http_client_factory(
            base_url=base, timeout=timeout, headers={"Accept": "application/json"}
        )

    @app.get("/health")
    async def health() -> JSONResponse:
        c = await _get_compactor()
        return JSONResponse({"status": "ok", "compactor": c.running})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        body: dict[str, Any] = await request.json()
        sift = await _get_compactor()
        messages = body.get("messages")
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict) and "content" in msg:
                    msg["content"] = _compact_content(sift, msg["content"])

        client = await _get_client()
        headers = _build_headers(request)
        stream = body.get("stream", False)

        if stream:
            async def iterate() -> AsyncIterator[bytes]:
                async with client.stream(
                    "POST", str(request.url), headers=headers,
                    content=json.dumps(body).encode(),
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

            return StreamingResponse(iterate(), media_type="text/event-stream")

        content = json.dumps(body).encode()
        resp = await client.post(
            str(request.url), headers=headers, content=content
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> Response:
        body: dict[str, Any] = await request.json()
        sift = await _get_compactor()
        inp = body.get("input")
        if isinstance(inp, str):
            body["input"] = sift(inp)
        elif isinstance(inp, list):
            body["input"] = [_compact_content(sift, x) for x in inp]

        client = await _get_client()
        headers = _build_headers(request)
        content = json.dumps(body).encode()
        resp = await client.post(
            str(request.url), headers=headers, content=content
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def passthrough(request: Request, path: str) -> Response:
        """Forward everything else untouched."""
        client = await _get_client()
        headers = _build_headers(request)
        resp = await client.request(
            request.method,
            str(request.url),
            headers=headers,
            content=await request.body(),
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )

    return app
