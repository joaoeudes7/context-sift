"""Tests for context-sift proxy (requires fastapi + uvicorn)."""

from __future__ import annotations

import json
import unittest
from collections.abc import AsyncIterator
from typing import Any

import httpx

from context_sift.proxy import _compact_content, create_app

HAS_FASTAPI = True
try:
    from fastapi.testclient import TestClient
except ImportError:
    HAS_FASTAPI = False


class _StubCompactor:
    """Lightweight compactor for proxy tests — no model load needed."""

    def __init__(self):
        self.running = True

    def __call__(self, text: str) -> str:
        words = text.split()
        if len(words) <= 3:
            return text
        return " ".join(words[:3])


class CompactContentTests(unittest.TestCase):
    """Unit tests for _compact_content (no FastAPI required)."""

    def test_string_is_compacted(self):
        def fake_sift(text: str) -> str:
            return "COMPACTED"

        result = _compact_content(fake_sift, "long verbose sentence here")
        self.assertEqual(result, "COMPACTED")

    def test_list_with_text_items(self):
        def fake_sift(text: str) -> str:
            return text.upper()

        content = [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
        ]
        result = _compact_content(fake_sift, content)
        self.assertEqual(result[0]["text"], "HELLO")
        self.assertEqual(result[1], content[1])  # image untouched

    def test_non_string_passthrough(self):
        def fake_sift(text: str) -> str:
            return text.upper()

        self.assertEqual(_compact_content(fake_sift, 42), 42)
        self.assertIsNone(_compact_content(fake_sift, None))


def _mock_upstream_response(
    status: int = 200, json_body: Any = None, headers: dict | None = None
) -> httpx.Response:
    content = json.dumps(json_body or {}).encode() if json_body is not None else b""
    return httpx.Response(status_code=status, content=content, headers=headers or {})


class _FakeTransport(httpx.AsyncBaseTransport):
    def __init__(self, response: httpx.Response):
        self._response = response
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._response


class _FakeSSEStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class _FakeStreamingTransport(httpx.AsyncBaseTransport):
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            status_code=200,
            stream=_FakeSSEStream(self._chunks),
            headers={"content-type": "text/event-stream"},
        )


class _FakeUpstreamTransport(httpx.AsyncBaseTransport):
    def __init__(self, response: httpx.Response):
        self._response = response
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._response


def _client_factory(transport: httpx.AsyncBaseTransport):
    def factory(**kwargs: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=transport, **kwargs)
    return factory


@unittest.skipUnless(HAS_FASTAPI, "fastapi not installed")
class ProxyHealthTests(unittest.TestCase):
    def test_health_returns_ok(self):
        compactor = _StubCompactor()
        app = create_app(compactor=compactor)
        client = TestClient(app)
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["compactor"])

    def test_passthrough_forwards_unknown_routes(self):
        compactor = _StubCompactor()
        transport = _FakeUpstreamTransport(_mock_upstream_response(200, {"ok": True}))
        app = create_app(compactor=compactor, http_client_factory=_client_factory(transport))
        client = TestClient(app)
        resp = client.get("/v1/models")
        self.assertEqual(resp.status_code, 200)


@unittest.skipUnless(HAS_FASTAPI, "fastapi not installed")
class ProxyChatCompletionsTests(unittest.TestCase):
    def test_compacts_message_content(self):
        compactor = _StubCompactor()
        upstream = _mock_upstream_response(200, {"choices": [{"message": {"content": "hi"}}]})
        transport = _FakeTransport(upstream)
        app = create_app(compactor=compactor, http_client_factory=_client_factory(transport))
        client = TestClient(app)

        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant with lots of context."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
        }
        resp = client.post("/v1/chat/completions", json=body)
        self.assertEqual(resp.status_code, 200)

        self.assertEqual(len(transport.requests), 1)
        upstream_body = json.loads(transport.requests[0].content)
        sys_msg = upstream_body["messages"][0]["content"]
        self.assertLess(len(sys_msg), len("You are a helpful assistant with lots of context."))
        self.assertEqual(upstream_body["model"], "gpt-4")

    def test_multimodal_content_preserves_images(self):
        compactor = _StubCompactor()
        upstream = _mock_upstream_response(200, {"choices": []})
        transport = _FakeTransport(upstream)
        app = create_app(compactor=compactor, http_client_factory=_client_factory(transport))
        client = TestClient(app)

        body = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please describe this image in great detail"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    ],
                }
            ],
        }
        resp = client.post("/v1/chat/completions", json=body)
        self.assertEqual(resp.status_code, 200)

        upstream_body = json.loads(transport.requests[0].content)
        content = upstream_body["messages"][0]["content"]
        self.assertLess(len(content[0]["text"]), len("Please describe this image in great detail"))
        self.assertEqual(content[1]["type"], "image_url")

    def test_stream_sse_passthrough(self):
        compactor = _StubCompactor()
        sse_chunks = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        transport = _FakeStreamingTransport(sse_chunks)
        app = create_app(compactor=compactor, http_client_factory=_client_factory(transport))
        client = TestClient(app)

        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": True,
        }
        resp = client.post("/v1/chat/completions", json=body)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/event-stream", resp.headers["content-type"])
        text = resp.text
        self.assertIn('"content":"Hello"', text)
        self.assertIn("[DONE]", text)

    def test_auth_override(self):
        import os

        compactor = _StubCompactor()
        upstream = _mock_upstream_response(200, {"choices": []})
        transport = _FakeTransport(upstream)
        app = create_app(compactor=compactor, http_client_factory=_client_factory(transport))
        client = TestClient(app)

        os.environ["CONTEXT_SIFT_UPSTREAM_KEY"] = "test-key-123"
        try:
            body = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
            resp = client.post("/v1/chat/completions", json=body)
            self.assertEqual(resp.status_code, 200)
            auth_header = transport.requests[0].headers.get("authorization")
            self.assertEqual(auth_header, "Bearer test-key-123")
        finally:
            del os.environ["CONTEXT_SIFT_UPSTREAM_KEY"]


@unittest.skipUnless(HAS_FASTAPI, "fastapi not installed")
class ProxyEmbeddingsTests(unittest.TestCase):
    def test_compacts_string_input(self):
        compactor = _StubCompactor()
        upstream = _mock_upstream_response(200, {"data": [{"embedding": [0.1, 0.2]}]})
        transport = _FakeTransport(upstream)
        app = create_app(compactor=compactor, http_client_factory=_client_factory(transport))
        client = TestClient(app)

        body = {
            "model": "text-embedding-3-small",
            "input": "A very long verbose sentence about the weather",
        }
        resp = client.post("/v1/embeddings", json=body)
        self.assertEqual(resp.status_code, 200)

        upstream_body = json.loads(transport.requests[0].content)
        self.assertLess(len(upstream_body["input"]), len(body["input"]))

    def test_compacts_list_input(self):
        compactor = _StubCompactor()
        upstream = _mock_upstream_response(200, {"data": []})
        transport = _FakeTransport(upstream)
        app = create_app(compactor=compactor, http_client_factory=_client_factory(transport))
        client = TestClient(app)

        body = {
            "model": "text-embedding-3-small",
            "input": [
                "Long text about topic A with lots of verbose words",
                "Long text about topic B with lots of verbose words",
            ],
        }
        resp = client.post("/v1/embeddings", json=body)
        self.assertEqual(resp.status_code, 200)

        upstream_body = json.loads(transport.requests[0].content)
        for i, orig in enumerate(body["input"]):
            self.assertLess(len(upstream_body["input"][i]), len(orig))


if __name__ == "__main__":
    unittest.main()
