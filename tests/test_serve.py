import io
import json
import os
import unittest

from context_sift.serve import acquire_singleton, serve, serve_socket, socket_is_live

from pathlib import Path
import tempfile
import asyncio


class _FakeService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str) -> str:
        self.calls.append(text)
        return text[:5]


def _run(lines: list[str]) -> tuple[list[dict], _FakeService]:
    service = _FakeService()
    out = io.StringIO()
    serve(io.StringIO("\n".join(lines)), out, service)
    responses = [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]
    return responses, service


class ServeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["CONTEXT_SIFT_GAIN_FILE"] = str(Path(self._tmp.name) / "gain.jsonl")

    def tearDown(self) -> None:
        os.environ.pop("CONTEXT_SIFT_GAIN_FILE", None)
        self._tmp.cleanup()

    def test_ready_line_then_roundtrip(self) -> None:
        responses, service = _run(['{"id":1,"text":"hello world"}'])

        self.assertEqual(responses[0], {"ready": True})
        self.assertEqual(responses[1]["id"], 1)
        self.assertEqual(responses[1]["text"], "hello")
        self.assertIn("ms", responses[1])
        self.assertEqual(responses[1]["ratio"], round(5 / 11, 4))
        self.assertEqual(service.calls, ["hello world"])

    def test_ids_correlated_across_lines(self) -> None:
        responses, _ = _run(['{"id":"a","text":"one"}', '{"id":2,"text":"two"}'])

        self.assertEqual([r["id"] for r in responses[1:]], ["a", 2])

    def test_bad_line_reports_error_and_continues(self) -> None:
        responses, service = _run(["not json", '{"id":9,"text":"keepme"}'])

        self.assertIn("error", responses[1])
        self.assertIsNone(responses[1]["id"])
        self.assertEqual(responses[2]["id"], 9)
        self.assertEqual(responses[2]["text"], "keepm")
        self.assertEqual(service.calls, ["keepme"])

    def test_missing_text_is_an_error_not_a_crash(self) -> None:
        responses, _ = _run(['{"id":3}', '{"id":4,"text":"ok"}'])

        self.assertIn("error", responses[1])
        self.assertEqual(responses[2]["text"], "ok")

    def test_empty_lines_are_skipped(self) -> None:
        responses, service = _run(["", "   ", '{"id":1,"text":"x"}'])

        self.assertEqual(len(responses), 2)
        self.assertEqual(service.calls, ["x"])

    def test_records_gain_from_meta(self) -> None:
        _run(['{"id":1,"text":"hello world","meta":{"source":"opencode","cwd":"/p","in_chars":100}}'])

        ledger = Path(os.environ["CONTEXT_SIFT_GAIN_FILE"]).read_text().splitlines()
        entry = json.loads(ledger[-1])
        self.assertEqual(entry["source"], "opencode")
        self.assertEqual(entry["cwd"], "/p")
        self.assertEqual(entry["in"], 100)
        self.assertEqual(entry["out"], 5)


class SocketServeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["CONTEXT_SIFT_GAIN_FILE"] = str(Path(self._tmp.name) / "gain.jsonl")

    def tearDown(self) -> None:
        os.environ.pop("CONTEXT_SIFT_GAIN_FILE", None)
        self._tmp.cleanup()

    async def test_roundtrip_then_idle_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s.sock"
            service = _FakeService()
            server = asyncio.create_task(serve_socket(path, service, idle_timeout=0.2))
            for _ in range(200):
                if path.exists():
                    break
                await asyncio.sleep(0.01)

            reader, writer = await asyncio.open_unix_connection(str(path))
            writer.write(b'{"id":1,"text":"hello world"}\n')
            await writer.drain()
            response = json.loads(await reader.readline())

            self.assertEqual(response["id"], 1)
            self.assertEqual(response["text"], "hello")
            self.assertEqual(service.calls, ["hello world"])

            writer.close()
            await asyncio.wait_for(server, timeout=5)
            self.assertFalse(path.exists())


class SingletonTests(unittest.TestCase):
    def test_socket_is_live_false_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(socket_is_live(Path(directory) / "nope.sock"))

    def test_acquire_singleton_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s.sock"
            first = acquire_singleton(path)
            self.assertIsNotNone(first)
            self.assertIsNone(acquire_singleton(path))  # already held
            os.close(first)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()