import io
import json
import unittest

from context_sift.serve import serve


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


if __name__ == "__main__":
    unittest.main()
