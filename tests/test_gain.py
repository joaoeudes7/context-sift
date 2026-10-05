import io
import json
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from context_sift.gain import gain_cli, read_entries, record, store_path


class GainTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._cwd = os.getcwd()
        os.environ["CONTEXT_SIFT_GAIN_FILE"] = str(Path(self._tmp.name) / "gain.jsonl")

    def tearDown(self) -> None:
        os.chdir(self._cwd)
        os.environ.pop("CONTEXT_SIFT_GAIN_FILE", None)
        self._tmp.cleanup()

    def _seed(self) -> None:
        record({"source": "opencode", "cwd": "/proj/a", "in": 1000, "out": 200, "ms": 10})
        record({"source": "proxy", "cwd": "/proj/b", "in": 500, "out": 500, "ms": 5})

    def test_store_path_honours_env(self) -> None:
        self.assertEqual(str(store_path()), os.environ["CONTEXT_SIFT_GAIN_FILE"])

    def test_record_appends_readable_lines(self) -> None:
        self._seed()
        entries = read_entries()
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["source"], "opencode")
        self.assertIn("ts", entries[0])

    def test_json_summary_math(self) -> None:
        self._seed()
        out = io.StringIO()
        with redirect_stdout(out):
            gain_cli(["--json"])
        data = json.loads(out.getvalue())
        self.assertEqual(data["summary"]["requests"], 2)
        self.assertEqual(data["summary"]["input_tokens"], 375)  # (1000+500)/4
        self.assertEqual(data["summary"]["output_tokens"], 175)  # (200+500)/4
        self.assertEqual(data["summary"]["saved_tokens"], 200)  # (800)/4
        self.assertEqual(data["summary"]["saved_pct"], 53.3)

    def test_project_filter(self) -> None:
        here = os.getcwd()
        record({"source": "opencode", "cwd": here, "in": 1000, "out": 200})
        record({"source": "opencode", "cwd": "/somewhere/else", "in": 1000, "out": 200})
        out = io.StringIO()
        with redirect_stdout(out):
            gain_cli(["--json", "--project"])
        data = json.loads(out.getvalue())
        self.assertEqual(data["summary"]["requests"], 1)

    def test_oneline(self) -> None:
        self._seed()
        out = io.StringIO()
        with redirect_stdout(out):
            gain_cli(["--oneline"])
        line = out.getvalue().strip()
        self.assertTrue(line.startswith("sift "))
        self.assertIn("saved", line)
        self.assertIn("2 reqs", line)

    def test_days_window_excludes_old(self) -> None:
        record({"source": "opencode", "cwd": "/p", "in": 100, "out": 10, "ts": time.time()})
        record({"source": "opencode", "cwd": "/p", "in": 100, "out": 10, "ts": time.time() - 10 * 86400})
        out = io.StringIO()
        with redirect_stdout(out):
            gain_cli(["--json", "--days", "1"])
        data = json.loads(out.getvalue())
        self.assertEqual(data["summary"]["requests"], 1)

    def test_spark_outputs_sparkline(self) -> None:
        self._seed()
        out = io.StringIO()
        with redirect_stdout(out):
            gain_cli(["--spark"])
        self.assertIn("Last 1d:", out.getvalue())

    def test_reset_requires_yes_and_clears(self) -> None:
        self._seed()
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(gain_cli(["--reset"]), 1)
        self.assertEqual(len(read_entries()), 2)

        with redirect_stdout(out):
            self.assertEqual(gain_cli(["--reset", "--yes"]), 0)
        self.assertEqual(read_entries(), [])


if __name__ == "__main__":
    unittest.main()
