import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RuntimeCliTests(unittest.TestCase):
    text = "Keep exact path src/auth/session.ts and timeout 30 seconds. " * 8

    def test_reads_stdin_with_zero_arguments(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "compact_dataset.runtime_cli"],
            input=self.text,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("src/auth/session.ts", completed.stdout)
        self.assertLessEqual(len(completed.stdout), len(self.text))

    def test_reads_utf8_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.txt"
            path.write_text(self.text, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-m", "compact_dataset.runtime_cli", str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertIn("30", completed.stdout)


if __name__ == "__main__":
    unittest.main()
