import unittest

from context_sift.git_diff import compact_git_diff
from context_sift.msc import Compactor


class GitDiffTests(unittest.TestCase):
    def test_public_compactor_routes_diff_without_loading_model(self):
        diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,4 +1,4 @@\n old context\n-old\n+new\n trailing context\n far context\n"
        compactor = object.__new__(Compactor)

        output = compactor(diff)

        self.assertEqual(output, compact_git_diff(diff))
        self.assertIn("-old", output)
        self.assertIn("+new", output)

    def test_preserves_headers_changes_and_nearby_context(self) -> None:
        diff = "\n".join([
            "diff --git a/src/auth.ts b/src/auth.ts",
            "--- a/src/auth.ts",
            "+++ b/src/auth.ts",
            "@@ -1,12 +1,12 @@",
            *[f" context_{index}" for index in range(5)],
            "-    return token != null",
            "+    return token?.length >= 32",
            *[f" tail_{index}" for index in range(5)],
        ])
        output = compact_git_diff(diff, context_lines=2)
        for line in (
            "diff --git a/src/auth.ts b/src/auth.ts",
            "--- a/src/auth.ts",
            "+++ b/src/auth.ts",
            "@@ -1,12 +1,12 @@",
            "-    return token != null",
            "+    return token?.length >= 32",
        ):
            self.assertIn(line, output)
        self.assertNotIn(" context_0", output)
        self.assertIn(" context_3", output)
        self.assertIn(" tail_1", output)
        self.assertNotIn(" tail_4", output)

    def test_preserves_change_whitespace_exactly(self) -> None:
        line = "+\tif (token) {  return true; }"
        output = compact_git_diff("diff --git a/a b/a\n@@ -1 +1 @@\n-old\n" + line)
        self.assertIn(line, output.splitlines())

    def test_non_diff_is_unchanged(self) -> None:
        text = "ordinary context"
        self.assertEqual(compact_git_diff(text), text)


if __name__ == "__main__":
    unittest.main()
