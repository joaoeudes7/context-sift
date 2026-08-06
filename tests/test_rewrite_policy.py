import unittest

from context_sift.rewrite_policy import validate_rewrite


class RewritePolicyTests(unittest.TestCase):
    def test_natural_text_may_rewrite_without_losing_facts(self) -> None:
        result = validate_rewrite(
            "Please ensure API latency stays below 200 ms before 15/08/2026.",
            "API latency below 200 ms by 15/08/2026.",
            "text",
        )
        self.assertTrue(result.allowed)

    def test_rejects_lost_or_invented_critical_value(self) -> None:
        lost = validate_rewrite("Deadline 15/08/2026.", "Deadline soon.", "text")
        invented = validate_rewrite("Deadline 15/08/2026.", "Deadline 16/08/2026.", "text")
        self.assertFalse(lost.allowed)
        self.assertFalse(invented.allowed)

    def test_code_and_diff_must_remain_copy_only(self) -> None:
        source = "diff --git a/a.py b/a.py\n-old()\n+new()"
        self.assertTrue(validate_rewrite(source, source, "git_diff").allowed)
        self.assertFalse(validate_rewrite(source, source.replace("new", "better"), "git_diff").allowed)

    def test_binary_rewrite_is_never_allowed(self) -> None:
        self.assertFalse(validate_rewrite("89504e47", "PNG image", "binary").allowed)

    def test_paths_must_remain_exact(self) -> None:
        result = validate_rewrite(
            "Edit tests/auth/session.test.ts after review.",
            "Edit session tests after review.",
            "text",
        )
        self.assertFalse(result.allowed)


if __name__ == "__main__":
    unittest.main()
