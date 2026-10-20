import unittest

from context_sift.checkpoint import keep_for_checkpoint
from context_sift.clause_dataset import split_clauses
from context_sift.rules import protected_spans, semantic_risk_spans


def _keep(text: str) -> dict[str, bool]:
    clauses = split_clauses(text)
    spans = protected_spans(text) + semantic_risk_spans(text)
    return {clause.text.strip(): flag for clause, flag in zip(clauses, keep_for_checkpoint(clauses, spans))}


class CheckpointProjectionTests(unittest.TestCase):
    def test_keeps_done_decision_and_next_drops_narration(self) -> None:
        text = (
            "Sure! I'd be happy to help. "
            "I implemented compare-and-swap in rotate_token.py. "
            "Decision: call rotate_atomically instead of a lock. "
            "Pending: run pytest tests/auth/test_rotation.py -q. "
            "I also reviewed unrelated files that turned out not to matter. "
        )
        kept = _keep(text)
        self.assertTrue(kept["I implemented compare-and-swap in rotate_token.py."])
        self.assertTrue(kept["Decision: call rotate_atomically instead of a lock."])
        self.assertTrue(kept["Pending: run pytest tests/auth/test_rotation.py -q."])
        self.assertFalse(kept["Sure!"])
        self.assertFalse(kept["I'd be happy to help."])
        self.assertFalse(kept["I also reviewed unrelated files that turned out not to matter."])

    def test_keeps_constraints_even_inside_narration_scope(self) -> None:
        text = "do not change the public API."
        self.assertTrue(_keep(text)["do not change the public API."])

    def test_keeps_failures_without_a_technical_anchor(self) -> None:
        text = "An error occurred during the migration. It was unrelated to the change."
        kept = _keep(text)
        self.assertTrue(kept["An error occurred during the migration."])
        self.assertFalse(kept["It was unrelated to the change."])

    def test_keeps_constraints_and_limits(self) -> None:
        for text in (
            "Only the maintainers can deploy to production.",
            "Restrict the API to read-only access.",
            "Cap the retry count at five.",
            "Somente os mantenedores podem fazer deploy.",
        ):
            self.assertTrue(_keep(text)[text], text)

    def test_drops_greeting_that_merely_contains_a_protected_word(self) -> None:
        # "question" is a protected anchor, but "Great question!" is still noise.
        text = "Great question!"
        self.assertFalse(_keep(text)["Great question!"])

    def test_drops_bare_label_but_keeps_status_label(self) -> None:
        text = "Here is the test run:\nPending: patch it\nResultados:\n"
        kept = _keep(text)
        self.assertFalse(kept["Here is the test run:"])
        self.assertTrue(kept["Pending: patch it"])
        self.assertFalse(kept["Resultados:"])

    def test_drops_omission_marker_and_pass_rows_keeps_summary(self) -> None:
        text = (
            "collected 31 items\n"
            "tests/a.py::test_ok PASSED\n"
            "  ...\n"
            "tests/b.py ....F\n"
            "1 failed, 30 passed in 2.0s\n"
        )
        kept = _keep(text)
        self.assertFalse(kept["collected 31 items"])
        self.assertFalse(kept["tests/a.py::test_ok PASSED"])
        self.assertFalse(kept["..."])
        self.assertTrue(kept["1 failed,"])
        self.assertTrue(kept["30 passed in 2.0s"])
        self.assertTrue(kept["tests/b.py ....F"])


if __name__ == "__main__":
    unittest.main()
