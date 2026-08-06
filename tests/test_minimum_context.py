import unittest

from context_sift.minimum_context import Candidate, select_minimum_context


class MinimumContextTests(unittest.TestCase):
    def test_selects_cheapest_evidence_covering_same_understanding(self) -> None:
        candidates = [
            Candidate("verbose", 20, frozenset({"cause", "action"})),
            Candidate("cause", 4, frozenset({"cause"})),
            Candidate("action", 3, frozenset({"action"})),
        ]
        self.assertEqual(
            select_minimum_context(candidates, {}, {"cause", "action"}),
            ["cause", "action"],
        )

    def test_keeps_protected_identifier_even_without_semantic_gain(self) -> None:
        candidates = [
            Candidate("core", 4, frozenset({"claim"})),
            Candidate("uuid", 1, frozenset(), protected=True),
        ]
        self.assertEqual(select_minimum_context(candidates, {}, {"claim"}), ["core", "uuid"])

    def test_fails_when_required_information_has_no_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "no evidence"):
            select_minimum_context([], {}, {"missing"})

    def test_optional_information_uses_marginal_gain_not_ratio(self) -> None:
        candidates = [
            Candidate("core", 4, frozenset({"claim"})),
            Candidate("useful", 2, frozenset({"limitation"})),
            Candidate("noise", 100, frozenset()),
        ]
        self.assertEqual(
            select_minimum_context(candidates, {"limitation": 1.0}, {"claim"}, min_optional_gain=0.4),
            ["core", "useful"],
        )
