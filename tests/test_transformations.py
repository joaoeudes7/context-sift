import unittest

from context_sift.transformations import (
    Operation,
    SourceUnit,
    Transformation,
    validate_transformations,
)


class TransformationTests(unittest.TestCase):
    def test_replace_can_shorten_equivalent_instruction(self) -> None:
        units = [SourceUnit("u1", "If JWT validation fails, stop authentication.", frozenset({"condition", "action"}))]
        transforms = [Transformation(
            Operation.REPLACE, ("u1",), "JWT validation fails: stop authentication.",
            frozenset({"condition", "action"}),
        )]
        self.assertEqual(validate_transformations(units, transforms, {"condition", "action"}), [])

    def test_rewrite_cannot_claim_fact_absent_from_sources(self) -> None:
        units = [SourceUnit("u1", "Authentication failed.", frozenset({"failure"}))]
        transforms = [Transformation(Operation.REPLACE, ("u1",), "Authentication failed.", frozenset({"cause"}))]
        self.assertIn("invented_fact_coverage:0", validate_transformations(units, transforms, set()))

    def test_protected_unit_cannot_be_dropped(self) -> None:
        units = [SourceUnit("u1", "trace=550e8400-e29b-41d4-a716-446655440000", frozenset(), True)]
        transforms = [Transformation(Operation.DROP, ("u1",), "", frozenset())]
        self.assertIn("invalid_drop:0", validate_transformations(units, transforms, set()))

    def test_merge_requires_all_mandatory_meaning(self) -> None:
        units = [
            SourceUnit("u1", "Token expired.", frozenset({"cause"})),
            SourceUnit("u2", "Renew token.", frozenset({"action"})),
        ]
        transforms = [Transformation(Operation.MERGE, ("u1", "u2"), "Token expired; renew token.", frozenset({"cause"}))]
        self.assertTrue(any(error.startswith("missing_required_facts") for error in validate_transformations(
            units, transforms, {"cause", "action"}
        )))


if __name__ == "__main__":
    unittest.main()
