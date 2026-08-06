import unittest

from context_sift.supervision import validate_supervision


SOURCE = "We test Model X. Dataset Y has 500 samples. Accuracy is 91%."


def valid_row() -> dict:
    return {
        "source": SOURCE,
        "central_message": "Model X reaches 91% accuracy on Dataset Y.",
        "units": [
            {"id": "u1", "start": 0, "end": 16, "text": SOURCE[0:16], "role": "core"},
            {"id": "u2", "start": 17, "end": 43, "text": SOURCE[17:43], "role": "mechanism", "protected": True},
            {"id": "u3", "start": 44, "end": len(SOURCE), "text": SOURCE[44:], "role": "evidence"},
        ],
        "target_unit_ids": ["u1", "u2", "u3"],
        "facts": [
            {"id": "f1", "claim": "Model X accuracy is 91%", "evidence_unit_ids": ["u1", "u3"], "central": True},
            {"id": "f2", "claim": "Dataset Y has 500 samples", "evidence_unit_ids": ["u2"], "required": True},
        ],
        "verification_questions": [
            {"question": "Which dataset and sample count?", "answer_fact_ids": ["f2"]},
        ],
    }


class SupervisionTests(unittest.TestCase):
    def test_accepts_grounded_reproducible_example(self) -> None:
        self.assertEqual(validate_supervision(valid_row()), [])

    def test_rejects_missing_protected_unit_and_required_fact(self) -> None:
        row = valid_row()
        row["target_unit_ids"] = ["u1", "u3"]
        errors = validate_supervision(row)
        self.assertTrue(any(error.startswith("missing_protected_units") for error in errors))
        self.assertTrue(any(error.startswith("required_fact_not_selected") for error in errors))
        self.assertTrue(any(error.startswith("unanswerable_verification_question") for error in errors))

    def test_rejects_evidence_not_matching_source_offsets(self) -> None:
        row = valid_row()
        row["units"][1]["text"] = "invented dataset"
        self.assertIn("unit_evidence_mismatch:u2", validate_supervision(row))

    def test_rejects_central_fact_without_selected_evidence(self) -> None:
        row = valid_row()
        row["units"][0]["role"] = "detail"
        row["target_unit_ids"] = ["u2"]
        self.assertIn("required_fact_not_selected:f1", validate_supervision(row))

    def test_rejects_selected_unit_without_factual_gain(self) -> None:
        row = valid_row()
        row["units"].append({
            "id": "u4", "start": 0, "end": 16, "text": SOURCE[0:16], "role": "detail",
        })
        row["target_unit_ids"].append("u4")
        self.assertTrue(any(error.startswith("selected_without_fact") for error in validate_supervision(row)))

    def test_rejects_non_atomic_fact_with_too_many_evidence_units(self) -> None:
        row = valid_row()
        row["facts"][0]["evidence_unit_ids"] = ["u1"] * 9
        self.assertIn("fact_evidence_not_atomic:f1", validate_supervision(row))


if __name__ == "__main__":
    unittest.main()
