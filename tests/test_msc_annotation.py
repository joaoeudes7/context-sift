import unittest

from context_sift.msc_annotation import materialize_annotation, source_chunks, teacher_units


class MSCAnnotationTests(unittest.TestCase):
    def test_long_source_chunks_are_lossless(self) -> None:
        source = ("alpha beta gamma\n" * 100).strip()
        chunks = source_chunks(source, 240)
        self.assertEqual("".join(chunks), source)
        self.assertTrue(all(len(chunk) <= 240 for chunk in chunks))

    def test_materializes_teacher_ids_with_local_offsets(self) -> None:
        source = "Token expired. Renew token now. Repeated background detail."
        units = teacher_units(source)
        annotation = {
            "central_message": "Expired token requires renewal.",
            "unit_roles": [
                {"unit_id": units[0]["id"], "role": "core"},
                {"unit_id": units[1]["id"], "role": "action"},
                {"unit_id": units[2]["id"], "role": "noise"},
            ],
            "target_unit_ids": [units[0]["id"], units[1]["id"]],
            "facts": [{
                "id": "f1", "claim": "Token expired and must be renewed",
                "evidence_unit_ids": [units[0]["id"], units[1]["id"]],
                "central": True, "required": True,
            }],
            "verification_questions": [],
        }
        row = materialize_annotation(source, annotation)
        self.assertEqual(row["units"][0]["text"], source[:14])

    def test_unmentioned_unit_defaults_to_noise(self) -> None:
        source = "First fact. Second fact."
        annotation = {
            "central_message": "Facts.",
            "unit_roles": [{"unit_id": "u0", "role": "core"}],
            "target_unit_ids": ["u0"],
            "facts": [{"id": "f1", "claim": "First", "evidence_unit_ids": ["u0"], "central": True}],
        }
        row = materialize_annotation(source, annotation)
        self.assertEqual(row["units"][1]["role"], "noise")

    def test_accepts_compact_role_mapping(self) -> None:
        source = "Token expired. Renew token."
        annotation = {
            "central_message": "Renew expired token.",
            "unit_roles": {"u0": "core", "u1": "action"},
            "target_unit_ids": ["u0", "u1"],
            "facts": [{
                "id": "f1", "claim": "Expired token needs renewal",
                "evidence_unit_ids": ["u0", "u1"], "central": True, "required": True,
            }],
        }
        self.assertEqual(len(materialize_annotation(source, annotation)["units"]), 2)

    def test_normalizes_teacher_role_aliases(self) -> None:
        source = "Background context."
        annotation = {
            "central_message": "Context.",
            "unit_roles": {"u0": "central_claim"},
            "facts": [{"id": "f1", "claim": "Context", "evidence_unit_ids": ["u0"], "central": True}],
        }
        self.assertEqual(materialize_annotation(source, annotation)["units"][0]["role"], "core")


if __name__ == "__main__":
    unittest.main()
