import unittest

from compact_dataset.msc_training import split_grouped_rows, training_windows


class FakeTokenizer:
    def encode(self, text: str, **kwargs) -> list[int]:
        return [1, *range(4, 4 + len(text.split())), 2]


class MSCTrainingTests(unittest.TestCase):
    def test_group_split_keeps_same_article_together(self) -> None:
        rows = [
            {"id": "a:0", "group_id": "Q1"}, {"id": "a:1", "group_id": "Q1"},
            {"id": "b:0", "group_id": "Q2"}, {"id": "c:0", "group_id": "Q3"},
        ]
        train, valid = split_grouped_rows(rows, 0.34, 7)
        train_groups = {row["group_id"] for row in train}
        valid_groups = {row["group_id"] for row in valid}
        self.assertFalse(train_groups & valid_groups)

    def test_reads_msc_target_unit_ids(self) -> None:
        row = {
            "id": "x", "source": "First. Second.",
            "units": [{"id": "u0", "text": "First."}, {"id": "u1", "text": "Second."}],
            "target_unit_ids": ["u1"],
        }
        window = training_windows(row, FakeTokenizer())[0]
        self.assertEqual(window.labels, [0, 1])

    def test_splits_long_documents_without_dropping_units(self) -> None:
        row = {
            "id": "x", "source": "ignored",
            "units": [{"text": str(index)} for index in range(5)],
            "labels": [0, 1, 0, 1, 1],
        }
        windows = training_windows(row, FakeTokenizer(), max_units=2)
        self.assertEqual([label for window in windows for label in window.labels], row["labels"])


if __name__ == "__main__":
    unittest.main()
