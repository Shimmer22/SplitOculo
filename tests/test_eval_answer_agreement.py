import unittest

from scripts.eval_answer_agreement import (
    extract_mcq_answer,
    score_answer_row,
    normalize_answer,
    is_answer_agreement,
    summarize_samples,
)


class TestAnswerAgreementUtils(unittest.TestCase):
    def test_extract_mcq_answer_prefers_option_letter(self):
        options = {"A": "cat", "B": "dog", "C": "bird", "D": "fish"}
        parsed = extract_mcq_answer("The answer is (C).", options=options)
        self.assertEqual(parsed["choice"], "C")
        self.assertEqual(parsed["answer"], "bird")

    def test_extract_mcq_answer_falls_back_to_option_text(self):
        options = {"A": "cat", "B": "dog", "C": "bird", "D": "fish"}
        parsed = extract_mcq_answer("I think it is a dog.", options=options)
        self.assertEqual(parsed["choice"], "B")
        self.assertEqual(parsed["answer"], "dog")

    def test_score_answer_row_returns_binary_losses(self):
        row = score_answer_row(
            label="B",
            teacher_output="B",
            student_output="A",
            options={"A": "cat", "B": "dog", "C": "bird", "D": "fish"},
        )
        self.assertTrue(row["teacher_correct"])
        self.assertFalse(row["student_correct"])
        self.assertEqual(row["teacher_label_loss"], 0.0)
        self.assertEqual(row["student_label_loss"], 1.0)
        self.assertEqual(row["distill_loss"], 1.0)

    def test_normalize_answer_removes_punctuation_and_case(self):
        self.assertEqual(normalize_answer("  The, QUICK-brown fox!! "), "the quick brown fox")

    def test_is_answer_agreement_uses_normalized_text(self):
        self.assertTrue(is_answer_agreement("A: 12.0", "a 12 0"))
        self.assertFalse(is_answer_agreement("cat", "dog"))

    def test_summarize_samples_aggregates_budget_and_subset_metrics(self):
        rows = [
            {
                "sample_id": "s1",
                "prediction": "A",
                "reference": "A",
                "payload_bytes": 1000,
                "subsets": ["ocr", "digits"],
            },
            {
                "sample_id": "s2",
                "prediction": "wrong",
                "reference": "B",
                "payload_bytes": 3000,
                "subsets": ["ocr"],
            },
            {
                "sample_id": "s3",
                "prediction": "Blue",
                "reference": "blue",
                "payload_bytes": 2000,
                "subsets": ["colors"],
            },
        ]

        summary = summarize_samples(rows)

        self.assertEqual(summary["num_samples"], 3)
        self.assertAlmostEqual(summary["agreement_rate"], 2 / 3)
        self.assertAlmostEqual(summary["avg_payload_bytes"], 2000.0)
        self.assertAlmostEqual(summary["avg_payload_bits"], 16000.0)

        self.assertEqual(summary["subsets"]["ocr"]["num_samples"], 2)
        self.assertAlmostEqual(summary["subsets"]["ocr"]["agreement_rate"], 0.5)
        self.assertEqual(summary["subsets"]["digits"]["num_samples"], 1)
        self.assertAlmostEqual(summary["subsets"]["digits"]["agreement_rate"], 1.0)
        self.assertEqual(summary["subsets"]["colors"]["num_samples"], 1)
        self.assertAlmostEqual(summary["subsets"]["colors"]["agreement_rate"], 1.0)

    def test_summarize_samples_aggregates_accuracy_and_losses(self):
        rows = [
            {
                "payload_bytes": 1000,
                "subsets": ["ai2d"],
                "teacher_correct": True,
                "student_correct": True,
                "teacher_label_loss": 0.0,
                "student_label_loss": 0.0,
                "distill_loss": 0.0,
            },
            {
                "payload_bytes": 1000,
                "subsets": ["ai2d"],
                "teacher_correct": True,
                "student_correct": False,
                "teacher_label_loss": 0.0,
                "student_label_loss": 1.0,
                "distill_loss": 1.0,
            },
        ]
        summary = summarize_samples(rows)
        self.assertAlmostEqual(summary["teacher_accuracy"], 1.0)
        self.assertAlmostEqual(summary["student_accuracy"], 0.5)
        self.assertAlmostEqual(summary["avg_teacher_label_loss"], 0.0)
        self.assertAlmostEqual(summary["avg_student_label_loss"], 0.5)
        self.assertAlmostEqual(summary["avg_distill_loss"], 0.5)


if __name__ == "__main__":
    unittest.main()
