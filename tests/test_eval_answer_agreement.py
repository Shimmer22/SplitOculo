import unittest

from scripts.eval_answer_agreement import (
    normalize_answer,
    is_answer_agreement,
    summarize_samples,
)


class TestAnswerAgreementUtils(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
