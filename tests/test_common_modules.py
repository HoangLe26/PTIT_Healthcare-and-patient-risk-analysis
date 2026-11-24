import unittest
import numpy as np
import pandas as pd
from benchmark.thresholds import find_optimal_f2_threshold
from benchmark.evaluation import (
    compute_binary_metrics,
    build_test_prediction_table,
    extract_extreme_errors,
)
from benchmark.reporting import (
    format_comparison_table_dataset_a,
    format_comparison_table_dataset_b,
)


class TestCommonModules(unittest.TestCase):
    def setUp(self):
        # Dữ liệu mẫu giả lập
        self.y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        self.y_prob = np.array([0.1, 0.2, 0.4, 0.6, 0.3, 0.7, 0.8, 0.9])
        self.sample_ids = [f"sample_{i}" for i in range(8)]

    def test_find_optimal_f2_threshold_basic(self):
        result = find_optimal_f2_threshold(self.y_true, self.y_prob, return_curve=True)
        self.assertIn("best_threshold", result)
        self.assertIn("best_f2", result)
        self.assertGreater(result["best_f2"], 0.0)
        self.assertTrue(0.0 <= result["best_threshold"] <= 1.0)
        self.assertIn("curve_df", result)

    def test_tie_breaker_picks_higher_threshold(self):
        # Giả lập trường hợp 2 ngưỡng cho cùng F2
        # y_true = [0, 1]
        # y_prob = [0.1, 0.8]
        # Thresh 0.2: TP=1, FP=0, FN=0 -> F2=1.0
        # Thresh 0.5: TP=1, FP=0, FN=0 -> F2=1.0
        # Thresh 0.8: TP=1, FP=0, FN=0 -> F2=1.0
        # Tie-breaker phải chọn 0.8 (ngưỡng cao nhất)
        y_true = np.array([0, 1])
        y_prob = np.array([0.1, 0.8])
        result = find_optimal_f2_threshold(y_true, y_prob)
        self.assertEqual(result["best_threshold"], 0.8)
        self.assertAlmostEqual(result["best_f2"], 1.0)

    def test_compute_binary_metrics(self):
        metrics = compute_binary_metrics(self.y_true, self.y_prob, threshold=0.5)
        self.assertIn("roc_auc", metrics)
        self.assertIn("average_precision", metrics)
        self.assertIn("brier_score", metrics)
        self.assertIn("f2", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("precision", metrics)
        self.assertEqual(metrics["tp"] + metrics["fn"], 4)
        self.assertEqual(metrics["tn"] + metrics["fp"], 4)

    def test_build_test_prediction_table(self):
        df = build_test_prediction_table(self.sample_ids, self.y_true, self.y_prob, threshold=0.5)
        self.assertEqual(len(df), 8)
        self.assertListEqual(list(df.columns), ["sample_id", "y_true", "y_prob", "threshold", "y_pred", "result_type"])
        self.assertTrue(set(df["result_type"].unique()).issubset({"TP", "FP", "TN", "FN"}))

    def test_extract_extreme_errors(self):
        df = build_test_prediction_table(self.sample_ids, self.y_true, self.y_prob, threshold=0.5)
        errors = extract_extreme_errors(df)
        self.assertIn("worst_fn", errors)
        self.assertIn("worst_fp", errors)
        
        # Với thresh=0.5:
        # y_true=1 mà prob < 0.5 là sample_4 (prob=0.3) -> FN duy nhất
        # y_true=0 mà prob >= 0.5 là sample_3 (prob=0.6) -> FP duy nhất
        if errors["worst_fn"] is not None:
            self.assertEqual(errors["worst_fn"]["sample_id"], "sample_4")
        if errors["worst_fp"] is not None:
            self.assertEqual(errors["worst_fp"]["sample_id"], "sample_3")

    def test_reporting_formatters(self):
        mock_results = {
            "KNN (Tùng)": compute_binary_metrics(self.y_true, self.y_prob, threshold=0.5),
            "Logistic Regression (Hoàng)": compute_binary_metrics(self.y_true, self.y_prob, threshold=0.4),
            "CatBoost (Khải)": compute_binary_metrics(self.y_true, self.y_prob, threshold=0.3),
        }
        df_a = format_comparison_table_dataset_a(mock_results)
        df_b = format_comparison_table_dataset_b(mock_results)
        self.assertEqual(len(df_a), 3)
        self.assertEqual(len(df_b), 3)


if __name__ == "__main__":
    unittest.main()
