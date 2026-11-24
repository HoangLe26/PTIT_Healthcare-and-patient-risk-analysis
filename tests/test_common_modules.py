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
from benchmark.conformal import (
    calibrate,
    prediction_sets,
    assign_conformal_prediction_labels,
    summarize,
)


class TestCommonModules(unittest.TestCase):
    def setUp(self):
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
        self.assertEqual(metrics["tp"] + metrics["fn"], 4)
        self.assertEqual(metrics["tn"] + metrics["fp"], 4)

    def test_conformal_prediction_and_ambiguous_labels(self):
        # Kiểm tra calibration và sinh nhãn phân vân với dữ liệu có ca khó phân loại
        y_cal = np.array([0] * 10 + [1] * 10)
        # Bao gồm cả một số ca khó (scores lớn) để q_90 >= 0.7
        p_cal = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                          0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
        q_90 = calibrate(y_cal, p_cal, 0.90)
        self.assertTrue(0.0 <= q_90 <= 1.0)
        
        # Mẫu test: p=0.5 (không chắc chắn, rơi vào giữa [1-q, q] -> Phân vân)
        p_test = np.array([0.01, 0.50, 0.99])
        set_strings, labels = assign_conformal_prediction_labels(p_test, q_90, ambiguous_name="Phân vân")
        self.assertEqual(len(labels), 3)
        self.assertEqual(labels[0], "0")
        self.assertEqual(labels[2], "1")
        self.assertEqual(labels[1], "Phân vân")

    def test_build_test_prediction_table_with_conformal(self):
        sets = ["{0}"] * 8
        labels = ["0"] * 8
        df = build_test_prediction_table(
            self.sample_ids, self.y_true, self.y_prob, threshold=0.5,
            conformal_sets_90=sets, conformal_labels_90=labels
        )
        self.assertIn("conformal_set_90", df.columns)
        self.assertIn("conformal_decision_90", df.columns)
        self.assertEqual(len(df), 8)


if __name__ == "__main__":
    unittest.main()
