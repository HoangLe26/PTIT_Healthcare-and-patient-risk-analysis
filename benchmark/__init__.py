"""Benchmark & Evaluation Toolkit for Healthcare ML Project.
Shared modules for all group members (Tùng, Hoàng, Khải).
"""

from .thresholds import find_optimal_f2_threshold
from .evaluation import (
    compute_binary_metrics,
    build_test_prediction_table,
    extract_extreme_errors,
)
from .reporting import (
    format_comparison_table_dataset_a,
    format_comparison_table_dataset_b,
)
from .data_loader import load_heart_data, load_readmission_data

__all__ = [
    "find_optimal_f2_threshold",
    "compute_binary_metrics",
    "build_test_prediction_table",
    "extract_extreme_errors",
    "format_comparison_table_dataset_a",
    "format_comparison_table_dataset_b",
    "load_heart_data",
    "load_readmission_data",
]
