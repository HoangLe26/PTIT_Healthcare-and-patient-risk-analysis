import pandas as pd
from typing import Dict, Any, List


def format_comparison_table_dataset_a(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Bảng đối chiếu 3 thuật toán cho Dataset A: Cleveland Heart Disease (N_test = 61).
    Chỉ số chính: ROC-AUC.
    Cần trình bày cả số ca TP, FP, TN, FN cụ thể do cỡ mẫu nhỏ.
    """
    rows = []
    for model_name, m in results.items():
        rows.append({
            "Mô hình": model_name,
            "ROC-AUC (Chính)": f"{m.get('roc_auc', 0.0):.4f}",
            "Average Precision": f"{m.get('average_precision', 0.0):.4f}",
            "Ngưỡng F2": f"{m.get('threshold', 0.5):.4f}",
            "F2 Score": f"{m.get('f2', 0.0):.4f}",
            "Recall / Sens": f"{m.get('recall', 0.0):.4f}",
            "Specificity": f"{m.get('specificity', 0.0):.4f}",
            "Precision": f"{m.get('precision', 0.0):.4f}",
            "Brier Score": f"{m.get('brier_score', 0.0):.4f}",
            "TP (ca)": m.get("tp", 0),
            "FP (ca)": m.get("fp", 0),
            "TN (ca)": m.get("tn", 0),
            "FN (ca)": m.get("fn", 0),
        })
    return pd.DataFrame(rows)


def format_comparison_table_dataset_b(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Bảng đối chiếu 3 thuật toán cho Dataset B: Diabetes 130-US Hospitals Readmission (N_test = 20.031).
    Chỉ số chính: Average Precision (AP).
    """
    rows = []
    for model_name, m in results.items():
        rows.append({
            "Mô hình": model_name,
            "Average Precision (Chính)": f"{m.get('average_precision', 0.0):.4f}",
            "ROC-AUC": f"{m.get('roc_auc', 0.0):.4f}",
            "Ngưỡng F2": f"{m.get('threshold', 0.5):.4f}",
            "F2 Score": f"{m.get('f2', 0.0):.4f}",
            "Recall / Sens": f"{m.get('recall', 0.0):.4f}",
            "Precision": f"{m.get('precision', 0.0):.4f}",
            "Specificity": f"{m.get('specificity', 0.0):.4f}",
            "Tỷ lệ gắn cờ": f"{m.get('flagged_rate', 0.0):.4f}",
            "Brier Score": f"{m.get('brier_score', 0.0):.4f}",
            "TP": m.get("tp", 0),
            "FP": m.get("fp", 0),
            "TN": m.get("tn", 0),
            "FN": m.get("fn", 0),
        })
    return pd.DataFrame(rows)
