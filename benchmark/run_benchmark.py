import os
import pandas as pd
from typing import Dict, Any, Optional

from benchmark.evaluation import compute_binary_metrics
from benchmark.reporting import (
    format_comparison_table_dataset_a,
    format_comparison_table_dataset_b,
)


def load_model_test_metrics(pred_csv_path: str) -> Optional[Dict[str, Any]]:
    """Đọc file test_predictions.csv của một thuật toán và tính lại bộ metrics chuẩn."""
    if not os.path.exists(pred_csv_path):
        return None
    df = pd.read_csv(pred_csv_path)
    thresh = float(df["threshold"].iloc[0])
    y_true = df["y_true"].values
    y_prob = df["y_prob"].values
    metrics = compute_binary_metrics(y_true, y_prob, threshold=thresh)
    return metrics


def generate_benchmark_tables(output_dir: str = "benchmark/results"):
    """
    Tự động quét kết quả dự đoán của 3 mô hình:
    - algorithm/KNN/
    - algorithm/Logistic/
    - algorithm/CatBoost/
    Sau đó xuất ra 2 bảng đối chiếu Markdown và CSV chuẩn cho báo cáo nhóm.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    models = {
        "KNN (Tùng)": "algorithm/KNN/artifacts",
        "Logistic Regression (Hoàng)": "algorithm/Logistic/artifacts",
        "CatBoost (Khải)": "algorithm/CatBoost/artifacts",
    }
    
    # 1. Dataset A: Heart Disease
    results_a = {}
    for name, path in models.items():
        pred_file = os.path.join(path, "heart", "test_predictions.csv")
        metrics = load_model_test_metrics(pred_file)
        if metrics:
            results_a[name] = metrics
            
    if results_a:
        df_a = format_comparison_table_dataset_a(results_a)
        df_a.to_csv(os.path.join(output_dir, "benchmark_dataset_a_heart.csv"), index=False)
        print("\n" + "=" * 60)
        print("BẢNG ĐỐI CHIẾU 3 THUẬT TOÁN: DATASET A (HEART DISEASE)")
        print("=" * 60)
        try:
            print(df_a.to_markdown(index=False))
        except (ImportError, Exception):
            print(df_a.to_string(index=False))
    else:
        print("Chưa tìm thấy file test_predictions.csv cho Dataset A.")
        
    # 2. Dataset B: Diabetes Readmission
    results_b = {}
    for name, path in models.items():
        pred_file = os.path.join(path, "readmission", "test_predictions.csv")
        metrics = load_model_test_metrics(pred_file)
        if metrics:
            results_b[name] = metrics
            
    if results_b:
        df_b = format_comparison_table_dataset_b(results_b)
        df_b.to_csv(os.path.join(output_dir, "benchmark_dataset_b_readmission.csv"), index=False)
        print("\n" + "=" * 60)
        print("BẢNG ĐỐI CHIẾU 3 THUẬT TOÁN: DATASET B (DIABETES READMISSION)")
        print("=" * 60)
        try:
            print(df_b.to_markdown(index=False))
        except (ImportError, Exception):
            print(df_b.to_string(index=False))
    else:
        print("Chưa tìm thấy file test_predictions.csv cho Dataset B.")


if __name__ == "__main__":
    generate_benchmark_tables()
