import os
import pandas as pd
from typing import Dict, Any, Optional, Tuple

from benchmark.evaluation import compute_binary_metrics
from benchmark.reporting import (
    format_comparison_table_dataset_a,
    format_comparison_table_dataset_b,
)


def load_model_predictions_and_metrics(
    pred_csv_path: str,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Đọc file test_predictions.csv của một thuật toán và tính lại bộ metrics chuẩn."""
    if not os.path.exists(pred_csv_path):
        return None, None
    df = pd.read_csv(pred_csv_path)
    thresh = float(df["threshold"].iloc[0])
    y_true = df["y_true"].values
    y_prob = df["y_prob"].values
    metrics = compute_binary_metrics(y_true, y_prob, threshold=thresh)
    return df, metrics


def verify_sample_alignment(
    pred_dfs: Dict[str, pd.DataFrame], dataset_name: str, *, verbose: bool = True,
) -> bool:
    """
    Kiểm tra mẫu đánh giá (sample_id và y_true) có khớp 100% giữa các thuật toán hay không.
    Đáp ứng yêu cầu kiểm tra khớp mẫu trước khi ghép bài trong QUY_UOC_VA_PHAN_CONG_NHOM.md.
    """
    if len(pred_dfs) < 2:
        return True
    names = list(pred_dfs.keys())
    ref_name = names[0]
    ref_df = pred_dfs[ref_name].sort_values("sample_id").reset_index(drop=True)
    for other_name in names[1:]:
        other_df = pred_dfs[other_name].sort_values("sample_id").reset_index(drop=True)
        if len(ref_df) != len(other_df):
            raise ValueError(
                f"[{dataset_name}] Lệch số mẫu test giữa {ref_name} ({len(ref_df)}) và {other_name} ({len(other_df)})!"
            )
        if not (ref_df["sample_id"].values == other_df["sample_id"].values).all():
            raise ValueError(f"[{dataset_name}] Danh sách sample_id không khớp giữa {ref_name} và {other_name}!")
        if not (ref_df["y_true"].values == other_df["y_true"].values).all():
            raise ValueError(f"[{dataset_name}] Nhãn thật y_true không khớp giữa {ref_name} và {other_name}!")
    if verbose:
        print(f"[{dataset_name}] XÁC NHẬN: Mẫu đánh giá (sample_id & y_true) khớp 100% giữa {len(names)} mô hình!")
    return True


def export_disagreement_cases(
    pred_dfs: Dict[str, pd.DataFrame],
    output_path: str,
) -> Optional[pd.DataFrame]:
    """
    Tìm các hồ sơ mà các thuật toán đưa ra nhãn dự đoán (y_pred) bất đồng nhau
    để phục vụ Case Study đối chiếu trực tiếp trong báo cáo chung.
    """
    if len(pred_dfs) < 2:
        return None
    names = list(pred_dfs.keys())
    merged = pred_dfs[names[0]][["sample_id", "y_true"]].copy()
    pred_cols = []
    for name in names:
        short = name.split(" ")[0]
        df_m = pred_dfs[name][["sample_id", "y_prob", "y_pred"]].rename(
            columns={"y_prob": f"prob_{short}", "y_pred": f"pred_{short}"}
        )
        merged = merged.merge(df_m, on="sample_id", how="inner")
        pred_cols.append(f"pred_{short}")
        
    # Lọc các dòng mà không phải tất cả mô hình đều đoán giống nhau
    disagree_mask = merged[pred_cols].nunique(axis=1) > 1
    disagree_df = merged[disagree_mask].copy()
    if len(disagree_df) > 0:
        disagree_df.to_csv(output_path, index=False)
    return disagree_df


def generate_benchmark_tables(output_dir: str = "benchmark/results"):
    """
    Tự động quét kết quả dự đoán của 3 mô hình:
    - algorithm/KNN/artifacts/
    - algorithm/Logistic/artifacts/
    - algorithm/CatBoost/artifacts/
    Kiểm tra khớp mẫu, trích xuất ca bất đồng và xuất 2 bảng đối chiếu chuẩn.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    models = {
        "KNN (Tùng)": "algorithm/KNN/artifacts",
        "Logistic Regression (Hoàng)": "algorithm/Logistic/artifacts",
        "CatBoost (Khải)": "algorithm/CatBoost/artifacts",
    }
    
    # 1. Dataset A: Heart Disease
    results_a, dfs_a = {}, {}
    for name, path in models.items():
        pred_file = os.path.join(path, "heart", "test_predictions.csv")
        df_p, metrics = load_model_predictions_and_metrics(pred_file)
        if metrics is not None:
            results_a[name] = metrics
            dfs_a[name] = df_p
            
    if results_a:
        verify_sample_alignment(dfs_a, "Heart Disease")
        dis_a = export_disagreement_cases(dfs_a, os.path.join(output_dir, "disagreement_cases_heart.csv"))
        df_a = format_comparison_table_dataset_a(results_a)
        df_a.to_csv(os.path.join(output_dir, "benchmark_dataset_a_heart.csv"), index=False)
        print("\n" + "=" * 60)
        print("BẢNG ĐỐI CHIẾU 3 THUẬT TOÁN: DATASET A (HEART DISEASE)")
        print("=" * 60)
        try:
            print(df_a.to_markdown(index=False))
        except (ImportError, Exception):
            print(df_a.to_string(index=False))
        if dis_a is not None:
            print(f"-> Tìm thấy {len(dis_a)} hồ sơ bất đồng giữa các mô hình (đã lưu disagreement_cases_heart.csv).")
    else:
        print("Chưa tìm thấy file test_predictions.csv cho Dataset A.")
        
    # 2. Dataset B: Diabetes Readmission
    results_b, dfs_b = {}, {}
    for name, path in models.items():
        pred_file = os.path.join(path, "readmission", "test_predictions.csv")
        df_p, metrics = load_model_predictions_and_metrics(pred_file)
        if metrics is not None:
            results_b[name] = metrics
            dfs_b[name] = df_p
            
    if results_b:
        verify_sample_alignment(dfs_b, "Diabetes Readmission")
        dis_b = export_disagreement_cases(dfs_b, os.path.join(output_dir, "disagreement_cases_readmission.csv"))
        df_b = format_comparison_table_dataset_b(results_b)
        df_b.to_csv(os.path.join(output_dir, "benchmark_dataset_b_readmission.csv"), index=False)
        print("\n" + "=" * 60)
        print("BẢNG ĐỐI CHIẾU 3 THUẬT TOÁN: DATASET B (DIABETES READMISSION)")
        print("=" * 60)
        try:
            print(df_b.to_markdown(index=False))
        except (ImportError, Exception):
            print(df_b.to_string(index=False))
        if dis_b is not None:
            print(f"-> Tìm thấy {len(dis_b)} hồ sơ bất đồng giữa các mô hình (đã lưu disagreement_cases_readmission.csv).")
    else:
        print("Chưa tìm thấy file test_predictions.csv cho Dataset B.")


if __name__ == "__main__":
    generate_benchmark_tables()
