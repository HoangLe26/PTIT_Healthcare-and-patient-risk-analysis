import os
import sys
import pandas as pd

# Thêm root vào path
sys.path.insert(0, os.path.abspath("."))

from algorithm.CatBoost.pipeline import (
    train_evaluate_catboost_heart,
    train_evaluate_catboost_readmission,
)


def main():
    print("=" * 65)
    print("CHẠY THỰC NGHIỆM CATBOOST CHUẨN CHỐNG LEAKAGE + NHÃN PHÂN VÂN")
    print("=" * 65)
    
    print("\n1. Thực nghiệm Dataset A: Cleveland Heart Disease...")
    res_heart = train_evaluate_catboost_heart()
    print(f"-> Xong Heart Disease trong {res_heart['runtime_seconds']}s!")
    print(f"   ROC-AUC (Chính): {res_heart['test_metrics']['roc_auc']:.4f}")
    print(f"   Ngưỡng F2: {res_heart['best_threshold']:.4f} (Recall: {res_heart['test_metrics']['recall']:.4f})")
    print(f"   Ngưỡng Conformal q_90: {res_heart['q_90']:.4f}, q_95: {res_heart['q_95']:.4f}")
    print("   Bảng tổng kết Conformal (Heart):")
    print(res_heart["conformal_summary"].to_string(index=False))
    
    print("\n" + "-" * 65)
    print("2. Thực nghiệm Dataset B: Diabetes Readmission...")
    res_readm = train_evaluate_catboost_readmission()
    print(f"-> Xong Diabetes Readmission trong {res_readm['runtime_seconds']}s!")
    print(f"   Average Precision (Chính): {res_readm['test_metrics']['average_precision']:.4f}")
    print(f"   ROC-AUC: {res_readm['test_metrics']['roc_auc']:.4f}")
    print(f"   Ngưỡng F2: {res_readm['best_threshold']:.4f} (Recall: {res_readm['test_metrics']['recall']:.4f})")
    print(f"   Ngưỡng Conformal q_90: {res_readm['q_90']:.4f}, q_95: {res_readm['q_95']:.4f}")
    print("   Bảng tổng kết Conformal (Readmission):")
    print(res_readm["conformal_summary"].to_string(index=False))
    
    # Xuất file gộp 4 dòng Conformal theo đúng quy ước gửi Hoàng (dòng 74 QUY_UOC)
    conformal_4rows = pd.concat(
        [res_heart["conformal_summary"], res_readm["conformal_summary"]],
        ignore_index=True,
    )
    conformal_4rows.to_csv("algorithm/CatBoost/artifacts/conformal_4rows_for_hoang.csv", index=False)
    
    print("\n" + "=" * 65)
    print("HOÀN TẤT! Đã xuất đầy đủ artifacts + conformal_4rows_for_hoang.csv!")


if __name__ == "__main__":
    main()
