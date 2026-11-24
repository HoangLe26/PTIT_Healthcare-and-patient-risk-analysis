import os
import sys

# Thêm root vào path
sys.path.insert(0, os.path.abspath("."))

from algorithm.CatBoost.pipeline import (
    train_evaluate_catboost_heart,
    train_evaluate_catboost_readmission,
)


def main():
    print("=" * 60)
    print("CHẠY THỰC NGHIỆM ĐỘC LẬP CATBOOST (KHẢI)")
    print("=" * 60)
    
    print("\n1. Thực nghiệm Dataset A: Cleveland Heart Disease...")
    res_heart = train_evaluate_catboost_heart()
    print("-> Xong Heart Disease!")
    print(f"   ROC-AUC: {res_heart['test_metrics']['roc_auc']:.4f}")
    print(f"   Ngưỡng F2: {res_heart['best_threshold']:.4f}")
    print(f"   Recall: {res_heart['test_metrics']['recall']:.4f}")
    
    print("\n2. Thực nghiệm Dataset B: Diabetes Readmission...")
    res_readm = train_evaluate_catboost_readmission()
    print("-> Xong Diabetes Readmission!")
    print(f"   Average Precision: {res_readm['test_metrics']['average_precision']:.4f}")
    print(f"   Ngưỡng F2: {res_readm['best_threshold']:.4f}")
    print(f"   Recall: {res_readm['test_metrics']['recall']:.4f}")
    
    print("\nToàn bộ kết quả đã được lưu tại algorithm/CatBoost/artifacts/!")


if __name__ == "__main__":
    main()
