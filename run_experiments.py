import os
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.healthcare_ml.catboost_pipeline import (
    run_catboost_heart_experiment,
    run_catboost_readmission_experiment,
    preprocess_features,
)
from src.healthcare_ml.data_loader import load_heart_data, load_readmission_data
from src.healthcare_ml.conformal import evaluate_split_conformal_binary


def run_conformal_heart(heart_results, output_dir="artifacts/catboost/heart"):
    """Đánh giá Split Conformal cho Heart Disease."""
    print("\n" + "=" * 50)
    print("ĐÁNH GIÁ SPLIT CONFORMAL: HEART DISEASE")
    print("=" * 50)
    df, splits, feature_cols, cat_cols = load_heart_data()
    X_all = preprocess_features(df, feature_cols, cat_cols)
    y_all = df["y"].values
    
    fit_idx = splits["fit"]
    calib_idx = splits["calibration"]
    test_idx = splits["test"]
    
    # Fit lại mô hình trên fit với best params
    best_params = heart_results["best_params"]
    fit_pool = Pool(X_all.iloc[fit_idx], y_all[fit_idx], cat_features=cat_cols)
    model = CatBoostClassifier(
        iterations=300,
        depth=best_params["depth"],
        learning_rate=best_params["learning_rate"],
        l2_leaf_reg=best_params["l2_leaf_reg"],
        random_seed=42,
        verbose=False,
        thread_count=2,
    )
    model.fit(fit_pool, verbose=False)
    
    calib_pool = Pool(X_all.iloc[calib_idx], cat_features=cat_cols)
    test_pool = Pool(X_all.iloc[test_idx], cat_features=cat_cols)
    
    prob_calib = model.predict_proba(calib_pool)[:, 1]
    prob_test = model.predict_proba(test_pool)[:, 1]
    
    y_calib = y_all[calib_idx]
    y_test = y_all[test_idx]
    
    conformal_df = evaluate_split_conformal_binary(
        y_calib=y_calib,
        prob_calib=prob_calib,
        y_test=y_test,
        prob_test=prob_test,
        alphas=[0.10, 0.05],
    )
    conformal_df["model"] = "CatBoost (Khải)"
    conformal_df["dataset"] = "Heart Disease (Cleveland)"
    conformal_df.to_csv(os.path.join(output_dir, "conformal_results.csv"), index=False)
    print(conformal_df.to_string(index=False))
    return conformal_df


def run_conformal_readmission(readmission_results, output_dir="artifacts/catboost/readmission"):
    """
    Đánh giá Split Conformal cho Diabetes Readmission.
    Quy tắc dòng 61: chọn 1 lượt cho mỗi bệnh nhân trong từng phần calibration/test
    có encounter_id nhỏ nhất để bảo đảm giả định hoán đổi (exchangeability).
    """
    print("\n" + "=" * 50)
    print("ĐÁNH GIÁ SPLIT CONFORMAL: DIABETES READMISSION")
    print("=" * 50)
    df, splits, feature_cols, cat_cols = load_readmission_data()
    X_all = preprocess_features(df, feature_cols, cat_cols)
    y_all = df["y"].values
    
    fit_idx = splits["fit"]
    calib_idx = splits["calibration"]
    test_idx = splits["test"]
    
    # Lọc 1 lượt encounter_id nhỏ nhất cho mỗi bệnh nhân trong calibration
    calib_df = df.iloc[calib_idx].copy()
    first_calib_indices = calib_df.sort_values(by="encounter_id").groupby("patient_nbr").head(1).index.values
    
    # Lọc 1 lượt encounter_id nhỏ nhất cho mỗi bệnh nhân trong test
    test_df = df.iloc[test_idx].copy()
    first_test_indices = test_df.sort_values(by="encounter_id").groupby("patient_nbr").head(1).index.values
    
    print(f"Số bệnh nhân duy nhất trong calibration: {len(first_calib_indices)} (từ {len(calib_idx)} lượt)")
    print(f"Số bệnh nhân duy nhất trong test: {len(first_test_indices)} (từ {len(test_idx)} lượt)")
    
    best_params = readmission_results["best_params"]
    fit_pool = Pool(X_all.iloc[fit_idx], y_all[fit_idx], cat_features=cat_cols)
    model = CatBoostClassifier(
        iterations=300,
        depth=best_params["depth"],
        learning_rate=best_params["learning_rate"],
        l2_leaf_reg=best_params["l2_leaf_reg"],
        random_seed=42,
        verbose=False,
        thread_count=4,
    )
    model.fit(fit_pool, verbose=False)
    
    calib_pool = Pool(X_all.iloc[first_calib_indices], cat_features=cat_cols)
    test_pool = Pool(X_all.iloc[first_test_indices], cat_features=cat_cols)
    
    prob_calib = model.predict_proba(calib_pool)[:, 1]
    prob_test = model.predict_proba(test_pool)[:, 1]
    
    y_calib = y_all[first_calib_indices]
    y_test = y_all[first_test_indices]
    
    conformal_df = evaluate_split_conformal_binary(
        y_calib=y_calib,
        prob_calib=prob_calib,
        y_test=y_test,
        prob_test=prob_test,
        alphas=[0.10, 0.05],
    )
    conformal_df["model"] = "CatBoost (Khải)"
    conformal_df["dataset"] = "Diabetes Readmission"
    conformal_df.to_csv(os.path.join(output_dir, "conformal_results.csv"), index=False)
    print(conformal_df.to_string(index=False))
    return conformal_df


def main():
    print("BẮT ĐẦU CHUỖI THỰC NGHIỆM CÁ NHÂN CỦA KHẢI (CATBOOST)")
    heart_res = run_catboost_heart_experiment()
    conf_heart = run_conformal_heart(heart_res)
    
    readm_res = run_catboost_readmission_experiment()
    conf_readm = run_conformal_readmission(readm_res)
    
    print("\nHOÀN TẤT TOÀN BỘ THỰC NGHIỆM CATBOOST VÀ CONFORMAL PREDICTION!")


if __name__ == "__main__":
    main()
