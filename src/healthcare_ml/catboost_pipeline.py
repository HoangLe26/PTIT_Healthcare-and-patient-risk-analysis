import os
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from catboost import CatBoostClassifier, Pool

from .data_loader import load_heart_data, load_readmission_data
from .thresholds import find_optimal_f2_threshold
from .evaluation import (
    compute_binary_metrics,
    build_test_prediction_table,
    extract_extreme_errors,
)


def preprocess_features(df: pd.DataFrame, feature_cols: List[str], cat_cols: List[str]) -> pd.DataFrame:
    """Chuẩn bị features cho CatBoost: điền missing và ép kiểu chuỗi cho categorical columns."""
    X = df[feature_cols].copy()
    for col in cat_cols:
        # Chuyển categorical sang string và điền missing rõ ràng
        X[col] = X[col].fillna("missing").astype(str)
        
    num_cols = [c for c in feature_cols if c not in cat_cols]
    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        # Điền median cho biến số
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            
    return X


def run_catboost_heart_experiment(
    output_dir: str = "artifacts/catboost/heart",
) -> Dict[str, Any]:
    """
    Thực nghiệm CatBoost trên Dataset A (Cleveland Heart Disease).
    - CV 5-fold phân tầng trên tập fit (169 mẫu).
    - Grid search siêu tham số theo ROC-AUC.
    - Tìm ngưỡng F2 trên tập threshold (37 mẫu).
    - Đánh giá trên tập test (61 mẫu).
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("CHẠY THỰC NGHIỆM CATBOOST: DATASET A (HEART DISEASE)")
    print("=" * 60)
    
    df, splits, feature_cols, cat_cols = load_heart_data()
    X_all = preprocess_features(df, feature_cols, cat_cols)
    y_all = df["y"].values
    
    fit_idx = splits["fit"]
    thresh_idx = splits["threshold"]
    test_idx = splits["test"]
    
    X_fit, y_fit = X_all.iloc[fit_idx], y_all[fit_idx]
    X_thresh, y_thresh = X_all.iloc[thresh_idx], y_all[thresh_idx]
    X_test, y_test = X_all.iloc[test_idx], y_all[test_idx]
    test_sample_ids = df["row_id"].iloc[test_idx].values
    
    print(f"Số mẫu: fit={len(fit_idx)}, threshold={len(thresh_idx)}, test={len(test_idx)}")
    
    # 1. Hyperparameter tuning với 5-fold Stratified CV trên tập fit
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Grid các siêu tham số
    param_grid = [
        {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 1},
        {"depth": 4, "learning_rate": 0.1, "l2_leaf_reg": 1},
        {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 5},
        {"depth": 4, "learning_rate": 0.1, "l2_leaf_reg": 5},
        {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 1},
        {"depth": 6, "learning_rate": 0.1, "l2_leaf_reg": 1},
        {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 5},
        {"depth": 6, "learning_rate": 0.1, "l2_leaf_reg": 5},
    ]
    
    best_score = -1.0
    best_params = None
    cv_records = []
    
    print("Bắt đầu Cross-Validation 5-fold trên tập fit:")
    for params in param_grid:
        fold_scores = []
        for train_fold_idx, val_fold_idx in skf.split(X_fit, y_fit):
            X_train_fold, y_train_fold = X_fit.iloc[train_fold_idx], y_fit[train_fold_idx]
            X_val_fold, y_val_fold = X_fit.iloc[val_fold_idx], y_fit[val_fold_idx]
            
            train_pool = Pool(X_train_fold, y_train_fold, cat_features=cat_cols)
            val_pool = Pool(X_val_fold, y_val_fold, cat_features=cat_cols)
            
            model = CatBoostClassifier(
                iterations=300,
                depth=params["depth"],
                learning_rate=params["learning_rate"],
                l2_leaf_reg=params["l2_leaf_reg"],
                random_seed=42,
                verbose=False,
                thread_count=2,
            )
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=30, verbose=False)
            val_probs = model.predict_proba(val_pool)[:, 1]
            fold_auc = roc_auc_score(y_val_fold, val_probs)
            fold_scores.append(fold_auc)
            
        mean_auc = float(np.mean(fold_scores))
        std_auc = float(np.std(fold_scores))
        cv_records.append({**params, "mean_roc_auc": mean_auc, "std_roc_auc": std_auc})
        print(f"  Params {params} -> Mean ROC-AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")
        
        # Tie-breaker: ưu tiên depth nhỏ hơn, sau đó l2_leaf_reg lớn hơn
        if mean_auc > best_score + 1e-5:
            best_score = mean_auc
            best_params = params
        elif abs(mean_auc - best_score) <= 1e-5:
            if params["depth"] < best_params["depth"]:
                best_params = params
                best_score = mean_auc
            elif params["depth"] == best_params["depth"] and params["l2_leaf_reg"] > best_params["l2_leaf_reg"]:
                best_params = params
                best_score = mean_auc
                
    print(f"\n=> Cấu hình tối ưu được chọn: {best_params} (Mean ROC-AUC: {best_score:.4f})")
    pd.DataFrame(cv_records).to_csv(os.path.join(output_dir, "cv_results.csv"), index=False)
    
    # 2. Refit mô hình tốt nhất trên toàn bộ tập fit
    full_fit_pool = Pool(X_fit, y_fit, cat_features=cat_cols)
    final_model = CatBoostClassifier(
        iterations=300,
        depth=best_params["depth"],
        learning_rate=best_params["learning_rate"],
        l2_leaf_reg=best_params["l2_leaf_reg"],
        random_seed=42,
        verbose=False,
        thread_count=2,
    )
    final_model.fit(full_fit_pool, verbose=False)
    
    # 3. Tìm ngưỡng tối ưu F2 trên tập threshold
    thresh_pool = Pool(X_thresh, y_thresh, cat_features=cat_cols)
    thresh_probs = final_model.predict_proba(thresh_pool)[:, 1]
    opt_thresh_info = find_optimal_f2_threshold(y_thresh, thresh_probs, return_curve=True)
    best_threshold = opt_thresh_info["best_threshold"]
    print(f"\n=> Ngưỡng F2 tối ưu trên tập threshold: {best_threshold:.4f} (F2 = {opt_thresh_info['best_f2']:.4f})")
    
    # Lưu curve threshold
    if "curve_df" in opt_thresh_info:
        opt_thresh_info["curve_df"].to_csv(os.path.join(output_dir, "threshold_curve.csv"), index=False)
        
    # 4. Đánh giá cuối cùng trên tập test
    test_pool = Pool(X_test, y_test, cat_features=cat_cols)
    test_probs = final_model.predict_proba(test_pool)[:, 1]
    
    test_metrics = compute_binary_metrics(y_test, test_probs, threshold=best_threshold)
    default_metrics = compute_binary_metrics(y_test, test_probs, threshold=0.5)
    
    # Bảng dự đoán test chi tiết
    pred_table = build_test_prediction_table(test_sample_ids, y_test, test_probs, threshold=best_threshold)
    pred_table.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
    
    # Trích xuất 1 ca FN và 1 ca FP sai nặng nhất
    extreme_errors = extract_extreme_errors(pred_table)
    
    # Feature importances
    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": final_model.get_feature_importance(),
    }).sort_values(by="importance", ascending=False)
    feat_imp.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
    
    print("\n" + "=" * 40)
    print("KẾT QUẢ TEST MÔ HÌNH CATBOOST (HEART DISEASE)")
    print("=" * 40)
    print(f"ROC-AUC (Chính):       {test_metrics['roc_auc']:.4f}")
    print(f"Average Precision:    {test_metrics['average_precision']:.4f}")
    print(f"Ngưỡng F2:            {best_threshold:.4f}")
    print(f"F2 Score:             {test_metrics['f2']:.4f} (ở ngưỡng 0.5: {default_metrics['f2']:.4f})")
    print(f"Recall / Sensitivity: {test_metrics['recall']:.4f}")
    print(f"Specificity:          {test_metrics['specificity']:.4f}")
    print(f"Precision:            {test_metrics['precision']:.4f}")
    print(f"Brier Score:          {test_metrics['brier_score']:.4f}")
    print(f"Confusion Matrix:     TP={test_metrics['tp']}, FP={test_metrics['fp']}, TN={test_metrics['tn']}, FN={test_metrics['fn']}")
    
    if extreme_errors["worst_fn"] is not None:
        fn_id = extreme_errors["worst_fn"]["sample_id"]
        fn_p = extreme_errors["worst_fn"]["y_prob"]
        print(f"\n[Ca FN sai nặng nhất]: Sample ID {fn_id}, P(y=1) = {fn_p:.4f} (Nhãn thật: 1 nhưng dự đoán 0)")
        
    if extreme_errors["worst_fp"] is not None:
        fp_id = extreme_errors["worst_fp"]["sample_id"]
        fp_p = extreme_errors["worst_fp"]["y_prob"]
        print(f"[Ca FP sai nặng nhất]: Sample ID {fp_id}, P(y=1) = {fp_p:.4f} (Nhãn thật: 0 nhưng dự đoán 1)")
        
    return {
        "dataset": "heart",
        "best_params": best_params,
        "best_threshold": best_threshold,
        "test_metrics": test_metrics,
        "default_metrics": default_metrics,
        "extreme_errors": extreme_errors,
        "feature_importance": feat_imp,
    }


def run_catboost_readmission_experiment(
    output_dir: str = "artifacts/catboost/readmission",
) -> Dict[str, Any]:
    """
    Thực nghiệm CatBoost trên Dataset B (Diabetes 130-US Hospitals Readmission).
    - CV 3-fold Stratified Group theo patient_nbr trên tập fit (59.372 mẫu).
    - Grid search siêu tham số theo Average Precision (AP).
    - Tìm ngưỡng F2 trên tập threshold (9.854 mẫu).
    - Đánh giá trên tập test (20.031 mẫu).
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("CHẠY THỰC NGHIỆM CATBOOST: DATASET B (DIABETES READMISSION)")
    print("=" * 60)
    
    df, splits, feature_cols, cat_cols = load_readmission_data()
    X_all = preprocess_features(df, feature_cols, cat_cols)
    y_all = df["y"].values
    patient_ids = df["patient_nbr"].values
    encounter_ids = df["encounter_id"].values
    
    fit_idx = splits["fit"]
    thresh_idx = splits["threshold"]
    test_idx = splits["test"]
    
    X_fit, y_fit = X_all.iloc[fit_idx], y_all[fit_idx]
    patient_fit = patient_ids[fit_idx]
    
    X_thresh, y_thresh = X_all.iloc[thresh_idx], y_all[thresh_idx]
    X_test, y_test = X_all.iloc[test_idx], y_all[test_idx]
    test_sample_ids = encounter_ids[test_idx]
    
    print(f"Số mẫu: fit={len(fit_idx)}, threshold={len(thresh_idx)}, test={len(test_idx)}")
    print(f"Số đặc trưng đầu vào: {len(feature_cols)} (trong đó {len(cat_cols)} biến phân loại)")
    
    # 1. Hyperparameter tuning với 3-fold StratifiedGroupKFold theo bệnh nhân trên tập fit
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Grid 4 cấu hình cốt lõi để tối ưu thời gian huấn luyện
    param_grid = [
        {"depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3},
        {"depth": 4, "learning_rate": 0.10, "l2_leaf_reg": 3},
        {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3},
        {"depth": 6, "learning_rate": 0.10, "l2_leaf_reg": 3},
    ]
    
    best_score = -1.0
    best_params = None
    cv_records = []
    
    print("Bắt đầu Cross-Validation 3-fold theo nhóm bệnh nhân trên tập fit:")
    for params in param_grid:
        fold_scores = []
        for train_fold_idx, val_fold_idx in sgkf.split(X_fit, y_fit, groups=patient_fit):
            X_train_fold, y_train_fold = X_fit.iloc[train_fold_idx], y_fit[train_fold_idx]
            X_val_fold, y_val_fold = X_fit.iloc[val_fold_idx], y_fit[val_fold_idx]
            
            train_pool = Pool(X_train_fold, y_train_fold, cat_features=cat_cols)
            val_pool = Pool(X_val_fold, y_val_fold, cat_features=cat_cols)
            
            model = CatBoostClassifier(
                iterations=250,
                depth=params["depth"],
                learning_rate=params["learning_rate"],
                l2_leaf_reg=params["l2_leaf_reg"],
                random_seed=42,
                verbose=False,
                thread_count=4,
            )
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=25, verbose=False)
            val_probs = model.predict_proba(val_pool)[:, 1]
            fold_ap = average_precision_score(y_val_fold, val_probs)
            fold_scores.append(fold_ap)
            
        mean_ap = float(np.mean(fold_scores))
        std_ap = float(np.std(fold_scores))
        cv_records.append({**params, "mean_average_precision": mean_ap, "std_average_precision": std_ap})
        print(f"  Params {params} -> Mean Average Precision (AP): {mean_ap:.4f} (+/- {std_ap:.4f})")
        
        # Tie-breaker: ưu tiên depth nhỏ hơn
        if mean_ap > best_score + 1e-5:
            best_score = mean_ap
            best_params = params
        elif abs(mean_ap - best_score) <= 1e-5 and params["depth"] < best_params["depth"]:
            best_params = params
            best_score = mean_ap
            
    print(f"\n=> Cấu hình tối ưu được chọn: {best_params} (Mean AP: {best_score:.4f})")
    pd.DataFrame(cv_records).to_csv(os.path.join(output_dir, "cv_results.csv"), index=False)
    
    # 2. Refit trên toàn bộ tập fit
    print("Huấn luyện mô hình tối ưu trên toàn bộ tập fit...")
    full_fit_pool = Pool(X_fit, y_fit, cat_features=cat_cols)
    final_model = CatBoostClassifier(
        iterations=300,
        depth=best_params["depth"],
        learning_rate=best_params["learning_rate"],
        l2_leaf_reg=best_params["l2_leaf_reg"],
        random_seed=42,
        verbose=False,
        thread_count=4,
    )
    final_model.fit(full_fit_pool, verbose=False)
    
    # 3. Tìm ngưỡng tối ưu F2 trên tập threshold
    thresh_pool = Pool(X_thresh, y_thresh, cat_features=cat_cols)
    thresh_probs = final_model.predict_proba(thresh_pool)[:, 1]
    opt_thresh_info = find_optimal_f2_threshold(y_thresh, thresh_probs, return_curve=True)
    best_threshold = opt_thresh_info["best_threshold"]
    print(f"\n=> Ngưỡng F2 tối ưu trên tập threshold: {best_threshold:.4f} (F2 = {opt_thresh_info['best_f2']:.4f})")
    
    if "curve_df" in opt_thresh_info:
        opt_thresh_info["curve_df"].to_csv(os.path.join(output_dir, "threshold_curve.csv"), index=False)
        
    # 4. Đánh giá cuối cùng trên tập test
    print("Đánh giá mô hình trên tập test (20.031 mẫu)...")
    test_pool = Pool(X_test, y_test, cat_features=cat_cols)
    test_probs = final_model.predict_proba(test_pool)[:, 1]
    
    test_metrics = compute_binary_metrics(y_test, test_probs, threshold=best_threshold)
    default_metrics = compute_binary_metrics(y_test, test_probs, threshold=0.5)
    
    # Bảng dự đoán test
    pred_table = build_test_prediction_table(test_sample_ids, y_test, test_probs, threshold=best_threshold)
    pred_table.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
    
    # Trích xuất 1 ca FN và 1 ca FP sai nặng nhất
    extreme_errors = extract_extreme_errors(pred_table)
    
    # Feature importances
    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": final_model.get_feature_importance(),
    }).sort_values(by="importance", ascending=False)
    feat_imp.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
    
    print("\n" + "=" * 40)
    print("KẾT QUẢ TEST MÔ HÌNH CATBOOST (DIABETES READMISSION)")
    print("=" * 40)
    print(f"Average Precision (Chính): {test_metrics['average_precision']:.4f}")
    print(f"ROC-AUC:                   {test_metrics['roc_auc']:.4f}")
    print(f"Ngưỡng F2:                {best_threshold:.4f}")
    print(f"F2 Score:                 {test_metrics['f2']:.4f} (ở ngưỡng 0.5: {default_metrics['f2']:.4f})")
    print(f"Recall / Sensitivity:     {test_metrics['recall']:.4f}")
    print(f"Precision:                {test_metrics['precision']:.4f}")
    print(f"Specificity:              {test_metrics['specificity']:.4f}")
    print(f"Tỷ lệ gắn cờ (Flagged):    {test_metrics['flagged_rate']:.4f}")
    print(f"Brier Score:              {test_metrics['brier_score']:.4f}")
    print(f"Confusion Matrix:         TP={test_metrics['tp']}, FP={test_metrics['fp']}, TN={test_metrics['tn']}, FN={test_metrics['fn']}")
    
    if extreme_errors["worst_fn"] is not None:
        fn_id = extreme_errors["worst_fn"]["sample_id"]
        fn_p = extreme_errors["worst_fn"]["y_prob"]
        print(f"\n[Ca FN sai nặng nhất]: Encounter ID {fn_id}, P(y=1) = {fn_p:.4f} (Nhãn thật: 1 nhưng dự đoán 0)")
        
    if extreme_errors["worst_fp"] is not None:
        fp_id = extreme_errors["worst_fp"]["sample_id"]
        fp_p = extreme_errors["worst_fp"]["y_prob"]
        print(f"[Ca FP sai nặng nhất]: Encounter ID {fp_id}, P(y=1) = {fp_p:.4f} (Nhãn thật: 0 nhưng dự đoán 1)")
        
    return {
        "dataset": "readmission",
        "best_params": best_params,
        "best_threshold": best_threshold,
        "test_metrics": test_metrics,
        "default_metrics": default_metrics,
        "extreme_errors": extreme_errors,
        "feature_importance": feat_imp,
    }
