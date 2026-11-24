import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from catboost import CatBoostClassifier, Pool

from benchmark.data_loader import load_heart_data, load_readmission_data
from benchmark.thresholds import find_optimal_f2_threshold
from benchmark.evaluation import (
    compute_binary_metrics,
    build_test_prediction_table,
    extract_extreme_errors,
)
from benchmark.conformal import (
    calibrate,
    prediction_sets,
    assign_conformal_prediction_labels,
    summarize as summarize_conformal,
)


def preprocess_features(df: pd.DataFrame, feature_cols: List[str], cat_cols: List[str]) -> pd.DataFrame:
    """Chuẩn bị features cho CatBoost: điền missing và ép kiểu chuỗi cho categorical columns."""
    X = df[feature_cols].copy()
    for col in cat_cols:
        X[col] = X[col].fillna("missing").astype(str)
        
    num_cols = [c for c in feature_cols if c not in cat_cols]
    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            
    return X


def train_evaluate_catboost_heart(
    output_dir: str = "algorithm/CatBoost/artifacts/heart",
) -> Dict[str, Any]:
    """Huấn luyện & Đánh giá CatBoost trên Dataset A (Heart Disease) có tích hợp Conformal & Nhãn phân vân."""
    os.makedirs(output_dir, exist_ok=True)
    df, splits, feature_cols, cat_cols = load_heart_data()
    X_all = preprocess_features(df, feature_cols, cat_cols)
    y_all = df["y"].values
    
    fit_idx = splits["fit"]
    thresh_idx = splits["threshold"]
    calib_idx = splits["calibration"]
    test_idx = splits["test"]
    
    X_fit, y_fit = X_all.iloc[fit_idx], y_all[fit_idx]
    X_thresh, y_thresh = X_all.iloc[thresh_idx], y_all[thresh_idx]
    X_calib, y_calib = X_all.iloc[calib_idx], y_all[calib_idx]
    X_test, y_test = X_all.iloc[test_idx], y_all[test_idx]
    test_sample_ids = df["row_id"].iloc[test_idx].values
    
    # 1. 5-fold Stratified CV trên tập fit
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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
            fold_scores.append(roc_auc_score(y_val_fold, val_probs))
            
        mean_auc = float(np.mean(fold_scores))
        std_auc = float(np.std(fold_scores))
        cv_records.append({**params, "mean_roc_auc": mean_auc, "std_roc_auc": std_auc})
        
        if mean_auc > best_score + 1e-5:
            best_score = mean_auc
            best_params = params
        elif abs(mean_auc - best_score) <= 1e-5 and params["depth"] < best_params["depth"]:
            best_params = params
            best_score = mean_auc
            
    pd.DataFrame(cv_records).to_csv(os.path.join(output_dir, "cv_results.csv"), index=False)
    
    # 2. Refit trên toàn bộ fit
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
    
    # 3. Tìm ngưỡng F2 trên threshold
    thresh_pool = Pool(X_thresh, y_thresh, cat_features=cat_cols)
    thresh_probs = final_model.predict_proba(thresh_pool)[:, 1]
    opt_thresh_info = find_optimal_f2_threshold(y_thresh, thresh_probs, return_curve=True)
    best_threshold = opt_thresh_info["best_threshold"]
    if "curve_df" in opt_thresh_info:
        opt_thresh_info["curve_df"].to_csv(os.path.join(output_dir, "threshold_curve.csv"), index=False)
        
    # 4. Hiệu chuẩn Conformal trên calibration & Tính toán nhãn phân vân
    calib_pool = Pool(X_calib, y_calib, cat_features=cat_cols)
    calib_probs = final_model.predict_proba(calib_pool)[:, 1]
    
    test_pool = Pool(X_test, y_test, cat_features=cat_cols)
    test_probs = final_model.predict_proba(test_pool)[:, 1]
    
    # Tính ngưỡng quantile theo code mẫu Conformal
    q_90 = calibrate(y_calib, calib_probs, 0.90)
    q_95 = calibrate(y_calib, calib_probs, 0.95)
    
    # Bảng tổng kết Conformal (theo chuẩn nhóm)
    conformal_summary_rows = summarize_conformal(y_calib, calib_probs, y_test, test_probs, levels=(0.90, 0.95))
    conformal_df = pd.DataFrame([
        dict(dataset="heart", model="CatBoost", **row)
        for row in conformal_summary_rows
    ])
    conformal_df.to_csv(os.path.join(output_dir, "conformal_results.csv"), index=False)
    
    # Gán nhãn Conformal và nhãn phân vân cho từng mẫu test
    sets_90, labels_90 = assign_conformal_prediction_labels(test_probs, q_90, label_0_name="0", label_1_name="1", ambiguous_name="Phân vân")
    sets_95, labels_95 = assign_conformal_prediction_labels(test_probs, q_95, label_0_name="0", label_1_name="1", ambiguous_name="Phân vân")
    
    # 5. Đánh giá trên test và xuất bảng dự đoán có nhãn phân vân
    test_metrics = compute_binary_metrics(y_test, test_probs, threshold=best_threshold)
    
    pred_table = build_test_prediction_table(
        sample_ids=test_sample_ids,
        y_true=y_test,
        y_prob=test_probs,
        threshold=best_threshold,
        conformal_sets_90=sets_90,
        conformal_labels_90=labels_90,
        conformal_sets_95=sets_95,
        conformal_labels_95=labels_95,
    )
    pred_table.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
    
    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": final_model.get_feature_importance(),
    }).sort_values(by="importance", ascending=False)
    feat_imp.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
    
    return {
        "best_params": best_params,
        "best_threshold": best_threshold,
        "test_metrics": test_metrics,
        "conformal_summary": conformal_df,
        "q_90": q_90,
        "q_95": q_95,
        "feature_importance": feat_imp,
    }


def train_evaluate_catboost_readmission(
    output_dir: str = "algorithm/CatBoost/artifacts/readmission",
) -> Dict[str, Any]:
    """Huấn luyện & Đánh giá CatBoost trên Dataset B (Diabetes Readmission) có tích hợp Conformal & Nhãn phân vân."""
    os.makedirs(output_dir, exist_ok=True)
    df, splits, feature_cols, cat_cols = load_readmission_data()
    X_all = preprocess_features(df, feature_cols, cat_cols)
    y_all = df["y"].values
    patient_ids = df["patient_nbr"].values
    encounter_ids = df["encounter_id"].values
    
    fit_idx = splits["fit"]
    thresh_idx = splits["threshold"]
    calib_idx = splits["calibration"]
    test_idx = splits["test"]
    
    X_fit, y_fit = X_all.iloc[fit_idx], y_all[fit_idx]
    patient_fit = patient_ids[fit_idx]
    X_thresh, y_thresh = X_all.iloc[thresh_idx], y_all[thresh_idx]
    X_test, y_test = X_all.iloc[test_idx], y_all[test_idx]
    test_sample_ids = encounter_ids[test_idx]
    
    # 1. 3-fold StratifiedGroupKFold theo bệnh nhân trên tập fit
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    param_grid = [
        {"depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3},
        {"depth": 4, "learning_rate": 0.10, "l2_leaf_reg": 3},
        {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3},
        {"depth": 6, "learning_rate": 0.10, "l2_leaf_reg": 3},
    ]
    
    best_score = -1.0
    best_params = None
    cv_records = []
    
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
            fold_scores.append(average_precision_score(y_val_fold, val_probs))
            
        mean_ap = float(np.mean(fold_scores))
        std_ap = float(np.std(fold_scores))
        cv_records.append({**params, "mean_average_precision": mean_ap, "std_average_precision": std_ap})
        
        if mean_ap > best_score + 1e-5:
            best_score = mean_ap
            best_params = params
        elif abs(mean_ap - best_score) <= 1e-5 and params["depth"] < best_params["depth"]:
            best_params = params
            best_score = mean_ap
            
    pd.DataFrame(cv_records).to_csv(os.path.join(output_dir, "cv_results.csv"), index=False)
    
    # 2. Refit trên toàn bộ fit
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
    
    # 3. Tìm ngưỡng F2 trên threshold
    thresh_pool = Pool(X_thresh, y_thresh, cat_features=cat_cols)
    thresh_probs = final_model.predict_proba(thresh_pool)[:, 1]
    opt_thresh_info = find_optimal_f2_threshold(y_thresh, thresh_probs, return_curve=True)
    best_threshold = opt_thresh_info["best_threshold"]
    if "curve_df" in opt_thresh_info:
        opt_thresh_info["curve_df"].to_csv(os.path.join(output_dir, "threshold_curve.csv"), index=False)
        
    # 4. Conformal Prediction trên bệnh nhân độc lập (theo code mẫu chuẩn nhóm)
    # Lọc encounter_id nhỏ nhất cho mỗi bệnh nhân trong calibration và test
    cal_df = df.iloc[calib_idx].sort_values("encounter_id").drop_duplicates("patient_nbr")
    test_sub_df = df.iloc[test_idx].sort_values("encounter_id").drop_duplicates("patient_nbr")
    assert set(cal_df["patient_nbr"]).isdisjoint(set(test_sub_df["patient_nbr"])), "Leakage giữa calib và test!"
    
    cal_pool = Pool(X_all.iloc[cal_df.index], cal_df["y"].values, cat_features=cat_cols)
    test_sub_pool = Pool(X_all.iloc[test_sub_df.index], test_sub_df["y"].values, cat_features=cat_cols)
    
    cal_probs = final_model.predict_proba(cal_pool)[:, 1]
    test_sub_probs = final_model.predict_proba(test_sub_pool)[:, 1]
    
    q_90 = calibrate(cal_df["y"].values, cal_probs, 0.90)
    q_95 = calibrate(cal_df["y"].values, cal_probs, 0.95)
    
    conformal_summary_rows = summarize_conformal(cal_df["y"].values, cal_probs, test_sub_df["y"].values, test_sub_probs, levels=(0.90, 0.95))
    conformal_df = pd.DataFrame([
        dict(dataset="readmission", model="CatBoost", **row)
        for row in conformal_summary_rows
    ])
    conformal_df.to_csv(os.path.join(output_dir, "conformal_results.csv"), index=False)
    
    # 5. Đánh giá trên toàn bộ 20.031 lượt test & Gán nhãn phân vân
    test_full_pool = Pool(X_test, y_test, cat_features=cat_cols)
    test_full_probs = final_model.predict_proba(test_full_pool)[:, 1]
    
    sets_90, labels_90 = assign_conformal_prediction_labels(test_full_probs, q_90, label_0_name="0", label_1_name="1", ambiguous_name="Phân vân")
    sets_95, labels_95 = assign_conformal_prediction_labels(test_full_probs, q_95, label_0_name="0", label_1_name="1", ambiguous_name="Phân vân")
    
    test_metrics = compute_binary_metrics(y_test, test_full_probs, threshold=best_threshold)
    
    pred_table = build_test_prediction_table(
        sample_ids=test_sample_ids,
        y_true=y_test,
        y_prob=test_full_probs,
        threshold=best_threshold,
        conformal_sets_90=sets_90,
        conformal_labels_90=labels_90,
        conformal_sets_95=sets_95,
        conformal_labels_95=labels_95,
    )
    pred_table.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
    
    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": final_model.get_feature_importance(),
    }).sort_values(by="importance", ascending=False)
    feat_imp.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
    
    return {
        "best_params": best_params,
        "best_threshold": best_threshold,
        "test_metrics": test_metrics,
        "conformal_summary": conformal_df,
        "q_90": q_90,
        "q_95": q_95,
        "feature_importance": feat_imp,
    }
