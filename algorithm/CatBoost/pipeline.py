import os
import time
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


def fit_preprocessor(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
) -> Dict[str, float]:
    """
    Học giá trị trung vị (median) cho các biến số CHỈ TRÊN TẬP HUẤN LUYỆN (chống data leakage).
    """
    num_cols = [c for c in feature_cols if c not in cat_cols]
    medians: Dict[str, float] = {}
    for col in num_cols:
        s = pd.to_numeric(df_train[col], errors="coerce")
        medians[col] = float(s.median()) if s.notna().any() else 0.0
    return medians


def transform_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
    medians: Dict[str, float],
) -> pd.DataFrame:
    """
    Áp dụng phép điền thiếu đã học từ train:
    - Biến số: điền trung vị đã học từ train.
    - Biến phân loại: mã hóa riêng giá trị thiếu thành '__missing__' và ép kiểu chuỗi.
    """
    X = df[feature_cols].copy()
    cat_set = set(cat_cols)
    for col in feature_cols:
        if col in cat_set:
            X[col] = X[col].fillna("__missing__").replace("", "__missing__").astype(str)
        else:
            s = pd.to_numeric(X[col], errors="coerce")
            X[col] = s.fillna(medians[col])
    return X


def save_extreme_errors(
    df_raw_test: pd.DataFrame,
    pred_table: pd.DataFrame,
    id_col: str,
    output_path: str,
) -> Dict[str, Any]:
    """Trích xuất và lưu 1 ca FN và 1 ca FP sai nặng nhất kèm đặc trưng bệnh nhân."""
    ext = extract_extreme_errors(pred_table)
    rows = []
    for err_type, key in [("FN", "worst_fn"), ("FP", "worst_fp")]:
        rec = ext[key]
        if rec is not None:
            sid = rec["sample_id"]
            raw_row = df_raw_test[df_raw_test[id_col] == sid].iloc[0].to_dict()
            rows.append({
                "error_type": err_type,
                "sample_id": sid,
                "y_true": int(rec["y_true"]),
                "y_prob": float(rec["y_prob"]),
                "threshold": float(rec["threshold"]),
                "y_pred": int(rec["y_pred"]),
                "conformal_decision_90": rec.get("conformal_decision_90", ""),
                "conformal_decision_95": rec.get("conformal_decision_95", ""),
                **raw_row,
            })
    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False)
    return ext


def train_evaluate_catboost_heart(
    output_dir: str = "algorithm/CatBoost/artifacts/heart",
) -> Dict[str, Any]:
    """Huấn luyện & Đánh giá CatBoost trên Dataset A (Heart Disease) chuẩn chống leakage + Nhãn phân vân."""
    t0 = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)
    df, splits, feature_cols, cat_cols = load_heart_data()
    y_all = df["y"].values
    
    fit_idx = splits["fit"]
    thresh_idx = splits["threshold"]
    calib_idx = splits["calibration"]
    test_idx = splits["test"]
    
    df_fit = df.iloc[fit_idx]
    df_thresh = df.iloc[thresh_idx]
    df_calib = df.iloc[calib_idx]
    df_test = df.iloc[test_idx]
    
    y_fit = y_all[fit_idx]
    y_thresh = y_all[thresh_idx]
    y_calib = y_all[calib_idx]
    y_test = y_all[test_idx]
    test_sample_ids = df_test["row_id"].values
    
    # 1. 5-fold Stratified CV trên tập fit (học median trong từng fold)
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
        fold_val_scores = []
        fold_train_scores = []
        for train_fold_idx, val_fold_idx in skf.split(df_fit, y_fit):
            df_tr = df_fit.iloc[train_fold_idx]
            df_va = df_fit.iloc[val_fold_idx]
            y_tr, y_va = y_fit[train_fold_idx], y_fit[val_fold_idx]
            
            # Học preprocessor chỉ trên train fold (chống leakage)
            medians_fold = fit_preprocessor(df_tr, feature_cols, cat_cols)
            X_tr = transform_features(df_tr, feature_cols, cat_cols, medians_fold)
            X_va = transform_features(df_va, feature_cols, cat_cols, medians_fold)
            
            train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
            val_pool = Pool(X_va, y_va, cat_features=cat_cols)
            
            model = CatBoostClassifier(
                iterations=300,
                depth=params["depth"],
                learning_rate=params["learning_rate"],
                l2_leaf_reg=params["l2_leaf_reg"],
                random_seed=42,
                verbose=False,
                thread_count=2,
                train_dir="algorithm/CatBoost/catboost_info",
            )
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=30, verbose=False)
            fold_train_scores.append(roc_auc_score(y_tr, model.predict_proba(train_pool)[:, 1]))
            fold_val_scores.append(roc_auc_score(y_va, model.predict_proba(val_pool)[:, 1]))
            
        mean_auc = float(np.mean(fold_val_scores))
        std_auc = float(np.std(fold_val_scores))
        mean_train_auc = float(np.mean(fold_train_scores))
        cv_records.append({
            **params,
            "mean_train_roc_auc": mean_train_auc,
            "mean_cv_roc_auc": mean_auc,
            "std_cv_roc_auc": std_auc,
        })
        
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
                
    pd.DataFrame(cv_records).to_csv(os.path.join(output_dir, "cv_results.csv"), index=False)
    
    # 2. Refit trên toàn bộ tập fit (học preprocessor chỉ trên fit)
    medians_fit = fit_preprocessor(df_fit, feature_cols, cat_cols)
    X_fit = transform_features(df_fit, feature_cols, cat_cols, medians_fit)
    X_thresh = transform_features(df_thresh, feature_cols, cat_cols, medians_fit)
    X_calib = transform_features(df_calib, feature_cols, cat_cols, medians_fit)
    X_test = transform_features(df_test, feature_cols, cat_cols, medians_fit)
    
    full_fit_pool = Pool(X_fit, y_fit, cat_features=cat_cols)
    final_model = CatBoostClassifier(
        iterations=300,
        depth=best_params["depth"],
        learning_rate=best_params["learning_rate"],
        l2_leaf_reg=best_params["l2_leaf_reg"],
        random_seed=42,
        verbose=False,
        thread_count=2,
        train_dir="algorithm/CatBoost/catboost_info",
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
    
    q_90 = calibrate(y_calib, calib_probs, 0.90)
    q_95 = calibrate(y_calib, calib_probs, 0.95)
    
    model_tag = f"CatBoost_d{best_params['depth']}_lr{best_params['learning_rate']}_l2_{best_params['l2_leaf_reg']}"
    conformal_summary_rows = summarize_conformal(y_calib, calib_probs, y_test, test_probs, levels=(0.90, 0.95))
    conformal_df = pd.DataFrame([
        dict(dataset="heart", model=model_tag, **row)
        for row in conformal_summary_rows
    ])
    conformal_df.to_csv(os.path.join(output_dir, "conformal_results.csv"), index=False)
    
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
    
    ext_errors = save_extreme_errors(df_test, pred_table, "row_id", os.path.join(output_dir, "extreme_errors.csv"))
    
    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": final_model.get_feature_importance(),
    }).sort_values(by="importance", ascending=False)
    feat_imp.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
    
    runtime_sec = round(time.perf_counter() - t0, 2)
    run_meta = {
        "dataset": "heart",
        "model": model_tag,
        "best_params": best_params,
        "best_cv_roc_auc": best_score,
        "best_threshold_f2": best_threshold,
        "q_90": float(q_90),
        "q_95": float(q_95),
        "n_features": len(feature_cols),
        "n_cat_features": len(cat_cols),
        "runtime_seconds": runtime_sec,
    }
    with open(os.path.join(output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)
    
    return {
        "best_params": best_params,
        "best_threshold": best_threshold,
        "test_metrics": test_metrics,
        "conformal_summary": conformal_df,
        "q_90": q_90,
        "q_95": q_95,
        "feature_importance": feat_imp,
        "extreme_errors": ext_errors,
        "runtime_seconds": runtime_sec,
    }


def train_evaluate_catboost_readmission(
    output_dir: str = "algorithm/CatBoost/artifacts/readmission",
) -> Dict[str, Any]:
    """Huấn luyện & Đánh giá CatBoost trên Dataset B (Diabetes Readmission) chuẩn chống leakage + Nhãn phân vân."""
    t0 = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)
    df, splits, feature_cols, cat_cols = load_readmission_data()
    y_all = df["y"].values
    patient_ids = df["patient_nbr"].values
    
    fit_idx = splits["fit"]
    thresh_idx = splits["threshold"]
    calib_idx = splits["calibration"]
    test_idx = splits["test"]
    
    df_fit = df.iloc[fit_idx]
    df_thresh = df.iloc[thresh_idx]
    df_calib = df.iloc[calib_idx]
    df_test = df.iloc[test_idx]
    
    # Kiểm tra rò rỉ bệnh nhân giữa các tập
    assert set(df_fit["patient_nbr"]).isdisjoint(set(df_test["patient_nbr"])), "Leakage fit-test!"
    assert set(df_calib["patient_nbr"]).isdisjoint(set(df_test["patient_nbr"])), "Leakage calib-test!"
    
    y_fit = y_all[fit_idx]
    patient_fit = patient_ids[fit_idx]
    y_thresh = y_all[thresh_idx]
    y_test = y_all[test_idx]
    test_sample_ids = df_test["encounter_id"].values
    
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
        fold_val_scores = []
        fold_train_scores = []
        for train_fold_idx, val_fold_idx in sgkf.split(df_fit, y_fit, groups=patient_fit):
            df_tr = df_fit.iloc[train_fold_idx]
            df_va = df_fit.iloc[val_fold_idx]
            y_tr, y_va = y_fit[train_fold_idx], y_fit[val_fold_idx]
            
            assert set(df_tr["patient_nbr"]).isdisjoint(set(df_va["patient_nbr"])), "Leakage trong fold CV!"
            
            medians_fold = fit_preprocessor(df_tr, feature_cols, cat_cols)
            X_tr = transform_features(df_tr, feature_cols, cat_cols, medians_fold)
            X_va = transform_features(df_va, feature_cols, cat_cols, medians_fold)
            
            train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
            val_pool = Pool(X_va, y_va, cat_features=cat_cols)
            
            model = CatBoostClassifier(
                iterations=250,
                depth=params["depth"],
                learning_rate=params["learning_rate"],
                l2_leaf_reg=params["l2_leaf_reg"],
                random_seed=42,
                verbose=False,
                thread_count=4,
                train_dir="algorithm/CatBoost/catboost_info",
            )
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=25, verbose=False)
            fold_train_scores.append(average_precision_score(y_tr, model.predict_proba(train_pool)[:, 1]))
            fold_val_scores.append(average_precision_score(y_va, model.predict_proba(val_pool)[:, 1]))
            
        mean_ap = float(np.mean(fold_val_scores))
        std_ap = float(np.std(fold_val_scores))
        mean_train_ap = float(np.mean(fold_train_scores))
        cv_records.append({
            **params,
            "mean_train_ap": mean_train_ap,
            "mean_cv_ap": mean_ap,
            "std_cv_ap": std_ap,
        })
        
        if mean_ap > best_score + 1e-5:
            best_score = mean_ap
            best_params = params
        elif abs(mean_ap - best_score) <= 1e-5 and params["depth"] < best_params["depth"]:
            best_params = params
            best_score = mean_ap
            
    pd.DataFrame(cv_records).to_csv(os.path.join(output_dir, "cv_results.csv"), index=False)
    
    # 2. Refit trên toàn bộ tập fit
    medians_fit = fit_preprocessor(df_fit, feature_cols, cat_cols)
    X_fit = transform_features(df_fit, feature_cols, cat_cols, medians_fit)
    X_thresh = transform_features(df_thresh, feature_cols, cat_cols, medians_fit)
    X_test_full = transform_features(df_test, feature_cols, cat_cols, medians_fit)
    
    full_fit_pool = Pool(X_fit, y_fit, cat_features=cat_cols)
    final_model = CatBoostClassifier(
        iterations=300,
        depth=best_params["depth"],
        learning_rate=best_params["learning_rate"],
        l2_leaf_reg=best_params["l2_leaf_reg"],
        random_seed=42,
        verbose=False,
        thread_count=4,
        train_dir="algorithm/CatBoost/catboost_info",
    )
    final_model.fit(full_fit_pool, verbose=False)
    
    # 3. Tìm ngưỡng F2 trên threshold
    thresh_pool = Pool(X_thresh, y_thresh, cat_features=cat_cols)
    thresh_probs = final_model.predict_proba(thresh_pool)[:, 1]
    opt_thresh_info = find_optimal_f2_threshold(y_thresh, thresh_probs, return_curve=True)
    best_threshold = opt_thresh_info["best_threshold"]
    if "curve_df" in opt_thresh_info:
        opt_thresh_info["curve_df"].to_csv(os.path.join(output_dir, "threshold_curve.csv"), index=False)
        
    # 4. Conformal Prediction trên bệnh nhân độc lập (encounter_id nhỏ nhất mỗi bệnh nhân)
    cal_sub_df = df_calib.sort_values("encounter_id").drop_duplicates("patient_nbr")
    test_sub_df = df_test.sort_values("encounter_id").drop_duplicates("patient_nbr")
    assert set(cal_sub_df["patient_nbr"]).isdisjoint(set(test_sub_df["patient_nbr"])), "Leakage calib-test!"
    
    X_cal_sub = transform_features(cal_sub_df, feature_cols, cat_cols, medians_fit)
    X_test_sub = transform_features(test_sub_df, feature_cols, cat_cols, medians_fit)
    
    cal_probs = final_model.predict_proba(Pool(X_cal_sub, cat_features=cat_cols))[:, 1]
    test_sub_probs = final_model.predict_proba(Pool(X_test_sub, cat_features=cat_cols))[:, 1]
    
    q_90 = calibrate(cal_sub_df["y"].values, cal_probs, 0.90)
    q_95 = calibrate(cal_sub_df["y"].values, cal_probs, 0.95)
    
    model_tag = f"CatBoost_d{best_params['depth']}_lr{best_params['learning_rate']}_l2_{best_params['l2_leaf_reg']}"
    conformal_summary_rows = summarize_conformal(
        cal_sub_df["y"].values, cal_probs, test_sub_df["y"].values, test_sub_probs, levels=(0.90, 0.95)
    )
    conformal_df = pd.DataFrame([
        dict(dataset="readmission", model=model_tag, **row)
        for row in conformal_summary_rows
    ])
    conformal_df.to_csv(os.path.join(output_dir, "conformal_results.csv"), index=False)
    
    # 5. Đánh giá trên toàn bộ 20.031 lượt test & Gán nhãn phân vân
    test_full_probs = final_model.predict_proba(Pool(X_test_full, cat_features=cat_cols))[:, 1]
    
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
    
    ext_errors = save_extreme_errors(df_test, pred_table, "encounter_id", os.path.join(output_dir, "extreme_errors.csv"))
    
    feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": final_model.get_feature_importance(),
    }).sort_values(by="importance", ascending=False)
    feat_imp.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
    
    runtime_sec = round(time.perf_counter() - t0, 2)
    run_meta = {
        "dataset": "readmission",
        "model": model_tag,
        "best_params": best_params,
        "best_cv_ap": best_score,
        "best_threshold_f2": best_threshold,
        "q_90": float(q_90),
        "q_95": float(q_95),
        "n_features": len(feature_cols),
        "n_cat_features": len(cat_cols),
        "runtime_seconds": runtime_sec,
    }
    with open(os.path.join(output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)
    
    return {
        "best_params": best_params,
        "best_threshold": best_threshold,
        "test_metrics": test_metrics,
        "conformal_summary": conformal_df,
        "q_90": q_90,
        "q_95": q_95,
        "feature_importance": feat_imp,
        "extreme_errors": ext_errors,
        "runtime_seconds": runtime_sec,
    }
