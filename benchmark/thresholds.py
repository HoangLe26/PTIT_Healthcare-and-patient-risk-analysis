import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional


def find_optimal_f2_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    return_curve: bool = False,
) -> Dict[str, Any]:
    """
    Find optimal threshold that maximizes F2 score on threshold development set.
    
    Quy tắc từ QUY_UOC_VA_PHAN_CONG_NHOM.md (Dòng 48):
    - Tối đa hóa F2 trên tập `threshold`.
    - Xét các điểm xác suất phân biệt.
    - Dự đoán dương khi xác suất >= ngưỡng.
    - Tie-breaking: Khi bằng F2, chọn ngưỡng cao hơn (giúp giảm FP khi F2 tương đương).
    
    Parameters:
    -----------
    y_true : 1D array-like of binary ground truth labels {0, 1}
    y_prob : 1D array-like of predicted positive class probabilities [0, 1]
    return_curve : bool, whether to return the evaluation table for all evaluated thresholds
    
    Returns:
    --------
    dict containing:
        - best_threshold (float)
        - best_f2 (float)
        - precision (float)
        - recall (float)
        - tp, fp, tn, fn (int)
        - curve_df (pd.DataFrame, optional)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    
    if len(y_true) != len(y_prob):
        raise ValueError(f"Lengths do not match: len(y_true)={len(y_true)}, len(y_prob)={len(y_prob)}")
    
    total_positives = np.sum(y_true == 1)
    total_negatives = np.sum(y_true == 0)
    
    if total_positives == 0 or total_negatives == 0:
        # Edge case: all 0 or all 1
        return {
            "best_threshold": 0.5,
            "best_f2": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tp": 0,
            "fp": 0,
            "tn": total_negatives,
            "fn": total_positives,
        }
    
    # Xét các điểm xác suất phân biệt
    unique_probs = np.unique(y_prob)
    # Bao gồm cả cận dưới và trên nếu cần
    candidate_thresholds = np.sort(unique_probs)
    
    best_threshold = 0.5
    best_f2 = -1.0
    best_metrics = {}
    
    records = []
    
    # Quét qua từng ngưỡng
    for thresh in candidate_thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F2 = 5 * P * R / (4 * P + R) = 5 * TP / (5 * TP + 4 * FN + FP)
        denom = (5 * tp + 4 * fn + fp)
        f2 = (5.0 * tp / denom) if denom > 0 else 0.0
        
        if return_curve:
            records.append({
                "threshold": float(thresh),
                "f2": float(f2),
                "precision": float(prec),
                "recall": float(rec),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn
            })
            
        # Tie-breaker: if f2 is strictly greater, OR if f2 == best_f2 but thresh > best_threshold
        # Chúng ta dùng dung sai số thực nhỏ 1e-9 để so sánh bằng
        if f2 > best_f2 + 1e-9:
            best_f2 = f2
            best_threshold = thresh
            best_metrics = {
                "precision": prec,
                "recall": rec,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        elif abs(f2 - best_f2) <= 1e-9 and thresh > best_threshold:
            # Tie-break: chọn ngưỡng cao hơn
            best_threshold = thresh
            best_metrics = {
                "precision": prec,
                "recall": rec,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
            
    result = {
        "best_threshold": float(best_threshold),
        "best_f2": float(best_f2),
        "precision": float(best_metrics["precision"]),
        "recall": float(best_metrics["recall"]),
        "tp": int(best_metrics["tp"]),
        "fp": int(best_metrics["fp"]),
        "tn": int(best_metrics["tn"]),
        "fn": int(best_metrics["fn"]),
    }
    
    if return_curve:
        result["curve_df"] = pd.DataFrame(records)
        
    return result
