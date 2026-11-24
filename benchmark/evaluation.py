import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)


def compute_binary_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_prob: Union[np.ndarray, List[float]],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Tính toàn diện bộ chỉ số phân loại nhị phân theo đặc tả môn học và quy ước nhóm.
    
    Bao gồm:
    - Chỉ số độc lập ngưỡng: ROC-AUC, Average Precision (AP), Brier Score, Log Loss.
    - Chỉ số phụ thuộc ngưỡng: Sensitivity/Recall, Specificity, Precision, F1, F2,
      Accuracy, Balanced Accuracy, Flagged Rate.
    - Ma trận nhầm lẫn: TP, FP, TN, FN.
    
    Parameters:
    -----------
    y_true : 1D array-like of binary ground truth labels {0, 1}
    y_prob : 1D array-like of predicted positive class probabilities [0, 1]
    threshold : float, decision threshold (default 0.5)
    
    Returns:
    --------
    dict of evaluated metrics
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    
    n_samples = len(y_true)
    if n_samples == 0:
        raise ValueError("y_true and y_prob must not be empty.")
        
    n_positives = int(np.sum(y_true == 1))
    n_negatives = int(np.sum(y_true == 0))
    prevalence = float(n_positives / n_samples)
    
    # 1. Các chỉ số độc lập ngưỡng
    # ROC-AUC & AP
    if n_positives > 0 and n_negatives > 0:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        ap = float(average_precision_score(y_true, y_prob))
    else:
        roc_auc = np.nan
        ap = np.nan
        
    # Brier Score
    brier = float(brier_score_loss(y_true, y_prob))
    
    # Log loss (kẹp xác suất tránh log(0))
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
    if n_positives > 0 and n_negatives > 0:
        ll = float(log_loss(y_true, y_prob_clipped))
    else:
        ll = np.nan
        
    # 2. Các chỉ số phụ thuộc ngưỡng
    y_pred = (y_prob >= threshold).astype(int)
    
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    
    # F1
    denom_f1 = (2 * tp + fp + fn)
    f1 = float(2.0 * tp / denom_f1) if denom_f1 > 0 else 0.0
    
    # F2
    denom_f2 = (5 * tp + 4 * fn + fp)
    f2 = float(5.0 * tp / denom_f2) if denom_f2 > 0 else 0.0
    
    accuracy = float((tp + tn) / n_samples)
    balanced_acc = float((sensitivity + specificity) / 2.0)
    flagged_rate = float((tp + fp) / n_samples)
    
    return {
        "n_samples": n_samples,
        "n_positives": n_positives,
        "n_negatives": n_negatives,
        "prevalence": prevalence,
        "threshold": float(threshold),
        "roc_auc": roc_auc,
        "average_precision": ap,
        "brier_score": brier,
        "log_loss": ll,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "sensitivity": sensitivity,
        "recall": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "f2": f2,
        "flagged_rate": flagged_rate,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def build_test_prediction_table(
    sample_ids: Union[np.ndarray, List[Any]],
    y_true: Union[np.ndarray, List[int]],
    y_prob: Union[np.ndarray, List[float]],
    threshold: float,
    conformal_sets_90: Optional[List[str]] = None,
    conformal_labels_90: Optional[List[str]] = None,
    conformal_sets_95: Optional[List[str]] = None,
    conformal_labels_95: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Tạo bảng dự đoán mẫu test chuẩn hóa theo quy ước nhóm (dòng 50 QUY_UOC_VA_PHAN_CONG_NHOM.md):
    Gồm ID dòng, nhãn thật, xác suất lớp dương, ngưỡng, nhãn dự đoán và loại kết quả TP/FP/TN/FN.
    Bổ sung các cột Conformal Prediction và 'NHÃN PHÂN VÂN' ở mức 90% và 95%.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    
    result_types = []
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            result_types.append("TP")
        elif t == 0 and p == 1:
            result_types.append("FP")
        elif t == 0 and p == 0:
            result_types.append("TN")
        elif t == 1 and p == 0:
            result_types.append("FN")
        else:
            result_types.append("UNKNOWN")
            
    data_dict = {
        "sample_id": sample_ids,
        "y_true": y_true,
        "y_prob": y_prob,
        "threshold": threshold,
        "y_pred": y_pred,
        "result_type": result_types,
    }
    
    if conformal_sets_90 is not None:
        data_dict["conformal_set_90"] = conformal_sets_90
    if conformal_labels_90 is not None:
        data_dict["conformal_decision_90"] = conformal_labels_90
        
    if conformal_sets_95 is not None:
        data_dict["conformal_set_95"] = conformal_sets_95
    if conformal_labels_95 is not None:
        data_dict["conformal_decision_95"] = conformal_labels_95
        
    df = pd.DataFrame(data_dict)
    return df


def extract_extreme_errors(
    pred_df: pd.DataFrame,
) -> Dict[str, Optional[pd.Series]]:
    """
    Trích xuất 1 ca FN và 1 ca FP sai nặng nhất (xác suất nghiêng mạnh nhất về lớp sai).
    Quy ước từ QUY_UOC_VA_PHAN_CONG_NHOM.md (Dòng 52):
    - FN (y_true=1, y_pred=0): chọn ca có y_prob THẤP NHẤT (càng gần 0 càng tự tin sai).
    - FP (y_true=0, y_pred=1): chọn ca có y_prob CAO NHẤT (càng gần 1 càng tự tin sai).
    """
    fn_candidates = pred_df[pred_df["result_type"] == "FN"]
    fp_candidates = pred_df[pred_df["result_type"] == "FP"]
    
    worst_fn = None
    worst_fp = None
    
    if len(fn_candidates) > 0:
        worst_fn = fn_candidates.sort_values(by="y_prob", ascending=True).iloc[0]
        
    if len(fp_candidates) > 0:
        worst_fp = fp_candidates.sort_values(by="y_prob", ascending=False).iloc[0]
        
    return {
        "worst_fn": worst_fn,
        "worst_fp": worst_fp,
        "n_fn": len(fn_candidates),
        "n_fp": len(fp_candidates),
    }
