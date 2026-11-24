"""Split Conformal Prediction toolkit.
Được chuẩn hóa theo code mẫu từ Conformal_Prediction/conformal_prediction.py.
Bổ sung cơ chế sinh 'nhãn phân vân' (Ambiguous prediction sets) trong phân loại lâm sàng.
"""
from math import ceil
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union


def _prob(values):
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or not p.size or not np.all(np.isfinite(p) & (p >= 0) & (p <= 1)):
        raise ValueError("Expected a nonempty vector of probabilities in [0, 1]")
    return p


def _labels(values, p):
    y = np.asarray(values)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected one binary label per probability")
    return y.astype(int)


def calibrate(y_cal, p_cal, coverage):
    """
    Tính ngưỡng nonconformity quantile q ở mức bao phủ coverage (ví dụ 0.90 hoặc 0.95).
    scores = 1 - P(nhãn thật).
    """
    if not 0 < coverage < 1:
        raise ValueError("coverage must lie between 0 and 1")
    p = _prob(p_cal)
    y = _labels(y_cal, p)
    scores = np.where(y == 1, 1 - p, p)
    rank = ceil((len(scores) + 1) * coverage)
    return np.inf if rank > len(scores) else np.partition(scores, rank - 1)[rank - 1]


def prediction_sets(p_positive, q):
    """
    Trả về ma trận boolean (N x 2):
    Cột 0: Nhãn 0 có thuộc tập dự đoán hay không (1 - (1-p) <= q <=> p <= q).
    Cột 1: Nhãn 1 có thuộc tập dự đoán hay không (1 - p <= q).
    """
    p = _prob(p_positive)
    if not (0 <= q <= 1 or q == np.inf):
        raise ValueError("invalid conformal threshold")
    return np.column_stack((p <= q, 1 - p <= q))


def assign_conformal_prediction_labels(
    p_positive: np.ndarray,
    q: float,
    label_0_name: str = "0",
    label_1_name: str = "1",
    ambiguous_name: str = "Phân vân",
    empty_name: str = "Rỗng",
) -> Tuple[List[str], List[str]]:
    """
    Chuyển đổi tập dự đoán Conformal thành nhãn phân loại có hỗ trợ 'NHÃN PHÂN VÂN'.
    
    Quy tắc gán nhãn:
    - Tập chứa {0}: Dự đoán dứt khoát nhãn 0 (Khỏe mạnh / Không tái nhập viện).
    - Tập chứa {1}: Dự đoán dứt khoát nhãn 1 (Có bệnh / Tái nhập viện).
    - Tập chứa {0, 1}: NHÃN PHÂN VÂN (Ambiguous) - Mô hình không đủ tự tin để đưa ra
      1 nhãn duy nhất ở mức tin cậy yêu cầu; cần chuyển bác sĩ / chuyên gia hội chẩn thêm.
    - Tập chứa {}: Tập rỗng (Empty) - Ca dị biệt ngoài phân phối.
    
    Returns:
    --------
    set_strings : list of str (ví dụ: "{0}", "{1}", "{0, 1}", "{}")
    decision_labels : list of str (ví dụ: "0", "1", "Phân vân", "Rỗng")
    """
    sets = prediction_sets(p_positive, q)
    set_strings = []
    decision_labels = []
    
    for inc_0, inc_1 in sets:
        if inc_0 and inc_1:
            set_strings.append("{0, 1}")
            decision_labels.append(ambiguous_name)
        elif inc_0 and not inc_1:
            set_strings.append("{0}")
            decision_labels.append(label_0_name)
        elif not inc_0 and inc_1:
            set_strings.append("{1}")
            decision_labels.append(label_1_name)
        else:
            set_strings.append("{}")
            decision_labels.append(empty_name)
            
    return set_strings, decision_labels


def summarize(y_cal, p_cal, y_test, p_test, levels=(0.90, 0.95)):
    """
    Tổng hợp kết quả Conformal Prediction theo đúng cấu trúc của nhóm:
    target, q, n_cal, n_test, coverage, empty, singleton, ambiguous, mean_size.
    """
    p = _prob(p_test)
    y = _labels(y_test, p)
    rows = []
    for level in levels:
        q = calibrate(y_cal, p_cal, level)
        sets = prediction_sets(p, q)
        sizes = sets.sum(axis=1)
        rows.append(dict(
            target=level,
            q=float(q),
            n_cal=len(y_cal),
            n_test=len(y),
            coverage=float(sets[np.arange(len(y)), y].mean()),
            empty=float((sizes == 0).mean()),
            singleton=float((sizes == 1).mean()),
            ambiguous=float((sizes == 2).mean()),
            mean_size=float(sizes.mean()),
        ))
    return rows
