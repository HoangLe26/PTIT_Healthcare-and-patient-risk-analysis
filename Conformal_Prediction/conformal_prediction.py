"""Split conformal cho nhãn 0/1. p là vector xác suất lớp 1 của mô hình đã fit.

Ví dụ với pipeline sklearn đã fit và hai tập đã chọn đúng quy ước nhóm:

    col = list(model.classes_).index(1)
    p_cal = model.predict_proba(X_cal)[:, col]
    p_test = model.predict_proba(X_test)[:, col]
    rows = summarize(y_cal, p_cal, y_test, p_test)

Mỗi dòng kết quả tương ứng một mức bao phủ mục tiêu. coverage là tỷ lệ
tập dự đoán chứa nhãn thật, ambiguous là tỷ lệ giữ cả hai nhãn.
Với Diabetes, chọn lượt có encounter_id nhỏ nhất mỗi bệnh nhân trong từng tập
trước khi gọi. Tập calibration độc lập với huấn luyện và chọn tham số.
"""
from math import ceil
import numpy as np


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
    """Return the finite-sample order statistic for the requested coverage."""
    if not 0 < coverage < 1:
        raise ValueError("coverage must lie between 0 and 1")
    p = _prob(p_cal)
    y = _labels(y_cal, p)
    scores = np.where(y == 1, 1 - p, p)
    rank = ceil((len(scores) + 1) * coverage)
    return np.inf if rank > len(scores) else np.partition(scores, rank - 1)[rank - 1]


def prediction_sets(p_positive, q):
    """Boolean matrix: column j indicates whether label j is included."""
    p = _prob(p_positive)
    if not (0 <= q <= 1 or q == np.inf):
        raise ValueError("invalid conformal threshold")
    return np.column_stack((p <= q, 1 - p <= q))


def summarize(y_cal, p_cal, y_test, p_test, levels=(0.90, 0.95)):
    p = _prob(p_test)
    y = _labels(y_test, p)
    rows = []
    for level in levels:
        q = calibrate(y_cal, p_cal, level)
        sets = prediction_sets(p, q)
        sizes = sets.sum(axis=1)
        rows.append(dict(target=level, q=q, n_cal=len(y_cal), n_test=len(y),
                         coverage=sets[np.arange(len(y)), y].mean(),
                         empty=(sizes == 0).mean(), singleton=(sizes == 1).mean(),
                         ambiguous=(sizes == 2).mean(), mean_size=sizes.mean()))
    return rows
