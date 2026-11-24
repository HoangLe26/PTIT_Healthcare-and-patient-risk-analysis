import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


def evaluate_split_conformal_binary(
    y_calib: np.ndarray,
    prob_calib: np.ndarray,
    y_test: np.ndarray,
    prob_test: np.ndarray,
    alphas: List[float] = [0.10, 0.05],
) -> pd.DataFrame:
    """
    Triển khai Split Conformal Prediction cho phân loại nhị phân theo Mục 5 QUY_UOC_VA_PHAN_CONG_NHOM.md.
    
    Quy tắc:
    1. alphas = [0.10, 0.05] ứng với độ bao phủ mục tiêu 90% và 95%.
    2. Điểm nonconformity s = 1 - p(nhãn thật) trên tập hiệu chỉnh (calibration).
    3. Sắp xếp n điểm tăng dần. Ngưỡng q ở hạng ceil((n+1) * (1 - alpha)), đếm từ 1.
       Nếu hạng vượt n, dùng q = +inf.
    4. Với mẫu test, nhãn y được giữ lại nếu 1 - p(y) <= q.
    5. Đánh giá: Độ bao phủ thực tế, Tỷ lệ đúng 1 nhãn (single), Cả 2 nhãn (both), Tập rỗng (empty).
    """
    y_calib = np.asarray(y_calib).astype(int)
    prob_calib = np.asarray(prob_calib).astype(float)
    y_test = np.asarray(y_test).astype(int)
    prob_test = np.asarray(prob_test).astype(float)
    
    # Xác suất của nhãn thật trên calibration
    # p(y=1) = prob_calib, p(y=0) = 1 - prob_calib
    p_true_calib = np.where(y_calib == 1, prob_calib, 1.0 - prob_calib)
    # Nonconformity score s = 1 - p(true)
    scores_calib = 1.0 - p_true_calib
    scores_calib_sorted = np.sort(scores_calib)
    n = len(scores_calib_sorted)
    
    results = []
    
    for alpha in alphas:
        target_coverage = 1.0 - alpha
        # Hạng ceil((n + 1) * (1 - alpha)), index 1-based
        rank = math.ceil((n + 1) * target_coverage)
        if rank > n:
            q = float("inf")
        else:
            q = float(scores_calib_sorted[rank - 1])
            
        # Đánh giá trên tập test
        # Với mỗi mẫu test, kiểm tra 1 - p(0) <= q và 1 - p(1) <= q
        p0_test = 1.0 - prob_test
        p1_test = prob_test
        
        include_0 = (1.0 - p0_test) <= (q + 1e-9)
        include_1 = (1.0 - p1_test) <= (q + 1e-9)
        
        # Kiểm tra tập nhãn chứa nhãn thật
        covered_0 = (y_test == 0) & include_0
        covered_1 = (y_test == 1) & include_1
        is_covered = covered_0 | covered_1
        
        set_size = include_0.astype(int) + include_1.astype(int)
        
        empirical_coverage = float(np.mean(is_covered))
        pct_single = float(np.mean(set_size == 1))
        pct_both = float(np.mean(set_size == 2))
        pct_empty = float(np.mean(set_size == 0))
        
        results.append({
            "target_coverage": target_coverage,
            "alpha": alpha,
            "quantile_q": q,
            "empirical_coverage": empirical_coverage,
            "single_label_rate": pct_single,
            "both_labels_rate": pct_both,
            "empty_set_rate": pct_empty,
            "n_calib": n,
            "n_test": len(y_test),
        })
        
    return pd.DataFrame(results)
