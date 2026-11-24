# BÁO CÁO KỸ THUẬT: CÁC MODULE DÙNG CHUNG (COMMON MODULES)

* **Người phụ trách xây dựng:** Khải
* **Dự án:** Healthcare and Patient Risk Analysis (Bài tập lớn Học máy - PTIT)
* **Vị trí mã nguồn:** `benchmark/`
* **Môi trường thực thi chuẩn:** `D:\Project\venv1\Scripts\python.exe`
* **Trạng thái kiểm thử:** Đạt 100% (6/6 unit tests)

---

## 1. Mục tiêu và phạm vi

Để đảm bảo kết quả thực nghiệm của 3 thành viên (**Tùng – KNN**, **Hoàng – Logistic Regression**, **Khải – CatBoost**) có tính nhất quán, có thể so sánh công bằng và không bị rò rỉ dữ liệu (Data Leakage), Khải chịu trách nhiệm thiết kế, lập trình và kiểm thử toàn bộ hệ thống tính toán dùng chung gồm:
1. **Module chọn ngưỡng tối ưu theo độ đo y tế $F_2$** ([thresholds.py](benchmark/thresholds.py)).
2. **Module tính bộ chỉ số đánh giá & Ma trận nhầm lẫn** ([evaluation.py](benchmark/evaluation.py)).
3. **Module định dạng và ghép 2 bảng đối chiếu kết quả** ([reporting.py](benchmark/reporting.py)).

---

## 2. Chi tiết Module 1: Chọn ngưỡng tối ưu $F_2$ (`thresholds.py`)

### 2.1. Động lực y tế và bài toán chọn ngưỡng
Ngưỡng mặc định $0.5$ thường dẫn đến việc bỏ sót nghiêm trọng các ca bệnh (False Negative cao) khi dữ liệu y tế bị mất cân bằng (như bài toán Tái nhập viện chỉ có ~11% ca dương tính). Nhóm thống nhất sử dụng độ đo **$F_2$ score** để chọn ngưỡng quyết định vì $F_2$ đặt trọng số cho **Recall cao gấp đôi Precision**, ưu tiên tối đa việc phát hiện ca bệnh.

### 2.2. Công thức tính toán
Với mỗi ngưỡng ứng viên $t$, mẫu được dự đoán dương tính nếu $P(y=1) \ge t$. Điểm $F_2$ được tính trực tiếp từ các phần tử của ma trận nhầm lẫn:
$$F_2 = (1 + 2^2) \times \frac{\text{Precision} \times \text{Recall}}{2^2 \times \text{Precision} + \text{Recall}} = \frac{5 \cdot TP}{5 \cdot TP + 4 \cdot FN + FP}$$

### 2.3. Quy ước kỹ thuật và Xử lý hòa điểm (Tie-breaker)
* **Tập dữ liệu áp dụng:** Chạy độc lập trên tập `threshold` (Dataset A: 37 mẫu; Dataset B: 9.854 mẫu). **Tuyệt đối không dùng tập test để tìm ngưỡng.**
* **Không gian quét:** Quét qua toàn bộ các giá trị xác suất phân biệt (unique sorted probabilities) có trong tập `threshold`.
* **Quy tắc Tie-breaker:** Khi có nhiều ngưỡng cùng đạt giá trị $F_2$ tối đa, thuật toán bắt buộc chọn **ngưỡng cao nhất** ($t^* = \max \{t \mid F_2(t) = F_{2,\max}\}$). Quy tắc này giúp triệt tiêu các ca dương tính giả (FP) không cần thiết khi điểm $F_2$ đã bão hòa.

---

## 3. Chi tiết Module 2: Tính toán chỉ số & Ma trận nhầm lẫn (`evaluation.py`)

Module cung cấp hàm chuẩn hóa `compute_binary_metrics(y_true, y_prob, threshold)` tính toán toàn diện 14 chỉ số phục vụ báo cáo:

### 3.1. Các chỉ số độc lập ngưỡng (Ranking & Calibration Metrics)
* **ROC-AUC:** Đánh giá khả năng xếp hạng phân biệt giữa người có bệnh và người khỏe mạnh (chỉ số chính của Dataset A).
* **Average Precision (AP / PR-AUC):** Đánh giá diện tích dưới đường cong Precision-Recall, đặc biệt nhạy và chuẩn xác với bài toán mất cân bằng lớp (chỉ số chính của Dataset B).
* **Brier Score:** Sai số bình phương trung bình của xác suất dự đoán, đo lường độ tin cậy thực tế của xác suất ($P \in [0, 1]$).
* **Log Loss:** Cross-entropy loss có kẹp epsilon ($10^{-15}$) chống lỗi số học vô cùng.

### 3.2. Các chỉ số phụ thuộc ngưỡng quyết định
* **Ma trận nhầm lẫn:** Đếm chính xác $TP, FP, TN, FN$.
* **Sensitivity / Recall:** $\frac{TP}{TP + FN}$ (Tỷ lệ bắt trúng bệnh nhân thực tế).
* **Specificity:** $\frac{TN}{TN + FP}$ (Tỷ lệ nhận diện đúng người bình thường).
* **Precision:** $\frac{TP}{TP + FP}$ (Độ tin cậy của cảnh báo nguy cơ).
* **$F_1$ score** và **$F_2$ score**.
* **Accuracy** và **Balanced Accuracy** ($\frac{\text{Sensitivity} + \text{Specificity}}{2}$).
* **Flagged Rate:** $\frac{TP + FP}{N}$ (Tỷ lệ lượt khám bị gắn cờ cảnh báo nguy cơ cao).

### 3.3. Chuẩn hóa đầu ra tập Test & Trích xuất lỗi cực biên
* Hàm `build_test_prediction_table`: Xuất bảng dự đoán chi tiết từng mẫu gồm: `sample_id, y_true, y_prob, threshold, y_pred, result_type (TP/FP/TN/FN)`.
* Hàm `extract_extreme_errors`: Tự động tìm ra **1 ca FN sai nặng nhất** ($P$ thấp nhất trong các ca có bệnh) và **1 ca FP sai nặng nhất** ($P$ cao nhất trong các ca không bệnh) để phục vụ phân tích sâu trong báo cáo.

---

## 4. Chi tiết Module 3: Ghép bảng đối chiếu kết quả (`reporting.py`)

Module cung cấp 2 hàm xuất bảng Markdown và DataFrame đối chiếu 3 mô hình (KNN, Logistic Regression, CatBoost):
* `format_comparison_table_dataset_a`: Định dạng bảng kết quả cho Cleveland Heart Disease (N=61), hiển thị chi tiết số ca đếm thực tế $TP, FP, TN, FN$ do cỡ mẫu nhỏ.
* `format_comparison_table_dataset_b`: Định dạng bảng kết quả cho Diabetes Readmission (N=20.031), hiển thị Average Precision, ROC-AUC, $F_2$, Brier score và tỷ lệ gắn cờ theo dõi.

---

## 5. Kết quả kiểm thử đơn vị (Unit Tests)

Bộ kiểm thử được viết tại [tests/test_common_modules.py](tests/test_common_modules.py), đã chạy kiểm chứng thành công trên cả 2 môi trường:
* **Môi trường hệ thống:** `Ran 6 tests in 0.273s - OK`
* **Môi trường ảo `D:\Project\venv1`:** `Ran 6 tests in 0.028s - OK`

| Tên Test Case | Mục đích kiểm tra | Kết quả |
|---|---|:---:|
| `test_find_optimal_f2_threshold_basic` | Kiểm tra thuật toán quét và tìm ngưỡng $F_2$ cơ bản | **PASS** |
| `test_tie_breaker_picks_higher_threshold` | Xác minh bắt buộc chọn ngưỡng cao hơn khi hòa điểm $F_2$ | **PASS** |
| `test_compute_binary_metrics` | Kiểm tra tính đúng đắn của 14 chỉ số và bảo toàn tổng mẫu | **PASS** |
| `test_build_test_prediction_table` | Kiểm tra định dạng bảng dự đoán tập test | **PASS** |
| `test_extract_extreme_errors` | Kiểm tra độ chính xác khi lọc ca FN/FP sai nặng nhất | **PASS** |
| `test_reporting_formatters` | Kiểm tra cấu trúc và tính toàn vẹn của bảng ghép kết quả | **PASS** |

---

## 6. Hướng dẫn các thành viên sử dụng

Hai thành viên Tùng (KNN) và Hoàng (Logistic Regression) chỉ cần import và sử dụng như sau:

```python
from benchmark.thresholds import find_optimal_f2_threshold
from benchmark.evaluation import compute_binary_metrics, build_test_prediction_table

# 1. Tìm ngưỡng F2 trên tập threshold (trước khi đánh giá test)
thresh_info = find_optimal_f2_threshold(y_thresh, prob_thresh)
best_threshold = thresh_info["best_threshold"]

# 2. Đánh giá mô hình trên tập test với ngưỡng vừa tìm được
test_metrics = compute_binary_metrics(y_test, prob_test, threshold=best_threshold)

# 3. Xuất bảng dự đoán test nộp cho Khải ghép bài
test_table = build_test_prediction_table(sample_ids, y_test, prob_test, threshold=best_threshold)
test_table.to_csv("test_predictions.csv", index=False)
```
