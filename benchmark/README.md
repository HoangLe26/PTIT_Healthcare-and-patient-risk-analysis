# HƯỚNG DẪN SỬ DỤNG MODULE BENCHMARK DÙNG CHUNG

Thư mục `benchmark/` chứa toàn bộ công cụ tải dữ liệu, tối ưu ngưỡng $F_2$, đánh giá chỉ số, gán **nhãn phân vân** (Conformal Prediction) và tổng hợp bảng đối chiếu dùng chung cho cả 3 thuật toán (**KNN – Tùng, Logistic Regression – Hoàng, CatBoost – Khải**).

---

## 1. Cấu trúc thư mục
```text
benchmark/
  ├── data_loader.py      # Tải dữ liệu chuẩn (giữ 'None', đọc từ metadata feature_availability)
  ├── thresholds.py       # Hàm chọn ngưỡng tối ưu F2 trên tập threshold (hòa chọn ngưỡng cao hơn)
  ├── evaluation.py       # Hàm tính 14 metrics, ma trận nhầm lẫn và xuất bảng test chuẩn
  ├── conformal.py        # Split Conformal Prediction chuẩn nhóm + hàm gắn 'Nhãn phân vân'
  ├── reporting.py        # Định dạng 2 bảng so sánh 3 thuật toán
  └── run_benchmark.py    # Tự động kiểm tra khớp mẫu, lọc ca bất đồng và ghép bảng tổng hợp
```

---

## 2. Cách các thành viên tích hợp vào mô hình của mình

### Bước 1: Tìm ngưỡng $F_2$ trên tập `threshold` (Trước khi đánh giá test)
```python
from benchmark.thresholds import find_optimal_f2_threshold

thresh_info = find_optimal_f2_threshold(y_thresh, prob_thresh)
best_threshold = thresh_info["best_threshold"]
```

### Bước 2: Tính Conformal & gắn Nhãn phân vân (Mức 90% và 95%)
```python
from benchmark.conformal import calibrate, assign_conformal_prediction_labels

q_90 = calibrate(y_calib, prob_calib, 0.90)
q_95 = calibrate(y_calib, prob_calib, 0.95)
sets_90, labels_90 = assign_conformal_prediction_labels(prob_test, q_90, ambiguous_name="Phân vân")
sets_95, labels_95 = assign_conformal_prediction_labels(prob_test, q_95, ambiguous_name="Phân vân")
```

### Bước 3: Đánh giá mô hình trên tập `test` và xuất bảng dự đoán
```python
from benchmark.evaluation import compute_binary_metrics, build_test_prediction_table

test_metrics = compute_binary_metrics(y_test, prob_test, threshold=best_threshold)
test_table = build_test_prediction_table(
    sample_ids, y_test, prob_test, threshold=best_threshold,
    conformal_sets_90=sets_90, conformal_labels_90=labels_90,
    conformal_sets_95=sets_95, conformal_labels_95=labels_95,
)
test_table.to_csv("test_predictions.csv", index=False)
```

---

## 3. Chạy tổng hợp bảng đối chiếu 3 thuật toán
Khi cả 3 người lưu `test_predictions.csv` vào:
- `algorithm/KNN/artifacts/{heart,readmission}/test_predictions.csv`
- `algorithm/Logistic/artifacts/{heart,readmission}/test_predictions.csv`
- `algorithm/CatBoost/artifacts/{heart,readmission}/test_predictions.csv`

Chạy 1 lệnh duy nhất từ thư mục gốc:
```powershell
python -m benchmark.run_benchmark
```
Script sẽ tự động:
1. Kiểm tra `sample_id` và `y_true` khớp 100% giữa 3 mô hình.
2. Trích xuất danh sách các ca 3 thuật toán bất đồng (`disagreement_cases_*.csv`) cho báo cáo chung.
3. Xuất 2 bảng đối chiếu Markdown & CSV tại `benchmark/results/`.
