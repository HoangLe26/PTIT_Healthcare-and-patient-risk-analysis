# HƯỚNG DẪN SỬ DỤNG MODULE BENCHMARK DÙNG CHUNG

Thư mục `benchmark/` chứa toàn bộ công cụ đánh giá, tối ưu ngưỡng và tổng hợp bảng kết quả dùng chung cho cả 3 thành viên (**Tùng, Hoàng, Khải**).

---

## 1. Cấu trúc thư mục
```text
benchmark/
  ├── thresholds.py       # Hàm chọn ngưỡng tối ưu F2 trên tập threshold
  ├── evaluation.py       # Hàm tính 14 metrics và ma trận nhầm lẫn
  ├── reporting.py        # Định dạng bảng so sánh 3 thuật toán
  ├── data_loader.py      # Tải dữ liệu và splits chuẩn
  └── run_benchmark.py    # Script tự động ghép kết quả 3 mô hình thành 2 bảng tổng hợp
```

---

## 2. Cách các thành viên tích hợp vào mô hình của mình

### Bước 1: Tìm ngưỡng $F_2$ trên tập `threshold` (Trước khi đánh giá test)
```python
from benchmark.thresholds import find_optimal_f2_threshold

# prob_thresh: vector xác suất P(y=1) trên tập threshold
thresh_info = find_optimal_f2_threshold(y_thresh, prob_thresh)
best_threshold = thresh_info["best_threshold"]
print("Ngưỡng F2 tối ưu:", best_threshold)
```

### Bước 2: Đánh giá mô hình trên tập `test` và xuất bảng dự đoán
```python
from benchmark.evaluation import compute_binary_metrics, build_test_prediction_table

# 1. Tính toán bộ metrics ở ngưỡng F2 vừa tìm được
test_metrics = compute_binary_metrics(y_test, prob_test, threshold=best_threshold)

# 2. Xuất bảng dự đoán test nộp cho Khải ghép bài
test_table = build_test_prediction_table(sample_ids, y_test, prob_test, threshold=best_threshold)
test_table.to_csv("test_predictions.csv", index=False)
```

---

## 3. Ghép bảng kết quả 3 thuật toán
Khi cả 3 người lưu file `test_predictions.csv` vào thư mục `artifacts/` của thuật toán mình:
- `algorithm/KNN/artifacts/`
- `algorithm/Logistic/artifacts/`
- `algorithm/CatBoost/artifacts/`

Chỉ cần chạy lệnh:
```powershell
python -m benchmark.run_benchmark
```
Hệ thống sẽ tự động tổng hợp ra 2 bảng Markdown và CSV tại `benchmark/results/`.
