# Khải – CatBoost và so sánh ba mô hình

**Bàn giao cho Hoàng:** phần CatBoost, hai bảng phân loại chính thức và một ca các mô hình dự đoán khác nhau.

1. Hoàn thiện [báo cáo CatBoost](README.md) cho cả hai dataset. Ghi cấu hình CV, cách dừng sớm, số vòng huấn luyện cuối và ảnh hưởng của các lựa chọn này. Giải thích ca báo nhầm và bỏ sót của từng dataset nếu có.
2. Mô tả cách chọn ngưỡng F2 trên tập threshold, cách xử lý hòa điểm và cách đếm TP, FP, TN, FN trong [phần đánh giá chung](../../reports/BAO_CAO_PHAN_DUNG_CHUNG_KHAI.md).
3. Sau khi đủ dự đoán test, chạy `python -m benchmark.run_benchmark` từ gốc repo. Kiểm tra [hai bảng kết quả](../../benchmark/results) đều có ba hàng KNN, Logistic và CatBoost. So sánh theo ROC-AUC trên Cleveland và Average Precision trên Diabetes.
4. Chọn một hồ sơ có dự đoán bất đồng. Ghi ID, nhãn thật, xác suất, ngưỡng và kết luận của từng mô hình. Gửi Hoàng hai bảng, phần giải thích và [bốn dòng Conformal của CatBoost](artifacts/conformal_4rows_for_hoang.csv).
