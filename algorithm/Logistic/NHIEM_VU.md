# Hoàng – Logistic Regression, Conformal Prediction và báo cáo cuối

**Hoàng chịu trách nhiệm viết, kiểm tra số liệu và hoàn thiện báo cáo cuối của nhóm** tại `reports/final_report.md`.

1. **Viết phần Logistic.** Dùng [kết quả đã chạy](artifacts) trên hai dataset để giải thích cách chọn C và L1/L2, ý nghĩa hệ số, ảnh hưởng của ngưỡng và các lỗi. Cleveland có 15 FP và 0 FN; chọn một FP để giải thích. Với Diabetes, phân tích một FP và một FN từ bảng dự đoán.
2. **Tổng hợp Conformal.** Ghép bốn dòng từ mỗi mô hình [KNN](../KNN/artifacts/conformal_4rows_for_hoang.csv), [Logistic](artifacts/conformal_4rows_for_hoang.csv) và [CatBoost](../CatBoost/artifacts/conformal_4rows_for_hoang.csv). Kiểm tra đủ 12 tổ hợp dataset × mô hình × mức 90%/95%. Giải thích độ bao phủ cùng tỷ lệ tập rỗng, một nhãn và hai nhãn.
3. **Nhận đóng góp.** Tùng gửi phần dữ liệu, KNN, phân tích lỗi và kết quả đối chiếu mẫu. Khải gửi phần CatBoost, quy tắc đánh giá, hai bảng so sánh và hồ sơ bất đồng. Hoàng kiểm tra số liệu trong bản thảo với các CSV chính thức.
4. **Viết báo cáo cuối.** Trình bày mục tiêu và nhãn của từng bài toán, dữ liệu và cách chia, ba phương pháp, hai bảng phân loại, phân tích lỗi, bảng Conformal, thảo luận, hạn chế và nguồn dữ liệu. Ghi rõ tập test từng được sử dụng ở đợt trước và sự khác nhau giữa số lượt nằm viện khi đánh giá phân loại với số bệnh nhân khi đánh giá Conformal. Gửi phần số liệu KNN và CatBoost cho từng người rà lại trước khi chốt bản nộp.
