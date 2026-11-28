# Tùng – KNN và dữ liệu dùng chung

**Bàn giao cho Hoàng:** phần dữ liệu và KNN để đưa vào báo cáo cuối, kèm bằng chứng ba mô hình được so sánh trên cùng mẫu.

1. Mô tả nhãn và tiêu chí chọn hồ sơ của Cleveland và Diabetes. Giải thích cách chia fold, cách giữ các lượt của một bệnh nhân trong cùng tập và cách chọn một lượt mỗi người khi đánh giá Conformal.
2. Từ [báo cáo KNN](README.md), trình bày ảnh hưởng của chuẩn hóa, số láng giềng và cách bỏ phiếu. Ghi cấu hình được chọn cùng điểm CV trên từng dataset.
3. Với từng dataset, giải thích một ca báo nhầm và một ca bỏ sót nếu có bằng đặc trưng và láng giềng. Gửi Hoàng phần nhận xét, hạn chế và [bốn dòng Conformal](artifacts/conformal_4rows_for_hoang.csv).
4. Khi Khải ghép bảng, chạy `python -m algorithm.KNN.run_knn --check-alignment` từ gốc repo. Kiểm tra [bảng đối chiếu](artifacts/test_alignment.csv) xác nhận ID và nhãn thật khớp với Logistic và CatBoost trên 61 hồ sơ Cleveland và 20.031 lượt Diabetes.
