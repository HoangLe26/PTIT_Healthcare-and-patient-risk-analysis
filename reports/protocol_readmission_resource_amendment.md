# Điều chỉnh tài nguyên trước test readmission

Ngày 11/10/2026, khoảng 18:15 giờ máy. Không mở test readmission tại thời điểm điều chỉnh. Heart đã được đánh giá bằng recipe riêng, không thay đổi mô hình/threshold/features của Heart.

Random Forest ban đầu tìm cả max_features=0.6 trên one-hot nhiều chiều, 6 cấu hình × 3 fold. Run vượt khoảng 15 phút chưa tạo artifact hoàn chỉnh; tiến trình huấn luyện riêng được dừng. Các model đã hoàn thành giữ nguyên checkpoint.

Điều chỉnh riêng search space RF: max_features='sqrt', n_estimators=80, max_depth ∈ {8,12}, min_samples_leaf ∈ {15,30}; 4 cấu hình × cùng 3 group folds. Không subsample cohort chính, không thay nhãn, cohort, split, primary metric, recipe calibration/threshold hoặc seed. RF có ngân sách nhỏ hơn các model khác; báo số cấu hình và runtime thực, không tuyên bố tuning công bằng tuyệt đối theo CPU.

Lý do là chi phí, không phải điểm test. Run bị ngắt là cancelled vì resource budget; không có metric RF giả từ run bị ngắt. Report cuối dùng run hoàn tất và giữ log stage_1_train_cancelled.log làm bằng chứng.

