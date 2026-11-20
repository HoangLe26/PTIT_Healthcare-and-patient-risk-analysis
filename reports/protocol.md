# Protocol chốt trước khi huấn luyện và đánh giá test

Seed 42; CPU tối đa 2 luồng/model. Mục tiêu: phân loại bệnh tim hiện diện (Cleveland) và readmitted `<30` tại xuất viện. Hai bài toán độc lập.

## Dữ liệu và cohort

Tải archive UCI ID 45/296. Heart: 303 dòng gốc, target num>0, không nhân bản. Readmission: raw CSV và IDS_mapping; loại mã xuất viện có mô tả expired/hospice trong bảng mã, loại gender Unknown/Invalid; giữ transfer và nhiều encounters. ID chỉ phục vụ group/audit, không dùng features. Các `None` xét nghiệm giữ nguyên, `?` chuyển missing. Cột weight, payer_code, medical_specialty bị loại theo thiết kế hạn chế thông tin thiếu và cardinality; không học quyết định từ test. Biến tại xuất viện, gồm thuốc/kết quả của chính encounter, không tổng hợp dòng tương lai.

## Chia tập

Heart: stratified 80% development, 20% test; development 5-fold. Readmission: fold 0 của StratifiedGroupKFold(5, shuffle=True, seed42) làm test; development group CV 3-fold. Nội bộ development tiếp tục có fit/tune, calibration và threshold holdout độc lập theo patient (Heart stratified). Vì Heart nhỏ, chất lượng threshold/calibration từ holdout sẽ có uncertainty lớn; ghi counts và không hứa constraint recall.

Holdout calibration/threshold không dùng chọn hyperparameters. Dùng tuning fit set để chọn họ model và siêu tham số. Model cuối fit trên fit/tune set và calibrator fit calibration; threshold chọn threshold set. Không refit sau đó trên toàn bộ development vì sẽ phá sự độc lập của calibrator/threshold. Full-data nghĩa là không subsample cohort chính, không phải model được fit trên test/calibration/threshold. Các họ model được chọn trước; số liệu CV chỉ là selection scores, không phải nested-CV unbiased assessment.

## Model selection

Heart primary ROC-AUC; readmission AP. Same fixed CV splits trong dataset. Randomized search seed42: tối đa 12 cấu hình/Heart, 6/readmission; giảm khi số tổ hợp nhỏ. LR, KNN, GaussianNB/QDA numerical, XGBoost cho Heart; LR, GaussianNB numerical, Random Forest, XGBoost cho readmission. Các ablation LR numerical, restricted Heart, engineered readmission, L1/L2/class_weight, KNN unscaled đều định trước. Lựa chọn theo mean CV primary metric; hòa trong 0.002 thì ưu tiên LR, GaussianNB, KNN, QDA, XGBoost/RF theo mức đơn giản. Không chọn winner bằng test.

## Calibration và threshold

Sigmoid calibration trên calibration holdout (logit raw probabilities, logistic unregularized). Dùng threshold holdout độc lập để so sánh raw/calibrated probabilities, báo Brier/log loss. Recipe deployment dùng sigmoid nếu Brier tốt hơn trên threshold holdout, nếu không dùng raw; quyết định này diễn ra trước test. Threshold mặc định tối đa F2, hòa chọn ngưỡng cao hơn. Báo thêm ngưỡng 0.5, sensitivity>=0.85 tối đa specificity và cost ratio 1/2/5, đều chọn bằng threshold holdout. Top-k test báo ngân sách 5/10/20% chốt trước. Không khẳng định đây là tiêu chuẩn lâm sàng.

## Thí nghiệm mở rộng

Split conformal Heart: model LR cố định dùng fit set, nonconformity `1-p(true class)` trên calibration set, test prediction sets ở alpha 0.05/0.10. Không dùng calibration set để tune LR conformal. Không áp dụng bảo đảm row-iid cho encounters lặp. TabPFN: kiểm tra nguồn/release và access; nếu không thể tải checkpoint công khai không tài khoản, dùng conformal và ghi lý do, không bịa thực nghiệm TabPFN.

## Đánh giá test cuối

Freeze: hashes pipeline, calibration, threshold, features, config và split. Evaluate một đợt recipe đóng băng; lưu predictions. Accuracy, balanced accuracy, precision/recall/specificity, F1/F2, ROC-AUC/AP, Brier/log loss; ngưỡng 0.5 và tuned. Bootstrap 1000 replicate fixed predictions: rows Heart; clusters patient với multiplicity readmission. CI95% percentile; báo invalid replicate. So sánh model paired cùng bootstrap draws. Subgroup cần >=30 rows, >=10 positive và >=10 negative để suy luận metric; các nhóm nhỏ gắn NA. Test errors hậu kiểm không dùng sửa model.

## Ablation, tài nguyên và giới hạn

E01–E12 theo đặc tả. E08 L1/L2 mean validation và nonzero; E09 k/distance/weights/scaling; E10 cùng numerical feature set; E11 NumPy gradient/loss curves; E12 encounter top-k. Learning curves LR ba cỡ train 30/60/100% trên cố định CV, lấy theo group ở readmission. Không ép KNN readmission full do chi phí distance; pilot matched-subsample có thể tùy chọn. Không có temporal/hospital external validation vì metadata không đủ. Test Heart nhỏ; data lịch sử; không dùng clinically.

