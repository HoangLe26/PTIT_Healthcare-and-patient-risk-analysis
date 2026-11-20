# Câu hỏi bảo vệ và trả lời theo thực nghiệm đã đóng băng

Số liệu đọc từ artifact cuối ngày 11/10/2026. Không lựa chọn hoặc điều chỉnh mô hình dựa trên test. File nguồn chính: `artifacts/metrics/*_test_metrics.csv`, `*_confidence_intervals.csv`, `*_threshold_policies.csv`, `*_controlled_ablations_summary.csv`, `optimization.json`, `heart_conformal.json` và manifest chọn mô hình.

## 1. Vì sao chọn hai bộ dữ liệu này?

Cleveland cho bài toán nhỏ về bệnh hiện diện; readmission cho bài toán encounter-level lớn, mất cân bằng và có nhiều lượt cùng bệnh nhân. Hai bộ kiểm tra hai bối cảnh khác nhau trong cùng khung đánh giá. Không ghép hai outcome, không trung bình hóa AUC của chúng, không gọi chúng là external validation của cùng một model.

## 2. Định nghĩa lớp dương và thời điểm dự đoán là gì?

Heart: `num > 0`, phân loại bệnh hiện diện khi đầu vào xét nghiệm đã sẵn có. Readmission: `readmitted == '<30'` tại xuất viện; `'>30'` và `'NO'` là âm. Dữ liệu không chứng minh nguy cơ mắc bệnh tim trong tương lai hoặc công cụ sàng lọc trước xét nghiệm.

## 3. Có thực sự 303 bệnh nhân Heart độc lập không?

Có 303 quan sát, không có patient ID xác thực để kiểm chứng người lặp. `patient_nbr` của Heart là khóa audit tạo từ hàng. Giả định các hàng độc lập cần được ghi là giả định, đặc biệt khi bootstrap/conformal. Readmission có patient_nbr ẩn danh thật để tách nhóm.

## 4. Cohort readmission được lọc thế nào?

101.766 lượt raw, loại 2.423 expired/hospice theo bảng mã xuất viện [11,13,14,19,20,21], sau đó loại 3 gender Unknown/Invalid; còn 99.340 lượt của 69.987 patient IDs. Giữ transfers và encounter lặp. Các bước lọc dựa trên nghĩa metadata, không dựa vào điểm test.

## 5. Mô hình thực sự học trên bao nhiêu hàng?

Heart fit 169, calibration 36, threshold 37, test 61. Readmission fit 59.372, calibration 10.083, threshold 9.854, test 20.031. Mô hình cuối không refit lên toàn bộ development. Cụm từ full cohort nghĩa không subsample cohort chính, không nghĩa dùng cả test/holdout để fit.

## 6. Vì sao group split readmission là bắt buộc?

Một người có thể có nhiều lượt với thông tin liên quan. Random row split có thể để cùng người ở train và test, gây đánh giá lạc quan. Dự án tách patient_nbr ở fit/calibration/threshold/test và fold tuning. Điểm test vẫn tính theo lượt, còn bootstrap lấy theo cluster bệnh nhân.

## 7. CV của dự án có phải nested CV hoặc OOF calibration không?

Không. CV chỉ chạy trên fit/tune để chọn model và hyperparameters, Heart 5 folds, readmission 3 group folds. Calibration và threshold dùng holdout riêng. Mean/SD CV là selection score, không phải 95% CI hoặc đánh giá nested độc lập. Đánh giá cuối lấy từ test đã khóa.

## 8. Chọn mô hình thắng bằng gì?

Theo mean CV metric chính: Heart ROC-AUC, readmission AP, có quy tắc ưu tiên model đơn giản khi chênh dưới 0,002. Heart chọn LR (CV AUC 0,9193); readmission chọn XGBoost (CV AP khoảng 0,2204). Test không quyết định winner.

## 9. Vì sao dùng AP cho readmission thay vì accuracy?

Test có 2.319/20.031 dương, prevalence 11,577%. Một model luôn âm đạt accuracy khoảng 88,423% nhưng bỏ mọi ca dương. AP xem ranking precision–recall; dummy prior AP bằng prevalence 0,11577. XGB AP 0,22894 cao hơn tham chiếu nhưng chưa đủ chứng minh ích lợi vận hành.

## 10. Kết quả Heart có phải hoàn hảo?

Không. LR AUC 0,96212, CI95% bootstrap [0,90857; 0,99451]. Tại threshold 0,269344: TP28, FN0, FP14, TN19; sensitivity quan sát 1 nhưng specificity 0,57576 và precision 0,66667. 42/61 hàng bị gắn cờ. Cỡ test nhỏ và FP đáng kể hạn chế kết luận.

## 11. Bootstrap sensitivity Heart bằng [1,1] có đủ tin cậy không?

Fixed predictions không có FN nên bootstrap không tạo ra kiểu sai chưa có. Khoảng suy biến không chứng minh sensitivity quần thể 100%. File `heart_sensitivity_exact_ci.json` bổ sung Clopper–Pearson 95% [0,87656;1] cho 28/28 dương, có điều kiện trên giả định positive observations iid. Khoảng này cũng không xử lý mọi shift hoặc sai lệch chọn mẫu.

## 12. Kết quả readmission có đủ tốt để dùng trong bệnh viện?

Chưa có bằng chứng. XGB AUC 0,66947, AP 0,22894, AP CI95% [0,20782;0,25007]. Threshold 0,090540 cho TP1788/FN531/FP9729/TN7983, recall 0,77102 nhưng precision 0,15525. Gắn cờ 11.517 lượt (57,50% test) để tìm 1.788 dương tạo tải false alarms cao. Cần kiểm định ngoài và đánh giá workflow/cost thật.

## 13. XGBoost hơn Logistic Regression chắc chắn ở mọi tiêu chí không?

Không. Paired cluster bootstrap trên cùng draws: AP chênh XGB−LR 0,012124, CI [0,006061;0,017717], AUC chênh 0,007767, CI [0,003977;0,011452]. Chênh F2 0,002925 có CI [−0,003339;0,008929], gồm 0. Bằng chứng ranking trong cohort khác với cải thiện quyết định threshold hoặc lợi ích lâm sàng.

## 14. Vì sao threshold thấp hơn 0,5?

Threshold tối đa F2 trên threshold holdout, ưu tiên recall hơn precision. Nhãn hiếm và xác suất phần lớn thấp nên 0,5 không thích hợp với mục tiêu F2. Threshold thấp tăng TP đồng thời tăng FP/gắn cờ. Các policy recall85 và cost ratio 1/2/5 cũng chọn trước test; chúng là minh họa chi phí, không phải chuẩn lâm sàng.

## 15. Nếu ngân sách chỉ cho xem 10% lượt thì sao?

`readmission_topk.csv`: top 10% tương ứng 2.004 lượt, tìm 546 dương; precision@k 27,2455%, recall@k 23,5446%. Top 5%: 1.002 lượt/321 dương, precision 32,0359%, recall 13,8422%. Top 20%: 4.007 lượt/899 dương, precision 22,4357%, recall 38,7667%. Top-k theo encounter, không tự chuyển thành ngân sách bệnh nhân duy nhất.

## 16. Calibration có luôn cải thiện không?

Không. Heart sigmoid được chọn trước test vì threshold-holdout Brier 0,148846 tốt hơn raw 0,149684 rất ít; calibration chỉ 36 hàng. Trên test raw Brier 0,085914/logloss 0,296926 so sigmoid 0,105827/0,369279, kém hơn. Dự án giữ recipe đã khóa; đây là bằng chứng uncertainty/small-sample overfit khả dĩ, không tự chứng minh cơ chế gây lỗi.

## 17. So raw và calibrated tại cùng threshold có công bằng không?

Đó là diagnostic ở cùng numeric threshold, không phải so sánh hai policy đã tune riêng tương đương. Sigmoid thay thang probability nên cùng con số có thể gắn cờ khác tập hàng. Brier/log loss thích hợp cho kiểm tra xác suất. Readmission raw Brier 0,0975052 vs sigmoid 0,0975069 gần như tương đương trên test; không tuyên bố cải thiện calibration đáng kể.

## 18. Class weighting giúp gì và có đánh đổi gì?

Controlled readmission E02 giữ folds/features/regularization: AP raw tăng nhẹ 0,213326 lên 0,214444, nhưng Brier tăng 0,095451 lên 0,222491. Đây là probability chưa post-hoc calibrate. Trọng số thay objective và mức xác suất; không đồng nhất cải thiện recall/ranking với calibration. Kết quả final tuned/calibrated phải báo riêng.

## 19. Feature engineering và bỏ xét nghiệm có luôn tốt không?

Không. Controlled readmission E05 AP full 0,213326 vs engineered 0,211206, chưa chứng minh cải thiện. Heart E06 AUC full 0,919262 vs restricted 0,888840: giảm thông tin chẩn đoán có thể làm giảm ranking nhưng đổi tính khả dụng. Cùng LR/folds giúp tách hiệu ứng feature set khỏi thuật toán; không suy ra causal effect của xét nghiệm.

## 20. KNN/Gaussian nối kiến thức môn học thế nào?

Controlled Heart E09 scaling False AUC 0,77810 vs True 0,91432 ở đối chứng giữ các lựa chọn khác; đơn vị đo chi phối khoảng cách. E10 cùng numerical features: GaussianNB 0,79844, LR 0,80395, QDA regularized 0,80725. Chênh nhỏ không chứng minh full covariance tốt hơn mọi bối cảnh. NB giả định độc lập có điều kiện; QDA ước lượng covariance và cần regularization.

## 21. Có thực sự tự viết và kiểm tra tối ưu LR không?

Có `optimization.json` và loss curve. NumPy chạy objective unpenalized mean Bernoulli NLL trên fit numerical Heart, 15.000 steps; gradient max error 6,517×10⁻¹¹ và prediction max difference với sklearn 5,576×10⁻⁹ trong điều kiện objective tương đương. Đây là minh chứng thuật toán, không thay pipeline cuối hoặc metric test.

## 22. Conformal coverage 100% có nghĩa chắc chắn không?

Không. Heart alpha 0,05: empirical coverage 1 nhưng mean set size 1,88525, ambiguous 88,5246%. Alpha 0,10: coverage 1, mean size 1,54098, ambiguous 54,0984%; không có empty sets. Nhiều tập `{0,1}` chứa nhãn thật nhưng ít thông tin phân loại. Calibration36/test61 và exchangeability giả định giới hạn bảo đảm. Không áp dụng row-iid conformal cho encounter readmission lặp.

## 23. Vì sao không chạy TabPFN, có bị lỗi không?

Không có run TabPFN nên không có lỗi thất bại. Trạng thái là skipped feasibility review theo phạm vi/ngân sách. Literature cập nhật Nature2025, TabPFN2.5 và report3.5 ngày15/09/2026. Repository bản mới mô tả auth/license flow, model card v2 công khai; chưa probe download. Conformal là mở rộng thực nghiệm đã có artifact, không gọi nó là phát minh mới nhất.

## 24. SHAP giải thích gì, có chỉ nguyên nhân y khoa không?

Không. Heart LR SHAP giải thích raw log-odds trước calibration, explicit background80 hàng fit. Readmission dùng XGBoost native TreeSHAP `pred_contribs=True`, background implicit training node cover fit59.372; reconstruction raw margin max error khoảng8,34×10⁻⁷. Không gọi native method là interventional80-row background. Permutation raw pipeline là global chính; mean absolute SHAP của vài ca lựa chọn không phải population importance.

## 25. Chọn ca lỗi có gây hiểu nhầm không?

Case studies là hậu kiểm từ CSV hiện hành. Heart không có FN nên chỉ trình bày FP/TP/TN. Ca được xếp theo probability của nhãn vận hành: predicted1 dùng p, predicted0 dùng1−p. Với threshold dưới0,5, `max(p,1−p)` có thể chọn nhầm FP probability thấp nên không dùng cách đó. Mô hình tự tin sai không chứng minh nhãn dữ liệu sai.

## 26. Tuning budget và train_seconds có công bằng tuyệt đối không?

Không. RF readmission search ban đầu bị dừng trước test vì chi phí cao, giảm còn4 configurations×3folds,80trees,sqrt,depth8/12,leaf15/30 theo amendment; cohort không subsample, các model cache giữ nguyên. Khoảng99,87s là search+CV+refit elapsed của search đã thu hẹp, không bao gồm mọi thời gian trial bị hủy. Các model có ngân sách khác nhau phải được ghi rõ.

## 27. Tái lập và demo có huấn luyện lại không?

Demo chỉ load `{dataset}_selected.joblib`, gọi `healthcare_ml.training.predict_artifact`, dùng threshold/calibrator đã lưu. Artifact và split có hash trong freeze, seed42 và cấu hình CPU có giới hạn. Demo không retune từ input/test, không load pickle do người dùng upload. Sample UI lấy từ fit, tách schema hai nhiệm vụ.

## 28. Những giới hạn lớn nhất và bước tiếp theo?

Dữ liệu lịch sử, nhãn quan sát, không xác minh bệnh nhân lặp Heart, kết quả readmission khiêm tốn, high false-alarm burden, thiếu hospital/time external validation và chưa chứng minh causal/clinical utility. Bước tiếp theo là một protocol mới với dữ liệu độc lập, định nghĩa cost/workflow cùng chuyên gia và đánh giá calibration/coverage dưới shift; không sửa model sau khi xem test rồi tái gọi cùng test là độc lập.
