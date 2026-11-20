# Outline 14 slide và ghi chú thuyết trình

Thời lượng gợi ý 12–15 phút. Tiêu đề và thứ tự đồng bộ trực tiếp với XML của `reports/presentation.pptx` cuối ngày 11/10/2026. Notes dưới đây là script hỗ trợ theo các chủ đề của deck. Số liệu từ artifact cuối; `train_seconds` là search+CV+refit elapsed.

## Slide 1. Healthcare and Patient Risk Analysis

Nội dung: Cleveland Heart Disease và tái nhập viện dưới30 ngày. Prototype học thuật trên hai bộ dữ liệu UCI.

Notes (30s): Hai nghiên cứu tình huống độc lập, chung nguyên tắc đánh giá. Không điền tên người thực hiện hoặc lớp khi chưa được cung cấp. Dữ liệu và kết quả không đủ thẩm định lâm sàng.


## Slide 2. Bài toán và thước đo

Nội dung: Heart `num>0`, bệnh hiện diện khi xét nghiệm đã có, primary ROC-AUC. Readmission `<30` tại xuất viện, primary AP. T–E–P cho từng nhiệm vụ.

Notes (45s): E là fit/development dùng học/chọn recipe, P báo độc lập trên test. Không gọi Heart nguy cơ tương lai, không gọi hai outcome là external validation. Nguồn slide1 trang7 và UCI.


## Slide 3. Dữ liệu và cohort

Nội dung: Heart303 hàng,139 dương. Readmission101.766 raw thành99.340 lượt,69.987 patient IDs,11.314 dương. Loại2.423 expired/hospice và3 invalid gender. Dữ liệu readmission1999–2008.

Notes (60s): Heart ID là synthetic row audit, chưa xác minh303 người duy nhất. Readmission patient_nbr có nghĩa grouping thật. Giữ `None` xét nghiệm thành category và `?` là missing, ID/target không vào features. Bảng nguồn `data/metadata/*_summary.json`.


## Slide 4. Các vai trò dữ liệu độc lập

Nội dung: Heart fit169/cal36/threshold37/test61. Readmission fit59.372/cal10.083/threshold9.854/test20.031. CV trên fit, calibrator trên calibration, policy trên threshold. Tách patient groups readmission.

Notes (75s): Không refit toàn bộ development. Đây là CV selection+holdout, không nestedCV/OOF calibration. Sigmoid recipe quyết định trước test, freeze rồi đánh giá một đợt. Cỡ Heart holdout nhỏ là đánh đổi. Nguồn protocol và split summaries.


## Slide 5. Kiến thức đã học và phần bổ sung

Nội dung: LR/MLE/L1–L2, KNN/distance/scaling, GaussianNB/QDA, pipeline/CV/group split, threshold/calibration/learning curves/bootstrap.

Notes (60s): Dẫn Slide2 gradient/regularization, Slide3 LR, Slide4 KNN, Slide5 probabilistic, Model-assessment. Phân biệt ensemble/SHAP/conformal/foundation models là bổ sung. Không gọi mọi calibration hoặc bootstrap là ngoài chương trình.


## Slide 6. Lựa chọn mô hình bằng development

Nội dung: Dummy/LR/GaussianNB/boosting cả hai; KNN/QDA numerical Heart; RF readmission. Heart chọnLR CVAUC0,9193, readmission chọnXGB CVAP0,2204. Full/numeric/restricted/engineered được gắn nhãn.

Notes (60s): CV mean/SD là điểm selection. RF được thu hẹp trước test vì tài nguyên thành4configs×3folds,80trees/sqrt/depth8–12/leaf15–30; cohort giữ nguyên. Báo unequal budgets, không coi search elapsed là thời gian fit riêng. Nguồn resource amendment và cv_results.


## Slide 7. Heart: AUC cao trên test nhỏ

Nội dung: LR AUC0,9621, CI95%[0,9086;0,9945]. Threshold0,269344: TP28/FN0/FP14/TN19. Precision0,6667, specificity0,5758, gắn cờ42/61.

Notes (75s): 28/28 dương được tìm trên test, không suy sensitivity quần thể100%. Bootstrap recall suy biến; exact conditional-binomial CI[0,8766;1]. Không có FN nên không dựng caFN. Hình ROC/PR và confusion từ artifactHeart.


## Slide 8. Readmission: ranking còn khiêm tốn

Nội dung: XGB AP0,22894 CI[0,20782;0,25007], AUC0,66947. Threshold0,090540: TP1788/FN531/FP9729/TN7983. Precision15,525%, recall77,102%, gắn cờ57,496%.

Notes (75s): Ranking khiêm tốn dù hơn dummyAP0,11577. Gắn cờ11.517 lượt để tìm1.788 dương tạo tải lớn. Paired XGB−LR AP+0,01212 CI[0,00606;0,01772], nhưng F2 difference CI gồm0. Không chứng minh lợi ích lâm sàng.


## Slide 9. Threshold và ngân sách gắn cờ

Nội dung: Bảng0,5/F2/recall85/cost policies từ `*_threshold_policies.csv`; giữ giá trị thật. Readmission top10%:2.004 lượt,546 dương, precision27,2455%, recall23,5446%.

Notes (60s): Policy chọn trên threshold holdout, không chọn ngưỡng bằng test. F2 ưu tiên recall; cost ratios chỉ minh họa. Top5%/20% có đánh đổi tương ứng. Ngân sách encounter không bằng ngân sách patient unique. Hình threshold tradeoff.


## Slide 10. Phân tích FP / FN và giải thích local

Nội dung: Case IDs/values đọc CSV mới nhất, HeartFP/TP/TN, readmissionFP/FN. Subgroup đủn≥30,dương≥10,âm≥10 mới diễn giải metric. Permutation là global; SHAP raw log-odds trước calibration.

Notes (75s): HeartSHAP background80fit; readmissionnativeTreeSHAP dùng training node cover fit-only, reconstruction error≈8,34×10⁻⁷. Không cùng background method, không diễn giải nhân quả. Case sorting theo xác suất nhãn được gán tại threshold: pred1p/pred0(1−p). Nhóm nhỏNA, không suy fairness được bảo đảm.


## Slide 11. Calibration có thể làm xác suất kém hơn

Nội dung: Heart threshold-holdout Brier raw0,149684→sigmoid0,148846; test raw0,085914→sigmoid0,105827. Test logloss0,296926→0,369279. Readmission Brier raw/cal gần tương đương0,097505/0,097507.

Notes (75s): Sigmoid được chọn trước test, vẫn giữ dù testHeart kém hơn. Calibration36 có uncertainty; overfit khả dĩ chưa phải kết luận causal. So raw/cal cùng numeric threshold chỉ diagnostic, không phải separately-retuned operating points. Dùng reliability diagram và Brier/logloss.


## Slide 12. Ablation và tối ưu từ slide

Nội dung: Heart scaling KNN AUC0,7781→0,9143; full→restrictedLR0,9193→0,8888. Readmission engineered không tăngAP (0,21333→0,21121); weighting tăngAP ít nhưng Brier raw0,09545→0,22249. Gradient error6,52×10⁻¹¹; NumPy/sklearn prediction gap5,58×10⁻⁹.

Notes (90s): Đối chứng giữ folds/settings, tách khỏi search leaderboard. E10 numericalNB/LR/QDA AUC0,7984/0,8039/0,8073. E08C/penalty cho tradeoff sparse/overfit; không quy mọi chênh cho thuật toán khi features khác. Demo NumPy unpenalized NLL không thay final pipeline. Hình loss và controlled curve.


## Slide 13. Split conformal và nghiên cứu gần đây

Nội dung: Heart alpha0,05/0,10 empirical coverage1/1, mean set size1,885/1,541, ambiguous88,525%/54,098%. TabPFN Nature2025, reports2.5 và3.5(15/09/2026), chưa thực nghiệm.

Notes (75s): Coverage cao có nhiều `{0,1}`, không perfect certainty. Calibration36, rows assumed exchangeable, không guarantee riêng subgroup/patient. Readmission row-iid không phù hợp. TabPFN là skipped feasibility review theo phạm vi/ngân sách, không failed run. V2modelcard công khai, repo phiên bản mới mô tả auth/license, chưa download probe. Nguồn literature_review và conformalJSON.


## Slide 14. Kết luận và giới hạn

Nội dung: Heart ranking cao trên test nhỏ, readmission modestAP/tảifalsealarm cao. Threshold thay đánh đổi, calibration chưa bảo đảm cải thiện, controlled comparisons và conformal có bằng chứng thật. Dữ liệu lịch sử, chưa external/temporal validation.

Notes (60s): Trả lời RQ bằng metric và giới hạn, không hứa deployment clinical. Kiểm định ngoài với protocol mới, xác định workflow/cost trước, nghiên cứu shift/calibration. Demo chỉ load recipe và threshold đã lưu, không train/retune. Kết thúc bằng câu hỏi bảo vệ, dùng `defense_qa.md`.
