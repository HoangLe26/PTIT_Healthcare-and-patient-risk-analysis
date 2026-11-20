# Nghiệm thu bài tập lớn — 11/10/2026

Các mục hoàn thành dưới đây có artifact hoặc log thực tế. Đặc tả nguồn: AI_AGENT_HEALTHCARE_ML_END_TO_END.md, mục16. Không coi TabPFN review là thí nghiệm đã chạy; không coi số files tồn tại thay cho kiểm tra nội dung/hiển thị.

## Dữ liệu và protocol

- [x] Hai bộ độc lập: UCI Cleveland303 hàng và Diabetes101.766 raw /99.340 cohort. `data/raw/`, `data/metadata/sources.json`.
- [x] Nguồn, DOI, licenseCCBY4.0, checksum archive/raw provenance, data cards/dictionaries: `data/metadata/`.
- [x] Target, thời điểm, đơn vị quan sát và cohort rõ: data cards, `reports/protocol.md` và cohort flow CSV.
- [x] Feature availability/ID exclusion; thảo luận ca/thal và target proxy: `feature_availability.csv`, báo cáo mục dữ liệu/giới hạn.
- [x] Heart gốc không nhân bản; independence là giả định vì ID audit tổng hợp, không xác minh303 người duy nhất.
- [x] Readmission group split mọi tầng và CV theo patient; tests disjoint đạt. `data/splits/readmission.json`.
- [x] Test không dùng model selection/threshold; freeze manifests và evaluated records trong `artifacts/`.
- [x] Imputer/encoder/scaler trong pipeline fold; validation/test không resample. Dự án không triển khai SMOTE, không claim có thí nghiệm resampling.

## Thực nghiệm và đánh giá

- [x] Đúng ma trận model: Heart Dummy/LR/KNN/GaussianNB/QDA/XGB; readmission Dummy/LR/GaussianNB/RF/XGB và ablation liên quan.
- [x] CV results, từng trial/hyperparameters và metric chính: `*_cv_results.csv`, `*_search_trials.csv`, `artifacts/runs/`.
- [x] Calibration và threshold holdout độc lập; giữ recipe dù calibration Heart xấu hơn test. `*_calibration_test.csv`, `*_threshold_policies.csv`.
- [x] E01 model comparison; E02 class weights; E03 threshold; E04 calibration: full metrics và controlled comparisons.
- [x] E05 readmission engineering; E06 restricted Heart; E08 regularization; E09 KNN geometry; E10 Gaussian numerical comparison: `*_controlled_ablations_summary.csv`.
- [x] E07 advanced thực tế: split conformal Heart tại alpha0.05/0.10. `heart_conformal.json`. TabPFN **skipped review**, không có run/metric giả.
- [x] E11 NumPy gradient/hội tụ/đối chiếu thư viện: `optimization.json`, `optimization_loss.csv/png`.
- [x] E12 top-k readmission: `readmission_topk.csv`.
- [x] Learning curves cả hai, validation curves C/k, optimization loss curve; phân biệt đúng ý nghĩa.
- [x] Test metrics độc lập, CI95% fixed-model và paired differences. Cluster bootstrap1000 theo patient với multiplicity ở readmission.
- [x] Thêm exact binomial CI sensitivity Heart để diễn giải bootstrap suy biến khi FN=0; không thay recipe sau test.
- [x] Runtime, phiên bản, seeds, split/model hashes và cảnh báo: `environment.json`, manifests, cancelled RF log.
- [x] RF resource amendment trước test readmission được ghi. Budget RF4 configs nhỏ hơn các model khác; không claim CPU tuning ngang nhau.

## Phân tích và tài liệu

- [x] FP/FN, numerical/missingness/subgroup/error summaries và cases cả hai. Heart FN=0 tại tuned threshold nên không dựng ca FN giả.
- [x] Permutation importance, LR coefficients, grouped local SHAP. XGB dùng native TreeSHAP; background tree-cover fit-only và raw-output units được ghi.
- [x] Subgroup có counts và điều kiện tối thiểu; không claim fairness/causality chắc chắn.
- [x] Báo cáo Markdown/PDF trả lời RQ1–RQ5, dữ liệu/code/artifact truy nguyên được: `final_report.md`, `final_report.pdf`.
- [x] PDF25 trang render toàn bộ bằng pypdfium2, kiểm tra tiếng Việt và layout; sửa trang figure bị tách để mỗi trang có nội dung rõ.
- [x] PowerPoint14 slide, native editable comparison charts/tables; toàn bộ render thực bằng PowerPoint COM read-only/ẩn và kiểm tra PNG.
- [x] Course alignment dẫn sáu PDFs/trang đúng; calibration/bootstrap là kiến thức đã học. `course_alignment.md`.
- [x] Literature review phân biệt bài phản biện/preprint/repository và thí nghiệm/review. `literature_review.md`.
- [x] Outline14 slide, notes và hơn15 câu hỏi bảo vệ. `slides_outline.md`, `defense_qa.md`.

## Kỹ thuật và bàn giao

- [x] 12 tests đạt; full core results tách smoke artifacts. PowerShell smoke script đã chạy lại thành công, sử dụng cache và kiểm tra hashes/predictions.
- [x] Bốn notebook đã thực thi kernel sạch, không error outputs; đã cập nhật theo cases/figures hậu kiểm mới.
- [x] Save/load prediction equivalence, threshold truy nguyên development và unknown-category handling được kiểm tra.
- [x] README, lock110 dependencies, pyproject và CLI/lệnh tái lập có thật. `pip check` không có broken requirements.
- [x] 54/54 artifact acceptance checks đạt. `artifacts/runs/acceptance_verification.json`.
- [x] Demo Streamlit local dùng pipeline frozen. Agent hỗ trợ chạy AppTest: hợp lệ/null/sai schema/kiểu số trên cả hai bộ đều đạt; dự đoán khớp predict_artifact. `artifacts/runs/demo_verification.json`.
- [x] TASKS/limitations/decision log phản ánh đúng thực nghiệm và giới hạn. Không publish/nộp bài thay người dùng.

## Phần không được khẳng định đã thực hiện

- Chưa chạy TabPFN hoặc các thuật toán conformal dưới shift gần đây; đây là literature/feasibility review.
- Chưa nested CV/cross-fitted calibration; dùng fit/calibration/threshold holdout rõ ràng và independent test.
- Chưa KNN readmission full, SVM/MLP/GMM/SMOTE; chúng không phải ma trận bắt buộc đã chốt.
- Chưa temporal/hospital external validation hoặc clinical validation; chưa thử Linux/macOS thực tế.
- Không dùng bundled artifact-tool vì không có runtime/tool dependency tương ứng; dùng thư viện xuất cục bộ và kiểm tra render thật.

