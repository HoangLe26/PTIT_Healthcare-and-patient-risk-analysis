# Trạng thái thực hiện

Hai Agent tối đa: Agent chính triển khai dữ liệu/ML/kiểm thử; Agent hỗ trợ nghiên cứu và tài liệu bàn giao. Không tạo Agent thứ ba.

| Giai đoạn | Trạng thái | Bằng chứng/điều kiện |
|---|---|---|
| 0. Môi trường và đọc đặc tả | Hoàn thành | Python 3.10, .venv, requirements-lock, pip check, protocol |
| 1. Dữ liệu/protocol | Hoàn thành | data/metadata: hai nguồn UCI, SHA-256, cohort; data/splits; reports/protocol.md |
| 2. Baseline/smoke/kiểm thử đầu | Hoàn thành | artifacts/smoke; 10 tests đạt; EDA đầy đủ theo sau huấn luyện |
| 3. Model comparison/E08–E11 | Hoàn thành | artifacts/metrics/*cv_results và *controlled_ablations; RF budget bổ sung trước test |
| 4. Calibration/threshold/conformal | Hoàn thành | models freeze cả hai; conformal Heart và policies đã có |
| 5. Test/errors/bootstrap | Hoàn thành | Test cả hai, 1000 paired bootstrap, cases/SHAP/subgroups; 12tests đạt |
| 6. Báo cáo/slide/demo/tái lập | Hoàn thành | PDF25trang; PPTX14slides/4nativecharts/6tables/14notes đã render;4notebooks;54checks;12tests;demo AppTest cả hai bộ đạt |

Các điều chỉnh được ghi trong decision_log và resource amendment RF trước test readmission. Review TabPFN không được ghi thành thí nghiệm đã chạy. Không có public deployment/nộp bài.

Đầu ra bàn giao: reports/final_report.md/PDF, reports/presentation.pptx, README, source/configs/tests/notebooks và output/healthcare_ml_submission.zip. Trạng thái chốt bằng artifact verification, demo/test verification và submission manifest.

