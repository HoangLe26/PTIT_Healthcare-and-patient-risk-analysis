# Healthcare and Patient Risk Analysis

Bài tập lớn Machine Learning: phân loại bệnh tim Cleveland và nguy cơ tái nhập viện `<30` ngày tại thời điểm xuất viện. Pipeline chống leakage, calibration/threshold độc lập và phép chia theo bệnh nhân ở readmission. Nội dung môn học được đối chiếu sáu PDF trong `Slide/`.

## Chạy và đọc kết quả Conformal Prediction

| File | Cách sử dụng |
|---|---|
| [conformal_prediction.py](Conformal_Prediction/conformal_prediction.py) | Từ thư mục dự án, dùng `from Conformal_Prediction.conformal_prediction import summarize`, rồi gọi hàm `summarize(y_cal, p_cal, y_test, p_test)` để đánh giá mô hình đã huấn luyện ở mức bao phủ 90% và 95%. Hai vector `p` chứa xác suất của lớp 1 theo đúng thứ tự nhãn. |
| [run_conformal_reference.py](Conformal_Prediction/run_conformal_reference.py) | Chạy `python Conformal_Prediction/run_conformal_reference.py` từ thư mục dự án để huấn luyện ba thuật toán với tham số cố định trên hai dataset và ghi lại bảng CSV. |
| [conformal_reference_results.csv](Conformal_Prediction/conformal_reference_results.csv) | Bảng tham chiếu gồm 12 dòng. Đọc `target` là mức bao phủ mục tiêu, `coverage` là tỷ lệ tập dự đoán chứa nhãn đúng và `ambiguous` là tỷ lệ hồ sơ nhận cả hai nhãn. Các tỷ lệ dùng thang 0–1. |

Khi nhóm chốt mô hình cuối, mỗi thành viên dùng cùng hàm trên dự đoán của mình và gửi kết quả cho Hoàng theo [quy ước và phân công nhóm](QUY_UOC_VA_PHAN_CONG_NHOM.md).

## Đọc kết quả

- `reports/final_report.md` / `reports/final_report.pdf`: báo cáo chính.
- `reports/presentation.pptx`: deck bảo vệ; `reports/slides_outline.md`: lời dẫn.
- `reports/defense_qa.md`: câu hỏi bảo vệ.
- `output/healthcare_ml_submission.zip`: gói bàn giao cả báo cáo, code, dữ liệu và artifact; SHA-256 trong `submission_manifest.json`.
- `reports/protocol.md`: protocol chốt trước test; `reports/acceptance_checklist.md`: nghiệm thu.
- `artifacts/metrics/*_test_metrics.csv`: điểm test; `*_cv_results.csv`: điểm lựa chọn development, không phải test.
- `artifacts/metrics/*_confidence_intervals.csv`: CI của recipe chọn bằng development và một đối chứng.
- `artifacts/metrics/*_controlled_ablations_summary.csv`: E02/E05/E06/E08/E09/E10 cùng folds.
- `data/metadata/`: provenance/checksum, data cards, cohort flow và schema.
- `notebooks/`: EDA, kết quả và tối ưu đã chạy từ kernel sạch.

## Cài và chạy trên Windows

Python 3.10 được kiểm chứng. Cài môi trường riêng từ project root (đường dẫn có khoảng trắng/tiếng Việt được hỗ trợ):

```powershell
$env:PYTHONUTF8 = '1'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1 -Smoke
```

Chạy đầy đủ:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1
.\.venv\Scripts\python.exe -X utf8 tools/build_notebooks.py
```

Linux/macOS: tạo `.venv`, cài dependencies và `pip install --no-deps -e .`, sau đó `bash scripts/run_all.sh`. Lock hiện có là snapshot môi trường Windows/Python3.10; các wheel hoặc dependency đặc thù hệ điều hành cần resolver tương thích nếu chuyển nền tảng. Không tuyên bố đã kiểm thử Linux/macOS.

## Demo cục bộ

```powershell
.\.venv\Scripts\python.exe -X utf8 -m streamlit run app/streamlit_app.py
```

Demo dùng artifact đã lưu, không train lại. Đầu vào mẫu chỉ giúp kiểm tra giao diện; probability là ước lượng từ dữ liệu lịch sử, không kết luận chẩn đoán/điều trị.

## Tái lập và cache

`download` dùng raw archive cache; `prepare` dựng cohort/splits deterministic; `train` checkpoint từng model. `freeze` khóa hash. Sau khi freeze, `train` tái sử dụng artifact; `evaluate` tái sử dụng đợt đánh giá test đã lưu để tránh test tuning. Muốn một nghiên cứu mới, tạo checkout/thư mục nghiên cứu khác và protocol mới; không xóa manifest để thử nhiều cấu hình trên cùng test rồi chọn điểm đẹp.

`artifacts/smoke/` tách biệt `artifacts/` full. Smoke không có mục đích ước lượng chất lượng mô hình; chỉ kiểm tra wiring, preprocessing và prediction. `tools/run_experiments.py` thực thi tuần tự các bước nghiên cứu với stage logs; báo cáo chỉ đọc artifact.

## Thiết kế cần biết

- Heart: 303 hàng, 61 test; development tách fit/tune 169, calibration 36, threshold 37.
- Readmission: group theo patient, không coi mỗi encounter độc lập. Cohort loại expired/hospice theo bảng mã và Unknown/Invalid gender.
- Model cuối giữ fit/tune set riêng; không refit vào calibration/threshold. Điều này giảm cỡ train nhưng cho phép kiểm toán độc lập rõ.
- Full nghĩa là không subsample cohort chính. Random search 12 cấu hình Heart, 6 readmission; folds 5/3, tối đa hai luồng xử lý.
- Gaussian models dùng numerical-only; bảng ghi feature_set để không nhầm thêm thông tin với thuật toán tốt hơn.
- Bootstrap fixed test predictions không chứa uncertainty từ model selection/retraining. Readmission resample toàn bệnh nhân với multiplicity.
- Đợt thực nghiệm ban đầu áp dụng Conformal cho Heart. Script tham chiếu của nhóm đánh giá cả Cleveland và Diabetes. Với Diabetes, script lấy lượt có `encounter_id` nhỏ nhất của mỗi bệnh nhân trong từng tập hiệu chỉnh và test.

## Kiểm tra và nguồn

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest -q
.\.venv\Scripts\python.exe -X utf8 -m pip check
```

Raw source: [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease), [UCI Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), CC BY 4.0. Citation/DOI trong data cards. Dữ liệu giảng viên chỉ đọc, giữ nguyên.

Xem `reports/limitations.md` cho giới hạn thật và `reports/decision_log.md` cho điều chỉnh môi trường, thời gian và thiết kế.

