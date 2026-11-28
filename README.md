# Healthcare and Patient Risk Analysis

Bài tập lớn môn Học máy của Tùng, Hoàng và Khải. Nhóm so sánh KNN, Logistic Regression và CatBoost trên hai bài toán phân loại. Mỗi bảng so sánh dùng cùng hồ sơ test của một dataset.

| Dataset | Mục tiêu dự đoán | Chỉ số chính |
|---|---|---|
| [UCI Heart Disease – Cleveland](https://doi.org/10.24432/C52P4X) | Phân loại tình trạng bệnh tim theo hồ sơ khám và xét nghiệm. Nhãn dương khi `num > 0`. | ROC-AUC |
| [UCI Diabetes 130-US Hospitals](https://doi.org/10.24432/C5230J) | Tại thời điểm xuất viện, dự đoán lượt điều trị có ghi nhận tái nhập viện trong dưới 30 ngày. | Average Precision |

## Ai đóng góp phần nào cho báo cáo cuối?

| Thành viên | Nội dung thành viên viết | Nguồn để tổng hợp |
|---|---|---|
| Tùng | Hai bộ dữ liệu, cách chia theo bệnh nhân, phương pháp KNN, chọn số láng giềng, kết quả và các ca dự đoán sai. | [Báo cáo KNN](algorithm/KNN/README.md), [kết quả KNN](algorithm/KNN/artifacts) và [kiểm tra dữ liệu](data/metadata/experiment_cohort_audit.json) |
| Hoàng | Logistic Regression, chọn L1/L2 và C, giải thích hệ số và lỗi. Phụ trách phần Conformal Prediction của cả ba thuật toán ở mức bao phủ 90% và 95%. | [Code Logistic](algorithm/Logistic/run_logistic.py), [kết quả Logistic](algorithm/Logistic/artifacts) và [hàm Conformal chung](Conformal_Prediction/conformal_prediction.py). Phần diễn giải Logistic cần viết từ các kết quả này. |
| Khải | CatBoost, chọn tham số, phân tích lỗi. Phụ trách cách chọn ngưỡng F2, tính chỉ số, so sánh ba mô hình trên từng dataset và chọn ca bất đồng. | [Báo cáo CatBoost](algorithm/CatBoost/README.md), [báo cáo phần dùng chung của Khải](reports/BAO_CAO_PHAN_DUNG_CHUNG_KHAI.md) và [bảng benchmark](benchmark/results) |

Hoàng chịu trách nhiệm viết và hoàn thiện báo cáo cuối tại `reports/final_report.md`. Tùng và Khải cung cấp phần thực nghiệm, nhận xét và hạn chế của mô hình mình. Xem nhiệm vụ của [Tùng](algorithm/KNN/NHIEM_VU.md), [Hoàng](algorithm/Logistic/NHIEM_VU.md) và [Khải](algorithm/CatBoost/NHIEM_VU.md). Báo cáo cuối dùng hai bảng phân loại (mỗi dataset ba thuật toán) và một bảng Conformal 12 dòng (ba thuật toán × hai dataset × hai mức bao phủ). Hoàng ghép bảng Conformal từ bốn dòng của từng mô hình cuối: [KNN](algorithm/KNN/artifacts/conformal_4rows_for_hoang.csv), [Logistic](algorithm/Logistic/artifacts/conformal_4rows_for_hoang.csv), [CatBoost](algorithm/CatBoost/artifacts/conformal_4rows_for_hoang.csv).

## Dữ liệu và quy trình đánh giá

Dùng [dữ liệu đã xử lý](data/processed) theo đúng thứ tự dòng và [bản chia của nhóm](data/splits). Chọn tham số trên `fit`, chọn ngưỡng F2 trên `threshold`, hiệu chỉnh Conformal trên `calibration`, đánh giá trên `test`. Với Diabetes, các lượt của cùng bệnh nhân thuộc cùng một tập. Riêng Conformal dùng lượt có mã encounter nhỏ nhất của mỗi bệnh nhân trong từng tập hiệu chỉnh và test. Ghi nhận trong báo cáo rằng nhóm đã xem các tập test này ở đợt thực nghiệm trước. Xem [quy ước và phân công](QUY_UOC_VA_PHAN_CONG_NHOM.md).

## Chạy từ thư mục gốc trên Windows

Dùng PowerShell và Python 3.10. Tạo môi trường khi cài lần đầu. File khóa hiện tại thiếu CatBoost nên cần cài thêm thư viện này.

```powershell
$env:PYTHONUTF8 = '1'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install catboost
```

Mỗi lệnh sau huấn luyện một thuật toán trên cả hai dataset và ghi lại kết quả vào thư mục `artifacts` của thuật toán đó.

```powershell
.\.venv\Scripts\python.exe -m algorithm.KNN.run_knn
.\.venv\Scripts\python.exe -m algorithm.Logistic.run_logistic
.\.venv\Scripts\python.exe -m algorithm.CatBoost.run_catboost
```

Khi cả ba đã xuất dự đoán, đối chiếu mẫu và tạo lại hai bảng phân loại. Kiểm tra mỗi bảng có đủ ba hàng trước khi đưa vào báo cáo.

```powershell
.\.venv\Scripts\python.exe -m algorithm.KNN.run_knn --check-alignment
.\.venv\Scripts\python.exe -m benchmark.run_benchmark
.\.venv\Scripts\python.exe -m pytest -q tests
```

## Dùng Conformal với mô hình cuối

Gọi `summarize(y_cal, p_cal, y_test, p_test)` từ [module dùng chung](Conformal_Prediction/conformal_prediction.py). Hai vector xác suất phải tương ứng với lớp 1 và cùng thứ tự với nhãn thật. Đọc `coverage` cùng với `singleton` và `ambiguous` để biết mô hình đạt độ bao phủ quan sát được với bao nhiêu hồ sơ nhận một hoặc cả hai nhãn.

[Script tham chiếu](Conformal_Prediction/run_conformal_reference.py) và [CSV tham chiếu](Conformal_Prediction/conformal_reference_results.csv) minh họa quy trình với KNN, Logistic và Decision Tree cấu hình cố định. Khi cần tái tạo ví dụ, chạy `.\.venv\Scripts\python.exe Conformal_Prediction/run_conformal_reference.py`.
