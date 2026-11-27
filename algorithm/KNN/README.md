# Thực nghiệm KNN của Tùng

Thực nghiệm dùng quy ước nhóm ngày 20/11/2026. Cleveland dự đoán tình trạng bệnh tim theo nhãn `num > 0` và dùng ROC-AUC làm chỉ số chính. Diabetes dự đoán tái nhập viện được ghi nhận trong dưới 30 ngày tại thời điểm xuất viện và dùng Average Precision (AP). [Quy ước thực nghiệm](../../QUY_UOC_VA_PHAN_CONG_NHOM.md)

## Cohort và danh sách dùng chung

Cleveland giữ **303 hồ sơ** với **139 dương tính (45,87%)** và 13 đặc trưng. Diabetes giữ **99.340 lượt của 69.987 bệnh nhân** với **11.314 dương tính (11,39%)** và 44 đặc trưng. Từ 101.766 lượt Diabetes, quy tắc xuất viện loại 2.423 lượt và bước lọc giới tính Unknown/Invalid loại thêm 3 lượt. Đã đối chiếu đặc trưng và nhãn của cohort với hàng gốc. Diabetes có 0 ID lượt trùng. Cả hai bộ có 0 hàng nằm trong nhóm trùng toàn bộ đặc trưng đầu vào. Các lượt khác nhau của cùng bệnh nhân vẫn được giữ trong cohort Diabetes. [Audit và danh sách đặc trưng](../../data/metadata/experiment_cohort_audit.json)

| Dataset | Phần | Số mẫu | Số bệnh nhân | Dương tính | Tỷ lệ dương |
|---|---|---:|---:|---:|---:|
| Cleveland | fit | 169 | 169 | 78 | 46,15% |
| Cleveland | threshold | 37 | 37 | 17 | 45,95% |
| Cleveland | calibration | 36 | 36 | 16 | 44,44% |
| Cleveland | test | 61 | 61 | 28 | 45,90% |
| Diabetes | fit | 59.372 | 42.013 | 6.649 | 11,20% |
| Diabetes | threshold | 9.854 | 6.999 | 1.152 | 11,69% |
| Diabetes | calibration | 10.083 | 7.024 | 1.194 | 11,84% |
| Diabetes | test | 20.031 | 13.951 | 2.319 | 11,58% |

[Bảng số mẫu CSV](../../data/metadata/experiment_split_counts.csv)

- Giữ nguyên danh sách vị trí dòng trong [heart.json](../../data/splits/heart.json) và [readmission.json](../../data/splits/readmission.json). Giao ID bệnh nhân giữa mọi cặp phần bằng 0.
- Dùng 5 fold phân tầng cho Cleveland và 3 fold phân tầng theo bệnh nhân cho Diabetes, xáo trộn với seed 42. Cột `row_position` là vị trí 0-based trong CSV processed, `row_id` dùng để đối chiếu và `fold` là fold validation 1-based. Với fold `f`, lấy train từ các dòng `fit` có `fold != f`. Giao ID bệnh nhân giữa train và validation bằng 0 trong từng fold. [Fold Cleveland](../../data/splits/heart_cv.csv) · [Fold Diabetes](../../data/splits/readmission_cv.csv)
- Danh sách Conformal dùng toàn bộ calibration và test ở Cleveland. Với Diabetes, mỗi bệnh nhân góp lượt có `encounter_id` nhỏ nhất trong từng phần, cho 7.024 bệnh nhân calibration và 13.951 bệnh nhân test. [Mẫu Cleveland](../../data/splits/heart_conformal.csv) · [Mẫu Diabetes](../../data/splits/readmission_conformal.csv)

## Phương pháp

Mỗi fold học trung vị, trung bình, độ lệch chuẩn và các mức phân loại từ train của fold. KNN chuẩn hóa biến số sau khi điền trung vị, dành một cột one-hot cho giá trị thiếu và ánh xạ hạng mục mới thành khối toàn 0. Giá trị chuỗi `None` của xét nghiệm Diabetes vẫn là một hạng mục. Mô hình chạy đủ sáu tổ hợp của `k = 5, 15, 31` và bỏ phiếu đều hoặc theo `1/d`. Khi có láng giềng ở khoảng cách 0, phiếu nghịch đảo chỉ tính các láng giềng ở khoảng cách 0.

Khoảng cách Euclidean cộng bình phương chênh lệch của các tọa độ. Nếu giữ đơn vị gốc, chênh 20 mg/dL cholesterol đóng góp 400, chênh 5 năm tuổi đóng góp 25 và hai hạng mục đã biết khác nhau đóng góp 2 trong một khối one-hot. Chuẩn hóa làm phần đóng góp của biến số phụ thuộc vào độ lệch chuẩn trên train. Ảnh hưởng của các biến phân loại vẫn phụ thuộc vào số khối bất đồng.

Tìm chính xác 31 láng giềng theo lô 192 mẫu, rồi lấy tiền tố tương ứng cho từng cấu hình. Khi bằng khoảng cách tính được, lấy `row_id` nhỏ hơn trước. Phép tính chia bình phương khoảng cách thành phần biến số và phần one-hot. Các cột one-hot thường gặp dùng tích ma trận đặc, các cột còn lại dùng tích ma trận thưa. Sáu cấu hình dùng chung một lần tìm láng giềng cho mỗi fold.

Chọn cấu hình bằng điểm CV trung bình. Khi hòa, ưu tiên `k` lớn hơn, sau đó ưu tiên bỏ phiếu đều. Fit lại trên toàn bộ `fit`. Tối đa hóa F2 trên `threshold`, xét từng xác suất phân biệt và giữ ngưỡng cao hơn khi hòa. Lấy `q` ở hạng `ceil((n_cal + 1) × target)` của điểm `1 − p(nhãn thật)` trên calibration. Giữ nhãn nếu điểm của nhãn nhỏ hơn hoặc bằng `q`. Cấu hình và ngưỡng của cả hai bộ được lưu trước khi sinh dự đoán test. [Khóa Cleveland](artifacts/heart/pretest_freeze.json) · [Khóa Diabetes](artifacts/readmission/pretest_freeze.json)

## Chọn k bằng cross-validation

| k | Cách bỏ phiếu | Cleveland ROC-AUC | Diabetes AP |
|---:|---|---:|---:|
| 5 | Đều | 0,8924 ± 0,0465 | 0,1367 ± 0,0053 |
| 5 | Nghịch đảo khoảng cách | 0,8998 ± 0,0440 | 0,1452 ± 0,0064 |
| 15 | Đều | 0,8993 ± 0,0234 | 0,1561 ± 0,0068 |
| 15 | Nghịch đảo khoảng cách | 0,9049 ± 0,0184 | 0,1675 ± 0,0076 |
| 31 | Đều | 0,9083 ± 0,0281 | 0,1709 ± 0,0047 |
| **31** | **Nghịch đảo khoảng cách** | **0,9112 ± 0,0227** | **0,1807 ± 0,0052** |

Các ô ghi trung bình ± độ lệch chuẩn qua fold. Ở cả hai bộ, trong lưới đã thử, điểm trung bình tăng khi tăng `k` và khi tăng trọng số cho láng giềng gần. Với cùng cách bỏ phiếu, `k = 31` làm một lượt lân cận đơn lẻ ít chi phối hơn `k = 5`. Khi `k` tăng, những hồ sơ xa hơn cũng tham gia bỏ phiếu và có thể làm mờ những nhóm nhỏ. Phiếu `1/d` giữ trọng số cao hơn cho các hồ sơ gần. Kết quả chỉ xét ba giá trị `k` trên hai bộ này. [Điểm từng fold Cleveland](artifacts/heart/cv_results.csv) · [Điểm từng fold Diabetes](artifacts/readmission/cv_results.csv)

## Cleveland

Test có 61 hồ sơ với 28 dương tính và 33 âm tính. Ngưỡng 0,153276 đạt F2 = 0,9043 trên phần threshold.

| ROC-AUC | Precision | Recall | F2 | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,9459 | 0,5833 (28/48) | 1,0000 (28/28) | 0,8750 | 28 | 20 | 13 | 0 |

- **FN:** 0/28 hồ sơ dương tính bị bỏ sót ở ngưỡng đã chọn.
- **FP, dòng 271:** Nhãn thật 0, xác suất lớp 1 là 0,737925 và vượt ngưỡng 0,153276. Có 23/31 láng giềng dương tính. Láng giềng gần nhất, dòng 174, có nhãn 1 và cách 2,236. Hai hồ sơ cùng `cp=4`, `thal=6`, `exang=0`, với `oldpeak` lần lượt là 2,3 và 2,0. Dòng 271 có `ca=0` và `slope=1`, còn dòng 174 có `ca=2` và `slope=2`. Dòng 0 cũng có `ca=0` và `oldpeak=2,3` cùng nhãn 0, nhưng xa hơn (2,667). Tổng phiếu của nhóm lân cận nghiêng về nhãn 1. Các đặc trưng trên mô tả khoảng cách trong biểu diễn đã dùng. Nhãn của một người và các biến chẩn đoán có thể khác nhau vì những yếu tố ngoài mô hình.

[Chỉ số và thời gian](artifacts/heart/run_summary.json) · [Dự đoán](artifacts/heart/test_predictions.csv) · [Ca sai và đặc trưng](artifacts/heart/extreme_errors.csv) · [Láng giềng ca sai](artifacts/heart/error_neighbors.csv)

## Diabetes

Test có 20.031 lượt với 2.319 dương tính (11,58%). Ngưỡng 0,033471 đạt F2 = 0,4066 trên phần threshold.

| Average Precision | Precision | Recall | F2 | TP | FP | TN | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,1839 | 0,1268 (2.011/15.865) | 0,8672 (2.011/2.319) | 0,3999 | 2.011 | 13.854 | 3.858 | 308 |

AP là 0,1839 với mốc tỷ lệ dương tính 0,1158. Ở ngưỡng F2, mô hình gắn cờ 15.865/20.031 lượt (79,20%), trong đó 13.854 lượt là FP.

- **FN, lượt 16187292:** Nhãn thật 1, xác suất lớp 1 bằng 0 và thấp hơn ngưỡng 0,033471. Có 92 FN cùng xác suất 0, đây là lượt có ID nhỏ nhất. Cả 31 láng giềng có nhãn 0. Lượt này có `number_outpatient=0`, `number_emergency=0`, `number_inpatient=0`. Hai láng giềng gần nhất, 13534602 và 60343620, cùng ba giá trị này, cùng mã nguồn nhập viện 7 và mã xuất viện 1. Mã chẩn đoán chính của cả hai khác mã 401 ở lượt FN. Trong biểu diễn one-hot, các cặp mã đã biết khác nhau có cùng mức phạt trong một biến, nên độ gần theo mô hình có thể khác độ tương đồng về lâm sàng.
- **FP, lượt 284109186:** Nhãn thật 0, xác suất lớp 1 là 0,906168 và vượt ngưỡng 0,033471. Có 28/31 láng giềng dương tính. Lượt này có 12 lần nội trú và 3 lần cấp cứu trước đó. Láng giềng gần nhất, lượt 156381930, có nhãn 1, cùng nhóm tuổi `[20-30)`, cùng 12 lần nội trú, 2 lần cấp cứu và cùng mã `diag_2=276`. Số lượt sử dụng dịch vụ góp vào độ gần của hai hồ sơ. Dữ liệu dựa trên những lần tái nhập viện được ghi nhận, nên nhãn 0 có thể bao gồm sự kiện xảy ra ngoài hệ thống quan sát.

[Chỉ số và thời gian](artifacts/readmission/run_summary.json) · [Dự đoán](artifacts/readmission/test_predictions.csv) · [Ca sai và đặc trưng](artifacts/readmission/extreme_errors.csv) · [Láng giềng ca sai](artifacts/readmission/error_neighbors.csv)

## Split Conformal Prediction

| Dataset | Mục tiêu | q | Calibration / test | Bao phủ thực đo | Một nhãn | Hai nhãn | Rỗng |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cleveland | 90% | 0,866197 | 36 / 61 | 100,00% (61) | 24,59% (15) | 75,41% (46) | 0,00% (0) |
| Cleveland | 95% | 0,940433 | 36 / 61 | 100,00% (61) | 8,20% (5) | 91,80% (56) | 0,00% (0) |
| Diabetes | 90% | 0,289715 | 7.024 / 13.951 | 90,51% (12.627) | 99,33% (13.858) | 0,00% (0) | 0,67% (93) |
| Diabetes | 95% | 0,902259 | 7.024 / 13.951 | 95,10% (13.267) | 64,66% (9.021) | 35,34% (4.930) | 0,00% (0) |

Mẫu số của các tỷ lệ là 61 hồ sơ Cleveland và 13.951 lượt Diabetes đã chọn, mỗi lượt thuộc một bệnh nhân riêng. Khi nâng mục tiêu từ 90% lên 95%, Cleveland vẫn bao phủ 61/61 hồ sơ và giảm từ 15 xuống 5 tập một nhãn. Ở Diabetes, mô hình đạt thêm 640 lượt được bao phủ và trả hai nhãn cho 4.930 lượt.

Ở Diabetes, toàn bộ các tập một nhãn ở hai mức đều là `{0}`. Mức 90% chứa nhãn thật cho 0/1.253 lượt dương tính trong tập test Conformal. Mức 95% chứa nhãn thật cho 569/1.253 lượt dương tính (45,41%), tất cả trong tập hai nhãn. Tỷ lệ một nhãn cao ở mức 90% chủ yếu phản ánh ưu thế của lớp âm trong tập đánh giá.

Bảo đảm lý thuyết là độ bao phủ biên khi mẫu calibration và test có thể hoán đổi. Việc lấy một lượt mỗi bệnh nhân trong Diabetes xử lý sự lặp bệnh nhân. Khác biệt bệnh viện và thời gian vẫn có thể ảnh hưởng giả định này. Hai tập test từng được sử dụng ở đợt trước. Cleveland chỉ có 61 hồ sơ test, nên kết quả bao phủ 61/61 vẫn dựa trên mẫu nhỏ.

## Thời gian và đối chiếu mẫu

Thời gian tính từ nạp dữ liệu qua CV, khóa ngưỡng và xuất kết quả test là 0,23 giây cho Cleveland và 172,63 giây cho Diabetes. Ba fold Diabetes lần lượt mất 29,88, 27,22 và 32,01 giây. Các giai đoạn chi tiết nằm trong hai `run_summary.json`.

ID và nhãn thật của KNN khớp bản chia test gốc và các bảng CatBoost hiện có trong repo, đủ 61 mẫu Cleveland và 20.031 mẫu Diabetes. Repo hiện chưa có bảng test Logistic Regression và Decision Tree để đối chiếu ba mô hình theo quy ước. [Kết quả đối chiếu](artifacts/test_alignment.csv)

## Chạy lại và bàn giao

Lần chạy này dùng Python 3.13.12, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3 và scikit-learn 1.7.2. Từ thư mục gốc của repo, dùng cùng môi trường để tái lập. Các tệp chia đã xuất có sẵn. Chạy lệnh audit để tái tạo và đối chiếu với raw. SHA-256 trong audit chuẩn hóa kết thúc dòng thành LF để dùng cùng bộ tệp trên Windows và Linux.

```powershell
python -m pip install numpy==2.2.6 pandas==2.3.3 scipy==1.15.3 scikit-learn==1.7.2
python -m benchmark.cohort
python -m algorithm.KNN.run_knn
python -m unittest discover -s tests -v
```

- **Hoàng** dùng [CSV Conformal 4 dòng](artifacts/conformal_4rows_for_hoang.csv). Cột `singleton` là tỷ lệ một nhãn, `ambiguous` là tỷ lệ cả hai nhãn. Các tỷ lệ dùng thang 0–1.
- **Khải** dùng hai bảng dự đoán đã liên kết ở phần kết quả. Cột `conformal_selected` cho biết các mẫu tham gia đánh giá Conformal. Dự đoán phân loại Diabetes bao gồm đủ 20.031 lượt.
- **Đối chiếu mẫu** chạy `python -m algorithm.KNN.run_knn --check-alignment` khi có bảng dự đoán của thành viên khác. [Trạng thái khớp ID và nhãn](artifacts/test_alignment.csv)
