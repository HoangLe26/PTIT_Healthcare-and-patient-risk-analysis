# Quy ước thực nghiệm và phân công nhóm

**Đề tài:** So sánh KNN, Logistic Regression và Decision Tree trong phân loại tình trạng bệnh tim và dự đoán tái nhập viện sớm. Dùng Conformal Prediction để đánh giá khi nào mỗi mô hình có thể đưa ra một nhãn duy nhất.

**Phạm vi đã chốt ngày 20/11/2026.** Đây là kế hoạch cho đợt thực nghiệm của Tùng, Hoàng và Khải. Khi tài liệu cũ nêu danh sách thuật toán hoặc quy trình hiệu chỉnh khác, nhóm dùng quy ước trong file này. Các kết quả từ đợt trước phải được ghi rõ nguồn nếu trích dẫn.

## 1. Hai câu hỏi dự đoán

| Quyết định | A · Cleveland | B · Diabetes 130-US Hospitals, 1999–2008 |
|---|---|---|
| Mục tiêu | Từ hồ sơ khám và xét nghiệm, nhận biết tình trạng bệnh tim hiện diện theo nhãn UCI | Tại thời điểm xuất viện, dự đoán lượt điều trị có được ghi nhận tái nhập viện trong dưới 30 ngày hay không |
| Đơn vị phân tích | Một hồ sơ bệnh nhân | Một lượt nằm viện |
| Nhãn dương / âm | Dương nếu `num > 0`, âm nếu `num = 0` | Dương nếu `readmitted = <30`, âm nếu giá trị là `>30` hoặc `NO` |
| Nguồn | [UCI Heart Disease, DOI 10.24432/C52P4X](https://doi.org/10.24432/C52P4X), dùng riêng Cleveland | [UCI Diabetes, DOI 10.24432/C5230J](https://doi.org/10.24432/C5230J) |
| Chỉ số chính | ROC-AUC | Average Precision (AP) |

Mỗi thành viên chạy thuật toán của mình trên cả A và B. So sánh ba thuật toán **trong từng bộ**, sau đó thảo luận xem thứ hạng có thay đổi khi chuyển bài toán hay không. Trình bày hai bảng kết quả riêng vì nhãn, tỷ lệ dương và ý nghĩa của một ca bỏ sót khác nhau.

## 2. Mẫu, đầu vào và cách chia

- **Cleveland:** dùng 303 hồ sơ cùng 13 đặc trưng trong [từ điển dữ liệu](data/metadata/heart_data_dictionary.csv). Các biến như `ca` và `thal` thuộc hồ sơ chẩn đoán. Phạm vi kết luận là phân loại với những thông tin đó.
- **Diabetes:** kế thừa quy tắc của [bản mô tả dữ liệu](data/metadata/readmission_data_card.md). Từ 101.766 lượt, loại các mã xuất viện tử vong/hospice và ba lượt có giới tính ghi Unknown/Invalid, còn 99.340 lượt của 69.987 bệnh nhân. Giữ các lượt khác nhau của cùng bệnh nhân. Dùng bộ biến `full` của dự án, loại nhãn gốc và nhãn dẫn xuất, ID dòng/lượt/bệnh nhân, `weight`, `payer_code` và `medical_specialty` khỏi đầu vào. Chỉ dùng thông tin có đến lúc xuất viện.
- **Kiểm tra trước khi chạy:** đối chiếu số dòng, tỷ lệ lớp và danh sách biến. Kiểm tra ID lượt trùng, ghi nhận các hồ sơ có giá trị giống nhau trước khi kết luận là bản sao. Với B, kiểm tra giao của các tập ID bệnh nhân bằng rỗng ở mọi lần chia.

Dùng đúng danh sách mẫu đã lưu trong [heart.json](data/splits/heart.json) và [readmission.json](data/splits/readmission.json). Tập phát triển gồm ba phần đầu của bảng. Khóa danh sách trước khi chạy cả ba thuật toán.

| Phần | A | B | Mục đích duy nhất |
|---|---:|---:|---|
| `fit` | 169 | 59.372 | Huấn luyện và chọn siêu tham số bằng cross-validation (CV) |
| `threshold` | 37 | 9.854 | Chọn ngưỡng đổi xác suất thành một nhãn |
| `calibration` | 36 | 10.083 | Tính ngưỡng Conformal để tạo tập nhãn |
| `test` | 61 | 20.031 | Đánh giá cấu hình đã cố định và phân tích lỗi |

Tùng lưu cùng một danh sách fold cho cả nhóm. Trong `fit`, dùng CV phân tầng 5 fold cho A và CV phân tầng theo nhóm bệnh nhân 3 fold cho B, xáo trộn với seed 42. Trong từng fold, học phép điền thiếu, mã hóa và chuẩn hóa từ phần huấn luyện. Điền trung vị cho biến số, mã hóa riêng giá trị thiếu và one-hot cho biến phân loại, cho phép gặp hạng mục mới khi dự đoán. Giữ cùng đầu vào cho ba thuật toán. Chuẩn hóa biến số cho KNN và Logistic Regression.

## 3. Sáu thí nghiệm và người chịu trách nhiệm

| Thành viên | Thí nghiệm cá nhân trên A và B | Phần dùng chung và sản phẩm bàn giao |
|---|---|---|
| **Tùng – KNN** | Thử `k ∈ {5, 15, 31}` và bỏ phiếu đều / theo nghịch đảo khoảng cách, dùng khoảng cách Euclidean. Giải thích tác động của k và ảnh hưởng của đơn vị đo lên khoảng cách. | Kiểm tra cohort, xuất danh sách fold và bảng số mẫu/dương tính mỗi tập. Với B, chạy dự đoán theo lô, tái sử dụng khoảng cách tới 31 láng giềng gần nhất trong mỗi fold cho cả sáu cấu hình để giảm chi phí. |
| **Hoàng – Logistic Regression** | Thử `C ∈ {0.01, 0.1, 1}` kết hợp L1/L2, dùng bộ giải hỗ trợ cả hai. Kiểm tra hội tụ, giải thích chiều tác động của một số hệ số và ảnh hưởng của điều chuẩn. | Viết hàm Split Conformal Prediction dùng chung và tổng hợp độ bao phủ thực tế, tỷ lệ kết quả mơ hồ ở hai mức mục tiêu 90% và 95%. |
| **Khải – Decision Tree** | Dùng Gini, thử độ sâu tối đa `{3, 5, 8}` và số mẫu tối thiểu mỗi lá `{5, 20}`. So sánh điểm train/CV để nhận diện quá khớp, minh họa một đường đi từ gốc đến lá. | Viết hàm chọn ngưỡng, tính chỉ số và lập ma trận nhầm lẫn dùng chung. Bàn giao bảng đối chiếu ba thuật toán trên từng dataset. |

**Quy tắc chọn mô hình.** Với từng thuật toán trên từng dataset, chọn cấu hình có chỉ số chính trung bình qua các fold cao nhất. Nếu bằng điểm, ưu tiên k lớn hơn và bỏ phiếu đều cho KNN, C nhỏ hơn rồi L2 cho Logistic, cây nông hơn rồi lá lớn hơn cho Decision Tree. Fit lại cấu hình đã chọn trên toàn bộ `fit`. Giữ mô hình này cho cả đánh giá phân loại và Conformal. Dùng xác suất gốc và trọng số lớp mặc định.

## 4. Chỉ số chung và phân tích lỗi

Khải chọn riêng một ngưỡng cho mỗi mô hình trên mỗi dataset bằng cùng quy tắc: tối đa hóa F2 trên `threshold`, xét các điểm xác suất phân biệt, dự đoán dương khi xác suất ≥ ngưỡng. Khi bằng F2, chọn ngưỡng cao hơn. F2 ưu tiên recall, phù hợp với câu hỏi về bỏ sót. Đây là lựa chọn cho bài tập.

Mỗi thành viên xuất một dòng cho mỗi mẫu test gồm ID dòng, nhãn thật, xác suất lớp dương, ngưỡng, nhãn dự đoán và loại kết quả TP/FP/TN/FN. Dùng chỉ số chính trên xác suất. Từ nhãn dự đoán, báo cáo precision, recall, F2, FP và FN. Ghi rõ mẫu số khi so sánh tỷ lệ. Với A, tập test chỉ có 61 hồ sơ nên cần trình bày cả số ca.

Mỗi người phân tích cho **từng dataset** một ca FN và một ca FP nếu có. Lấy ca sai có xác suất dự đoán nghiêng mạnh nhất về lớp đã chọn trong từng loại lỗi. Nếu một loại lỗi bằng 0, ghi số ca thực tế. Đối chiếu đặc trưng, xác suất và ngưỡng để giải thích hành vi mô hình. Ghi một hạn chế của cách giải thích. Trong báo cáo chung, chọn một hồ sơ mà ba thuật toán bất đồng để đối chiếu trực tiếp.

## 5. Phần mở rộng chung – Split Conformal Prediction

Hoàng triển khai hàm Conformal dùng chung, kiểm tra thứ tự hai lớp và cách xử lý các xác suất bằng nhau. Từng thành viên dùng hàm chung cho hai mô hình mình đã fit (một cho mỗi dataset) và tự giải thích kết quả.

1. Dùng các mức bao phủ mục tiêu 90% và 95%, tương ứng α = 0,10 và 0,05. Trên tập hiệu chỉnh, với mỗi mẫu tính điểm `s = 1 − p(nhãn thật)`.
2. Sắp xếp n điểm tăng dần. Lấy q ở hạng `ceil((n+1) × (1−α))`, đếm từ 1. Nếu hạng vượt n, dùng q = +∞.
3. Với mẫu mới, giữ mọi nhãn y thỏa `1 − p(y) ≤ q`. Tính tỷ lệ tập nhãn chứa nhãn thật (độ bao phủ), tỷ lệ tập có đúng một nhãn, cả hai nhãn và tập rỗng. Dấu ≤ giữ cả trường hợp bằng ngưỡng, thường gặp với KNN và cây.
4. Với A, dùng toàn bộ phần calibration và test. Với B, **riêng phép đánh giá Conformal**, chọn một lượt cho mỗi bệnh nhân trong từng phần calibration/test bằng quy tắc cố định lấy lượt có encounter_id nhỏ nhất. Cả nhóm dùng cùng danh sách. Báo số bệnh nhân và ghi rõ kết quả áp dụng cho lượt được chọn. Bảng phân loại chính vẫn đánh giá toàn bộ lượt test.
5. So sánh tỷ lệ trả về một nhãn giữa các mô hình tại cùng mức mục tiêu, đồng thời báo độ bao phủ thực đo. Thảo luận sự đánh đổi giữa độ bao phủ và độ cụ thể. Bảo đảm lý thuyết là độ bao phủ biên dưới giả định các mẫu hiệu chỉnh và đánh giá có thể hoán đổi. Việc lấy một lượt mỗi người xử lý sự lặp bệnh nhân, còn khác biệt bệnh viện và thời gian vẫn có thể ảnh hưởng giả định này.

Tài liệu phương pháp: [Angelopoulos và Bates, nhập môn Conformal Prediction](https://arxiv.org/abs/2107.07511).

### File dùng chung và cách bàn giao

| File | Ai dùng và dùng để làm gì? |
|---|---|
| [conformal_prediction.py](Conformal_Prediction/conformal_prediction.py) | Hoàng phụ trách hàm `summarize(y_cal, p_cal, y_test, p_test)`. Tùng, Hoàng và Khải truyền nhãn thật cùng xác suất lớp 1 của mô hình mình vào hàm. Mỗi lần gọi cho một dataset trả về hai dòng, tương ứng 90% và 95%. |
| [run_conformal_reference.py](Conformal_Prediction/run_conformal_reference.py) | Cả ba tham khảo cách đọc bản chia, chọn lượt theo bệnh nhân ở Diabetes và lấy đúng cột xác suất lớp 1. Chạy `python Conformal_Prediction/run_conformal_reference.py` để tái tạo kết quả với các tham số cố định trong script. |
| [conformal_reference_results.csv](Conformal_Prediction/conformal_reference_results.csv) | Hoàng dùng bảng tham chiếu 12 dòng để đối chiếu cấu trúc kết quả. Mỗi dòng ứng với một dataset, một thuật toán và một mức bao phủ. |

Khi đã chốt tham số, mỗi người gửi Hoàng **4 dòng kết quả** từ mô hình cuối của mình (2 dataset × 2 mức bao phủ), kèm tên dataset và cấu hình mô hình. Hoàng ghép thành bảng chính thức và lưu riêng để cả nhóm có thể đối chiếu với bảng tham chiếu.

## 6. Trình tự bàn giao và tiêu chí hoàn thành

1. **Trước huấn luyện:** Tùng bàn giao danh sách dòng, đặc trưng, fold và kết quả kiểm tra theo bệnh nhân. Hoàng và Khải có thể thử các hàm đánh giá trên xác suất giả lập trong lúc chờ.
2. **Trước đánh giá test:** mỗi người bàn giao cấu hình đã chọn, bảng điểm CV, ngưỡng F2 và hai ngưỡng Conformal cho từng dataset. Ghi thời gian chạy. Lưu các lựa chọn trước khi xem bảng so sánh. Các tập test này đã được dùng trong đợt trước, cần công bố việc tái sử dụng khi viết phần hạn chế. Mọi điều chỉnh sau khi xem test phải ghi là phân tích thăm dò.
3. **Khi ghép bài:** mỗi người nộp code chạy lại được, hai bảng dự đoán, kết quả Conformal, phân tích lỗi và slide giải thích phương pháp của mình. Khải ghép hai bảng phân loại với ba hàng thuật toán mỗi bảng. Hoàng ghép bảng Conformal gồm 12 hàng (3 thuật toán × 2 dataset × 2 mức bao phủ). Tùng kiểm tra mẫu đánh giá khớp nhau giữa ba thuật toán.
4. **Khi bảo vệ:** Tùng giải thích khoảng cách và ảnh hưởng của k. Hoàng giải thích điều chuẩn và độ bao phủ Conformal. Khải giải thích quy tắc phân nhánh, quá khớp và tác động của ngưỡng lên số ca bỏ sót. Mỗi người trình bày ít nhất một lỗi quan sát được hoặc một loại lỗi có số ca bằng 0 cùng kết quả Conformal của mô hình mình.
