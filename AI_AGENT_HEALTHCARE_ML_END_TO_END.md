# Đặc tả thực thi end-to-end cho AI Agent: Healthcare and Patient Risk Analysis

> **Cách dùng:** Giao toàn bộ file này cho AI Agent có quyền truy cập thư mục dự án, terminal và Internet. Yêu cầu Agent thực thi cho đến khi đáp ứng tiêu chí nghiệm thu ở cuối file. Đây là đặc tả công việc, không phải báo cáo kết quả đã thực nghiệm.
>
> **Ngôn ngữ:** Báo cáo, diễn giải và nội dung thuyết trình bằng tiếng Việt; tên biến, module, API và thuật ngữ chuẩn có thể dùng tiếng Anh.
>
> **Ngày chuẩn bị:** 11/10/2026. Phiên bản thư viện, quyền truy cập mô hình và nghiên cứu mới phải được kiểm tra lại khi thực hiện.

> **Cập nhật theo tài liệu môn học:** Đã đối chiếu 6 PDF trong `Slide/` (236 trang). Các tham chiếu trang dưới đây dùng **số thứ tự trang PDF, bắt đầu từ 1**, không dùng số in ở chân slide vì một số file có hai hệ số khác nhau. Giữ nguyên tài liệu giảng viên.

## 1. Đọc hiểu đề bài và mục tiêu

Đề bài môn Machine Learning yêu cầu:

1. Tự tìm kiếm dữ liệu, sử dụng **tối thiểu hai bộ dữ liệu**.
2. Tổng hợp và vận dụng những kiến thức đã học.
3. Phân tích kết quả và lỗi của mô hình.
4. Có điểm cộng khi bổ sung kiến thức ngoài chương trình, đặc biệt hướng nghiên cứu ML mới.
5. Chủ đề: **Healthcare and patient risk analysis**.

Mục tiêu của Agent là tạo một bài tập lớn hoàn chỉnh, có thể chạy lại, giải thích được và bảo vệ trước giảng viên. Thành công không được định nghĩa bằng một mức accuracy tùy ý. Thành công là thực nghiệm đúng, kết quả trung thực, lập luận có bằng chứng và sản phẩm đáp ứng đầy đủ đề bài.

**Tên đề tài đề xuất:**

> Phân tích nguy cơ bệnh nhân bằng học máy trên dữ liệu y tế dạng bảng: phân loại bệnh tim và dự đoán tái nhập viện sớm ở bệnh nhân đái tháo đường.

Hai bộ dữ liệu tạo thành hai nghiên cứu tình huống độc lập, cùng sử dụng một khung thực nghiệm. Không ghép hàng hoặc cột của hai bộ thành một bài toán chung vì chúng có quần thể, biến đầu vào và nhãn khác nhau. Không gọi việc đánh giá hai nhãn khác nhau là external validation của cùng một mô hình.

### 1.1. Ma trận đáp ứng yêu cầu

| Yêu cầu | Công việc bắt buộc | Bằng chứng bàn giao |
|---|---|---|
| Tự tìm dữ liệu | Tìm, kiểm chứng và tải từ nguồn có xuất xứ | Data cards, URL, giấy phép, ngày tải, SHA-256 |
| Ít nhất hai bộ | Thực nghiệm hoàn chỉnh trên hai bộ riêng biệt | Hai pipeline, hai bảng kết quả, hai phần phân tích |
| Tổng hợp kiến thức | Định nghĩa bài toán, tối ưu/regularization, Logistic Regression, KNN, Gaussian/Naive Bayes, đánh giá | Code chạy được và bảng ánh xạ tới trang slide |
| Phân tích kết quả | So sánh baseline, metric phù hợp, độ bất định, thời gian chạy | Bảng CSV, biểu đồ, luận giải |
| Phân tích lỗi | FN/FP, nhóm bệnh nhân, confidence, missingness, overfitting | Error analysis và case studies |
| Kiến thức mở rộng | Explainability, mô hình ensemble bổ sung và một thí nghiệm TabPFN hoặc conformal | Phân biệt rõ kiến thức trong slide, mở rộng triển khai và nghiên cứu gần đây |
| Tính tái lập | Seed, cấu hình, môi trường, split cố định, lệnh chạy | README, manifest, tests, logs |

### 1.2. Phạm vi đã xác nhận từ thư mục Slide

| Tài liệu | Số trang | Nội dung đã xác nhận và trang PDF | Áp dụng trong bài tập lớn |
|---|---:|---|---|
| [Slide1.pdf](Slide/Slide1.pdf) | 23 | T–E–P: 7; supervised/classification: 9–12; pipeline, loss/objective/metric, train/inference: 18–23 | Định nghĩa riêng T, E, P cho hai bài toán; đóng gói preprocessing và model |
| [Slide2.pdf](Slide/Slide2.pdf) | 28 | Linear Regression; gradient descent: 7–17; normal equation: 18–19; polynomial: 20; metrics/residuals: 21–24; Ridge/LASSO: 25–28 | Giải thích tối ưu, scaling, regularization và liên hệ với Logistic Regression; không ép OLS thành mô hình xác suất nhị phân |
| [Slide3.pdf](Slide/Slide3.pdf) | 59 | Sigmoid/log-odds: 5–16; MLE/log loss/gradient: 17–24; L1/L2: 25–28, 37; pipeline/CV: 29–36; threshold/cost/top-k: 40–48; ROC/AP/calibration/class weights: 49–54; multiclass: 55–59 | Logistic Regression là baseline cốt lõi; thực nghiệm L1/L2, class weights, threshold, xác suất và ngân sách gắn cờ |
| [Slide4-KNN.pdf](Slide/Slide4-KNN.pdf) | 30 | KNN và khoảng cách: 4–13; scaling/weights: 14–18; k, bias–variance, imbalance, chiều dữ liệu, chi phí: 19–24; pipeline/search: 25–28 | KNN bắt buộc trên Heart; nghiên cứu k, khoảng cách, weighting, scaling và prediction latency |
| [Slide5-Prob-model.pdf](Slide/Slide5-Prob-model.pdf) | 42 | Uncertainty: 4; conditional independence/Bayes: 13–21; Gaussian/GMM: 26–29; MLE/MAP: 33–36; Gaussian classifier/Naive Bayes: 37–40; EM: 41 | Gaussian Naive Bayes; so sánh covariance đầy đủ có regularization với covariance chéo trên Heart; giải thích giả định xác suất |
| [Model-assessment.pdf](Slide/Model-assessment.pdf) | 54 | Selection/assessment: 5–7; metrics: 11–19; CV: 20–28; leakage/group split: 29–32; threshold/OOF/calibration: 33–37; bias–variance/learning curves: 38–41; bootstrap: 42–49; workflow: 50–54 | Protocol độc lập test, pipeline trong fold, group split, OOF, learning curves và cluster bootstrap |

`Slide1.pdf`, trang PDF 2, ghi phần final là “projects with slide” và chiếm 70% điểm môn học. Vì vậy bài thuyết trình là sản phẩm cần hoàn thiện. Đây là tỷ trọng điểm môn học trong tài liệu, **không phải rubric chi tiết chấm từng hạng mục của bài tập lớn**.

Tạo `reports/course_alignment.md` với các cột: `concept`, `source_file`, `pdf_page`, `evidence_summary`, `application`, `experiment_id`, `artifact_path`, `status`. Điền nguồn dựa trên bảng trên và kiểm tra lại nếu file được cập nhật. Có thể thêm hash các PDF để truy nguyên phiên bản.

### 1.3. Phân biệt kiến thức trong môn học và kiến thức bổ sung

- **Đã có trong slide:** Logistic Regression, KNN, Gaussian classifier, Gaussian Naive Bayes, GMM/EM, MLE/MAP, L1/L2, preprocessing, CV, group split, threshold/cost, class weights, discrimination/calibration, learning curves và bootstrap.
- **Mở rộng cách triển khai:** post-hoc sigmoid/isotonic calibration và cross-fitting phức tạp đi sâu hơn phần kiểm tra calibration trong slide. Không gọi toàn bộ chủ đề calibration là ngoài chương trình.
- **Chưa thấy được dạy chi tiết trong sáu file:** Decision Tree/Random Forest/boosting, SVM, SHAP, conformal prediction và tabular foundation models. Có thể sử dụng, nhưng ghi là kiến thức bổ sung trong phạm vi tài liệu hiện có. Một hình cây hoặc một tên thuật toán xuất hiện trong ví dụ chưa chứng minh đã có bài giảng riêng về thuật toán đó.
- **Không phải hướng mới chỉ vì chưa học:** Random Forest, boosting, SHAP và conformal cơ bản. Phần “ML mới” cần literature review gần đây và thí nghiệm phù hợp, ưu tiên TabPFN như mục 11.
- PCA và clustering chỉ thêm khi giúp trả lời câu hỏi cụ thể. GMM đã có trong Slide5 nên không tính GMM là đóng góp mới ngoài môn học.

Không yêu cầu mọi chương phải có một bài toán riêng. Hai outcome chính là phân loại; kiến thức Linear Regression được vận dụng qua tối ưu, regularization và so sánh bản chất hàm đầu ra/loss. Nếu thêm regression phải đặt câu hỏi và thời điểm dự đoán riêng, không chuyển `readmitted` thành số rồi chạy OLS để “đủ chương”.

## 2. Vai trò, quyền tự chủ và nguyên tắc làm việc

Bạn là AI Agent phụ trách đồng thời nghiên cứu dữ liệu, triển khai ML, kiểm định thực nghiệm, phân tích và viết báo cáo. Hãy trực tiếp thực hiện công việc, không chỉ đưa ra kế hoạch hoặc tạo khung code có TODO.

### 2.1. Phạm vi đã được người dùng ủy quyền

Trong phạm vi bài tập lớn này, bạn được tự chủ:

- Đọc và tạo/sửa file dự án; tổ chức thư mục, viết chương trình, notebook, tài liệu và tests.
- Truy cập Internet, tìm tài liệu, tải dữ liệu công khai, tải trọng số được phép sử dụng.
- Tạo môi trường ảo trong dự án và cài dependency cần thiết; tự sửa lỗi tương thích.
- Sử dụng CPU/GPU sẵn có, chạy các thí nghiệm, tinh chỉnh tham số và sinh artifact.
- Tự quyết định những lựa chọn kỹ thuật thông thường: framework, cấu trúc code, seed, cấu hình, thuật toán và cách trình bày.
- Tạo Git repository/nhánh/commit cục bộ nếu phù hợp; bảo toàn công việc người dùng đang có.
- Tạo demo chạy cục bộ, báo cáo, slide và gói bàn giao.
- Tiếp tục xử lý lỗi, chọn phương án thay thế hợp lý và ghi nhận quyết định, không hỏi xác nhận ở từng bước.

Quyền này áp dụng cho nhiệm vụ và tài nguyên được cấp; không tự tạo ra quyền vượt giới hạn của công cụ hay hệ thống. Không tự thanh toán, thuê GPU có phí, tiết lộ credentials, xóa dữ liệu ngoài phạm vi, gửi thư cho người khác, xuất bản công khai hoặc nộp bài thay người dùng khi chưa có chỉ dẫn cụ thể. Nếu gặp một bước như vậy, hoàn thiện phần còn lại và chuẩn bị sản phẩm cụ thể trước khi cần người dùng can thiệp.

### 2.2. Quy tắc vận hành

1. Đọc hướng dẫn repository hiện có. Không ghi đè file của người dùng một cách mù quáng.
2. Lưu trạng thái ở `TASKS.md`; mỗi task có trạng thái, đầu ra và điều kiện hoàn thành.
3. Ghi quyết định đáng kể vào `reports/decision_log.md` và vấn đề thực sự chưa giải quyết vào `reports/limitations.md`.
4. Khi một lệnh lỗi, xem log, xác định nguyên nhân rồi sửa. Không lặp cùng lệnh vô hạn.
5. Không bịa metric, biểu đồ, kết quả test, tài liệu tham khảo hoặc thông tin cấu hình máy.
6. Phân biệt `completed`, `failed`, `skipped`, `blocked`; một tính năng bị skip không được mô tả là đã thực hiện.
7. Cập nhật tiến độ sau mỗi mốc lớn; tiếp tục đến khi xong các yêu cầu cốt lõi.
8. Ưu tiên chất lượng khoa học và tính tái lập hơn số lượng mô hình hoặc giao diện đẹp.

## 3. Câu hỏi nghiên cứu và định nghĩa bài toán

Trả lời bằng kết quả thực nghiệm:

- RQ1: Mô hình nào có chất lượng dự đoán tốt nhất trong từng bài toán theo metric đã chốt trước?
- RQ2: Mất cân bằng nhãn và lựa chọn threshold ảnh hưởng thế nào đến FN, FP và số ca bị gắn cờ?
- RQ3: Xác suất dự đoán có được hiệu chỉnh tốt không; calibration có cải thiện không?
- RQ4: Mô hình sai ở những nhóm hoặc tình huống nào, và có bằng chứng gì về nguyên nhân?
- RQ5: Phương pháp nâng cao có mang lại giá trị so với baseline khi xét hiệu năng, tài nguyên và khả năng giải thích?

Trong báo cáo, khai báo T–E–P theo Slide1 trang PDF 7: T là từng nhiệm vụ phân loại; E là dữ liệu development dùng học và chọn mô hình; P là metric chính trên đánh giá độc lập. Bổ sung câu hỏi phụ về ảnh hưởng của regularization, khoảng cách KNN và giả định độc lập có điều kiện; trả lời bằng E08–E11 ở mục 12.

### 3.1. Dataset A: UCI Heart Disease — Cleveland

Nguồn: [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease), DOI `10.24432/C52P4X`.

UCI mô tả Cleveland với 303 quan sát, 13 biến đầu vào, có giá trị thiếu; nhãn gốc `num` nhận các giá trị 0–4. Trang nguồn ghi giấy phép CC BY 4.0. Xác nhận lại schema và số hàng thực tế sau tải.

**Thiết kế của dự án:**

- Dùng riêng bản Cleveland có nguồn gốc rõ ràng. Không thay bằng bản sao đã nhân bản thành hàng nghìn dòng.
- Đặt `y = 1` khi `num > 0`; `y = 0` khi `num == 0`.
- Đây là phân loại sự hiện diện của bệnh trong dữ liệu chẩn đoán; không tuyên bố là nguy cơ mắc bệnh trong 10 năm hay tiên lượng tương lai.
- Bối cảnh mặc định: hỗ trợ phân loại khi các biến xét nghiệm đầu vào đã sẵn có.
- Kiểm tra thời điểm thu nhận các biến và mối liên hệ của chúng với quy trình xác định nhãn. Các biến giàu thông tin chẩn đoán như `ca`, `thal` cần thảo luận riêng; không mô tả mô hình đầy đủ là công cụ sàng lọc trước xét nghiệm.
- Ablation hữu ích: bộ biến đầy đủ so với bộ biến hạn chế, dễ thu nhận hơn. Xác định danh sách trước thực nghiệm và có lý do cho từng biến.
- Primary metric: ROC-AUC; secondary gồm AP, sensitivity, specificity, F1, balanced accuracy và Brier score.

### 3.2. Dataset B: Diabetes 130-US Hospitals for Years 1999–2008

Nguồn: [UCI Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), DOI `10.24432/C5230J`.

Nguồn mô tả 101.766 lượt điều trị, có các ID `encounter_id`, `patient_nbr`, và cung cấp `diabetic_data.csv`, `IDS_mapping.csv`; giấy phép CC BY 4.0. Đối chiếu các trường với file gốc khi tải.

**Thiết kế của dự án:**

- Một hàng là một lượt điều trị, không mặc định là một bệnh nhân độc lập.
- Nhãn: `readmitted == '<30'` là 1; `'>30'` và `'NO'` là 0. Kiểm tra các giá trị nhãn thực tế trước ánh xạ; không tự ánh xạ nhãn lạ thành 0.
- Thời điểm dự đoán: lúc xuất viện. Chỉ sử dụng thông tin có trước hoặc tại thời điểm này.
- Định nghĩa outcome theo mã `<30` của dữ liệu; không tự đổi thành `≤30`.
- Giữ `patient_nbr` làm group để chia tập và bootstrap; loại khỏi feature matrix. Loại `encounter_id` khỏi features.
- Giữ nhiều lượt điều trị nếu chia nhóm theo bệnh nhân đúng. Không tự coi thứ tự ID là thứ tự thời gian.
- Định nghĩa cohort chính là những lượt xuất viện còn khả năng theo dõi tái nhập viện. Đọc `IDS_mapping.csv` để quyết định loại các tình trạng tử vong/hospice; không đoán mã ID. Ghi rõ tiêu chí và số hàng bị loại.
- Không tự loại transfer hoặc nhóm phức tạp chỉ để tăng metric; nếu cần, lập luận và phân tích độ nhạy.
- `'NO'` được hiểu theo nhãn quan sát, không đảm bảo bệnh nhân không tái nhập viện ở ngoài hệ thống.
- Primary metric: **Average Precision (AP)**. Báo cáo rõ tên và cách tính; không đồng nhất AP với diện tích PR tính bằng quy tắc hình thang.
- Secondary: ROC-AUC, sensitivity, specificity, precision, F1/F2, balanced accuracy, Brier score và tỷ lệ gắn cờ.

### 3.3. Điều kiện thay dataset

Thử nguồn chính, thư viện chính thức và mirror có thể xác minh trước khi thay. Nếu thật sự không truy cập được, chọn một dataset y tế công khai khác có target rõ, giấy phép phù hợp và đủ metadata; cập nhật toàn bộ thiết kế, data card, nguồn và câu hỏi nghiên cứu.

Hai bản xử lý khác nhau của cùng dataset không đáp ứng yêu cầu hai bộ độc lập. Không sử dụng dữ liệu tổng hợp để thay thế hai bộ thật. Không chọn dữ liệu yêu cầu phê duyệt truy cập kéo dài làm phụ thuộc bắt buộc.

## 4. Khảo sát môi trường và cấu trúc sản phẩm

### 4.1. Khởi tạo

- Kiểm tra OS, Python, RAM, CPU, GPU, dung lượng ổ đĩa và Internet.
- Dùng Python phiên bản được bộ thư viện lựa chọn hỗ trợ; tạo `.venv` riêng.
- Kiểm tra tài liệu tương thích tại thời điểm cài, pin phiên bản sau khi xác nhận pipeline chạy.
- Bộ thư viện nền: numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, joblib, pyyaml, pytest và công cụ tải HTTP phù hợp.
- Tùy nhu cầu: imbalanced-learn, xgboost hoặc lightgbm hoặc catboost, shap, tabpfn, jupyter, nbconvert, streamlit.
- Không cài cả ba thư viện boosting nếu một thư viện đã đáp ứng tốt.
- Thiết kế dùng đường dẫn tương đối từ project root và chạy được với đường dẫn Windows chứa khoảng trắng/Unicode.
- Giới hạn luồng xử lý để tránh nested parallelism làm treo máy.

### 4.2. Cấu trúc đề xuất

```text
project/
  AI_AGENT_HEALTHCARE_ML_END_TO_END.md
  README.md
  TASKS.md
  pyproject.toml
  requirements-lock.txt
  .gitignore
  configs/
    heart.yaml
    readmission.yaml
    smoke.yaml
  Slide/                         # tài liệu môn học, chỉ đọc
  data/
    raw/heart/
    raw/readmission/
    processed/
    splits/
    metadata/
      sources.json
      heart_data_card.md
      readmission_data_card.md
      feature_availability.csv
  src/healthcare_ml/
    data.py
    cohort.py
    splitting.py
    preprocessing.py
    models.py
    training.py
    calibration.py
    thresholds.py
    evaluation.py
    explainability.py
    error_analysis.py
    reporting.py
    cli.py
  scripts/
    run_all.ps1
    run_all.sh
  notebooks/
    01_heart_eda.ipynb
    02_readmission_eda.ipynb
    03_results_and_errors.ipynb
    04_optimization_and_course_links.ipynb
  tests/
  artifacts/
    runs/
    models/
    predictions/
    metrics/
    figures/
  reports/
    protocol.md
    course_alignment.md
    literature_review.md
    decision_log.md
    limitations.md
    final_report.md
    final_report.pdf
    slides_outline.md
    presentation.pptx
    defense_qa.md
    model_cards/
    acceptance_checklist.md
  app/
    streamlit_app.py
```

Có thể điều chỉnh module để code gọn; không tạo hàng loạt file rỗng cho đủ cây thư mục. Notebook gọi lại logic trong `src/`, tránh hai bản tiền xử lý khác nhau. File PDF/PPTX là đầu ra cần hoàn thiện nếu môi trường có công cụ hỗ trợ; nếu bị chặn thực sự, bàn giao Markdown hoàn chỉnh và ghi rõ artifact chưa tạo, không tạo file giả có đuôi tương ứng.

## 5. Thu thập, kiểm định và mô tả dữ liệu

### 5.1. Acquisition và provenance

1. Tải từ UCI; có thể dùng `ucimlrepo` với ID 45 và 296 sau khi kiểm tra API đang dùng.
2. Với readmission, bảo đảm có `patient_nbr` và bảng mã; một API chỉ trả features/targets có thể bỏ ID cần thiết. Dùng raw CSV hoặc lấy IDs riêng rồi kiểm tra căn chỉnh hàng.
3. Lưu bản raw bất biến. Không sửa trực tiếp raw để xử lý missing.
4. Tạo `sources.json`: URL trang giới thiệu, URL tải thực tế, DOI, giấy phép, thời điểm tải UTC, tên file, kích thước, SHA-256, phương thức tải.
5. Nếu tải archive, kiểm tra đường dẫn bên trong trước giải nén; chỉ giải nén vào thư mục dữ liệu tương ứng.
6. Tải lại phải idempotent: kiểm tra checksum/cache, không nhân đôi dữ liệu.

### 5.2. Data card bắt buộc cho từng bộ

- Nguồn, mục đích gốc, thời kỳ dữ liệu, đối tượng, đơn vị quan sát.
- Target, ý nghĩa lớp dương, thời điểm dự đoán, population mục tiêu của thực nghiệm.
- Số hàng/cột raw và sau từng bước lọc; số bệnh nhân nếu có ID.
- Data dictionary: tên, kiểu, đơn vị, mã categorical, missing token, phạm vi hợp lệ.
- Missingness, duplicate, imbalance, biến cần loại, lý do loại.
- Quy tắc cohort, nguy cơ leakage, bias lấy mẫu và giới hạn ứng dụng.
- Giấy phép và cách trích dẫn.

### 5.3. Kiểm tra chất lượng

- Kiểm tra `?`, chuỗi rỗng, NaN và token đặc biệt theo từng cột.
- Không dùng cấu hình parser biến mọi chuỗi `None` thành missing: ở một số biến xét nghiệm, đây có thể là trạng thái chưa xét nghiệm có ý nghĩa. Đối chiếu tài liệu và giữ thành category khi phù hợp.
- Số 0 không mặc định là missing.
- Biến mã hóa bằng số không mặc định là liên tục; ví dụ mã nhập viện hoặc nhóm triệu chứng cần xử lý categorical.
- Giá trị bất thường được ghi nhận trước khi quyết định loại/cắt; ngưỡng học từ dữ liệu phải fit trên train.
- Phân biệt duplicate do sao chép file với hai quan sát thật có cùng features. Không tùy tiện loại bệnh nhân giống nhau về thuộc tính.
- Kiểm tra ID trùng và xung đột nhãn. Nếu không có ID, trình bày giới hạn của việc phát hiện cùng bệnh nhân.
- Tạo cohort flow table: `step`, `reason`, `rows_before`, `rows_removed`, `rows_after`, `positive_rate`.

## 6. Protocol đánh giá và chống data leakage

**Phải ghi `reports/protocol.md` trước khi tuning và trước khi mở kết quả test.** Chốt dataset, cohort, features, split, metric chính, search budget, tiêu chí chọn mô hình và threshold.

### 6.1. Heart Disease

- Giữ test khoảng 20% bằng stratified split, seed mặc định 42; lưu danh sách row key.
- 80% còn lại là development set. Model selection bằng stratified 5-fold CV trong development; giảm số fold nếu số ca dương/âm không đủ và ghi lý do.
- Khi đủ tài nguyên, dùng nested CV trên development: outer 5 folds để ước lượng ổn định, inner 3 folds để tuning. Không trộn điểm inner tuning với điểm outer evaluation.
- Vì dữ liệu nhỏ, test có khoảng vài chục bệnh nhân; phải trình bày uncertainty lớn và không suy ra ưu thế chắc chắn từ chênh lệch nhỏ.
- Nếu phát hiện duplicate do sao chép, xử lý trước khi chia theo quy tắc có căn cứ; nếu có nhóm liên quan đã xác minh thì chia theo group.

### 6.2. Readmission

- Tất cả phép chia phải tách biệt `patient_nbr`.
- Gợi ý: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, chọn trước một fold cố định làm test khoảng 20%; development là các fold còn lại.
- Trên development, dùng stratified group CV 3 folds cho sàng lọc và 5 folds cho vòng cuối nếu tài nguyên cho phép.
- Các tỷ lệ là xấp xỉ do nhóm có nhiều hàng. Báo cáo tỷ lệ nhãn, số encounter và số bệnh nhân theo split.
- Kiểm tra giao tập patient ID bằng assertion ở mọi lớp train/validation/test, kể cả các split nội bộ phục vụ calibration và tuning.
- Không chọn một test fold khác vì điểm tốt hơn hoặc phân phối thuận lợi hơn.
- Nếu không có timestamp đáng tin cậy thì không tuyên bố có temporal validation. Nếu không có hospital ID thì không tuyên bố kiểm định tổng quát hóa giữa bệnh viện.

### 6.3. Quy trình lựa chọn và khóa test

1. Raw schema checks và quy tắc cohort xác định theo nguồn được áp dụng trước chia tập.
2. Sau khi chia, EDA định hướng lựa chọn model/features chỉ dùng development.
3. Với mỗi fold, fit imputer, encoder, scaler, selector và sampler trên phần train của fold.
4. Chọn hyperparameters và họ mô hình bằng CV development theo metric chính. Khi gần tương đương, ưu tiên mô hình đơn giản, ổn định và ít tốn tài nguyên hơn theo quy tắc đã ghi.
5. Xác suất dùng để chọn threshold phải là xác suất trên các quan sát không được dùng để fit mô hình sinh ra chúng.
6. Nếu thêm calibration, các dự đoán đánh giá calibration cũng phải từ quan sát không được dùng để fit calibrator. Dùng held-out calibration/threshold partitions tách nhóm, hoặc nested cross-fitting.
7. Đối với dữ liệu nhỏ, ưu tiên cross-fitting thay vì chia quá nhiều holdout khiến mỗi phần không đủ ca dương. Mô tả rõ các tầng fit và predict.
8. Một cách triển khai đúng: outer development fold là tập đánh giá; bên trong outer-train mới tune model, tạo inner OOF predictions để fit calibrator; sau đó fit base model trên outer-train và predict outer-validation qua calibrator. Gom outer predictions để chọn threshold cuối.
9. Fit artifact cuối trên development bằng recipe đã chốt. Nếu dùng calibrator học từ OOF nhưng base model refit toàn bộ development, ghi nhận khác biệt cỡ tập huấn luyện; không coi điểm trên tập fit calibrator là đánh giá độc lập.
10. Khóa pipeline, feature list, calibrator và threshold; tạo manifest/hash. Chạy một đợt đánh giá test cuối cho danh sách model/ablation đã chốt.
11. Phân tích lỗi trên test là hậu kiểm. Mọi thay đổi gợi ý từ đó ghi là công việc tiếp theo; không chỉnh model rồi tiếp tục dùng cùng test để báo cáo một kết quả “độc lập”.

Lưu split và protocol đủ chi tiết để kiểm tra bằng code. Không dựa vào lời khẳng định “đã tránh leakage” trong báo cáo.

## 7. EDA và feature engineering

### 7.1. EDA có câu hỏi

Sinh trên development: phân bố nhãn, missingness theo biến và nhãn, phân bố numerical/categorical, outlier, quan hệ đơn biến với target, tương quan numerical và số lượt điều trị mỗi bệnh nhân.

Mỗi biểu đồ trong báo cáo có kết luận cụ thể, kèm giới hạn. Tránh hàng chục biểu đồ không được diễn giải. Test chỉ được dùng cho thống kê và hậu kiểm sau khi đóng băng protocol.

### 7.2. Pipeline mặc định

- Numerical: median imputation; cân nhắc missing indicator; standard scaling cho Logistic Regression, SVM, KNN.
- Categorical: hạng mục missing rõ ràng hoặc most frequent theo cấu hình; one-hot với xử lý category chưa xuất hiện ở train.
- Cây/boosting: không cần scaling nếu thuật toán không yêu cầu; bảo đảm hỗ trợ sparse/dense đúng.
- Nhóm category hiếm và bỏ cột missing quá cao là phép biến đổi cần học ngưỡng/quy tắc từ train, trừ cột bị loại theo metadata độc lập.
- Target encoding chỉ dùng khi có cross-fitting đúng và group-aware; không mặc định sử dụng.
- PCA chỉ là ablation hợp lý cho nhánh numerical đã chuẩn hóa, không bắt buộc cho mọi pipeline.

### 7.3. Feature engineering cho readmission

Các ý tưởng cần kiểm chứng, không mặc định tất cả sẽ cải thiện:

- Gộp ICD diagnosis thành nhóm theo bảng mã có nguồn; xử lý mã dạng chữ và missing riêng, không ép tất cả sang float.
- Tổng số lần sử dụng dịch vụ trước đây từ các biến outpatient/emergency/inpatient nếu ý nghĩa được xác nhận.
- Số loại thuốc thay đổi, trạng thái dùng thuốc, trạng thái xét nghiệm.
- Tuổi theo khoảng: ordinal hoặc categorical; nếu dùng trung điểm phải nói rõ đó là xấp xỉ.
- Không suy luận lịch sử bệnh nhân bằng cách nhìn sang encounter tương lai hoặc tổng hợp toàn bộ các dòng của một người.

Tạo `feature_availability.csv` với `feature`, `available_at_prediction`, `evidence`, `include`, `reason`. Feature engineering phải có ablation so với feature set cơ sở.

## 8. Hệ thống mô hình và tìm siêu tham số

### 8.1. Mức tối thiểu

Chọn mô hình theo nội dung slide, không dùng toàn ensemble khiến phần đã học bị mờ nhạt. Ma trận bắt buộc:

| Mô hình | Heart | Readmission | Vai trò và tham số |
|---|---|---|---|
| DummyClassifier | Bắt buộc | Bắt buộc | prior, most_frequent |
| Logistic Regression | Bắt buộc | Bắt buộc | Đã học; C, L1/L2, class_weight, solver tương thích |
| KNN | Bắt buộc | Thí nghiệm có giới hạn ngân sách nếu khả thi | Đã học; k, Manhattan/Euclidean, uniform/distance |
| Gaussian Naive Bayes | Bắt buộc | Bắt buộc | Đã học; var_smoothing, class priors học trên train |
| Gaussian classifier covariance đầy đủ có regularization | Bắt buộc trên nhánh numerical | Tùy chọn | Liên hệ Slide5 trang 37–40; có thể dùng QDA, tune reg_param |
| Random Forest | Tùy chọn | Bắt buộc | Mở rộng ensemble; n_estimators, depth, leaf size |
| Một thuật toán boosting | Bắt buộc | Bắt buộc | Baseline mở rộng mạnh; learning_rate, depth/leaves, số vòng |

Heart có bốn họ chính Logistic Regression, KNN, Gaussian generative và boosting; GaussianNB và Gaussian classifier đầy đủ là hai biến thể trong cùng họ, không đếm thành hai họ độc lập. Readmission có Logistic Regression, GaussianNB, Random Forest và boosting. Decision Tree riêng, SVM và MLP là tùy chọn sau khi hoàn thành các model trên.

**Thiết kế Gaussian có ý nghĩa:**

- GaussianNB và Gaussian classifier đầy đủ chạy trên một nhánh numerical được xác định từ data dictionary, impute trong fold. Không coi mã categorical hoặc one-hot là biến Gaussian hợp lệ mà không nêu xấp xỉ.
- So sánh GaussianNB với Gaussian classifier đầy đủ trên **cùng numerical features và cùng folds** ở Heart. Thêm Logistic Regression trên cùng nhánh numerical để kiểm soát ảnh hưởng của feature set.
- Bảng tổng hợp gắn cột `feature_set`; không quy mọi chênh lệch với model dùng full features cho thuật toán.
- Dùng regularization để kiểm soát covariance gần suy biến. Ghi convergence/numerical warnings; không ép nghịch đảo ma trận lỗi.
- Có thể mở rộng mixed Naive Bayes (Gaussian cho numerical, categorical likelihood có smoothing cho categorical), nhưng đây là phần triển khai thêm; kiểm chứng riêng trước khi thay baseline.

**Thiết kế KNN:**

- Heart: tune k trong tập như `{3, 5, 7, 11, 15, 21}`, chỉ dùng k không vượt số mẫu train-fold; `p ∈ {1, 2}`, `weights ∈ {uniform, distance}`.
- Scaling numerical, encode categorical trong pipeline; giải thích hình học khoảng cách khi ghép numerical với one-hot.
- Readmission: thử pilot chỉ trên development, đo chi phí rồi quyết định. Nếu phải subsample, lấy theo bệnh nhân và chạy cả Logistic Regression/GaussianNB trên cùng mẫu để so sánh công bằng; báo riêng khỏi full-data leaderboard.
- Giải thích bias–variance khi đổi k, nguy cơ chiều cao và độ trễ inference theo Slide4; không chỉ ghi điểm KNN cuối.

### 8.1a. Thực nghiệm nối kiến thức tối ưu và xác suất

Tạo `04_optimization_and_course_links.ipynb`, sử dụng development của Heart và các ví dụ nhỏ có nhãn minh họa rõ:

1. Giải thích khác biệt Linear Regression và Logistic Regression: đầu ra, squared error so với Bernoulli negative log-likelihood; liên hệ Slide2 và Slide3.
2. Viết Logistic Regression bằng NumPy ở mức gọn: sigmoid ổn định số, mean log loss, vectorized gradients, cập nhật đồng thời w/b và tùy chọn L2 không phạt intercept. Có loss curve theo iteration, đối chiếu vài learning rates trên development train/validation.
3. Kiểm tra gradient bằng finite differences trên ví dụ nhỏ; đối chiếu predictions/log loss với triển khai thư viện khi cùng dữ liệu, objective, mức regularization và điều kiện hội tụ. Không giả định `lambda = 1/C` đúng dưới mọi quy ước chuẩn hóa loss/API.
4. Dùng triển khai thư viện làm pipeline chính; bản NumPy chứng minh hiểu thuật toán, không cần nhân đôi toàn bộ hệ thống tuning.
5. Giải thích MLE, MAP và liên hệ L2 với Gaussian prior dưới giả định phù hợp; phân biệt point estimate với posterior đầy đủ. Logistic Regression thường dùng không tự trở thành full Bayesian inference.
6. Trình bày công thức Gaussian class score và diagonal covariance của GaussianNB; đối chiếu giả định với tương quan và phân phối development thực tế.

GMM/EM có thể dùng làm phân tích phân nhóm bệnh nhân bổ sung trên numerical features: fit chỉ bằng development, không dùng target để dựng cụm, giải thích soft assignments và tránh gọi cụm là chẩn đoán. Đây là kiến thức đã học, không thay thế các bài toán phân loại bắt buộc.

### 8.2. Imbalance

- So sánh cấu hình không trọng số và `class_weight`/trọng số tương đương.
- Threshold tuning là một thí nghiệm riêng, không thay thế việc đánh giá ranking.
- Chỉ thêm resampling khi có câu hỏi rõ. Dùng `imblearn.Pipeline` bên trong CV.
- Không SMOTE trước chia tập. Không SMOTE trực tiếp lên one-hot rồi diễn giải các giá trị fractional như bệnh nhân thật; với dữ liệu hỗn hợp cân nhắc SMOTENC trước one-hot hoặc random oversampling.
- Validation/test giữ phân phối tự nhiên. Calibration thực hiện trên phân phối đích chưa được resample.
- Không cộng dồn class weighting, oversampling và threshold tuning mà không có ablation để biết tác dụng từng phần.

### 8.3. Tuning và giới hạn tài nguyên

- Dùng randomized search hoặc công cụ tối ưu tương đương, seed cố định.
- Khởi đầu 10–20 cấu hình cho model phù hợp; mô hình đơn giản có thể dùng grid nhỏ.
- Sàng lọc rồi tập trung khoảng 20–40 trial cho 1–2 ứng viên tốt nếu ngân sách cho phép.
- Budget là mặc định vận hành, không phải chỉ tiêu phải tiêu hết. Sau smoke run, ước lượng thời gian thực tế rồi điều chỉnh và ghi log.
- Có timeout/checkpoint cho từng thí nghiệm; khi quá đắt thì giảm search space, folds hoặc số cây trước khi bỏ hẳn.
- Early stopping chỉ được nhìn vào validation nội bộ, không dùng test.
- So sánh model cùng split và ngân sách tuning đủ công bằng; ghi rõ chênh lệch tài nguyên.
- Một metric bất thường như AUC gần 1 phải kích hoạt audit target proxy, duplicate, ID, leakage và split; không mặc định xem đó là thành tích.

## 9. Đánh giá, calibration và threshold

### 9.1. Metric bắt buộc

Cho mỗi model: prevalence, ROC-AUC, AP, accuracy, balanced accuracy, precision, recall/sensitivity, specificity, F1, Brier score, log loss, confusion matrix, training time, inference time và số features sau biến đổi. Ghi feature set để phân biệt numerical-only với full features.

- `Sensitivity = TP / (TP + FN)`.
- `Specificity = TN / (TN + FP)`.
- `Precision = TP / (TP + FP)`.
- `F2` ưu tiên recall hơn F1; dùng bổ sung cho readmission.
- Brier score đánh giá sai số xác suất, không chỉ riêng calibration.
- Metric không xác định phải ghi NA và lý do; không im lặng gán 0 làm kết luận sai.
- Dummy prior có xác suất hằng; AP tham chiếu gắn với prevalence. Tránh gọi accuracy cao ở bài toán mất cân bằng là tốt nếu recall lớp dương thấp.

Tách bảng điểm CV development khỏi bảng test cuối. Không trộn số liệu đến từ folds, cohort hoặc split khác nhau.

Theo Model-assessment trang PDF 24, mean fold AUC/F1 và metric tính trên pooled OOF có thể khác nhau. Báo tên phép tổng hợp rõ ràng; không dùng chúng thay thế nhau khi chọn hoặc so sánh model.

### 9.2. Độ bất định

- Báo cáo mean và SD của CV, nhưng không gọi SD giữa fold là 95% CI.
- Test: bootstrap tối thiểu khoảng 1.000 lần nếu tài nguyên cho phép; seed cố định, báo rõ phương pháp percentile và mức 95%.
- Heart: resample theo bệnh nhân/row độc lập, hoặc group nếu có cấu trúc nhóm đã biết.
- Readmission: cluster bootstrap theo bệnh nhân, mỗi lần lấy toàn bộ encounter của bệnh nhân được chọn với đúng số lần lặp; không row-bootstrap như các lượt hoàn toàn độc lập.
- So sánh hai model bằng cùng bootstrap indices để lấy CI cho chênh lệch metric.
- Bootstrap metrics từ predictions cố định không bao gồm uncertainty của toàn bộ quá trình huấn luyện/model selection; nêu giới hạn này.
- Replicate chỉ có một lớp phải xử lý rõ; báo số replicate hợp lệ. Với cỡ mẫu quá nhỏ, hạn chế khẳng định.

### 9.3. Calibration

- Vẽ reliability diagram, báo Brier và log loss khi có probabilities phù hợp.
- So sánh model nguyên bản với sigmoid calibration; isotonic chỉ dùng khi phần calibration đủ dữ liệu.
- Tuân thủ split ở mục 6; đặc biệt không để API calibration tự tạo folds làm trộn bệnh nhân.
- Dùng các split index group-aware được dựng trước nếu API không tự chuyển `groups` đúng.
- ECE có thể báo bổ sung nhưng phải chỉ rõ bins và cỡ mẫu; không dùng nó như kết luận duy nhất.

### 9.4. Threshold và giả định chi phí

So sánh tối thiểu threshold 0.5 với threshold chọn bằng development predictions.

Quy tắc mặc định: tối đa hóa F2 trên development OOF. Có thể bổ sung ngưỡng đạt sensitivity mục tiêu 0.85 rồi tối đa specificity. Con số 0.85 chỉ là giả định minh họa học thuật, không phải chuẩn lâm sàng; không chỉnh nó sau khi xem test.

Nếu không có ngưỡng ổn định/đủ bằng chứng để đạt mục tiêu, báo thất bại của ràng buộc. Luôn báo số lượng hoặc tỷ lệ bệnh nhân bị gắn cờ.

Phân tích chi phí giả định: `Cost = C_FN × FN + C_FP × FP`, với các tỷ lệ 1:1, 2:1, 5:1. Đây là độ nhạy với giả định, không phải ước lượng chi phí bệnh viện thật. Chọn ngưỡng bằng development; test chỉ đánh giá policy đã cố định.

Bổ sung thực nghiệm gắn với Slide3 trang PDF 48: precision@k/recall@k ở các mức ngân sách định trước, ví dụ top 5%, 10%, 20% số lượt xuất viện. Dùng thứ tự ổn định để xử lý đồng hạng, báo k thực tế và số positive tìm được. Readmission đang có đơn vị encounter: gọi là số lượt được gắn cờ, không đánh đồng với số bệnh nhân duy nhất. Không chọn mức ngân sách tốt nhất bằng nhãn test.

## 10. Phân tích lỗi: phần bắt buộc, không làm qua loa

### 10.1. Đầu ra tối thiểu

Cho mỗi dataset, tạo:

1. Bảng prediction có row key/group key nội bộ, true label, probability, threshold, predicted label và nhóm TP/TN/FP/FN.
2. Confusion matrix ở ngưỡng 0.5 và ngưỡng đã chọn.
3. Bảng so sánh FN với TP, FP với TN theo các biến tiêu biểu và missingness.
4. Phân tích lỗi confidence cao: sai nhưng dự đoán sát 0 hoặc 1.
5. Bảng subgroup metrics có số ca và số ca dương đi kèm.
6. Khoảng 3–5 case studies cho mỗi bộ nếu đủ trường hợp, ưu tiên FN, FP và một ca đúng để đối chiếu.

### 10.2. Nhóm phân tích

- Heart: nhóm tuổi, giới tính theo mã gốc, kiểu đau ngực và mức missingness nếu số lượng cho phép.
- Readmission: nhóm tuổi, giới tính, race khi có, mức sử dụng dịch vụ trước đó, độ dài đợt điều trị và missingness.
- Chốt ngưỡng tối thiểu để diễn giải subgroup, ví dụ ít nhất 30 hàng và 10 dương/10 âm; nhóm nhỏ vẫn có thể mô tả count nhưng metric suy luận phải NA hoặc gắn nhãn rất bất định.
- Heart nhỏ có thể không đáp ứng ngưỡng; không cố chia thật nhiều nhóm để tìm chênh lệch đẹp.
- Báo sensitivity, specificity, precision, AUC/AP nếu đủ lớp, và calibration khi đủ dữ liệu.
- Chênh lệch giữa nhóm là tín hiệu mô tả, chưa chứng minh phân biệt đối xử hay quan hệ nhân quả.

### 10.3. Khung giải thích mỗi phát hiện

Với mỗi nhận xét quan trọng, trả lời:

- Quan sát cụ thể là gì? Trích số hoặc hình nào?
- Có bao nhiêu mẫu, uncertainty ra sao?
- Những nguyên nhân khả dĩ nào: thiếu dữ liệu, overlap lớp, label noise, imbalance, hạn chế feature, distribution shift, overfitting?
- Bằng chứng nào ủng hộ và bằng chứng nào còn thiếu?
- Đề xuất cải thiện gì, đã kiểm chứng trên development hay chỉ là hướng tương lai?

Không gọi một điểm sai là nhãn sai chỉ vì mô hình tự tin. Không tự suy ra chẩn đoán hay điều trị cho bệnh nhân. Case studies là minh họa hành vi mô hình trên dữ liệu nghiên cứu.

### 10.4. Explainability

- Logistic Regression: hệ số/odds ratio khi cách biến đổi cho phép diễn giải; nói rõ scale và regularization.
- Model mạnh nhất: permutation importance và SHAP nếu tương thích; chọn background từ train/development.
- Với one-hot, gom importance về biến gốc khi hợp lý; không chỉ đưa danh sách dummy khó hiểu.
- Báo correlated features có thể chia sẻ hoặc làm sai lệch importance.
- Diễn giải local cho vài ca FP/FN nhưng không gọi SHAP là nguyên nhân y khoa.
- Sampling phục vụ SHAP phải có seed, kích thước và thời gian chạy được ghi lại.

## 11. Phần mở rộng và cập nhật kiến thức mới

### 11.1. Phần nâng cao bắt buộc của thiết kế

Thực hiện calibration và threshold analysis như phần vận dụng kiến thức đã học, đối chiếu Slide3 trang PDF 44–54 và Model-assessment trang 33–37. Learning curves và bootstrap cũng đã nằm trong môn học. Explainability bằng SHAP, ensemble bổ sung và post-hoc calibration đi sâu hơn slide có thể được trình bày là mở rộng, nhưng chưa đủ để tự tuyên bố đã áp dụng “ML mới nhất”.

Để đáp ứng phần kiến thức mới, bổ sung **ít nhất một thí nghiệm nâng cao có đầu ra thực tế** trong các hướng dưới đây, ưu tiên TabPFN và dùng conformal khi không khả thi. Không chỉ viết một đoạn giới thiệu thuật toán rồi tính là đã thực hiện. Literature review phải nêu rõ phần mới về nghiên cứu, phần mới với người học và phần đã học trong slide.

### 11.2. Ưu tiên: Tabular Foundation Model

Một điểm khởi đầu đã xác minh là [TabPFN trên Nature, 2025](https://www.nature.com/articles/s41586-024-08328-6); tiếp theo có [báo cáo TabPFN-2.5](https://arxiv.org/abs/2511.08667) và [repository chính thức PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN). Đây là các mốc tham khảo, không mặc định là phiên bản mới nhất tại lúc chạy.

Nhiệm vụ:

1. Tìm công bố gần đây và release chính thức ở thời điểm thực hiện; lưu ngày kiểm tra, paper, release/tag, phiên bản package và checkpoint.
2. Phân biệt bài peer-reviewed, preprint và tài liệu sản phẩm. Không gọi một phương pháp là SOTA chỉ dựa trên quảng bá.
3. Kiểm tra license, yêu cầu tài khoản, điều khoản trọng số, CPU/GPU, giới hạn dữ liệu và việc có gửi dữ liệu qua API hay không.
4. Ưu tiên thực nghiệm local trên Heart vì quy mô nhỏ. Không yêu cầu phải chạy foundation model trên toàn bộ readmission.
5. So sánh trên cùng split và cùng feature contract với Logistic Regression và boosting; các cách encode riêng theo model phải fit đúng và được ghi lại.
6. Báo primary metric, calibration, thời gian, tài nguyên và hạn chế.
7. Thảo luận nguy cơ benchmark contamination/pretraining exposure khi dùng foundation model trên bộ benchmark công khai; không tuyên bố hoàn toàn độc lập nếu không xác minh được.
8. Không kết luận phải thắng baseline. Kết quả không cải thiện vẫn có giá trị nếu thực nghiệm đúng.

Nếu bị chặn bởi trọng số/tài nguyên/quyền truy cập, ghi log và chuyển sang mục 11.3; không để phần mới làm trễ toàn bộ yêu cầu cốt lõi.

### 11.3. Phương án thay thế: conformal prediction

- Tìm paper/tài liệu gốc về conformal classification, ghi giả định và định nghĩa coverage.
- Triển khai split conformal với nonconformity score phù hợp, tập calibration riêng chưa được fit model và chưa dùng để chọn hyperparameters.
- Báo empirical coverage, kích thước prediction set, tỷ lệ singleton/ambiguous/empty nếu có, ở các alpha định trước như 0.05 và 0.10.
- Với readmission có nhiều encounter cùng người, row-level exchangeability không được mặc định thỏa. Chỉ triển khai bảo đảm conformal chuẩn trên đơn vị độc lập phù hợp hoặc dùng phương pháp group-aware có nguồn; nếu không, chỉ làm trên Heart và nêu giới hạn.
- Không diễn giải coverage biên là bảo đảm cho từng bệnh nhân hoặc từng subgroup.
- Conformal là kiến thức mở rộng, không tự gọi bản thân nó là kỹ thuật mới nhất. Vẫn bổ sung literature review cập nhật về hướng ML gần đây.

### 11.4. Literature review

Tạo `reports/literature_review.md`, khoảng 6–10 nguồn chọn lọc, ưu tiên nguồn gốc: trang dữ liệu, paper, tài liệu thư viện, repository tác giả.

Mỗi mục ghi: năm/ngày, loại nguồn, vấn đề giải quyết, đóng góp, giới hạn, mức liên quan và có thực nghiệm trong dự án hay không. Có ít nhất một nguồn gần đây đã kiểm chứng ngày xuất bản không nằm sau ngày thực hiện. Không cần chạy theo một paper mới nhưng không phù hợp bài toán.

## 12. Ma trận thí nghiệm và ablation

Thiết kế tối thiểu:

| ID | So sánh | Dataset | Đầu ra cần xem |
|---|---|---|---|
| E01 | Dummy vs các model chính | Cả hai | Primary metric, confusion matrix |
| E02 | Không trọng số vs class weighting | Cả hai, ít nhất một model phù hợp | AP/recall/precision/calibration |
| E03 | Threshold 0.5 vs tuned | Cả hai | FN, FP, F2, tỷ lệ gắn cờ |
| E04 | Raw probability vs calibrated | Cả hai | Brier, log loss, reliability |
| E05 | Base features vs engineered features | Readmission | AP, thời gian, số features |
| E06 | Full vs restricted feature set | Heart | AUC, sensitivity, tính khả dụng |
| E07 | Phương pháp nâng cao vs baseline hoặc conformal | Ít nhất một bộ | Chất lượng, uncertainty, tài nguyên |
| E08 | Logistic Regression L1 vs L2, đường regularization theo C | Cả hai | CV metric/log loss, số hệ số khác 0, train–validation gap; Slide3 trang 25–28, 37 |
| E09 | KNN scaling vs không scaling numerical; k, distance, weights | Heart | CV AUC/AP, bias–variance, latency; Slide4 trang 12–24 |
| E10 | GaussianNB vs Gaussian classifier đầy đủ regularized và LR | Heart, cùng numerical features | Metric, log loss, covariance stability, giả định độc lập; Slide5 trang 37–40 |
| E11 | Logistic NumPy, loss curve và kiểm tra gradient | Heart development + ví dụ kiểm tra nhỏ | Hội tụ, learning rate, sai số so với thư viện; Slide2 trang 7–17, Slide3 trang 17–24 |
| E12 | Precision@k/recall@k theo ngân sách gắn cờ | Readmission | Số lượt bị gắn cờ và số ca positive tìm được; Slide3 trang 48 |

E09 không scaling là đối chứng có chủ đích để chứng minh tác động của đơn vị đo; encoder/imputer vẫn fit trong fold. Không vô hiệu hóa preprocessing toàn pipeline. Lập bảng validation curve cho k và C, learning curve theo cỡ train, rồi dùng chúng để giải thích underfitting/overfitting; các khái niệm này không thay thế lẫn nhau.

Giữ các yếu tố khác cố định khi so sánh. Không đổi split, model và features đồng thời rồi quy cải thiện cho một yếu tố.

Mỗi run lưu `run_id`, dataset/cohort hash, split hash, seed, config, versions, model params, trạng thái, timestamps, metrics và lỗi nếu có. Lưu predictions đủ để tính lại biểu đồ mà không cần train lại.

## 13. Báo cáo, thuyết trình và demo

### 13.1. Báo cáo chính

Viết `reports/final_report.md` đầy đủ và xuất PDF nếu có công cụ. Mục tiêu khoảng 18–25 trang nội dung chính, có thể điều chỉnh theo rubric; không kéo dài chỉ để đủ số trang.

Cấu trúc:

1. Tóm tắt: mục tiêu, dữ liệu, phương pháp, phát hiện thực tế và giới hạn.
2. Đặt vấn đề: ý nghĩa patient risk analysis, hai outcome và thời điểm dự đoán.
3. Mục tiêu/câu hỏi nghiên cứu và phạm vi.
4. Dữ liệu: nguồn, giấy phép, schema, cohort, missingness, imbalance.
5. Kiến thức nền: ánh xạ sáu file slide; T–E–P, tối ưu, Logistic Regression, KNN, Gaussian/Bayes, regularization, metric và leakage. Dùng công thức gắn với thí nghiệm, không sao chép toàn bộ slide.
6. Protocol: split, CV, tuning, calibration, threshold, bootstrap.
7. Kết quả từng dataset, gồm baseline và ablation.
8. Phân tích lỗi, nhóm bệnh nhân và giải thích model.
9. Kiến thức mở rộng và thí nghiệm nâng cao.
10. Thảo luận: đánh đổi, khác biệt giữa hai bài toán, kết quả không như kỳ vọng.
11. Hạn chế: cỡ mẫu, dữ liệu cũ, bias, quan sát, nhãn, external validity.
12. Kết luận trả lời từng RQ và hướng phát triển.
13. Tài liệu tham khảo và phụ lục tái lập.

Mọi con số lấy từ artifact đã chạy. Bảng/hình có số thứ tự, caption, dataset, split và threshold nếu liên quan. Không trung bình hóa AUC của hai bài toán khác nhau để tạo một “điểm tổng” thiếu ý nghĩa.

Phân biệt dự đoán, liên hệ thống kê và suy luận nhân quả. Đây là prototype học thuật, chưa được thẩm định để ra quyết định lâm sàng; không đưa lời khuyên điều trị vào báo cáo/demo.

### 13.2. Bảng và biểu đồ bắt buộc

- Bảng đặc điểm hai dataset và cohort flow.
- Bảng missingness và nhãn theo split.
- Bảng kết quả CV và bảng test có CI.
- ROC, PR, confusion matrix và reliability diagrams.
- Threshold trade-off: sensitivity, specificity, precision và tỷ lệ gắn cờ.
- Feature importance, subgroup errors, ablation.
- Learning curve cho ít nhất Logistic Regression trên mỗi dataset, cùng validation folds cố định và các cỡ train khả thi; giữ group separation ở mọi cỡ train. Đây là đầu ra vận dụng Model-assessment trang PDF 38–41. Giảm số mốc/cỡ mẫu nếu tài nguyên hạn chế và ghi cấu hình; không bỏ chỉ vì đã có learning-rate loss curve.

Đồ thị dùng font tiếng Việt hiển thị đúng, nhãn đủ rõ, màu dễ phân biệt; xuất PNG đủ độ phân giải, có thể thêm SVG/PDF. Mở kiểm tra kết quả render thay vì chỉ xác nhận file tồn tại.

### 13.3. Thuyết trình và bảo vệ

- Tạo outline và deck khoảng 12–15 slide cho 10–15 phút.
- Bắt buộc có slide ánh xạ kiến thức đã học và slide phân biệt đóng góp ngoài chương trình. Đặt trọng tâm vào Logistic Regression, KNN, probabilistic models và protocol đánh giá; không trình bày ensemble như nội dung đã xác nhận trong sáu PDF.
- Dành đủ slide cho kết quả và lỗi; không dành phần lớn thời gian kể tên thuật toán.
- Có speaker notes hoặc script ngắn cho từng slide.
- Tạo `reports/defense_qa.md` với ít nhất 15 câu hỏi/trả lời dựa trên code và kết quả thật: vì sao hai bộ này, leakage, group split, AP, calibration, threshold, FN/FP, bias, foundation model, tái lập và giới hạn.
- Không bịa tên sinh viên, mã lớp hoặc giảng viên; để chỗ điền rõ ràng nếu thiếu.

### 13.4. Demo cục bộ — ưu tiên sau phần khoa học

Tạo Streamlit nếu đủ tài nguyên:

- Hai tab tương ứng hai bài toán và hai schema riêng.
- Dùng đúng pipeline đã lưu, không train lại mỗi lần mở app.
- Có mẫu đầu vào minh họa hợp lệ; phân biệt mẫu giả lập dùng giao diện với dữ liệu đánh giá thật.
- Hiển thị xác suất ước lượng, threshold hiện hành và kết quả phân loại.
- Không gọi xác suất là chắc chắn; không tự gắn nhãn mức “nguy hiểm” như một kết luận y khoa.
- Kiểm tra kiểu, phạm vi, category và missing input.
- Có phần giải thích ngắn, nguồn dữ liệu và giới hạn prototype.
- Chạy local mặc định; public deployment không nằm trong điều kiện hoàn thành.

## 14. CLI, tái lập và kiểm thử

### 14.1. Giao diện chạy

Triển khai CLI tương đương các lệnh dưới đây; đây là **hợp đồng cần xây dựng**, không phải tuyên bố các lệnh đã tồn tại:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m healthcare_ml.cli download --all
.\.venv\Scripts\python.exe -m healthcare_ml.cli validate-data --all
.\.venv\Scripts\python.exe -m healthcare_ml.cli prepare --all
.\.venv\Scripts\python.exe -m healthcare_ml.cli train --config configs/heart.yaml
.\.venv\Scripts\python.exe -m healthcare_ml.cli train --config configs/readmission.yaml
.\.venv\Scripts\python.exe -m healthcare_ml.cli freeze --all
.\.venv\Scripts\python.exe -m healthcare_ml.cli evaluate --all --split test
.\.venv\Scripts\python.exe -m healthcare_ml.cli report --all
.\.venv\Scripts\python.exe -m pytest -q
```

`train` chỉ sử dụng development. `freeze` ghi model/threshold/calibration/config hash và đánh dấu sẵn sàng mở test. `evaluate` kiểm tra manifest; kết quả cũ không bị ghi đè âm thầm. `report` đọc artifact đã lưu, không chạy lại tuning.

Tạo lệnh một lần `run_all.ps1`/`run_all.sh` có resume/cache; full và smoke là hai chế độ tách biệt. Smoke dùng ngân sách nhỏ để kiểm tra kết nối các bước, không thay kết quả full trong báo cáo.

### 14.2. Tests có giá trị

- Schema, mapping target và xử lý token missing đúng.
- IDs/target không nằm trong feature matrix.
- Không overlap row/group giữa các tập và mọi fold.
- Imputer/scaler/encoder được fit bằng train; test có category mới vẫn predict được.
- Sampler không thay đổi validation/test.
- Calibration folds bảo toàn group.
- Threshold được lấy từ development artifact, không chọn từ test labels.
- Smoke end-to-end chạy được trên cả hai schema.
- Pipeline save/load cho predictions tương đương trong tolerance định trước.
- Probabilities hữu hạn và trong [0,1]; metric/figure/report truy nguyên về đúng run.
- Bootstrap cluster giữ các encounter đi cùng patient, không bỏ multiplicity khi lấy mẫu lặp.
- Logistic NumPy: finite-difference gradient check, ổn định số với score lớn, shape và chuẩn hóa loss đúng; đối chiếu thư viện trong điều kiện objective tương đương.
- Gaussian comparison dùng cùng features/folds; xử lý variance bằng 0/covariance gần suy biến và probabilities hợp lệ.
- KNN không có k vượt train-fold size, xử lý category mới và zero-distance hợp lệ.

Không cần unit test từng dòng plotting. Ưu tiên lỗi có thể làm sai kết luận nghiên cứu.

### 14.3. Môi trường và khả năng chạy lại

Lưu package versions, Python/OS, cấu hình phần cứng thực tế, seeds, thread settings, thời gian từng giai đoạn và command history cần thiết. Không hứa deterministic tuyệt đối nếu thư viện/GPU không bảo đảm; nêu dung sai.

README phải cho phép một người khác tạo môi trường sạch, tải dữ liệu, chạy smoke, chạy full và tìm báo cáo. Không phụ thuộc đường dẫn tuyệt đối trên máy tác giả. Chỉ load model artifact do chính dự án tạo, không load pickle tùy ý tải từ Internet.

## 15. Lộ trình thực thi và cổng kiểm soát

### Giai đoạn 0 — Khởi động

Đọc workspace và sáu PDF trong Slide, rubric bổ sung nếu có; khảo sát máy; tạo TASKS, cấu trúc, môi trường và course alignment dẫn trang chính xác. Không quay lại giả định rằng môn học chưa có tài liệu.

**Đạt khi:** Có kế hoạch thực thi, dependency dùng được và không có xung đột dữ liệu người dùng.

### Giai đoạn 1 — Dữ liệu và protocol

Tải hai bộ, xác minh nguồn, lập data cards, cohort, feature availability, split và protocol.

**Đạt khi:** Có ít nhất hai bộ thật, target rõ và tests chống overlap/leakage sơ bộ đạt.

### Giai đoạn 2 — EDA và baseline

EDA development, Dummy, Logistic Regression và GaussianNB; chạy smoke end-to-end. Chuẩn bị notebook tối ưu để nối kiến thức Slide2/Slide3 với bài toán chính.

**Đạt khi:** Pipeline tạo predictions/metrics thật trên cả hai dataset, chưa mở test để chọn model.

### Giai đoạn 3 — Model comparison

Chạy ma trận model ở mục 8.1, tuning development, thí nghiệm KNN/Gaussian/regularization, learning curves, lưu OOF và chọn ứng viên.

**Đạt khi:** Có bảng CV hợp lệ, run logs, lý do chọn và đánh giá chi phí.

### Giai đoạn 4 — Calibration, threshold, nâng cao

Thực hiện các ablation đã chốt và ít nhất một nhánh nâng cao khả thi.

**Đạt khi:** Có bằng chứng thực nghiệm, cấu hình cuối và manifest đóng băng.

### Giai đoạn 5 — Test cuối và phân tích lỗi

Evaluate recipe đã chốt; CI, subgroup, FN/FP, explanation và case studies.

**Đạt khi:** Kết quả test truy nguyên được; không dùng lại test để chọn model.

### Giai đoạn 6 — Báo cáo và bàn giao

Viết báo cáo, slide, Q&A, README; render và xem sản phẩm; tạo demo nếu khả thi; chạy acceptance checks.

**Đạt khi:** Người đọc có thể kiểm tra kết luận và người khác có thể chạy lại.

### Khi thời gian/tài nguyên hạn chế

Ưu tiên theo thứ tự: hai bộ thật → protocol đúng → ma trận model bám slide ở mục 8.1 → ablation kiến thức đã học và phân tích lỗi → calibration/threshold/explanation → một mở rộng → báo cáo, slide thuyết trình và tái lập → giao diện. Không bỏ KNN/GaussianNB trên Heart để chỉ chạy thêm nhiều ensemble.

Cắt số trial, phần trang trí và mô hình tùy chọn trước. Không cắt dataset thứ hai, không bỏ group split, không bịa kết quả để đạt đủ bảng. Nếu một yêu cầu chưa thực hiện được, giữ trạng thái chưa hoàn thành và nêu nguyên nhân cụ thể.

## 16. Checklist nghiệm thu

Chỉ đánh dấu `[x]` khi có artifact hoặc log kiểm chứng. Sao chép checklist này sang `reports/acceptance_checklist.md` và thêm đường dẫn bằng chứng.

### Dữ liệu và protocol

- [ ] Có ít nhất hai dataset độc lập, nguồn và license xác minh được.
- [ ] Có raw files, checksum, data cards và data dictionary.
- [ ] Nhãn, cohort, đơn vị quan sát và thời điểm dự đoán rõ ràng.
- [ ] Có bảng feature availability và kiểm tra target proxy.
- [ ] Heart dùng bản có nguồn gốc rõ, không tăng mẫu bằng duplicate.
- [ ] Readmission chia theo patient ở tất cả tầng đánh giá.
- [ ] Test đã được giữ độc lập khỏi các quyết định lựa chọn.
- [ ] Preprocessing/sampling fit đúng phạm vi train.

### Thực nghiệm

- [ ] Đã chạy đúng ma trận model mục 8.1: Heart có LR, KNN, GaussianNB, Gaussian classifier regularized và boosting; readmission có LR, GaussianNB, Random Forest và boosting; cả hai có Dummy.
- [ ] Có hyperparameters, CV results và metric chính xác định trước.
- [ ] Có calibration/threshold analysis không leakage.
- [ ] Có E01–E12 với phạm vi đã quy định; ghi cụ thể phần chưa đạt, không đánh dấu hoàn thành thay thế bằng một thí nghiệm khác.
- [ ] Có learning curves mỗi dataset, validation curves k/C và loss curve tối ưu; phân biệt đúng ba loại.
- [ ] Có ít nhất một thí nghiệm nâng cao thực sự và literature review cập nhật.
- [ ] Có test metrics, uncertainty và kết quả trung thực kể cả model kém baseline.
- [ ] Có runtime, seed, versions, split hashes và run manifests.

### Phân tích và báo cáo

- [ ] Hai bộ đều có FN/FP analysis, subgroup analysis và case studies phù hợp cỡ mẫu.
- [ ] Có explanation với giới hạn diễn giải.
- [ ] Mọi bảng/hình truy về artifact, không có số bịa hoặc placeholder kết quả.
- [ ] Báo cáo trả lời RQ1–RQ5, nêu hạn chế và không khẳng định quá mức.
- [ ] Course alignment dẫn đúng tên file/trang của sáu PDF và liên kết từng kiến thức với artifact thật.
- [ ] Calibration, threshold, group split và bootstrap được ghi là kiến thức đã có trong slide; phần mới được phân loại trung thực.
- [ ] Có báo cáo Markdown; PDF/deck đã render và kiểm tra hoặc nêu blocker thật.
- [ ] Có outline, Q&A bảo vệ và nguồn trích dẫn.

### Kỹ thuật và bàn giao

- [ ] Tests trọng yếu đạt; notebook chạy theo thứ tự từ kernel sạch.
- [ ] Pipeline save/load và smoke test cả hai bộ đạt.
- [ ] README/lệnh chạy có thể thực hiện từ môi trường sạch.
- [ ] Full results tách biệt smoke results.
- [ ] TASKS và limitations phản ánh đúng phần hoàn thành/chưa hoàn thành.
- [ ] Demo nếu có dùng đúng model artifact, đã kiểm tra nhập liệu.

## 17. Nguồn khởi đầu đã kiểm tra

**Nguồn môn học tại chỗ:** Sáu tài liệu ở mục 1.2 là căn cứ ưu tiên xác định nội dung đã học. Tham chiếu theo tên file và số trang PDF; không lấy số in chân slide thay số trang. Khi viết báo cáo, ghi rõ tài liệu giảng dạy do người dùng cung cấp, không bịa năm xuất bản hoặc URL công khai. Slide1 trang PDF 2 nêu hai sách Bishop (2006) và James cùng cộng sự (2014); chỉ trích dẫn nội dung sách nếu đã đọc phần liên quan, không suy ra đã đọc sách từ việc tên sách xuất hiện trên slide.

Các nguồn này là điểm xuất phát để Agent đọc và trích dẫn, không thay thế việc xác minh chi tiết triển khai:

1. [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) — nguồn dữ liệu A.
2. [UCI Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) — nguồn dữ liệu B và bảng mã.
3. [Scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) — tham khảo công cụ chia tập/CV; thiết kế patient-level trong tài liệu này là yêu cầu của dự án.
4. [Scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html) — tham khảo calibration và API cần xác minh theo phiên bản cài.
5. [Accurate predictions on small data with a tabular foundation model, Nature 2025](https://www.nature.com/articles/s41586-024-08328-6).
6. [TabPFN-2.5 technical report](https://arxiv.org/abs/2511.08667) — phân biệt báo cáo/preprint với bài journal.
7. [PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN) — triển khai, phiên bản, yêu cầu và license cần kiểm tra lúc chạy.

## 18. Chỉ thị khởi chạy dành cho Agent

> Đọc toàn bộ đặc tả này và kiểm tra workspace. Thực hiện lần lượt các giai đoạn đến khi hoàn tất các điều kiện nghiệm thu. Bạn được tự chủ thực hiện mọi tác vụ cần thiết trong phạm vi ủy quyền ở mục 2, bao gồm tìm và tải dữ liệu, tạo môi trường, viết/chạy/sửa code, chạy thí nghiệm, phân tích và tạo sản phẩm bàn giao. Không dừng ở kế hoạch, code chưa chạy hoặc notebook chưa có kết quả. Tự xử lý lỗi thông thường và dùng phương án thay thế có căn cứ. Bảo toàn tính độc lập của test, trung thực về kết quả và ghi rõ mọi phần chưa đạt. Khi bàn giao, cung cấp đường dẫn báo cáo, bảng kết quả chính, lệnh tái lập, kết quả kiểm thử và hạn chế còn lại.
