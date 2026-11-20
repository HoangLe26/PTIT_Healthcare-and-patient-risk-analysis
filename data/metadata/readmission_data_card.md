# Data card: readmission

Nguồn: [readmission](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008); DOI 10.24432/C5230J; license CC BY 4.0.

Raw 101,766 hàng; cohort 99,340 hàng; 69,987 đơn vị group; positive 11,314 (11.39%).

Target: readmitted == <30; nguy cơ tại lúc xuất viện; NO không đảm bảo không tái nhập viện ngoài hệ thống.

Schema/mã categorical/missing: xem `readmission_data_dictionary.csv`, `readmission_feature_availability.csv` và raw documentation. Numerical Heart: age (years), trestbps (mmHg), chol (mg/dL), thalach (beats/min), oldpeak (ST depression). Numerical readmission: counts hoặc days theo tên biến. Missing ? chuyển NaN; None xét nghiệm giữ category. Không tự coi 0 là missing.

Các splits/counts: `readmission_summary.json`. Group splitting: patient_nbr; mọi encounter cùng người thuộc một split. Train-only transformations, test không dùng chọn mô hình.

Duplicate raw cả hàng: 0; không bỏ hàng chỉ vì giống features. Features không chứa ID/nhãn. Nhóm cohort loại có flow CSV. Dữ liệu lịch sử, selection bias, label uncertainty và giới hạn tổng quát hóa; không phải mô hình đã thẩm định lâm sàng.
