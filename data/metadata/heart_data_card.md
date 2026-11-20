# Data card: heart

Nguồn: [heart](https://archive.ics.uci.edu/dataset/45/heart+disease); DOI 10.24432/C52P4X; license CC BY 4.0.

Raw 303 hàng; cohort 303 hàng; 303 đơn vị group; positive 139 (45.87%).

Target: num > 0; bệnh hiện diện tại quá trình chẩn đoán, không tiên lượng tương lai.

Schema/mã categorical/missing: xem `heart_data_dictionary.csv`, `heart_feature_availability.csv` và raw documentation. Numerical Heart: age (years), trestbps (mmHg), chol (mg/dL), thalach (beats/min), oldpeak (ST depression). Numerical readmission: counts hoặc days theo tên biến. Missing ? chuyển NaN; None xét nghiệm giữ category. Không tự coi 0 là missing.

Các splits/counts: `heart_summary.json`. Group splitting: row là đơn vị độc lập do không có patient identifier được xác minh. Train-only transformations, test không dùng chọn mô hình.

Duplicate raw cả hàng: 0; không bỏ hàng chỉ vì giống features. Features không chứa ID/nhãn. Nhóm cohort loại có flow CSV. Dữ liệu lịch sử, selection bias, label uncertainty và giới hạn tổng quát hóa; không phải mô hình đã thẩm định lâm sàng.
