# THUẬT TOÁN CATBOOST: KẾT QUẢ VÀ HƯỚNG DẪN THỰC THI

* **Người phụ trách:** Khải
* **Thư mục:** `algorithm/CatBoost/`
* **Vị trí Artifacts:** `algorithm/CatBoost/artifacts/`

---

## 1. Tổng quan phương pháp
CatBoost là thuật toán **Gradient Boosting trên cây quyết định đối xứng (Oblivious Trees)**, là đại diện cho kỹ thuật Học máy nâng cao ngoài chương trình slide, mang lại điểm cộng lớn cho bài tập lớn:
1. **Xử lý trực tiếp biến phân loại (Categorical):** Dùng kỹ thuật *Ordered Target Statistics* giúp hạn chế target leakage và tránh bùng nổ số chiều so với One-Hot Encoding thông thường.
2. **Chống Data Leakage tuyệt đối:** Phép điền trung vị biến số (`fit_preprocessor`) được học độc lập trong từng fold huấn luyện; các phép chia Diabetes đều được kiểm tra `assert` không trùng `patient_nbr`.

---

## 2. Kết quả thực nghiệm của Khải

### Dataset A: Cleveland Heart Disease (N=303, Test=61, 13 biến gồm 8 biến phân loại)
* **Cấu hình tối ưu (5-fold CV):** `depth=6`, `learning_rate=0.1`, `l2_leaf_reg=5` (`CatBoost_d6_lr0.1_l2_5`).
* **Ngưỡng $F_2$ tối ưu:** `0.0092`.
* **Thời gian chạy:** `91.21s`.
* **Kết quả Test:**
  * **ROC-AUC (Chính):** **`0.9329`**
  * **Recall / Độ nhạy:** **`100%`** (Bắt trúng 28/28 ca bệnh, **FN = 0 ca**).
  * **Brier Score:** `0.1103`.

### Dataset B: Diabetes Readmission (N=99.340, Test=20.031, 44 biến gồm 36 biến phân loại)
* **Cấu hình tối ưu (3-fold Group CV theo bệnh nhân):** `depth=4`, `learning_rate=0.1`, `l2_leaf_reg=3` (`CatBoost_d4_lr0.1_l2_3`).
* **Ngưỡng $F_2$ tối ưu:** `0.0767`.
* **Thời gian chạy:** `300.88s`.
* **Kết quả Test:**
  * **Average Precision (Chính):** **`0.2340`** (Tăng vượt trội so với baseline ngẫu nhiên ~11%).
  * **ROC-AUC:** **`0.6695`**.
  * **Recall:** **`79.60%`**.

---

## 3. Đánh giá Split Conformal Prediction & Nhãn phân vân

Thay vì ép mô hình phải đưa ra nhãn dứt khoát $0$ hoặc $1$ đối với những ca bệnh phức tạp, Conformal Prediction cho phép mô hình gán **"Nhãn phân vân"** (Ambiguous Prediction Set $\{0, 1\}$) để khuyến nghị bác sĩ thăm khám kỹ hơn:

| Bộ dữ liệu | Cấu hình | Mức tin cậy | Ngưỡng $q$ | Độ bao phủ thực tế | Đơn nhãn (0/1) | **Nhãn phân vân** ({0, 1}) | Tập rỗng ({}) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Heart Disease** | `CatBoost_d6_lr0.1_l2_5` | 90% | 0.9788 | **96.72%** | 44.26% | **55.74%** | 0.00% |
| **Heart Disease** | `CatBoost_d6_lr0.1_l2_5` | 95% | 0.9995 | **100.00%** | 1.64% | **98.36%** | 0.00% |
| **Readmission** | `CatBoost_d4_lr0.1_l2_3` | 90% | 0.3208 | **90.60%** | **99.28%** | **0.00%** | 0.72% |
| **Readmission** | `CatBoost_d4_lr0.1_l2_3` | 95% | 0.8991 | **95.02%** | **72.98%** | **27.02%** | 0.00% |

* File 4 dòng gửi Hoàng ghép bảng 12 dòng: `algorithm/CatBoost/artifacts/conformal_4rows_for_hoang.csv`.
* File chi tiết ca FN/FP sai nặng nhất: `algorithm/CatBoost/artifacts/{heart,readmission}/extreme_errors.csv`.
* Các cột nhãn Conformal (`conformal_set_90`, `conformal_decision_90`, `conformal_set_95`, `conformal_decision_95`) đã được lưu chi tiết cho từng bệnh nhân tại `test_predictions.csv`.

---

## 4. Lệnh chạy thực nghiệm
Từ thư mục gốc dự án:
```powershell
& "D:\Project\venv1\Scripts\python.exe" algorithm/CatBoost/run_catboost.py
```
