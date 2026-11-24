# THUẬT TOÁN CATBOOST: KẾT QUẢ VÀ HƯỚNG DẪN THỰC THI

* **Người phụ trách:** Khải
* **Thư mục:** `algorithm/CatBoost/`
* **Vị trí Artifacts:** `algorithm/CatBoost/artifacts/`

---

## 1. Tổng quan phương pháp
CatBoost là thuật toán **Gradient Boosting trên cây quyết định đối xứng (Oblivious Trees)**, là đại diện cho kỹ thuật Học máy nâng cao ngoài chương trình slide, mang lại điểm cộng lớn cho bài tập lớn:
1. **Xử lý trực tiếp biến phân loại (Categorical):** Dùng kỹ thuật *Ordered Target Statistics* giúp hạn chế target leakage và tránh bùng nổ số chiều so với One-Hot Encoding thông thường.
2. **Kháng Overfitting cao:** Cấu trúc cây đối xứng giúp tính toán nhanh và tổng quát hóa tốt trên dữ liệu bảng y tế.

---

## 2. Kết quả thực nghiệm của Khải

### Dataset A: Cleveland Heart Disease (N=303, Test=61)
* **Cấu hình tối ưu (5-fold CV):** `depth=6`, `learning_rate=0.1`, `l2_leaf_reg=5` (Mean ROC-AUC: `0.9129`).
* **Ngưỡng $F_2$ tối ưu:** `0.0092`.
* **Kết quả Test:**
  * **ROC-AUC (Chính):** **`0.9329`**
  * **Recall / Độ nhạy:** **`100%`** (Bắt trúng 28/28 ca bệnh, **FN = 0 ca**).
  * **Brier Score:** `0.1103`.
* **Top 3 đặc trưng quan trọng:** `thal` (16.55%), `ca` (14.39%), `cp` (12.02%).

### Dataset B: Diabetes Readmission (N=99.340, Test=20.031)
* **Cấu hình tối ưu (3-fold Group CV theo bệnh nhân):** `depth=6`, `learning_rate=0.1`, `l2_leaf_reg=3` (Mean AP: `0.2200`).
* **Ngưỡng $F_2$ tối ưu:** `0.0774`.
* **Kết quả Test:**
  * **Average Precision (Chính):** **`0.2238`** (Vượt trội so với baseline ngẫu nhiên ~11%).
  * **Recall:** **`79.30%`** (Bắt trúng 1.839 ca tái nhập viện sớm).
  * **Brier Score:** `0.0979`.
* **Top 3 đặc trưng quan trọng:** `number_inpatient` (17.30%), `discharge_disposition_id` (12.81%), `diag_1` (10.14%).

---

## 3. Lệnh chạy thực nghiệm
Từ thư mục gốc dự án:
```powershell
& "D:\Project\venv1\Scripts\python.exe" algorithm/CatBoost/run_catboost.py
```
Toàn bộ bảng điểm CV, đường cong ngưỡng, bảng dự đoán test và feature importance sẽ được xuất vào `algorithm/CatBoost/artifacts/`.
