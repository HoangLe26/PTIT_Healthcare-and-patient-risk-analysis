# Nhật ký quyết định

- Đọc đặc tả cập nhật theo 6 PDF; giữ nguyên Slide.
- Giới hạn hai Agent đang chạy, gồm Agent chính và một Agent hỗ trợ tài liệu.
- Môi trường Python 3.10 riêng; scikit-learn 1.7.2 để API L1/L2 nhất quán. CPU hai luồng, không yêu cầu GPU.
- Chọn XGBoost làm boosting duy nhất. Không cài nhiều boosting packages.
- Protocol dùng fit/tune, calibration, threshold, test độc lập; thay nested cross-fitting bằng holdout rõ ràng để tránh leakage và kiểm toán dễ. Ghi giảm cỡ train và uncertainty đặc biệt ở Heart.
- Thực nghiệm full không subsample cohort chính; budget 12/6 cấu hình và 5/3 fold theo cỡ dữ liệu. Không xem test để tăng ngân sách.


- Parser IDS_mapping tách section bằng dòng dấu phẩy; thêm assertion mã expired/hospice 11/13/14/19/20/21 trước huấn luyện. Cohort cuối 99.340 lượt.
- Windows Unicode path: bật PYTHONUTF8=1 cho editable install; pin threadpoolctl3.6.0 vì3.7 snapshot path làm hỏng đường dẫn tiếng Việt. Smoke/test sau sửa đạt.
- ICD grouping tham khảo [Strack2014 Table2](https://pmc.ncbi.nlm.nih.gov/articles/PMC3996476/); mã V/E tách supplementary, decimal thuộc nhóm theo tiền tố. Kiểm tra cohort không có mã decimal 785/786/787/788 chịu ảnh hưởng của cập nhật prefix.
- RF readmission vượt ngân sách thời gian với max_features0.6 trên one-hot; dừng riêng subprocess và thu hẹp search trước khi xem test readmission. Xem protocol_readmission_resource_amendment.md. Giữ checkpoint model khác, không đổi recipe Heart.
- Thêm Clopper-Pearson CI để diễn giải sensitivity Heart vì bootstrap từ predictions có FN=0 tạo CI suy biến. Đây là phân tích uncertainty bổ sung, không đổi mô hình/threshold sau test.
- SHAP0.49 không đọc vector base_score của XGBoost3.2; dùng native exact TreeSHAP pred_contribs=True, kiểm tra tổng đóng góp+base khớp raw margin. Không sửa model. Background native là tree node cover học từ fit, khác background sample80 của LinearExplainer Heart.
- QA hậu kiểm sửa lựa chọn case confidence: dùng xác suất của nhãn được gắn tại operating threshold, không dùng max(p,1-p) cho FP khi threshold<0.5. Refresh case/SHAP từ predictions đã lưu, không đổi metric/model. Reliability diagram readmission zoom trục theo bin means để dễ đọc.
- Runtime artifact-tool không có trong môi trường (MODULE_NOT_FOUND, không có load_workspace_dependencies). Trong phạm vi quyền lựa chọn framework đã được giao, dùng ReportLab và python-pptx cục bộ; đã chạy operation markers của kỹ năng PDF/Presentations. PDF render bằng pypdfium2; PPTX render thực bằng PowerPoint COM read-only/ẩn, bảng và charts chính là native editable objects. Không tuyên bố đã dùng bundled artifact runtime.
