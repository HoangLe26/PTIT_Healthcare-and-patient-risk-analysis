# Giới hạn thực nghiệm

1. Hai bài toán có outcome khác nhau, không phải external validation của cùng model. Không ghép thành một patient-risk score chung.
2. Heart chỉ 303 hàng, test 61, calibration36 và threshold37; estimates/threshold có độ bất định cao. Không xác minh cùng bệnh nhân ở nhiều hàng vì không có patient identifier đáng tin; synthetic row IDs chỉ là audit keys.
3. Readmission có nhiều encounters cùng bệnh nhân; group split và cluster bootstrap đã được thiết kế. Không có ngày thực đáng tin/hospital ID để chứng minh temporal hoặc hospital-level generalization.
4. Nhãn NO không loại trừ tái nhập viện ngoài hệ thống; dữ liệu historical không bảo đảm áp dụng hiện nay. Cohort có selection bias, outcomes quan sát không chứng minh causal effects.
5. Model cuối fit trên fit/tune portion, calibration và threshold giữ riêng. Chưa triển khai nested cross-fitting; CV là model selection, không phải unbiased nested-CV estimate. Independent test mới là assessment.
6. Budget search có giới hạn theo phần cứng; không tuyên bố tối ưu toàn cục. GPU4GB chưa dùng cho pipeline chính; CPU hai luồng.
7. Gaussian assumptions chỉ áp dụng nhánh numerical; so sánh full-features cần nhận biết feature_set. Subgroup nhỏ được báo count/NA, không kết luận fairness chắc chắn.
8. Fixed-prediction bootstrap không bao gồm training/model selection uncertainty, CI percentile hạn chế với mẫu nhỏ. Permutation importance và SHAP không có ý nghĩa nhân quả.
9. Conformal coverage là marginal dưới exchangeability, không phải bảo đảm cho mỗi người hoặc từng nhóm. Không áp dụng bảo đảm iid trên repeated encounter rows.
10. TabPFN chỉ có thể được báo là khảo sát nếu không chạy; tình trạng access và quyết định fallback cần evidence trong literature review. Không có metric giả của foundation model.
11. Repository hỗ trợ PowerShell/đường dẫn Unicode được kiểm chứng; Linux/macOS chưa được chạy thực tế.
12. Prototype phục vụ môn học; chưa clinical validation, chưa được dùng cho điều trị hoặc triển khai công khai.

