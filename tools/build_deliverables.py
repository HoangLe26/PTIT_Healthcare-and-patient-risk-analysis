"""Build Vietnamese PDF, Markdown and editable PowerPoint from frozen evidence.

This exporter never trains models or retunes test decisions. Local publishing
libraries provide a fallback when the bundled artifact runtime is unavailable.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import html
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("heart", "readmission")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value, digits=4):
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{n:.{digits}f}" if math.isfinite(n) else "NA"


def table(rows, columns):
    if not rows:
        return "Chưa có artifact cho phân tích này."
    def cell(value):
        return str(value).replace("|", "/").replace("\n", " ")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(cell(row.get(c, "NA")) for c in columns) + " |" for row in rows)
    return "\n".join(lines)


def load_evidence(root=ROOT):
    evidence = {}
    suffixes = ["cv_results", "test_metrics", "confidence_intervals", "paired_differences",
                "threshold_policies", "calibration_test", "subgroups", "ablation",
                "controlled_ablations_summary", "learning_curve", "case_studies",
                "error_features", "error_missingness", "permutation_importance",
                "local_shap_grouped", "topk"]
    for name in DATASETS:
        evidence[name] = {suffix: read_csv(root / f"artifacts/metrics/{name}_{suffix}.csv") for suffix in suffixes}
        evidence[name]["summary"] = read_json(root / f"data/metadata/{name}_summary.json")
        evidence[name]["manifest"] = read_json(root / f"artifacts/runs/{name}_training_manifest.json")
        evidence[name]["shap_status"] = read_json(root / f"artifacts/metrics/{name}_shap_status.json")
    evidence["optimization"] = read_json(root / "artifacts/metrics/optimization.json")
    evidence["conformal"] = read_json(root / "artifacts/metrics/heart_conformal.json")
    evidence["sensitivity_exact_ci"] = read_json(root / "artifacts/metrics/heart_sensitivity_exact_ci.json")
    return evidence


def selected_test(evidence, dataset):
    selected = evidence[dataset]["manifest"].get("selected")
    return next((row for row in evidence[dataset]["test_metrics"] if row["model"] == selected), None)


def result_paragraph(evidence, dataset):
    row = selected_test(evidence, dataset)
    if row is None:
        return "Chưa có kết quả test của recipe đã chọn trên development."
    metric = "roc_auc" if dataset == "heart" else "ap"
    counts = {key: number(row.get(key), 0) for key in ("tp", "tn", "fp", "fn")}
    total = sum(float(row[key]) for key in counts)
    alerts = (float(row["tp"]) + float(row["fp"])) / total
    text = (f"Mô hình {row['model']} được chọn bằng CV development, không chọn bằng test. "
            f"Test {metric}={number(row.get(metric))}, ROC-AUC={number(row.get('roc_auc'))}, "
            f"AP={number(row.get('ap'))}. Tại threshold {number(row.get('threshold'), 6)}, "
            f"TP={counts['tp']}, TN={counts['tn']}, FP={counts['fp']}, FN={counts['fn']}; "
            f"recall={number(row.get('recall'))}, precision={number(row.get('precision'))}, "
            f"specificity={number(row.get('specificity'))}, tỷ lệ gắn cờ={alerts:.2%}.")
    if dataset == "heart":
        text += " Test nhỏ nên recall quan sát bằng 1 không chứng minh sensitivity quần thể bằng 100%. Không dựng ca FN khi không có FN."
    else:
        text += " Đánh giá encounter-level trên bệnh nhân chưa xuất hiện ở phần fit. Ranking còn khiêm tốn và tải false alarm cao; chưa chứng minh hiệu ích lâm sàng."
    return text


SCIENTIFIC_NOTES = """
Mô hình chỉ fit trên fit/tune set, không refit toàn bộ development. Calibration
và threshold dùng holdout riêng; readmission tách patient_nbr ở tất cả các phần.
CV mean/SD là điểm lựa chọn, không phải nested-CV unbiased estimate hay test CI.
train_seconds là search + CV + refit elapsed, không phải thời gian một lần fit.
Heart patient_nbr là khóa audit tổng hợp từ hàng, không xác nhận số bệnh nhân duy
nhất. Cleveland giả định quan sát độc lập; không kiểm chứng encounter lặp.
Readmission là nhãn <30 tại xuất viện trên dữ liệu Mỹ năm 1999–2008, không có
temporal hoặc external hospital validation. Heart phân loại bệnh hiện diện sau
khi thông tin xét nghiệm đã có, không phải nguy cơ tương lai hoặc trước xét nghiệm.
Sigmoid được chọn trên threshold holdout trước test. Nếu test calibration kém
hơn vẫn giữ recipe đóng băng. So raw/calibrated ở cùng numeric threshold chỉ
là diagnostic, không phải hai operating points được retune công bằng.
Permutation importance dùng ranking của raw pipeline. LR SHAP giải thích raw
log-odds trước calibration, background từ fit. Native XGBoost TreeSHAP dùng
training node cover, không gọi là background interventional 80 hàng.
Case confidence là xác suất của nhãn vận hành đã gán: pred1 => p, pred0 => 1-p.
Đây là probability của assigned label, không phải chắc chắn chẩn đoán.
Conformal coverage biên cần exchangeability, không đảm bảo từng bệnh nhân.
Coverage cao với nhiều tập {0,1} thể hiện đánh đổi độ bao phủ và độ thông tin.
TabPFN chưa chạy: skipped feasibility review theo phạm vi/ngân sách, không có
failed/blocked run. Model card v2 công khai; repo bản mới mô tả auth/license flow.
RF readmission giảm search budget trước test theo resource amendment, giữ cohort
và folds. Báo ngân sách không bằng nhau, không quy chênh lệch cho thuật toán riêng.
""".strip()


def frame(evidence,name,key):
    return pd.DataFrame(evidence[name].get(key,[]))

def compact(df,columns,digits=3):
    out=df[columns].copy()
    for c in columns:
        if c not in ['model','feature_set','metric','probability','policy','group','feature','experiment','variant','error_type']:
            out[c]=out[c].map(lambda v:number(v,digits))
    return out

def make_sections(e):
    h=selected_test(e,'heart');r=selected_test(e,'readmission')
    hs=e['heart']['summary'];rs=e['readmission']['summary']
    sections=[]
    def section(title,paragraphs,tables=None,figures=None):
        sections.append({'title':title,'paragraphs':paragraphs,'tables':tables or [],'figures':figures or []})
    section('Tóm tắt và câu hỏi nghiên cứu',[
        'Đề tài nghiên cứu hai nhiệm vụ phân loại trên dữ liệu y tế dạng bảng: nhận diện bệnh tim đã hiện diện trong Cleveland và xếp hạng khả năng tái nhập viện sớm theo mã <30 ở các lượt xuất viện có đái tháo đường. Hai tập có nhãn, thời kỳ và quần thể khác nhau; chúng tạo hai nghiên cứu tình huống, không phải external validation của một mô hình chung.',
        result_paragraph(e,'heart'),result_paragraph(e,'readmission'),
        'Đóng góp chính là quy trình kiểm toán: chia dữ liệu theo đơn vị quan sát, học toàn bộ preprocessing trong fold, tách calibration và threshold khỏi fit, khóa recipe trước test, lưu predictions và dùng bootstrap phù hợp cấu trúc bệnh nhân. Kết quả bao gồm cả phương pháp không cải thiện: feature engineering readmission giảm AP development; calibration Heart làm Brier test xấu hơn.',
        'RQ1 so sánh mô hình theo ROC-AUC/AP; RQ2 phân tích imbalance và operating threshold; RQ3 kiểm tra xác suất; RQ4 tìm FP/FN và nhóm dễ sai; RQ5 đánh giá giá trị mở rộng so với chi phí và giả định. Không đặt một mức accuracy tùy ý làm điều kiện thành công.',
        'T–E–P theo Slide1: T là từng bài toán phân loại; E là dữ liệu fit/development dùng học và lựa chọn; P là metric chính đánh giá độc lập. Prototype chỉ phục vụ nghiên cứu môn học, chưa được thẩm định để dùng lâm sàng.'
    ],[('Hai nghiên cứu tình huống',pd.DataFrame([
        {'Dataset':'Heart Cleveland','Raw':hs['raw_rows'],'Cohort':hs['rows'],'Test':hs['splits']['test']['rows'],'Primary':'ROC-AUC'},
        {'Dataset':'Readmission','Raw':rs['raw_rows'],'Cohort':rs['rows'],'Test':rs['splits']['test']['rows'],'Primary':'AP'}]))])
    section('Dữ liệu Heart và ý nghĩa dự đoán',[
        'Nguồn UCI ID45, DOI10.24432/C52P4X, CC BY4.0. Dùng riêng processed Cleveland với 303 hàng gốc và 13 biến; target y=1 nếu num>0, y=0 nếu num=0. Không dùng bản sao đã nhân hàng và không chuyển nhãn 0–4 thành một đại lượng regression.',
        'Numerical gồm age, trestbps, chol, thalach, oldpeak. Các mã cp, sex, fbs, restecg, exang, slope, ca, thal được coi là categorical theo data dictionary. Missing ? được đưa về NaN; imputer median/most frequent fit riêng trong train-fold. Số0 không bị coi là missing.',
        'Các đầu vào như ca và thal liên quan quy trình xét nghiệm/chẩn đoán. Thực nghiệm đầy đủ giả định thông tin này đã sẵn có, không chứng minh khả năng sàng lọc trước xét nghiệm. Ablation restricted bỏ ca/thal để khảo sát mức phụ thuộc vào xét nghiệm chuyên sâu; vẫn không xác lập một quy trình triển khai lâm sàng.',
        'ID audit được tạo theo hàng, không phải một định danh bệnh nhân đã xác thực. Mỗi hàng được giả định độc lập; không thể kiểm chứng người xuất hiện lặp bằng metadata hiện có. Test chỉ61 quan sát, vì vậy cần CI và tránh khẳng định ưu thế tuyệt đối từ chênh lệch nhỏ.',
        'Phân phối development cho thấy các lớp còn chồng lấn về tuổi, cholesterol và huyết áp. Các liên hệ này mô tả dữ liệu, không chứng minh tác động nhân quả. Thống kê outlier không được dùng để tùy tiện loại các ca khó.'
    ],figures=['heart_distributions','heart_missing'])
    flow=pd.DataFrame(rs['flow'])
    section('Dữ liệu tái nhập viện và xây dựng cohort',[
        'Nguồn UCI ID296, DOI10.24432/C5230J, CC BY4.0. Raw gồm101.766 lượt điều trị, không phải101.766 bệnh nhân độc lập. Target dương chỉ khi readmitted bằng <30; >30 và NO là âm. NO là nhãn quan sát trong hệ thống, không chứng minh không tái nhập viện ở ngoài hệ thống.',
        'Thời điểm dự đoán là xuất viện. Dùng thông tin của encounter hiện tại có sẵn đến thời điểm này; không tổng hợp các encounter tương lai. encounter_id và patient_nbr không nằm trong feature matrix. patient_nbr chỉ dùng cho group split, kiểm toán và cluster bootstrap.',
        'Đọc đúng section discharge_disposition_id trong IDS_mapping.csv; loại expired/hospice mã11,13,14,19,20,21 và3 hàng gender Unknown/Invalid. Giữ các trường hợp transfer; không bỏ nhóm phức tạp chỉ để tăng điểm. Cohort cuối có99.340 lượt và69.987 patient groups.',
        'Giữ None ở biến xét nghiệm như trạng thái không ghi nhận xét nghiệm, tránh parser tự biến thành missing. Loại weight, payer_code, medical_specialty theo protocol vì thiếu thông tin/cardinality và mục tiêu nguồn lực. Numerical là số ngày hoặc counts; mã nhập viện, tuổi theo khoảng và diagnosis xử lý categorical.',
        'Feature engineering gộp ICD theo nhóm trong nghiên cứu Strack2014, riêng mã V/E và missing; thêm tổng prior visits. Cách gộp có thể làm mất chi tiết chẩn đoán, nên phải có đối chứng cùng hyperparameters và folds.'
    ],[('Dòng chảy cohort',flow)],figures=['readmission_distributions'])
    splits=[]
    for name,label in [('heart','Heart'),('readmission','Readmission')]:
        for k in ['fit','calibration','threshold','test']:
            s=e[name]['summary']['splits'][k]
            splits.append({'Dataset':label,'Role':k,'Rows':s['rows'],'Groups':s['patients'],'Positive':s['positive'],'Prevalence':number(s['prevalence'],3)})
    section('Protocol, cấu trúc dữ liệu và chống leakage',[
        'Protocol được ghi trước tuning/test. Heart giữ test20% stratified seed42; phần development được chia fit/tune169, calibration36, threshold37. Readmission giữ fold0 của StratifiedGroupKFold5 làm test; mọi lần chia nội bộ và CV tiếp tục tách patient_nbr. ID không được đưa vào model.',
        'Model selection chỉ dùng fit/tune: Heart5 folds và readmission3 group folds, random search seed42. Imputer, encoder và scaler học lại trên train của mỗi fold. Các category mới ở validation/test được xử lý handle_unknown=ignore. Không resample validation/test; dự án không cần SMOTE để đáp ứng imbalance analysis.',
        'Recipe cuối giữ model fit trên fit/tune, calibrator fit calibration và chọn threshold trên holdout riêng. Không refit toàn bộ development sau calibration, vì khi đó xác suất/threshold không còn độc lập với bộ huấn luyện đã đổi. Đổi lại, cỡ train nhỏ hơn và calibration Heart bất định do chỉ36 quan sát.',
        'Sigmoid được giữ nếu Brier trên threshold holdout tốt hơn raw. Threshold tối đa F2; nếu hòa chọn ngưỡng lớn hơn. Freeze lưu hash models, split, processed data và mã biến đổi trước test. CV mean/SD là điểm lựa chọn, không phải estimate nested-CV không thiên lệch.',
        'RF readmission ban đầu vượt ngân sách thời gian với max_features0.6 trên one-hot. Trước khi mở test readmission, thu hẹp sang sqrt,80 cây, depth8/12, leaf15/30,4 cấu hình. Cohort/folds/nhãn giữ nguyên; báo resource amendment và log run bị ngắt. Không tuyên bố tuning công bằng tuyệt đối theo CPU.'
    ],[('Các vai trò dữ liệu độc lập',pd.DataFrame(splits))])
    section('Logistic Regression, tối ưu và regularization',[
        'Logistic Regression dùng score z=wᵀx+b và p=sigmoid(z). Log-odds tuyến tính theo features; xác suất liên tục và nhãn dự đoán là hai đối tượng khác nhau. Changing threshold không đổi p; class weighting hoặc regularization làm đổi model và có thể đổi p.',
        'Bernoulli negative log-likelihood trung bình: J=mean(logaddexp(0,z)-y*z). Gradient w=Xᵀ(p-y)/n; gradient intercept=mean(p-y). Với L2 thêm lambda*w vào gradient w, không phạt intercept trong bản NumPy. Tính cả hai gradient trước khi cập nhật.',
        f'Bản NumPy dùng development numerical Heart, kiểm tra finite differences: sai số gradient tối đa {e["optimization"].get("gradient_max_error",float("nan")):.3g}. Khi cùng objective không regularization và hội tụ, xác suất lệch tối đa {e["optimization"].get("prediction_max_difference_sklearn",float("nan")):.3g} so với LBFGS của thư viện. Đây là kiểm tra thuật toán, không phải điểm test mới.',
        'L2 co hệ số; L1 có thể làm nhiều hệ số bằng0. C nhỏ tăng regularization trong API được cài; không mặc định lambda=1/C cho mọi quy ước chuẩn hóa objective. Quan hệ MAP/Gaussian prior chỉ đúng khi đặt likelihood/prior và scaling tương ứng; point estimate không phải posterior đầy đủ.',
        'Scaling dùng train statistics. Learning-rate loss curve theo iteration chẩn đoán tối ưu; validation curve theo C đánh giá regularization; learning curve theo số quan sát đánh giá tác dụng thêm dữ liệu. Ba loại curve trả lời các câu hỏi khác nhau.'
    ],figures=['optimization_loss'])
    gauss=frame(e,'heart','controlled_ablations_summary');gauss=gauss[gauss.experiment=='E10']
    section('KNN và mô hình xác suất đã học',[
        'KNN phụ thuộc biểu diễn và khoảng cách: Manhattan dùng tổng sai khác tuyệt đối; Euclidean dùng căn tổng bình phương. Biến có đơn vị lớn có thể chi phối distance nếu không scaling. Numerical được chuẩn hóa trong fold, categorical one-hot; hình học ghép này vẫn là lựa chọn mô hình cần giải thích.',
        'Thử k3/5/7/11/15/21, p1/2 và uniform/distance weighting trên cùng folds. k nhỏ nhạy với noise; k lớn làm trơn biên. Không chỉ nhìn training accuracy của k=1. Chiều cao và brute-force prediction cost khiến KNN readmission full là tùy chọn, không ép chạy để tăng số tên thuật toán.',
        'Gaussian classifier dùng class priors và class-conditional Gaussian. Score gồm distance, covariance determinant và prior. GaussianNB giả định độc lập có điều kiện nên covariance chéo; QDA dùng covariance đầy đủ có regularization. Không coi các mã bệnh/categorical là Gaussian liên tục.',
        'E10 so GaussianNB, QDA và LR trên cùng numerical features/folds. So với full-feature LR phải ghi feature_set để phân biệt hiệu quả thuật toán và lượng thông tin. Tương quan development gợi ý giả định độc lập có thể không hoàn hảo, nhưng không tự chứng minh NB sẽ kém ở mọi split.',
        'GMM/EM xuất hiện trong Slide5 nhưng không cần triển khai một bài clustering riêng để đủ chương. Việc kết hợp supervised, xác suất, tối ưu và đánh giá đúng phù hợp hai outcome đã chọn.'
    ],[('E10: cùng numerical features',compact(gauss,['variant','roc_auc','ap','brier']))],['heart_knn_controlled_curve'])
    for name,label in [('heart','Heart'),('readmission','Readmission')]:
        cv=frame(e,name,'cv_results')
        section(f'Lựa chọn mô hình trên development: {label}',[
            'Các họ model đã được định trước, dùng cùng CV splits trong dataset. Dummy prior/most frequent là mốc tham chiếu; các mô hình Gaussian dùng nhánh numerical. Các ablation weighted/restricted/engineered được ghi rõ, không trộn thành những thuật toán độc lập giả.',
            f'Metric chính là {"ROC-AUC" if name=="heart" else "Average Precision"}. Selected bằng mean CV, quy tắc gần hòa trong0.002 ưu tiên đơn giản đã chốt. Winner development: {e[name]["manifest"].get("selected")}. Không chọn lại winner bằng test.',
            'Mean fold và pooled OOF metric có thể khác. Bảng này chỉ báo mean/SD giữa folds; SD không phải CI95%. Tuning/feature comparison có selection optimism; assessment cuối dùng test độc lập. train_seconds bao gồm search, CV và refit, không phải latency một fit duy nhất.',
            'Search budget hữu hạn: Heart tối đa12 cấu hình, readmission6, riêng RF4 sau điều chỉnh tài nguyên. Không có tuyên bố tối ưu toàn cục hoặc SOTA. Params đầy đủ và từng trial được lưu ở artifacts/runs để chạy lại, kèm warnings và dữ liệu/split hash.'
        ],[('CV và nguồn lực',compact(cv,['model','feature_set','mean_score','std_score','train_seconds']))], [f'{name}_learning_curve'])
    for name,label in [('heart','Heart'),('readmission','Readmission')]:
        tm=frame(e,name,'test_metrics');ci=frame(e,name,'confidence_intervals');winner=e[name]['manifest']['selected']
        primary='roc_auc' if name=='heart' else 'ap';selected_ci=ci[(ci.model==winner)&ci.metric.isin([primary,'roc_auc','brier'])]
        paragraphs=[result_paragraph(e,name),
            'Bảng dùng threshold của từng recipe được chọn trên development; không phải accuracy tại mặc định0.5. Dummy tối đa F2 gắn cờ tất cả, nên accuracy của nó ở operating point này khác always-negative0.5. Ranking AP/AUC vẫn là baseline không phụ thuộc threshold.',
            'Bootstrap1000 fixed-prediction resamples giữ label/probability đi cùng. Heart dùng rows giả định độc lập; readmission lấy bệnh nhân có hoàn lại và giữ toàn bộ encounters với multiplicity. CI không bao gồm biến động do training/model selection hoặc distribution shift.',
            ('AUC cao ở test nhỏ hơn mức CV không chứng minh mô hình luôn tốt hơn. Kiểm tra IDs/target proxies và train-only pipeline đã đạt; các xét nghiệm chuyên sâu vẫn là hạn chế deployment. Sensitivity test1 từ28 positives có binomial exact CI95% khoảng0.877–1.000 dưới giả định độc lập. Bootstrap recall suy biến1–1 không chứng minh population sensitivity100%.' if name=='heart' else
             'AP0.229 so prevalence0.116 cho thấy ranking có thông tin, nhưng chất lượng absolute còn khiêm tốn. Không tăng điểm bằng chia ngẫu nhiên encounter lặp. Chênh lệch XGB–LR AP khoảng0.012 với CI paired dương; chênh lệch F2 có CI chứa0, nên tăng ranking không bảo đảm cải thiện vận hành cùng mức chắc chắn.')]
        if name=='heart':
            section(f'Đánh giá độc lập: {label}',[paragraphs[0],paragraphs[1],paragraphs[3]],
                [('Test của các recipe',compact(tm,['model','feature_set','roc_auc','ap','recall','precision']))])
            section('Heart: ranking và độ bất định',[
                paragraphs[2],
                'ROC mô tả true-positive rate theo false-positive rate; PR thể hiện precision theo recall. Các đường cong dùng probabilities cố định, không chọn ngưỡng mới. AP và AUC đánh giá ranking; confusion matrix của ngưỡng đã chốt trả lời câu hỏi khác.',
                'CV AUC của LR khoảng0.919 nhưng test0.962 có thể phản ánh sampling fluctuation của split nhỏ. Test có28 positives và33 negatives. Mỗi cặp dương/âm ảnh hưởng AUC, vì vậy cần trình bày khoảng tin cậy thay vì chỉ chọn ba chữ số đẹp.',
                'Recipe calibrated LR và LR balanced có ranking test bằng nhau trên split này. Paired bootstrap của hai recipe không cho bằng chứng một phiên bản tốt hơn về AUC. Điều đó không làm mất giá trị model comparison, vì các objective và probability quality vẫn khác.',
                'Trong bootstrap fixed predictions, FN=0 làm recall CI suy biến. Binomial exact CI95% sensitivity có lower bound khoảng0.877 dưới giả định quan sát dương độc lập, cho thấy chưa thể suy sensitivity quần thể100%. Chưa đưa training/model selection uncertainty vào các interval.'
            ],[('CI95% của recipe chọn bằng development',compact(selected_ci,['metric','point','lower','upper']))],['heart_roc_pr'])
        else:
            section(f'Đánh giá độc lập: {label}',paragraphs,
                [('Test của các recipe',compact(tm,['model','feature_set','roc_auc','ap','recall','precision'])),
                 ('CI95% của recipe chọn bằng development',compact(selected_ci,['metric','point','lower','upper']))], [f'{name}_roc_pr'])
    calibration=[]
    for name in DATASETS:
        for row in e[name]['calibration_test']:
            calibration.append({'Dataset':name,'Probability':row['probability'],'Brier':number(row['brier']),'Log_loss':number(row['log_loss'])})
    section('Calibration: chất lượng xác suất có thể suy giảm',[
        'Discrimination đánh giá xếp hạng dương/âm; calibration kiểm tra xác suất so với tỷ lệ quan sát. ROC-AUC/AP tốt không tự chứng minh p có nghĩa như tần suất. Brier đo tổng sai số xác suất, không tách riêng calibration; dùng cùng log loss và reliability diagram.',
        'Sigmoid fit bằng calibration holdout; raw/calibrated được so Brier trên threshold holdout trước test. Heart chọn sigmoid vì cải thiện rất nhỏ khoảng0.00084 trên37 quan sát. Test Brier raw0.0859 nhưng sigmoid0.1058; log loss cũng xấu hơn. Đây là kết quả bất lợi cần giữ và giải thích, không đổi recipe sau khi xem test.',
        'Một giả thuyết hợp lý là calibration set36 quan sát không đủ ổn định, có sampling fluctuation hoặc khác mức phân bố xác suất. Chưa có bằng chứng để khẳng định duy nhất overfitting là nguyên nhân. Hướng tương lai là cross-fitting hoặc calibration set lớn hơn, cần một đánh giá mới độc lập.',
        'Readmission có calibration10.083 encounters tách theo patient, thuận lợi hơn về cỡ mẫu. Reliability plot dùng quantile bins; axes readmission zoom theo bin means để đọc rõ vùng xác suất thấp. Raw và calibrated ở cùng numeric threshold chỉ là chẩn đoán, không đại diện hai policy được tối ưu lại công bằng.'
    ],[('Chất lượng xác suất trên test',pd.DataFrame(calibration))],['heart_calibration','readmission_calibration'])
    operating=[]
    for name in DATASETS:
        selected=e[name]['manifest']['selected']
        for row in e[name]['threshold_policies']:
            if row['model']==selected and row['policy'] in ['0.5','f2']:
                operating.append({'Dataset':name,'Policy':row['policy'],'Threshold':number(row['threshold']),
                    'Recall':number(row['recall']),'Precision':number(row['precision']),'FP':row['fp'],'FN':row['fn'],'Alert':number(row['alert_rate'])})
    section('Threshold, bỏ sót và tải cảnh báo',[
        'Threshold0.5 là mặc định, không phải tối ưu phổ quát. Quy tắc F2 ưu tiên recall, được chọn trên threshold holdout độc lập. Khi giảm threshold, alert count/recall thường tăng; precision không bảo đảm giảm đơn điệu. Probability và label quyết định cần được trình bày riêng.',
        'Readmission tại0.5 chỉ TP32, FN2287 và FP25: bỏ sót phần lớn ca tái nhập viện sớm. Ngưỡng F2 khoảng0.0905 đưa TP lên1788 nhưng FP lên9729, hơn57% encounters được gắn cờ. Vì vậy sensitivity77% không đủ để đánh giá policy hữu ích khi tải xử lý bị giới hạn.',
        'Heart tuned threshold phát hiện28/28 positives nhưng có14 FP, specificity57.6%. Đây là đánh đổi được chốt trên development, không phải chuẩn lâm sàng. Ràng buộc recall>=0.85 chỉ là scenario học thuật; recall validation không bảo đảm đạt cùng mức trên test.',
        'Các cost ratios C_FN:C_FP=1:1,2:1,5:1 được chọn threshold trên development và đánh giá cố định trên test. Đây là phân tích giả định, không phải số tiền bệnh viện hoặc utility lâm sàng thật. Không dùng test để chọn tỷ lệ thuận lợi nhất.'
    ],[('Operating points đã chốt',pd.DataFrame(operating))],['readmission_confusion'])
    topk=frame(e,'readmission','topk')
    section('Ngân sách gắn cờ và giá trị của ranking',[
        'Theo Slide3 trang48, một danh sách có ngân sách cố định cần precision@k và recall@k. K được xác định trước bằng5/10/20% encounters test, sort ổn định để xử lý ties. Không chọn mức k bằng nhãn test.',
        'Top5% tương ứng1002 encounters, tìm321 positives: precision32.0% nhưng recall13.8%. Top10% là2004 lượt, tìm546 positives: precision27.2%, recall23.5%. Top20% là4007 lượt, tìm899: precision22.4%, recall38.8%. Mở rộng ngân sách tìm thêm positives nhưng tăng số ca âm được xem xét.',
        'Đơn vị là encounter, không phải số bệnh nhân duy nhất; một người có thể có nhiều lượt trong test. Để xây policy bệnh nhân cần định nghĩa thời gian, hợp nhất alerts và có metadata theo dõi phù hợp. Dữ liệu hiện tại chưa đáp ứng temporal deployment claim.',
        'AP tổng hợp chất lượng positive ranking trên nhiều ngưỡng; không đồng nhất AP với PR-AUC hình thang. Top-k bổ sung ý nghĩa vận hành vì một bệnh viện có thể không xử lý được57% encounters dù threshold F2 đạt recall cao.',
        'Phát hiện này trả lời RQ2: metric tối ưu và cơ chế quyết định phải phù hợp mục tiêu. Một ngưỡng học thuật tốt không thay thế xác định nguồn lực, mức chấp nhận FN/FP và kiểm chứng triển khai.'
    ],[('Ngân sách định trước trên test',compact(topk,list(topk.columns)))],['readmission_threshold_tradeoff'])
    for name,label in [('heart','Heart'),('readmission','Readmission')]:
        cases=frame(e,name,'case_studies');casecols=['row_id','y','probability','prediction','error_type','confidence']
        error=frame(e,name,'error_features')
        section(f'Phân tích lỗi và case studies: {label}',[
            ('Recipe Heart không có FN tại threshold tuned. Báo cáo không tạo ca FN giả; chọn2 FP,1 TP và1 TN để đối chiếu. Không có FN trên61 test rows chưa chứng minh không bỏ sót ở dữ liệu mới.' if name=='heart' else
             'Recipe readmission có531 FN và9729 FP. Chọn2 FN,2 FP,1 TP và1 TN từ predictions cố định để phân tích, ưu tiên xác suất của nhãn được gắn. Những ca này minh họa model behavior, không đại diện toàn bộ quần thể.'),
            'Confidence ở đây là p khi operating label=1 và1-p khi label=0. Với threshold<0.5, max(p,1-p) có thể ưu tiên FP có xác suất dương thấp nhưng policy vẫn gắn cờ; vì vậy không dùng max(p,1-p) để gọi đó là FP tự tin.',
            'So sánh FN với TP và FP với TN bằng counts, numerical features và missingness, không suy ra nguyên nhân từ một quan sát. Sai confidence cao có thể do overlap lớp, thiếu yếu tố dự báo, outcome không được quan sát ngoài hệ thống hoặc shift; chưa thể kết luận nhãn sai.',
            'Local explanations dùng raw model output trước calibration. Nếu một biến góp phần tăng score cho FP, nó mô tả cách model sử dụng thông tin, không chứng minh biến đó gây tái nhập viện/bệnh tim. Các đề xuất thu thập feature tốt hơn cần thực nghiệm mới, không retune bằng error cases test.',
            'Bảng dưới giữ anonymous audit keys để truy nguyên artifact; không giải thích lâm sàng về điều trị. Với readmission, lịch sử number_inpatient là tín hiệu mạnh nhưng không thể giải thích mọi ca; một người chưa có nhiều nhập viện vẫn có thể thuộc lớp dương.'
        ],[('Các ca đối chiếu từ recipe frozen',compact(cases,casecols)),
            ('Numerical means theo nhóm lỗi',compact(error,list(error.columns)))])
    groups=[]
    for name in DATASETS:
        for row in e[name]['subgroups']:
            if str(row['interpretable']).lower()=='true' and row['feature'] in (['sex'] if name=='heart' else ['gender','race']):
                groups.append({'Dataset':name,'Feature':row['feature'],'Group':row['group'],'n':row['n'],'Positive':row['positive'],
                               'Recall':number(row.get('recall')),'Precision':number(row.get('precision')),'AP':number(row.get('ap'))})
    section('Subgroup analysis và giới hạn suy luận fairness',[
        'Phân tích trước các nhóm tuổi/giới, race khi có, sử dụng dịch vụ trước đó và missingness. Chỉ diễn giải metric khi nhóm có ít nhất30 observations và10 positives/10 negatives. Nhóm không đủ count vẫn có thể mô tả số lượng nhưng các metric suy luận gắn NA.',
        'Heart nhỏ nên nhiều nhóm không đạt yêu cầu; không cố chia nhiều khoảng để tìm chênh lệch đẹp. patient identity không xác thực khiến giới hạn giả định độc lập vẫn còn. Không gọi thiếu chênh lệch quan sát là bằng chứng công bằng.',
        'Readmission là dữ liệu hệ thống chăm sóc lịch sử; nhãn và khả năng tái nhập viện quan sát được có thể khác theo nhóm. Chênh lệch precision có thể chịu ảnh hưởng prevalence ngoài chất lượng phân loại. Cần counts và uncertainty thay vì chỉ so hai số recall.',
        'Subgroup analysis là hậu kiểm mô tả, không chứng minh phân biệt đối xử hay tác động nhân quả. Những nhóm nhỏ, dữ liệu missing và outcome ngoài hệ thống cần được xem là hạn chế. Chưa làm subgroup-calibrated deployment hoặc thử nghiệm prospective.',
        'Kết quả đủ hỗ trợ RQ4 về nơi model dễ sai, nhưng chưa đủ kết luận một policy lâm sàng công bằng. CSV đầy đủ có các nhóm age/missingness và điều kiện đủ count để kiểm tra lại.'
    ],[('Ví dụ nhóm đáp ứng count tối thiểu',pd.DataFrame(groups))])
    importance=[]
    for name in DATASETS:
        for row in e[name]['permutation_importance'][:5]:importance.append({'Dataset':name,'Feature':row['feature'],'Decrease':number(row['importance']),'SD':number(row['sd'])})
    section('Giải thích mô hình và các yếu tố liên quan',[
        'Permutation importance dùng5 lần hoán vị, sample tối đa2000 test rows seed47 và primary ranking metric của raw pipeline. Đây là hậu kiểm không chọn lại features. SD của phép hoán vị không phải CI tổng quát của tác động biến.',
        'Biến tương quan có thể chia sẻ importance, làm individual importance thấp hoặc không ổn định. Các feature permutation có thể tạo tổ hợp ít xuất hiện; do đó importance đo model reliance, không xác lập quan hệ nhân quả.',
        'Heart dùng LinearExplainer với background80 fit rows; contributions ở raw log-odds. Readmission dùng native exact TreeSHAP vì SHAP0.49 không đọc vector base_score của XGB3.2. Background native là training node cover, không gọi là interventional80-row sample. Tổng contributions+base được đối chiếu với raw margins.',
        'Gom one-hot contributions về biến gốc bằng tổng trong từng case. Mean absolute SHAP trên các case được chọn chỉ là minh họa local, không phải global population importance. Bảng và code giữ encoded values để kiểm toán.',
        'Numerical LR odds ratio được diễn giải theo1 SD train. Với full one-hot, category contrast cần chênh lệch hai hệ số; exp(coef) riêng không phải odds ratio tự nhiên so reference. Regularization và covariance giữa biến cũng hạn chế diễn giải.'
    ],[('Các biến có permutation importance cao',pd.DataFrame(importance))],['readmission_importance'])
    controlled=[]
    keep=['E02_weighted_False','E02_weighted_True','E05_full','E05_engineered','E06_full','E06_restricted','E09_scaling_True','E09_scaling_False']
    for name in DATASETS:
        for row in e[name]['controlled_ablations_summary']:
            if row['variant'] in keep:controlled.append({'Dataset':name,'Variant':row['variant'],'AUC':number(row['roc_auc']),'AP':number(row['ap']),'Brier':number(row['brier'])})
    section('Ablation và các phương pháp không cải thiện',[
        'E02 giữ cùng C/penalty/features/folds khi đổi class weights; E05/E06 giữ cùng hyperparameters khi đổi feature set; E09 giữ k/distance/weights khi đổi scaling. Các đối chứng này giúp tránh quy chênh lệch do nhiều yếu tố cùng đổi cho một nguyên nhân.',
        'KNN Heart scaled có CV AUC cao hơn unscaled rõ rệt. Tác động khớp nội dung khoảng cách/đơn vị đo trong Slide4, nhưng vẫn phụ thuộc encoder và feature geometry. Việc k lớn hơn tốt ở miền đã thử không chứng minh k nên tăng vô hạn.',
        'LR readmission balanced hơi tăng AP nhưng raw Brier tăng từ khoảng0.095 lên0.222 trong đối chứng development. Weighting đổi objective/phân phối học, nên probability quality cần kiểm tra và threshold phải chọn lại; không chỉ công bố recall cải thiện.',
        'Engineered feature set không cải thiện AP development so full; gộp diagnoses có thể giảm chi tiết và không tạo thêm tín hiệu độc lập. Restricted Heart bỏ ca/thal giảm CV AUC, phù hợp việc giảm thông tin xét nghiệm, nhưng diễn giải chỉ là đánh đổi trong thiết kế này.',
        'E08 cho thấy C lớn tăng training score nhưng không nhất thiết tăng validation score; L1 tạo thưa rõ trên readmission. Kết quả không cải thiện có giá trị vì giúp trả lời câu hỏi thay vì chỉ giữ thí nghiệm thuận lợi.'
    ],[('Đối chứng development giữ điều kiện còn lại',pd.DataFrame(controlled))],['readmission_regularization'])
    conformal=pd.DataFrame(e['conformal'].get('results',[]))
    section('Mở rộng thực nghiệm: split conformal',[
        'Dùng raw Logistic Regression Heart đã chọn/tune bằng fit; conformal calibration set riêng36 observations, score=1-p(true class). Quantile dùng rank ceil((n+1)*(1-alpha)); nếu rank vượt n, dùng bound1. Alpha0.05 và0.10 định trước.',
        'Prediction set gồm nhãn c nếu1-p(c)<=qhat. Coverage test quan sát bằng1 ở cả hai alpha nhưng average set size khoảng1.885/1.541; ambiguous rate88.5%/54.1%. Coverage cao đi cùng nhiều tập{0,1}, nên không thể gọi là phân loại hoàn hảo hoặc certainty.',
        'Conformal chỉ có bảo đảm biên dưới exchangeability giữa calibration/test; không đảm bảo cho từng người hoặc subgroup. Heart không có patient ID được xác minh, nên independence/exchangeability là giả định. Không áp dụng bảo đảm row-iid cho readmission encounters lặp.',
        'Phần này vượt phạm vi sáu slide nhưng conformal nền tảng không phải kỹ thuật mới nhất. Literature review cập nhật nghiên cứu distribution shift và các thế hệ tabular foundation models; phân biệt thuật toán đã chạy với paper đã đọc.',
        'RQ5: lợi ích bổ sung là trình bày ambiguity và coverage/size trade-off, không nhất thiết tăng accuracy. Chưa có prospective validation hoặc conformal dưới shift được triển khai; các phương pháp gần đây là hướng nghiên cứu tiếp theo.'
    ],[('Conformal test, raw LR cố định',compact(conformal,['alpha','coverage','mean_set_size','singleton_rate','ambiguous_rate','empty_rate']))])
    section('Kiến thức mới và tổng hợp với môn học',[
        'Sáu slide xác nhận Logistic Regression, KNN, Gaussian/Naive Bayes, MLE/MAP, regularization, CV/group split, threshold, calibration, learning curves và bootstrap. Không lấy calibration/bootstrap làm ví dụ kiến thức hoàn toàn ngoài môn học. Post-hoc calibration/cross-fitting đi sâu hơn phần kiểm tra cơ bản.',
        'Nature2025 giới thiệu TabPFN cho small/medium tabular data; technical reports2.5 và3.5 là các mốc nghiên cứu sau đó đã được kiểm chứng trước11/10/2026. Các báo cáo benchmark thuộc tác giả, không phải metric của hai splits hiện tại; không khẳng định model mới chắc chắn tốt hơn boosting trong dự án.',
        'TabPFN ở đây là skipped feasibility review theo phạm vi/ngân sách, chưa tải checkpoint hoặc chạy, không có failed/blocked run. Model card v2 công khai; repository bản mới mô tả auth/license flow. Không suy v2 gated chỉ từ bản mới và không ghi một thông báo lỗi tải giả.',
        'Nghiên cứu conformal dưới non-exchangeability/distribution shift2025–2026 giúp hiểu giới hạn bảo đảm hiện có. Dự án không triển khai optimal transport hoặc shift diagnostics của các paper mới, nên chỉ trích dẫn như literature review.',
        'Ánh xạ course_alignment chỉ rõ PDF page từ1, không số chân slide. GMM/EM đã học nhưng không bắt buộc một thí nghiệm clustering rời rạc. Ensemble/SHAP là bổ sung; phần cập nhật mới được phân loại riêng, không dùng marketing SOTA hoặc cộng trung bình AUC hai outcome.'
    ])
    section('Trả lời câu hỏi nghiên cứu và hạn chế',[
        'RQ1: LR là lựa chọn development cho Heart; XGB cho readmission. Test assessment hỗ trợ ranking có thông tin ở cả hai, nhưng mức hiệu năng và độ bất định rất khác. Không thể xếp hai outcome trên một thang clinical utility chung.',
        'RQ2: threshold0.5 bỏ sót đa số readmission positives; F2 tăng detection nhưng tạo nhiều FP. Top-k cho thấy mức nguồn lực và phần ca dương tìm được. Policy vận hành cần mục tiêu/chi phí được xác định ngoài test.',
        'RQ3: calibration Heart xấu hơn trên test; giữ kết quả bất lợi và recipe. RQ4: FN/FP, missingness, subgroup và explanations cho thấy model limitations; không chứng minh label noise hoặc nhân quả. RQ5: conformal tăng khả năng biểu đạt uncertainty với ambiguity lớn, còn TabPFN chỉ review.',
        'Giới hạn: dữ liệu lịch sử và cohort selection; Heart test61, cal36; IDs Heart tổng hợp; readmission outcomes ngoài hệ thống; thiếu timestamps/hospital ID đáng tin. Chưa temporal/external validation. CV selection có optimism; bootstrap chỉ fixed model; cỡ subgroup và correlation hạn chế inference.',
        'Thời gian/tài nguyên ảnh hưởng search budget RF. Model cuối không fit toàn bộ development; chưa triển khai nested cross-fitting. Hướng tiếp theo có thể tăng calibration data, bổ sung features/metadata hợp lệ, prospective validation và thử foundation model local theo license; mọi thay đổi cần đánh giá độc lập mới.'
    ])
    section('Tái lập, kiểm định và sản phẩm bàn giao',[
        'Python3.10.11, scikit-learn1.7.2, CPU tối đa2 luồng/model. Môi trường riêng .venv và requirements-lock.txt. Windows Unicode paths cần PYTHONUTF8=1 cho editable install; threadpoolctl3.6.0 tránh lỗi snapshot path3.7. Không hứa deterministic tuyệt đối trên mọi nền tảng.',
        '12 tests kiểm tra target/schema, group disjoint ở tất cả partitions/folds, missing None, ID exclusion, unknown categories, train-only statistics, save/load predictions, cluster multiplicity, gradient, frozen hashes và threshold truy nguyên development. Bốn notebook đã chạy từ kernel sạch, không có error outputs.',
        'README mô tả tạo môi trường, smoke/full commands, CLI và cache. Smoke chỉ kiểm tra wiring, tách artifacts/smoke khỏi full. Sau freeze, train/evaluate tái sử dụng artifact để tránh test tuning. Một nghiên cứu mới cần checkout/protocol/evaluation mới, không xóa manifest để chọn điểm đẹp trên cùng test.',
        'Data cards/source.json chứa URLs, DOI, license, SHA-256, raw/processed/split provenance. Runs chứa trial params, warnings, durations và cancelled RF evidence. Báo cáo đọc CSV/JSON/PNG đã chạy, không huấn luyện lại hoặc chèn metric giả.',
        'PDF và PPTX được sinh cục bộ khi artifact runtime không có sẵn, dùng font hỗ trợ tiếng Việt. Các bảng/chart so sánh trong deck chỉnh sửa được; đồ thị khoa học gốc kèm PNG/CSV/code. Kiểm tra render thực tế; không đánh giá chỉ bằng file tồn tại. Demo chạy local, chưa publish/nộp thay sinh viên.'
    ])
    return sections


def write_markdown(sections):
    lines=['# Healthcare and Patient Risk Analysis','',
      '## Bài tập lớn Machine Learning','',
      'Sinh viên / mã lớp: điền thông tin trước khi nộp. Ngày thực nghiệm: 11/10/2026.','',
      'Hai nghiên cứu tình huống: Heart Cleveland và Diabetes 130-US Hospitals. Số liệu lấy từ artifact thực nghiệm đã khóa.','']
    for i,s in enumerate(sections,1):
        lines.extend([f'## {i}. {s["title"]}',''])
        for p in s['paragraphs']:lines.extend([p,''])
        for caption,df in s['tables']:lines.extend([f'**{caption}**','',df.to_markdown(index=False),''])
        for figure in s['figures']:lines.extend([f'![{figure}](../artifacts/figures/{figure}.png)',''])
    lines.extend(['## Tài liệu tham khảo','',
       (ROOT/'reports/literature_review.md').read_text(encoding='utf-8'),'',
       'Tài liệu giảng dạy: Slide1.pdf, Slide2.pdf, Slide3.pdf, Slide4-KNN.pdf, Slide5-Prob-model.pdf, Model-assessment.pdf do người dùng cung cấp; dẫn PDF page trong course_alignment.md.','',
       'ICD grouping: Strack và cộng sự (2014), https://pmc.ncbi.nlm.nih.gov/articles/PMC3996476/ .',''])
    (ROOT/'reports/final_report.md').write_text('\n'.join(lines),encoding='utf-8')


def export_pdf(sections):
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from PIL import Image as PILImage
    fonts=Path('C:/Windows/Fonts')
    if (fonts/'arial.ttf').exists():
        font_files=[('Report',fonts/'arial.ttf'),('ReportBold',fonts/'arialbd.ttf'),('ReportItalic',fonts/'ariali.ttf')]
    else:
        import matplotlib
        fonts=Path(matplotlib.get_data_path())/'fonts/ttf'
        font_files=[('Report',fonts/'DejaVuSans.ttf'),('ReportBold',fonts/'DejaVuSans-Bold.ttf'),('ReportItalic',fonts/'DejaVuSans-Oblique.ttf')]
    for name,file in font_files:pdfmetrics.registerFont(TTFont(name,str(file)))
    styles={
      'title':ParagraphStyle('title',fontName='ReportBold',fontSize=25,leading=33,textColor=colors.HexColor('#153448'),spaceAfter=20),
      'head':ParagraphStyle('head',fontName='ReportBold',fontSize=17,leading=22,textColor=colors.HexColor('#153448'),spaceAfter=12),
      'body':ParagraphStyle('body',fontName='Report',fontSize=10.2,leading=14,spaceAfter=9),
      'small':ParagraphStyle('small',fontName='Report',fontSize=8.1,leading=10.4,spaceAfter=6),
      'caption':ParagraphStyle('caption',fontName='ReportBold',fontSize=9.2,leading=12,spaceAfter=6)}
    width=A4[0]-3.5*cm
    def para(text,kind='body'):return Paragraph(html.escape(str(text)),styles[kind])
    story=[Spacer(1,2*cm),para('Healthcare and Patient Risk Analysis','title'),para('Phân loại bệnh tim và phân tích nguy cơ tái nhập viện sớm','head'),Spacer(1,.7*cm),
           para('BÀI TẬP LỚN MACHINE LEARNING'),para('Sinh viên / mã lớp: ........................................................'),para('Ngày thực nghiệm: 11/10/2026'),Spacer(1,1.2*cm),
           para('Báo cáo dựa trên hai bộ dữ liệu UCI và các recipe đánh giá đã khóa. Nội dung kết hợp kiến thức trong sáu tài liệu môn học với uncertainty, explainability và literature review cập nhật.'),
           para('Prototype nghiên cứu môn học; chưa được thẩm định lâm sàng. Số liệu, thí nghiệm bất lợi và hạn chế được trình bày trung thực.'),PageBreak()]
    for i,s in enumerate(sections,1):
        story.append(para(f'{i}. {s["title"]}','head'))
        for p in s['paragraphs']:story.append(para(p))
        for caption,df in s['tables']:
            story.append(para(caption,'caption'))
            data=[[para(c,'small') for c in df.columns]]+[[para(v,'small') for v in row] for row in df.astype(str).values]
            n=len(df.columns);weights=np.array([1.65 if c in ['model','variant','Feature','feature','Policy','Probability','Dataset','Role'] else 1 for c in df.columns]);weights=weights/weights.sum()*width
            tb=Table(data,colWidths=weights.tolist(),repeatRows=1,hAlign='LEFT')
            tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9f0f3')),('VALIGN',(0,0),(-1,-1),'TOP'),
              ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#2a788e')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#dce4e8')),
              ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
            story.extend([tb,Spacer(1,8)])
        figs=[]
        for f in s['figures']:
            path=ROOT/f'artifacts/figures/{f}.png'
            if not path.exists():continue
            iw,ih=PILImage.open(path).size
            w=width if len(s['figures'])==1 else width*.48
            height=w*ih/iw
            if height>6.3*cm:w*=6.3*cm/height;height=6.3*cm
            figs.append(Image(str(path),width=w,height=height))
        if len(figs)==2:story.append(Table([figs],colWidths=[width/2]*2))
        elif figs:story.append(figs[0])
        story.append(PageBreak())
    story.append(para('Tài liệu tham khảo và nguồn kiểm chứng','head'))
    review=(ROOT/'reports/literature_review.md').read_text(encoding='utf-8')
    for line in review.splitlines():
        if line.startswith('|') and not line.startswith('|---') and 'Nguồn và ngày' not in line:
            columns=line.strip('|').split('|');text=' — '.join(c.strip() for c in columns[:2])
            text=re.sub(r'\[([^]]+)\]\(([^)]+)\)',r'\1: \2',text);story.append(para(text,'small'))
    story.append(para('Tài liệu môn học: sáu PDF trong Slide/; xem course_alignment.md để dẫn trang chính xác. Không bịa năm xuất bản/URL công khai của tài liệu giảng dạy.','small'))
    story.append(para('Strack và cộng sự2014: ICD grouping và dữ liệu readmission. https://pmc.ncbi.nlm.nih.gov/articles/PMC3996476/','small'))
    def footer(canvas,doc):
        canvas.setFont('Report',8);canvas.setFillColor(colors.HexColor('#627681'))
        canvas.drawString(1.75*cm,1.05*cm,'Machine Learning • Healthcare and Patient Risk Analysis')
        canvas.drawRightString(A4[0]-1.75*cm,1.05*cm,str(doc.page))
    SimpleDocTemplate(str(ROOT/'reports/final_report.pdf'),pagesize=A4,leftMargin=1.75*cm,rightMargin=1.75*cm,topMargin=1.5*cm,bottomMargin=1.7*cm).build(story,onFirstPage=footer,onLaterPages=footer)


def export_deck(e):
    from pptx import Presentation
    from pptx.util import Inches,Pt
    from pptx.dml.color import RGBColor
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE,XL_LEGEND_POSITION
    from pptx.enum.text import PP_ALIGN
    prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
    navy=RGBColor.from_string('153448');teal=RGBColor.from_string('2A788E');gray=RGBColor.from_string('546C7A')
    def text(slide,value,x,y,w,h,size=22,bold=False,color=navy):
        box=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=box.text_frame;tf.word_wrap=True
        for i,line in enumerate(value.split('\n')):
            p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name='Arial';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=color;p.space_after=Pt(12)
        return box
    def slide(title,note):
        sl=prs.slides.add_slide(prs.slide_layouts[6]);text(sl,title,.65,.38,12.1,.9,32,True)
        text(sl,str(len(prs.slides)),12.2,7.12,.5,.2,11,color=gray)
        sl.notes_slide.notes_text_frame.text=note+'\nNguồn: reports/protocol.md; artifacts/metrics CSVs; course_alignment.md; literature_review.md. 11/10/2026.'
        return sl
    def native_table(sl,df,x=.7,y=1.65,w=11.9,h=3.4,size=18):
        table=sl.shapes.add_table(len(df)+1,len(df.columns),Inches(x),Inches(y),Inches(w),Inches(h)).table
        for i,row in enumerate([list(df.columns),*df.astype(str).values.tolist()]):
            for j,value in enumerate(row):
                cell=table.cell(i,j);cell.text=str(value);cell.margin_left=Inches(.07);cell.margin_right=Inches(.07)
                cell.fill.solid();cell.fill.fore_color.rgb=RGBColor.from_string('E9F0F3' if i==0 else 'FFFFFF')
                for p in cell.text_frame.paragraphs:p.font.name='Arial';p.font.size=Pt(size);p.font.bold=i==0;p.font.color.rgb=navy
        return table
    def chart(sl,labels,values,series,x=.7,y=1.6,w=7.6,h=4.7):
        cd=CategoryChartData();cd.categories=labels;cd.add_series(series,values)
        ch=sl.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED,Inches(x),Inches(y),Inches(w),Inches(h),cd).chart
        ch.has_legend=False;ch.chart_style=10
        ch.category_axis.tick_labels.font.name='Arial';ch.category_axis.tick_labels.font.size=Pt(17)
        ch.value_axis.tick_labels.font.name='Arial';ch.value_axis.tick_labels.font.size=Pt(14)
        ch.value_axis.minimum_scale=0
        if 'ROC-AUC' in series:ch.value_axis.maximum_scale=1.0
        ch.plots[0].has_data_labels=True;ch.plots[0].data_labels.font.name='Arial';ch.plots[0].data_labels.font.size=Pt(16);ch.plots[0].data_labels.number_format='0.000'
        return ch
    h=selected_test(e,'heart');r=selected_test(e,'readmission')
    sl=slide('Healthcare and Patient Risk Analysis','Giới thiệu hai outcome độc lập. Không hứa clinical readiness, không cộng AUC hai task.')
    text(sl,'Phân loại bệnh tim và nguy cơ\ntái nhập viện sớm',.85,1.8,11.4,2.0,42,True)
    text(sl,'Bài tập lớn Machine Learning\nHai bộ dữ liệu UCI • Thực nghiệm 11/10/2026',.9,4.3,11,1.2,24)
    text(sl,'Sinh viên / mã lớp: điền trước khi nộp',.9,6.25,11,.5,18,color=gray)
    sl=slide('Bài toán và thước đo','T–E–P theo Slide1 trang7. Heart nhận diện bệnh hiện diện sau xét nghiệm; readmission at discharge.')
    native_table(sl,pd.DataFrame([['Heart','num > 0','Sau xét nghiệm','ROC-AUC'],['Readmission','<30','Lúc xuất viện','AP']],columns=['Dataset','Nhãn dương','Thời điểm','Metric']),h=2.0)
    text(sl,'Xác suất và nhãn quyết định có vai trò khác nhau.\nHai task tạo hai nghiên cứu tình huống, không phải external validation.',.85,4.25,11.5,1.5,24)
    sl=slide('Dữ liệu và cohort','Raw source UCI licenseCCBY4.0; expired/hospice mapping kiểm tra trước train. Heart row IDs tổng hợp.')
    native_table(sl,pd.DataFrame([['Heart',303,303,61],['Readmission',101766,99340,20031]],columns=['Dataset','Raw rows','Cohort','Test']),h=2.0)
    text(sl,'Readmission: loại expired/hospice và gender không hợp lệ.\nMột bệnh nhân có nhiều encounters: patient_nbr dùng để chia nhóm.\nHeart: thiếu patient ID xác thực, độc lập giữa hàng là giả định.',.85,4.05,11.5,2.0,22)
    sl=slide('Các vai trò dữ liệu độc lập','Mọi group disjoint assertion đạt. Calibration/threshold không dùng fit; không refit tất cả development. CV selection khác independent assessment.')
    native_table(sl,pd.DataFrame([['Fit / tune',169,59372],['Calibration',36,10083],['Threshold',37,9854],['Test',61,20031]],columns=['Role','Heart rows','Readmission rows']),h=3.15)
    text(sl,'Heart: stratified CV5 folds. Readmission: group CV3 folds.\nFreeze model, threshold, preprocessing và hashes trước test.',.85,5.3,11.5,1.25,22)
    sl=slide('Kiến thức đã học và phần bổ sung','Source course_alignment sixPDFs. Calibration/bootstrap đã học. RF limitedbeforetest amend no optimize test.')
    text(sl,'Đã học\nLogistic Regression • KNN • Gaussian/Naive Bayes\nMLE/MAP, L1/L2, pipeline, CV và group split\nThreshold, calibration, learning curves và bootstrap',.9,1.65,11.5,2.8,24)
    text(sl,'Bổ sung\nXGBoost / Random Forest, SHAP và split conformal\nReview cập nhật TabPFN; chưa có run TabPFN',.9,4.55,11.5,1.8,24)
    sl=slide('Lựa chọn mô hình bằng development','Scores CVselection not unbiasedestimate. Gaussian numerical-only. Budget12/6 exceptRF4 after resource amendment.')
    rows=[]
    for name in DATASETS:
        for a in e[name]['cv_results']:
            if a['model'] in ['logistic','xgboost','knn','random_forest']:rows.append([name,a['model'],number(a['mean_score'],3),number(a['std_score'],3)])
    native_table(sl,pd.DataFrame(rows,columns=['Dataset','Model','CV primary','Fold SD']),h=4.1)
    text(sl,'Chọn LR cho Heart, XGBoost cho readmission.\nSD giữa fold không phải CI95%.',.85,6.0,11.4,.9,21)
    sl=slide('Heart: AUC cao trên test nhỏ','Independenttest61; sensitivityn28. BootstrapAUC .909–.995; binomial exactrecalllower .877. Diagnostic features available, no prescreenclaim.')
    models=['logistic','knn','xgboost','gaussian_nb'];lookup={a['model']:a for a in e['heart']['test_metrics']}
    chart(sl,models,[float(lookup[m]['roc_auc']) for m in models],'Test ROC-AUC')
    text(sl,'LR ROC-AUC\n0,962\nCI95% 0,909–0,995',8.7,1.7,3.8,2.0,28,True)
    text(sl,'Threshold0,269\nTP28 • FN0\nFP14 • TN19\nKhông suy ra sensitivity\nquần thể100%',8.7,4.0,3.8,2.4,21)
    sl=slide('Readmission: ranking còn khiêm tốn','Test20031 heldoutpatients; APbaselineprevalence.1158. XGBAP.2289 pairedCI+.006..+.018 aboveLR. Highalertburden.')
    models=['dummy','logistic','random_forest','xgboost'];lookup={a['model']:a for a in e['readmission']['test_metrics']}
    chart(sl,models,[float(lookup[m]['ap']) for m in models],'Test AP')
    text(sl,'XGBoost\nAP0,229\nROC-AUC0,669',8.7,1.8,3.8,1.9,29,True)
    text(sl,'Recall77,1%\nPrecision15,5%\n9729 cảnh báo sai\n531 ca bị bỏ sót',8.7,4.05,3.8,2.1,23)
    sl=slide('Threshold và ngân sách gắn cờ','Allthresholds chosen beforetest. F2recallhigh versusprecisionburden; topk encounter not patient.')
    data=[]
    for a in e['readmission']['threshold_policies']:
        if a['model']=='xgboost' and a['policy'] in ['0.5','f2']:data.append([a['policy'],number(a['recall'],3),number(a['precision'],3),a['fp'],a['fn']])
    native_table(sl,pd.DataFrame(data,columns=['Policy','Recall','Precision','FP','FN']),h=2.1)
    text(sl,'Top10%: 2004 encounters, tìm546 ca dương.\nPrecision27,2%; recall23,5%.\nF2 gắn cờ57,5% encounters: cần cân nhắc tải xử lý.',.9,4.45,11.5,1.8,24)
    sl=slide('Phân tích FP / FN và giải thích local','Case selection operating-label probability, not maxclass whenthreshold<.5. SHAP rawlogodds notcausal; nativeXGBtreecoverfitonly.')
    local=frame(e,'readmission','local_shap_grouped');cases=frame(e,'readmission','case_studies');fn=cases[cases.error_type=='FN'].iloc[0]
    block=local[local.row_id.astype(str)==str(fn.row_id)].copy()
    if block.empty:block=local[local.row_id.astype(float)==float(fn.row_id)].copy()
    block['magnitude']=pd.to_numeric(block.shap).abs();block=block.nlargest(6,'magnitude')
    ch=chart(sl,block.feature.tolist(),pd.to_numeric(block.shap).tolist(),'Raw log-odds SHAP',w=7.8)
    ch.value_axis.minimum_scale=None
    text(sl,f'Ca FN: {fn.row_id}\np={number(fn.probability,3)}\nThreshold0,091',8.9,1.85,3.5,1.65,24,True)
    text(sl,'Đóng góp vào score\nkhông phải nguyên nhân.\nSai tự tin không\nchứng minh nhãn sai.',8.9,4.05,3.5,2.1,22)
    sl=slide('Calibration có thể làm xác suất kém hơn','Heart sigmoid selected thresholdholdout gain.00084; testrawBrier.0859 vs.1058 calibrated. Keep frozenrecipe; no testretuning.')
    chart(sl,['Heart raw','Heart sigmoid','Readmission raw','Readmission sigmoid'],[float(a['brier']) for name in DATASETS for a in e[name]['calibration_test']],'Test Brier',w=7.8)
    text(sl,'Heart: Brier\n0,086 → 0,106\nCalibrator chỉ36 ca',8.9,1.9,3.5,2.0,25,True)
    text(sl,'Giữ recipe đã khóa.\nCải thiện development\nkhông bảo đảm test.\nKhông retune sau test.',8.9,4.2,3.5,2.0,22)
    sl=slide('Ablation và tối ưu từ slide','Controlledkdistanceweights scaling; L1/L2 Cvalidcurve. NumPy gradients verified6.5e-11 andpred5.6e-9. Engineeredfeaturesdidnotimprove.')
    text(sl,'KNN Heart: scaling cải thiện CV rõ rệt.\nL1/L2: C lớn tăng train score, có thể giảm validation.\nClass weights: tăng recall/AP không bảo đảm Brier tốt.\nGộp ICD readmission không cải thiện AP development.',.9,1.65,11.5,2.85,25)
    text(sl,'Logistic NumPy\nGradient error6,5×10⁻¹¹\nPrediction difference5,6×10⁻⁹ so thư viện',.9,4.85,11.5,1.5,25,True)
    sl=slide('Split conformal và nghiên cứu gần đây','Conformal rawLR independentcal36,exchangeability;coverage1 butambiguous. TabPFNNature2025/technical3.52026 reviewonlynotexecuted.')
    rows=[[a['alpha'],number(a['coverage'],3),number(a['mean_set_size'],3),number(a['ambiguous_rate'],3)] for a in e['conformal']['results']]
    native_table(sl,pd.DataFrame(rows,columns=['Alpha','Coverage','Mean set size','Ambiguous']),h=2.2)
    text(sl,'Coverage cao đi cùng nhiều tập{0,1}.\nKhông bảo đảm riêng từng bệnh nhân hoặc khi distribution shift.\nTabPFN cập nhật trong review; không có metric giả của model chưa chạy.',.9,4.3,11.5,2.0,22)
    sl=slide('Kết luận và giới hạn','Answer RQ1–RQ5. Historicaldata/Heartsmall/groups external/temporal validationmissing. Noclinicalclaim. Reprodata hashesfrozenpreds12tests4notebooks.')
    text(sl,'LR cho Heart; XGBoost cho readmission theo development.\nThreshold và calibration cần đánh giá riêng với ranking.\nError analysis làm rõ bỏ sót, cảnh báo sai và uncertainty.',.9,1.65,11.5,2.1,26)
    text(sl,'Dữ liệu lịch sử • Heart test nhỏ • Chưa external/temporal validation\n12 tests đạt; 4 notebook chạy sạch; artifact và code tái lập.\nPrototype nghiên cứu, chưa thẩm định lâm sàng.',.9,4.45,11.5,2.0,23)
    assert len(prs.slides)==14
    prs.save(ROOT/'reports/presentation.pptx')


def render_pdf():
    import pypdfium2 as pdfium
    from PIL import Image,ImageDraw
    target=ROOT/'tmp/qa/report';target.mkdir(parents=True,exist_ok=True)
    pdf=pdfium.PdfDocument(str(ROOT/'reports/final_report.pdf'));paths=[]
    for i,page in enumerate(pdf):
        im=page.render(scale=1.4).to_pil();path=target/f'page_{i+1:02}.png';im.save(path);paths.append(path)
    cols=4;tw,th=300,445;rows=math.ceil(len(paths)/cols)
    canvas=Image.new('RGB',(cols*tw,rows*th),'#ddd');draw=ImageDraw.Draw(canvas)
    for i,path in enumerate(paths):
        im=Image.open(path);im.thumbnail((tw-10,th-25));x=i%cols*tw;y=i//cols*th;canvas.paste(im,(x+5,y+20));draw.text((x+8,y+3),str(i+1),fill='black')
    canvas.save(ROOT/'tmp/qa/report_montage.png')
    (ROOT/'artifacts/runs/report_render.json').write_text(json.dumps({'pages':len(paths),'renderer':'pypdfium2','all_pages_rendered':True}),encoding='utf-8')
    print('Rendered report pages:',len(paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--evidence-only", action="store_true", help="Print actual selected test results without writing artifacts")
    args = parser.parse_args()
    evidence = load_evidence(args.root)
    for dataset in DATASETS:
        print(result_paragraph(evidence, dataset))
    if not args.evidence_only:
        sections=make_sections(evidence);write_markdown(sections);export_pdf(sections);export_deck(evidence);render_pdf()
        print('Created final_report.md, final_report.pdf and editable presentation.pptx (14 slides)')


if __name__ == "__main__":
    main()
