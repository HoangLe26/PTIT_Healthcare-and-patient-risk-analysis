from pathlib import Path
import sys
import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
BOOT='''from pathlib import Path
import sys
root = Path.cwd()
if not (root / "src").exists(): root = root.parent
sys.path.insert(0, str(root / "src"))
import pandas as pd, numpy as np
from IPython.display import display, Image, Markdown
from healthcare_ml.data import load_dataset, feature_columns
'''

def build(execute=True):
    specs=[]
    for name,title,file in [('heart','Bệnh tim Cleveland','01_heart_eda.ipynb'),('readmission','Tái nhập viện sớm','02_readmission_eda.ipynb')]:
        cells=[nbf.v4.new_markdown_cell(f'# EDA development: {title}\nChỉ dùng development để tìm hiểu dữ liệu. Không chọn feature/model bằng test.'),
               nbf.v4.new_code_cell(BOOT),nbf.v4.new_code_cell(f'df, split = load_dataset("{name}")\ndev = df.iloc[split["development"]]\ndisplay(dev[feature_columns(df,"{name}")].head())\ndisplay(dev.y.value_counts().rename("count"))\ndisplay(dev[feature_columns(df,"{name}")].isna().mean().sort_values(ascending=False).head(15))'),
               nbf.v4.new_markdown_cell('## Phân phối, missingness và các câu hỏi\nSo sánh theo nhãn chỉ là liên hệ thống kê. Giữ `None` xét nghiệm như một trạng thái; mọi imputer/encoder học riêng trong train-fold.'),
               nbf.v4.new_code_cell(f'for suffix in ["labels", "missing", "distributions", "correlation"]:\n    display(Image(filename=str(root / "artifacts/figures" / f"{name}_{{suffix}}.png")))'),
               nbf.v4.new_markdown_cell('## Split kiểm toán\nTập test giữ riêng; group patient tách biệt. Fit, calibration, threshold và test không giao nhau.'),
               nbf.v4.new_code_cell('parts = ["fit", "calibration", "threshold", "test"]\nfor i,a in enumerate(parts):\n    for b in parts[i+1:]:\n        assert not set(df.iloc[split[a]].patient_nbr) & set(df.iloc[split[b]].patient_nbr)\nprint("Verified disjoint groups", {k:len(split[k]) for k in parts})')]
        specs.append((file,cells))
    specs.append(('03_results_and_errors.ipynb',[
        nbf.v4.new_markdown_cell('# Kết quả test cuối và phân tích lỗi\nNotebook đọc predictions của recipe đã đóng băng, không huấn luyện hoặc chọn lại mô hình.'),nbf.v4.new_code_cell(BOOT),
        nbf.v4.new_code_cell('for name in ["heart", "readmission"]:\n    display(Markdown(f"## {name}"))\n    display(pd.read_csv(root/f"artifacts/metrics/{name}_cv_results.csv"))\n    display(pd.read_csv(root/f"artifacts/metrics/{name}_test_metrics.csv"))\n    display(pd.read_csv(root/f"artifacts/metrics/{name}_confidence_intervals.csv"))'),
        nbf.v4.new_markdown_cell('## FN, FP, calibration và giải thích\nSai confidence cao không chứng minh nhãn sai. Permutation importance/coefficients không phải tác động nhân quả.'),
        nbf.v4.new_code_cell('for name in ["heart", "readmission"]:\n    for suffix in ["roc_pr", "confusion", "calibration", "importance", "learning_curve"]:\n        display(Image(filename=str(root/f"artifacts/figures/{name}_{suffix}.png")))\n    display(pd.read_csv(root/f"artifacts/metrics/{name}_case_studies.csv"))\n    display(pd.read_csv(root/f"artifacts/metrics/{name}_subgroups.csv"))'),
        nbf.v4.new_markdown_cell('## Hướng mở rộng\nSplit conformal trên Heart; coverage biên cần exchangeability, không bảo đảm cho từng bệnh nhân. TabPFN được khảo sát trong literature review, không có số thực nghiệm nếu không chạy.'),
        nbf.v4.new_code_cell('import json\ndisplay(json.loads((root/"artifacts/metrics/heart_conformal.json").read_text(encoding="utf-8")))\ndisplay(pd.read_csv(root/"artifacts/metrics/readmission_topk.csv"))')]))
    specs.append(('04_optimization_and_course_links.ipynb',[
        nbf.v4.new_markdown_cell('# Tối ưu và liên hệ kiến thức môn học\nSlide2: gradient descent và scaling. Slide3: Bernoulli likelihood, mean log loss, gradient p-y và L1/L2. Slide5: MLE/MAP và conditional independence.'),nbf.v4.new_code_cell(BOOT),
        nbf.v4.new_code_cell('from healthcare_ml.optimization import objective_gradient, fit_numpy\nfrom scipy.special import expit\nX=np.array([[0.],[1.]])\ny=np.array([0,1])\nw,b,loss=fit_numpy(X,y,rate=1,steps=40)\nprint("loss start/end",loss[0],loss[-1],"probabilities",expit(X@w+b))'),
        nbf.v4.new_markdown_cell('## Công thức\n`z=Xw+b`, `p=sigmoid(z)`, `J=mean(logaddexp(0,z)-y*z)+lambda*||w||²/2`. Gradient `X.T@(p-y)/n+lambda*w`; intercept không bị phạt. Cập nhật đồng thời từ cùng bước. Không đồng nhất lambda với 1/C khi quy ước loss khác nhau.'),
        nbf.v4.new_code_cell('import json\ndisplay(json.loads((root/"artifacts/metrics/optimization.json").read_text(encoding="utf-8")))\nlosses=pd.read_csv(root/"artifacts/metrics/optimization_loss.csv")\nimport matplotlib.pyplot as plt\nfig,ax=plt.subplots(figsize=(8,4))\nfor rate,block in losses.groupby("rate"):\n    ax.plot(block.iteration,block.train_loss,label=str(rate))\nax.set(ylim=(0, min(3,losses.train_loss.max())),xlabel="iteration",ylabel="mean train log loss")\nax.legend(title="learning rate");plt.show()'),
        nbf.v4.new_markdown_cell('## Regularization và mô hình xác suất\nL2 tương ứng Gaussian prior khi likelihood/prior và scale được đặt phù hợp; MAP là point estimate, chưa phải posterior đầy đủ. GaussianNB dùng covariance chéo, QDA dùng covariance đầy đủ có regularization. So sánh E10 cùng numerical features để tránh nhầm hiệu quả thuật toán với thêm biến.'),
        nbf.v4.new_code_cell('for name in ["heart","readmission"]:\n    display(pd.read_csv(root/f"artifacts/metrics/{name}_controlled_ablations_summary.csv"))\n    display(Image(filename=str(root/f"artifacts/figures/{name}_regularization.png")))\ndisplay(Image(filename=str(root/"artifacts/figures/heart_knn_controlled_curve.png")))')]))
    for file,cells in specs:
        book=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Healthcare ML (.venv)','language':'python','name':'python3'}})
        path=ROOT/'notebooks'/file
        if execute:
            client=NotebookClient(book,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}},allow_errors=False)
            # Explicit kernel command avoids accidentally executing with global dependencies.
            from jupyter_client import KernelManager
            km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-X','utf8','-m','ipykernel_launcher','-f','{connection_file}']
            client.km=km;client.execute()
        nbf.write(book,path);print(f'Wrote executed {path.name}',flush=True)

if __name__=='__main__':build()

