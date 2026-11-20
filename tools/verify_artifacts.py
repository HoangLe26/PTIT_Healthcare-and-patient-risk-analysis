from pathlib import Path
import json
import pandas as pd
import nbformat
from healthcare_ml.common import ROOT,dump_json,sha256

def verify():
    records=[]
    def check(label,condition,evidence):
        records.append({'check':label,'passed':bool(condition),'evidence':evidence})
    for name in ['heart','readmission']:
        metrics=ROOT/f'artifacts/metrics/{name}_test_metrics.csv'
        check(f'{name}: independent test metrics',metrics.exists(),str(metrics.relative_to(ROOT)))
        if metrics.exists():
            df=pd.read_csv(metrics)
            check(f'{name}: required models',set(['dummy','logistic','gaussian_nb','xgboost',*(['knn','qda'] if name=='heart' else ['random_forest'])]).issubset(df.model),str(metrics.relative_to(ROOT)))
            check(f'{name}: finite probability metrics',df[['roc_auc','ap','brier','log_loss']].notna().all().all(),str(metrics.relative_to(ROOT)))
        for suffix in ['cv_results','controlled_ablations_summary','learning_curve','confidence_intervals','subgroups','case_studies','permutation_importance']:
            p=ROOT/f'artifacts/metrics/{name}_{suffix}.csv';check(f'{name}: {suffix}',p.exists(),str(p.relative_to(ROOT)))
        freeze=ROOT/f'artifacts/models/{name}_freeze.json';check(f'{name}: frozen',freeze.exists(),str(freeze.relative_to(ROOT)))
        for suffix in ['roc_pr','confusion','calibration','importance','learning_curve','threshold_tradeoff','regularization']:
            p=ROOT/f'artifacts/figures/{name}_{suffix}.png';check(f'{name}: {suffix} figure',p.exists(),str(p.relative_to(ROOT)))
    for filename in ['heart_conformal.json','optimization.json','readmission_topk.csv']:
        p=ROOT/'artifacts/metrics'/filename;check(filename,p.exists(),str(p.relative_to(ROOT)))
    expected_books=['01_heart_eda.ipynb','02_readmission_eda.ipynb','03_results_and_errors.ipynb','04_optimization_and_course_links.ipynb']
    for filename in expected_books:
        p=ROOT/'notebooks'/filename
        check(filename+': exists',p.exists(),str(p.relative_to(ROOT)))
        if not p.exists():continue
        book=nbformat.read(p,as_version=4);codes=[c for c in book.cells if c.cell_type=='code']
        check(p.name+': executed',all(c.execution_count is not None for c in codes) and not any(o.output_type=='error' for c in codes for o in c.outputs),str(p.relative_to(ROOT)))
    for filename in ['final_report.md','final_report.pdf','presentation.pptx','course_alignment.md','literature_review.md','defense_qa.md','slides_outline.md']:
        p=ROOT/'reports'/filename;check(filename,p.exists() and p.stat().st_size>100,str(p.relative_to(ROOT)))
    dump_json(ROOT/'artifacts/runs/acceptance_verification.json',records)
    passed=sum(r['passed'] for r in records)
    print(f'{passed}/{len(records)} artifact checks passed')
    if passed!=len(records):
        for r in records:
            if not r['passed']:print('FAILED',r['check'],r['evidence'])
        raise SystemExit(1)

if __name__=='__main__':verify()

