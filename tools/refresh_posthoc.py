"""Refresh diagnostic visuals/case selection from frozen saved predictions only."""
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from healthcare_ml.common import ROOT,load_json,save_csv
from healthcare_ml.data import load_dataset
from healthcare_ml.analysis import savefig

for name in ['heart','readmission']:
    df,split=load_dataset(name);selected=load_json(ROOT/f'artifacts/models/{name}_freeze.json')['selected']
    art=joblib.load(ROOT/f'artifacts/models/{name}_selected.joblib')
    preds=pd.read_csv(ROOT/f'artifacts/predictions/{name}_{selected}_test.csv')
    test=df.iloc[split['test']].copy();merged=test.merge(preds[['row_id','probability','prediction','error_type']],on='row_id')
    merged['confidence']=np.where(merged.prediction==1,merged.probability,1-merged.probability)
    cases=[]
    for error in ['FN','FP','TP','TN']:
        cases.append(merged[merged.error_type==error].sort_values('confidence',ascending=False).head(2 if error in ['FN','FP'] else 1))
    save_csv(pd.concat(cases),ROOT/f'artifacts/metrics/{name}_case_studies.csv')
    save_csv(merged.groupby('error_type').agg(n=('y','size'),mean_probability=('probability','mean'),mean_assigned_class_probability=('confidence','mean')).reset_index(),ROOT/f'artifacts/metrics/{name}_error_summary.csv')
    raw=preds.raw_probability.to_numpy();cal=art['sigmoid_candidate'].predict(raw)
    fig,ax=plt.subplots(figsize=(6,5));displays=[]
    for label,p in [('raw',raw),('sigmoid candidate',cal)]:displays.append(CalibrationDisplay.from_predictions(preds.y,p,n_bins=6,strategy='quantile',name=label,ax=ax))
    if name=='readmission':
        bound=min(1,max(max(d.prob_true.max(),d.prob_pred.max()) for d in displays)*1.2)
        ax.set_xlim(0,bound);ax.set_ylim(0,bound);ax.set_title('Calibration: quantile bin means (axes zoomed)')
    savefig(f'{name}_calibration')
print('Post-hoc case selection and calibration figures refreshed; frozen predictions unchanged')

