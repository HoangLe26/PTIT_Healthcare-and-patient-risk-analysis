"""Controlled development-only comparisons in addition to tuned comparisons."""
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from .common import ROOT,save_csv,sha256,load_json,dump_json
from .data import load_dataset
from .models import pipeline_for
from .training import folds_for
from .evaluation import metrics
from .analysis import savefig

def run_ablations(name):
    cache=ROOT/f'artifacts/runs/{name}_controlled_ablations.json'
    signature={'split':sha256(ROOT/f'data/splits/{name}.json'),'model':sha256(ROOT/f'artifacts/models/{name}_logistic.joblib'),
               'source':sha256(ROOT/'src/healthcare_ml/ablations.py')}
    if cache.exists() and load_json(cache)==signature:
        print(f'{name}: controlled ablation cache reused',flush=True);return
    df,split=load_dataset(name);ids=np.array(split['fit']);folds=folds_for(df,ids,name,5 if name=='heart' else 3)
    artifact=joblib.load(ROOT/f'artifacts/models/{name}_logistic.joblib')
    params=artifact['params'];baseC=params['model__C'];penalty=params['model__penalty'];specs=[]
    for reg in ['l1','l2']:
        for C in [.01,.1,1,10]:specs.append((f'E08_{reg}_C{C}','E08','logistic','full',True,False,{'model__C':C,'model__penalty':reg}))
    for weighted in [False,True]:specs.append((f'E02_weighted_{weighted}','E02','logistic','full',True,weighted,{'model__C':baseC,'model__penalty':penalty}))
    for fs in (['full','restricted'] if name=='heart' else ['full','engineered']):
        eid='E06' if name=='heart' else 'E05'
        specs.append((f'{eid}_{fs}',eid,'logistic',fs,True,False,{'model__C':baseC,'model__penalty':penalty}))
    if name=='heart':
        bestknn=joblib.load(ROOT/'artifacts/models/heart_knn.joblib')['params']
        for scaled in [True,False]:specs.append((f'E09_scaling_{scaled}','E09','knn','full',scaled,False,bestknn))
        for k in [3,5,7,11,15,21]:
            for distance in [1,2]:
                for weight in ['uniform','distance']:
                    specs.append((f'E09_k{k}_p{distance}_{weight}','E09','knn','full',True,False,
                        {'model__n_neighbors':k,'model__p':distance,'model__weights':weight}))
        for kind in ['logistic','gaussian_nb','qda']:
            source=joblib.load(ROOT/f'artifacts/models/heart_{"logistic_numeric" if kind=="logistic" else kind}.joblib')
            specs.append((f'E10_{kind}','E10',kind,'numeric',True,False,source['params']))
    rows=[]
    with threadpool_limits(limits=2):
        for label,eid,kind,fs,scaled,weighted,changes in specs:
            for i,(tr,va) in enumerate(folds):
                train=df.iloc[ids[tr]];val=df.iloc[ids[va]]
                model=pipeline_for(df,name,kind,fs,scaled,weighted).set_params(**changes)
                start=time.perf_counter();model.fit(train,train.y)
                p=model.predict_proba(val)[:,1];pt=model.predict_proba(train)[:,1]
                estimator=model.named_steps['model'];nz=int(np.count_nonzero(estimator.coef_)) if hasattr(estimator,'coef_') else None
                rows.append({'experiment':eid,'variant':label,'fold':i,'model':kind,'feature_set':fs,
                     'scaled':scaled,'weighted':weighted,'C':changes.get('model__C'),
                     'penalty':changes.get('model__penalty'),'k':changes.get('model__n_neighbors'),
                     'p':changes.get('model__p'),'weights':changes.get('model__weights'),
                     'nonzero':nz,'seconds':time.perf_counter()-start,
                     'train_score':metrics(train.y,pt)['roc_auc' if name=='heart' else 'ap'],**metrics(val.y,p)})
    result=pd.DataFrame(rows);save_csv(result,ROOT/f'artifacts/metrics/{name}_controlled_ablations.csv')
    summary=result.groupby(['experiment','variant','feature_set'],dropna=False).agg(
          roc_auc=('roc_auc','mean'),ap=('ap','mean'),brier=('brier','mean'),log_loss=('log_loss','mean'),
          train_score=('train_score','mean'),nonzero=('nonzero','mean'),seconds=('seconds','mean')).reset_index()
    save_csv(summary,ROOT/f'artifacts/metrics/{name}_controlled_ablations_summary.csv')
    lr=result[result.experiment=='E08'];score='roc_auc' if name=='heart' else 'ap'
    plt.figure(figsize=(7,4))
    for penalty,t in lr.groupby('penalty'):
        g=t.groupby('C')[score].mean();plt.plot(g.index,g.values,'o-',label=penalty)
    plt.xscale('log');plt.xlabel('C');plt.ylabel(f'CV {score}');plt.title('Controlled L1/L2 comparison');plt.legend();savefig(f'{name}_regularization')
    if name=='heart':
        krows=result[(result.experiment=='E09')&result.k.notna()&result.variant.str.startswith('E09_k')]
        plt.figure(figsize=(7,4))
        for (p,w),t in krows.groupby(['p','weights']):
            g=t.groupby('k').roc_auc.mean();plt.plot(g.index,g.values,'o-',label=f'p={p}, {w}')
        plt.xlabel('k');plt.ylabel('CV ROC-AUC');plt.legend();plt.title('KNN distance/weight validation curve');savefig('heart_knn_controlled_curve')
    print(f'{name}: controlled ablations saved',flush=True)
    dump_json(cache,signature)

