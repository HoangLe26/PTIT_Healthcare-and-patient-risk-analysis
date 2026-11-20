import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score,average_precision_score,accuracy_score,balanced_accuracy_score,
    precision_score,recall_score,f1_score,fbeta_score,brier_score_loss,log_loss,confusion_matrix)

def metrics(y,p,t=.5):
    y=np.asarray(y);p=np.asarray(p);pred=p>=t
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    both=len(np.unique(y))==2
    return {'roc_auc':roc_auc_score(y,p) if both else np.nan,'ap':average_precision_score(y,p) if both else np.nan,
      'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred) if both else np.nan,
      'precision':precision_score(y,pred,zero_division=0) if (tp+fp) else np.nan,'recall':tp/(tp+fn) if tp+fn else np.nan,
      'specificity':tn/(tn+fp) if tn+fp else np.nan,'f1':f1_score(y,pred,zero_division=0),
      'f2':fbeta_score(y,pred,beta=2,zero_division=0),'brier':brier_score_loss(y,p),'log_loss':log_loss(y,np.clip(p,1e-12,1-1e-12),labels=[0,1]),
      'threshold':float(t),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'prevalence':float(y.mean()),'alert_rate':float(pred.mean())}

def select_threshold(y,p,policy='f2',ratio=1):
    y=np.asarray(y);p=np.asarray(p)
    candidates=np.unique(np.r_[0,p,1])
    # vectorized cumulative confusion counts for sorted descending probabilities
    order=np.argsort(-p,kind='stable');sp=p[order];sy=y[order];cum=np.r_[0,np.cumsum(sy)]
    counts=np.searchsorted(-sp,-candidates,side='right');tp=cum[counts];fp=counts-tp;fn=y.sum()-tp;tn=len(y)-y.sum()-fp
    if policy=='f2':scores=np.divide(5*tp,5*tp+4*fn+fp,out=np.zeros(len(tp)),where=(5*tp+4*fn+fp)!=0)
    elif policy=='cost':scores=-(ratio*fn+fp)
    else:
        feasible=tp/max(y.sum(),1)>=.85
        scores=np.where(feasible,tn/max(len(y)-y.sum(),1),-np.inf)
    valid=np.isfinite(scores)
    if not valid.any():return None
    best=scores.max();return float(candidates[np.flatnonzero(scores==best)[-1]])

def cluster_indices(groups,rng):
    groups=np.asarray(groups);unique,inverse=np.unique(groups,return_inverse=True)
    draws=rng.integers(0,len(unique),len(unique));multiplicity=np.bincount(draws,minlength=len(unique))
    return np.repeat(np.arange(len(groups)),multiplicity[inverse])

def bootstrap(y,probabilities,thresholds,groups,B=1000):
    rng=np.random.default_rng(42);samples={name:[] for name in probabilities};invalid=0
    for _ in range(B):
        ids=cluster_indices(groups,rng)
        if len(np.unique(np.asarray(y)[ids]))<2:invalid+=1;continue
        for name,p in probabilities.items():samples[name].append(metrics(np.asarray(y)[ids],np.asarray(p)[ids],thresholds[name]))
    rows=[];differences=[]
    for name,values in samples.items():
        point=metrics(y,probabilities[name],thresholds[name]);table=pd.DataFrame(values)
        for m in ['roc_auc','ap','recall','specificity','precision','f1','f2','brier','log_loss']:
            good=table[m].dropna();lo,hi=np.quantile(good,[.025,.975]) if len(good) else (np.nan,np.nan)
            rows.append({'model':name,'metric':m,'point':point[m],'lower':lo,'upper':hi,'valid':len(good),'repetitions':B,'invalid_one_class':invalid})
    names=list(samples)
    if len(names)>1:
        a,b=names[:2];ta,tb=pd.DataFrame(samples[a]),pd.DataFrame(samples[b])
        for m in ['roc_auc','ap','brier','f2']:
            delta=ta[m]-tb[m];lo,hi=np.quantile(delta.dropna(),[.025,.975])
            differences.append({'comparison':f'{a} - {b}','metric':m,'point':metrics(y,probabilities[a],thresholds[a])[m]-metrics(y,probabilities[b],thresholds[b])[m],
                                'lower':lo,'upper':hi,'repetitions':B})
    return pd.DataFrame(rows),pd.DataFrame(differences)

