import time
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay,PrecisionRecallDisplay,ConfusionMatrixDisplay
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from .common import ROOT,dump_json,load_json,save_csv,sha256
from .data import load_dataset,feature_columns
from .training import predict_artifact,folds_for
from .evaluation import metrics,bootstrap

plt.rcParams.update({'font.family':'DejaVu Sans','figure.dpi':120,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
def savefig(name):
    plt.tight_layout();plt.savefig(ROOT/f'artifacts/figures/{name}.png',bbox_inches='tight');plt.close()

def development_analysis(name):
    manifest_path=ROOT/f'artifacts/runs/{name}_development_analysis.json'
    signature={'split':sha256(ROOT/f'data/splits/{name}.json'),'model':sha256(ROOT/f'artifacts/models/{name}_logistic.joblib')}
    if manifest_path.exists() and load_json(manifest_path)==signature:
        print(f'{name}: development analysis cache reused',flush=True);return
    df,split=load_dataset(name);dev=df.iloc[split['development']]
    for suffix in ['metrics','figures']:(ROOT/f'artifacts/{suffix}').mkdir(parents=True,exist_ok=True)
    plt.figure(figsize=(7,4));dev.y.value_counts().sort_index().plot.bar(color=['#2a788e','#e5a030']);plt.xticks([0,1],['Âm tính','Dương tính'],rotation=0);plt.ylabel('Số quan sát development');plt.title(f'{name}: phân bố nhãn');savefig(f'{name}_labels')
    miss=dev[feature_columns(df,name)].isna().mean().sort_values(ascending=False).head(12)
    plt.figure(figsize=(8,4));miss.plot.bar(color='#2a788e');plt.ylabel('Tỷ lệ missing');plt.title('Missingness trên development');savefig(f'{name}_missing')
    pd.DataFrame({'feature':miss.index,'missing_rate':miss.values}).to_csv(ROOT/f'artifacts/metrics/{name}_missingness.csv',index=False)
    numeric=['age','trestbps','chol','thalach','oldpeak'] if name=='heart' else ['time_in_hospital','num_medications','number_inpatient','num_lab_procedures']
    fig,axes=plt.subplots(2,3 if name=='heart' else 2,figsize=(11,6));axes=axes.ravel()
    for ax,c in zip(axes,numeric):
        for y,color in [(0,'#2a788e'),(1,'#e5a030')]:ax.hist(dev.loc[dev.y==y,c].dropna(),bins=16,alpha=.55,density=True,label=f'y={y}',color=color)
        ax.set_title(c);ax.legend(fontsize=8)
    for ax in axes[len(numeric):]:ax.axis('off')
    savefig(f'{name}_distributions')
    corr=dev[numeric].corr();plt.figure(figsize=(6,5));plt.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm');plt.xticks(range(len(numeric)),numeric,rotation=45,ha='right');plt.yticks(range(len(numeric)),numeric);plt.colorbar();plt.title('Tương quan numerical development');savefig(f'{name}_correlation')
    # Learning curves reuse locked folds and group-level subsets.
    art=joblib.load(ROOT/f'artifacts/models/{name}_logistic.joblib');fit=np.array(split['fit']);n=5 if name=='heart' else 3
    folds=folds_for(df,fit,name,n);rows=[];rng=np.random.default_rng(46)
    with threadpool_limits(limits=2):
        for i,(tr,va) in enumerate(folds):
            train=df.iloc[fit[tr]];validation=df.iloc[fit[va]];groups=train.patient_nbr.unique();rng.shuffle(groups)
            for frac in [.3,.6,1.]:
                small=train[train.patient_nbr.isin(groups[:max(2,int(len(groups)*frac))])]
                model=clone(art['pipeline']);start=time.perf_counter();model.fit(small,small.y)
                for label,X in [('train',small),('validation',validation)]:
                    score=metrics(X.y,model.predict_proba(X)[:,1])['roc_auc' if name=='heart' else 'ap']
                    rows.append({'fold':i,'fraction':frac,'rows':len(small),'groups':small.patient_nbr.nunique(),'split':label,'score':score,'seconds':time.perf_counter()-start})
    lc=pd.DataFrame(rows);save_csv(lc,ROOT/f'artifacts/metrics/{name}_learning_curve.csv')
    plt.figure(figsize=(7,4))
    for kind in ['train','validation']:
        g=lc[lc.split==kind].groupby('fraction').agg(rows=('rows','mean'),score=('score','mean'),sd=('score','std'))
        plt.errorbar(g.rows,g.score,yerr=g.sd,marker='o',capsize=3,label=kind)
    plt.ylabel('ROC-AUC' if name=='heart' else 'AP');plt.xlabel('Số quan sát train');plt.title('Learning curve: Logistic Regression');plt.legend();savefig(f'{name}_learning_curve')
    trial_path=ROOT/f'artifacts/metrics/{name}_search_trials.csv'
    trials=pd.read_csv(trial_path if trial_path.exists() else ROOT/f'artifacts/runs/{name}_logistic_trials.csv')
    for model,col,label in [('logistic','param_model__C','C'),('knn','param_model__n_neighbors','k')]:
        t=trials[trials.model==model]
        if t.empty or col not in t:continue
        g=t.groupby(col).mean(numeric_only=True)
        plt.figure(figsize=(7,4));plt.plot(g.index,g.mean_train_score,'o-',label='train');plt.plot(g.index,g.mean_test_score,'o-',label='validation')
        if label=='C':plt.xscale('log')
        plt.xlabel(label);plt.ylabel('CV primary score');plt.title(f'Validation curve ({model}; average sampled settings)');plt.legend();savefig(f'{name}_{model}_validation_curve')
    dump_json(manifest_path,signature)

def conformal():
    name='heart';df,split=load_dataset(name);art=joblib.load(ROOT/'artifacts/models/heart_logistic.joblib')
    ca=df.iloc[split['calibration']];test=df.iloc[split['test']]
    # Use raw fixed LR; no calibrator fitting to same conformal set.
    pc=predict_artifact(art,ca,False);pt=predict_artifact(art,test,False)
    scores=1-np.where(ca.y.to_numpy()==1,pc,1-pc);rows=[]
    for alpha in [.05,.1]:
        rank=int(np.ceil((len(ca)+1)*(1-alpha)))
        q=float(np.sort(scores)[rank-1]) if rank<=len(ca) else 1.
        sets=np.column_stack([pt,1-pt])<=q
        coverage=float(sets[np.arange(len(test)),test.y.to_numpy()].mean());size=sets.sum(axis=1)
        rows.append({'alpha':alpha,'qhat':q,'coverage':coverage,'mean_set_size':float(size.mean()),
          'singleton_rate':float((size==1).mean()),'ambiguous_rate':float((size==2).mean()),'empty_rate':float((size==0).mean()),
          'calibration_n':len(ca),'test_n':len(test),'method':'split conformal, score=1-p_true, finite-sample rank'})
    dump_json(ROOT/'artifacts/metrics/heart_conformal.json',{'results':rows,'exchangeability_assumption':'Cleveland rows assumed independent; no patient grouping metadata to verify',
              'model':'Logistic Regression raw, trained on fit only','calibration_ids_hash':sha256(ROOT/'data/splits/heart.json')})

def evaluate(name,B=1000):
    frozen=load_json(ROOT/f'artifacts/models/{name}_freeze.json');marker=ROOT/f'artifacts/runs/{name}_evaluated.json'
    if marker.exists():print(f'{name}: independent test assessment already saved; reused',flush=True);return
    for filename,digest in frozen['models'].items():assert sha256(ROOT/'artifacts/models'/filename)==digest,'Model changed after freeze'
    for filename,digest in frozen.get('pipeline_source_hashes',{}).items():assert sha256(ROOT/'src/healthcare_ml'/filename)==digest,'Feature/calibration source changed after freeze'
    assert sha256(ROOT/f'data/splits/{name}.json')==frozen['split_hash']
    assert sha256(ROOT/f'data/processed/{name}.csv')==frozen['processed_hash']
    df,split=load_dataset(name);test=df.iloc[split['test']].copy();y=test.y.to_numpy();rows=[];probabilities={};thresholds={};threshold_rows=[]
    cv=pd.read_csv(ROOT/f'artifacts/metrics/{name}_cv_results.csv')
    for model in cv.model:
        art=joblib.load(ROOT/f'artifacts/models/{name}_{model}.joblib');start=time.perf_counter();p=predict_artifact(art,test);seconds=time.perf_counter()-start
        row={'model':model,'feature_set':art['feature_set'],**metrics(y,p,art['threshold']),
             'train_seconds':art['train_seconds'],'inference_seconds':seconds,'calibration_selected':art['calibration_selected'],
             'features_after_transform':len(art['pipeline'].named_steps['prepare'].get_feature_names_out())}
        rows.append(row);probabilities[model]=p;thresholds[model]=art['threshold']
        pred=pd.DataFrame({'row_id':test.row_id,'patient_nbr':test.patient_nbr,'y':y,'probability':p,
             'raw_probability':predict_artifact(art,test,False),'threshold':art['threshold'],'prediction':(p>=art['threshold']).astype(int)})
        pred['error_type']=np.select([(y==1)&(pred.prediction==1),(y==0)&(pred.prediction==0),(y==0)&(pred.prediction==1)],['TP','TN','FP'],default='FN')
        save_csv(pred,ROOT/f'artifacts/predictions/{name}_{model}_test.csv')
        threshold_rows.append({'model':model,'policy':'0.5',**metrics(y,p,.5)})
        for label,t in art['policies'].items():
            if t is not None:threshold_rows.append({'model':model,'policy':label,**metrics(y,p,t)})
    save_csv(pd.DataFrame(rows),ROOT/f'artifacts/metrics/{name}_test_metrics.csv')
    save_csv(pd.DataFrame(threshold_rows),ROOT/f'artifacts/metrics/{name}_threshold_policies.csv')
    selected=frozen['selected'];p=probabilities[selected];t=thresholds[selected]
    # confidence intervals on selected plus LR (or strongest distinct development runner-up)
    other='logistic' if selected!='logistic' else cv[cv.model!=selected].sort_values('mean_score',ascending=False).iloc[0].model
    names=[selected,other]
    print(f'{name}: bootstrap {B} paired patient resamples for {names}',flush=True)
    ci,deltas=bootstrap(y,{m:probabilities[m] for m in names},{m:thresholds[m] for m in names},test.patient_nbr.to_numpy(),B)
    save_csv(ci,ROOT/f'artifacts/metrics/{name}_confidence_intervals.csv');save_csv(deltas,ROOT/f'artifacts/metrics/{name}_paired_differences.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for m in dict.fromkeys([selected,'logistic','dummy']):
        RocCurveDisplay.from_predictions(y,probabilities[m],name=m,ax=axes[0]);PrecisionRecallDisplay.from_predictions(y,probabilities[m],name=m,ax=axes[1])
    axes[0].plot([0,1],[0,1],'k--',alpha=.4);axes[1].axhline(y.mean(),ls='--',color='gray');savefig(f'{name}_roc_pr')
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,threshold in zip(axes,[.5,t]):
        ConfusionMatrixDisplay.from_predictions(y,p>=threshold,display_labels=['Âm','Dương'],ax=ax,colorbar=False)
        ax.set_title(f'{selected}, threshold={threshold:.3f}')
    savefig(f'{name}_confusion')
    art=joblib.load(ROOT/f'artifacts/models/{name}_selected.joblib');raw=predict_artifact(art,test,False);cal=art['sigmoid_candidate'].predict(raw)
    fig,ax=plt.subplots(figsize=(6,5));displays=[]
    for label,probs in [('raw',raw),('sigmoid candidate',cal)]:displays.append(CalibrationDisplay.from_predictions(y,probs,n_bins=6,strategy='quantile',name=label,ax=ax))
    if name=='readmission':
        bound=min(1,max(max(d.prob_true.max(),d.prob_pred.max()) for d in displays)*1.2)
        ax.set_xlim(0,bound);ax.set_ylim(0,bound);ax.set_title('Calibration: quantile bin means (axes zoomed)')
    savefig(f'{name}_calibration')
    calibration=pd.DataFrame([{'model':selected,'probability':kind,**metrics(y,probs,t)} for kind,probs in [('raw',raw),('sigmoid_candidate',cal)]])
    save_csv(calibration,ROOT/f'artifacts/metrics/{name}_calibration_test.csv')
    curve=[]
    for threshold in np.linspace(0,1,101):curve.append(metrics(y,p,threshold))
    curve=pd.DataFrame(curve);save_csv(curve,ROOT/f'artifacts/metrics/{name}_threshold_sweep.csv')
    plt.figure(figsize=(7,4))
    for c in ['recall','specificity','precision','alert_rate']:plt.plot(curve.threshold,curve[c],label=c)
    plt.axvline(t,ls='--',color='gray');plt.legend();plt.xlabel('Threshold');plt.ylabel('Metric');savefig(f'{name}_threshold_tradeoff')
    subgroup=[]
    cols=['age_group','sex','cp','missing_group'] if name=='heart' else ['age','gender','race','prior_inpatient_group','missing_group']
    if name=='heart':test['age_group']=pd.cut(test.age,[0,45,60,120],labels=['<45','45–59','60+'],right=False)
    else:test['prior_inpatient_group']=pd.cut(test.number_inpatient,[-1,0,1,100],labels=['0','1','2+'])
    test['missing_group']=np.where(test[feature_columns(df,name)].isna().sum(axis=1)>0,'has_missing','complete')
    for c in cols:
        for value,block in test.groupby(c,dropna=False,observed=True):
            positions=np.flatnonzero(test.index.isin(block.index));yg=y[positions];pg=p[positions];valid=len(yg)>=30 and yg.sum()>=10 and (len(yg)-yg.sum())>=10
            row={'feature':c,'group':str(value),'n':len(yg),'positive':int(yg.sum()),'negative':int(len(yg)-yg.sum()),'interpretable':valid}
            row.update(metrics(yg,pg,t) if valid else {k:np.nan for k in ['roc_auc','ap','recall','specificity','precision','brier']})
            subgroup.append(row)
    sub=pd.DataFrame(subgroup);save_csv(sub,ROOT/f'artifacts/metrics/{name}_subgroups.csv')
    valid=sub[sub.interpretable]
    if len(valid):
        plt.figure(figsize=(9,5));labels=valid.feature+'='+valid.group;plt.barh(labels,valid.recall,color='#2a788e');plt.xlabel('Sensitivity (groups meeting minimum counts)');savefig(f'{name}_subgroups')
    preds=pd.read_csv(ROOT/f'artifacts/predictions/{name}_{selected}_test.csv')
    merged=test.merge(preds[['row_id','probability','prediction','error_type']],on='row_id')
    # Confidence must refer to the operating label, not the most probable class:
    # with a threshold below0.5, an alert can have p<0.5.
    merged['confidence']=np.where(merged.prediction==1,merged.probability,1-merged.probability)
    cases=[]
    for error in ['FN','FP','TP','TN']:
        block=merged[merged.error_type==error].sort_values('confidence',ascending=False).head(2 if error in ['FN','FP'] else 1)
        cases.append(block)
    save_csv(pd.concat(cases),ROOT/f'artifacts/metrics/{name}_case_studies.csv')
    save_csv(merged.groupby('error_type').agg(n=('y','size'),mean_probability=('probability','mean'),mean_assigned_class_probability=('confidence','mean')).reset_index(),ROOT/f'artifacts/metrics/{name}_error_summary.csv')
    nums=['age','trestbps','chol','thalach','oldpeak'] if name=='heart' else ['time_in_hospital','num_medications','number_inpatient','number_emergency','num_lab_procedures']
    save_csv(merged.groupby('error_type')[nums].mean().reset_index(),ROOT/f'artifacts/metrics/{name}_error_features.csv')
    merged['missing_count']=merged[feature_columns(df,name)].isna().sum(axis=1)
    save_csv(merged.groupby('error_type').agg(n=('y','size'),missing_count_mean=('missing_count','mean'),
             any_missing_rate=('missing_count',lambda x:float((x>0).mean()))).reset_index(),ROOT/f'artifacts/metrics/{name}_error_missingness.csv')
    ablation=pd.DataFrame(rows);save_csv(ablation,ROOT/f'artifacts/metrics/{name}_ablation.csv')
    sample=test.sample(min(len(test),2000),random_state=47)
    with threadpool_limits(limits=2):
        imp=permutation_importance(art['pipeline'],sample,sample.y,scoring='roc_auc' if name=='heart' else 'average_precision',n_repeats=5,random_state=47,n_jobs=1)
    important=pd.DataFrame({'feature':test.columns,'importance':imp.importances_mean,'sd':imp.importances_std})
    important=important[important.feature.isin(feature_columns(df,name,art['feature_set']))].sort_values('importance',ascending=False)
    save_csv(important,ROOT/f'artifacts/metrics/{name}_permutation_importance.csv')
    plt.figure(figsize=(8,5));top=important.head(12).iloc[::-1];plt.barh(top.feature,top.importance,xerr=top.sd,color='#2a788e');plt.xlabel('Decrease in primary metric');plt.title('Permutation importance, post-hoc test analysis');savefig(f'{name}_importance')
    if art['kind']=='logistic':
        prep=art['pipeline'].named_steps['prepare'];coeff=art['pipeline'].named_steps['model'].coef_[0]
        encoded=prep.get_feature_names_out()
        save_csv(pd.DataFrame({'feature':encoded,'coefficient':coeff,'odds_ratio':np.exp(np.clip(coeff,-30,30)),
            'interpretation':['numerical: per 1 train SD; conditional association' if f.startswith('num__') else 'full one-hot: category contrast requires coefficient difference, exp(coef) alone not reference OR' for f in encoded]}),ROOT/f'artifacts/metrics/{name}_logistic_coefficients.csv')
    if name=='readmission':
        topk=[];order=np.argsort(-p,kind='stable')
        for fraction in [.05,.1,.2]:
            k=max(1,int(np.ceil(len(y)*fraction)));found=int(y[order[:k]].sum())
            topk.append({'fraction':fraction,'k':k,'positive_found':found,'precision_at_k':found/k,'recall_at_k':found/y.sum()})
        save_csv(pd.DataFrame(topk),ROOT/'artifacts/metrics/readmission_topk.csv')
    else:
        conformal()
        from scipy.stats import binomtest
        chosen=metrics(y,p,t);interval=binomtest(chosen['tp'],chosen['tp']+chosen['fn']).proportion_ci(confidence_level=.95,method='exact')
        dump_json(ROOT/'artifacts/metrics/heart_sensitivity_exact_ci.json',{'tp':chosen['tp'],'positives':chosen['tp']+chosen['fn'],
             'point':chosen['recall'],'lower':interval.low,'upper':interval.high,'method':'Clopper-Pearson 95% binomial exact; conditional on iid positive observations',
             'note':'Bootstrap recall interval can collapse when no false negatives; finite n does not establish 100% population sensitivity'})
    card=f'''# Model card: {name}

Selected: **{selected}**, selected exclusively by development CV. Feature set: {art['feature_set']}. Threshold {t:.6f}; calibration selected: {art['calibration_selected']}.

Fit/calibration/threshold/test roles: data/splits/{name}.json. No test tuning. Numerical/full-feature comparisons labelled in metrics. Test metrics: artifacts/metrics/{name}_test_metrics.csv; confidence intervals fixed model only, clustered by patient where needed. Predictions and artifact hashes saved.

Intended use: course prototype, {'diagnostic classification after tests available' if name=='heart' else 'encounter-level risk ranking at discharge'}. Not clinically validated. Historical data, label noise, population shift, missing external outcomes, limited Heart sample. Do not infer causality from coefficients/importance.
'''
    (ROOT/f'reports/model_cards/{name}.md').write_text(card,encoding='utf-8')
    dump_json(marker,{'selected':selected,'test_n':len(test),'bootstrap_repetitions':B,'test_used_for_selection':False,'artifact_hash':sha256(ROOT/f'artifacts/models/{name}_selected.joblib')})
    print(f'{name}: assessment saved ({selected})',flush=True)

