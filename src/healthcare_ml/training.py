import datetime,time,warnings,json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold,RandomizedSearchCV,cross_validate
from sklearn.base import clone
from sklearn.metrics import brier_score_loss
from threadpoolctl import threadpool_limits
from .common import ROOT,dump_json,load_json,sha256,save_csv,environment
from .data import load_dataset,assert_disjoint
from .models import pipeline_for,search_space,model_specs
from .calibration import SigmoidCalibrator
from .evaluation import select_threshold,metrics

def folds_for(df,ids,name,n_splits):
    X=df.iloc[ids];y=X.y;g=X.patient_nbr
    cv=StratifiedKFold(n_splits,shuffle=True,random_state=42) if name=='heart' else StratifiedGroupKFold(n_splits,shuffle=True,random_state=42)
    folds=list(cv.split(X,y,g if name!='heart' else None))
    for tr,va in folds:assert_disjoint(tr,va,g.reset_index(drop=True))
    return folds

def train(config):
    name=config['dataset'];mode=config.get('mode','full');smoke=mode=='smoke'
    prefix=ROOT/('artifacts/smoke' if smoke else 'artifacts')
    for part in ['metrics','models','runs']: (prefix/part).mkdir(parents=True,exist_ok=True)
    if not smoke and (ROOT/f'artifacts/models/{name}_freeze.json').exists():
        print(f'{name}: frozen artifact exists, reusing; create new explicit study to retrain',flush=True);return
    df,split=load_dataset(name);ids=np.array(split['fit'])
    if smoke and name=='readmission':
        # choose whole patients without looking at test
        groups=df.iloc[ids].patient_nbr.unique()[:1500];ids=ids[df.iloc[ids].patient_nbr.isin(groups)]
    folds=folds_for(df,ids,name,config['folds']);X=df.iloc[ids];y=X.y
    primary='roc_auc' if name=='heart' else 'average_precision'
    cv_rows=[];all_trials=[];models={};warnings_log=[]
    t0=time.perf_counter()
    specs=model_specs(name)
    if smoke:specs=[s for s in specs if s[0] in ['dummy','logistic','gaussian_nb']]
    with threadpool_limits(limits=config.get('threads',2)):
        for label,kind,feature_set,scaled,weighted in specs:
            cached=prefix/f'models/{name}_{label}.joblib'
            entry=prefix/f'runs/{name}_{label}.json'
            if cached.exists() and entry.exists():
                row=load_json(entry);cv_rows.append(row);models[label]=joblib.load(cached)
                trial_file=prefix/f'runs/{name}_{label}_trials.csv'
                if trial_file.exists():all_trials.extend(pd.read_csv(trial_file).to_dict('records'))
                print(f'{name}/{label}: resumed',flush=True);continue
            print(f'{name}/{label}: training {len(X)} fit rows, {len(folds)} folds',flush=True)
            model=pipeline_for(df,name,kind,feature_set,scaled,weighted)
            space=search_space(kind)
            if kind=='random_forest' and config.get('rf_safe_budget',False):
                space={'model__max_features':['sqrt'],'model__n_estimators':[80],
                       'model__max_depth':[8,12],'model__min_samples_leaf':[15,30]}
            if label.startswith('logistic'):space=dict(space,model__class_weight=['balanced' if weighted else None])
            # Reserve class weighting as a deliberate ablation, not a hidden search option
            trial_count=min(config['search_trials'],int(np.prod([len(v) for v in space.values()])))
            search=RandomizedSearchCV(model,space,n_iter=trial_count,scoring=primary,cv=folds,
                                      n_jobs=1,random_state=42,refit=True,error_score='raise',return_train_score=True)
            start=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');search.fit(X,y)
            warns=sorted(set(str(w.message) for w in caught));warnings_log.extend({'model':label,'warning':w} for w in warns)
            elapsed=time.perf_counter()-start;best=search.best_estimator_
            ca=df.iloc[split['calibration']];th=df.iloc[split['threshold']]
            raw_ca=best.predict_proba(ca)[:,1];raw_th=best.predict_proba(th)[:,1]
            calibrator=SigmoidCalibrator().fit(raw_ca,ca.y)
            calibrated=calibrator.predict(raw_th)
            use_cal=label!='dummy' and brier_score_loss(th.y,calibrated)<brier_score_loss(th.y,raw_th)
            chosen_prob=calibrated if use_cal else raw_th
            threshold=select_threshold(th.y,chosen_prob)
            policies={'f2':threshold,'recall85':select_threshold(th.y,chosen_prob,'recall')}
            policies.update({f'cost_{r}':select_threshold(th.y,chosen_prob,'cost',r) for r in [1,2,5]})
            artifact={'pipeline':best,'calibrator':calibrator if use_cal else None,'sigmoid_candidate':calibrator,
                      'threshold':threshold,'policies':policies,'feature_set':feature_set,'model':label,
                      'kind':kind,'dataset':name,'calibration_selected':bool(use_cal),
                      'train_seconds':elapsed,'config':config,'params':search.best_params_}
            row={'model':label,'feature_set':feature_set,'mean_score':float(search.best_score_),
                 'std_score':float(search.cv_results_['std_test_score'][search.best_index_]),'params':json.dumps(search.best_params_),
                 'train_seconds':elapsed,'calibration_selected':bool(use_cal),'threshold':threshold,
                 'threshold_brier_raw':brier_score_loss(th.y,raw_th),'threshold_brier_calibrated':brier_score_loss(th.y,calibrated)}
            dump_json(entry,row);joblib.dump(artifact,cached,compress=3)
            trial=pd.DataFrame(search.cv_results_)
            trial['model']=label;trial['feature_set']=feature_set
            save_csv(trial,prefix/f'runs/{name}_{label}_trials.csv');all_trials.extend(trial.to_dict('records'))
            cv_rows.append(row);models[label]=artifact
            print(f'{label}: CV={row["mean_score"]:.4f}, threshold={threshold:.4f}, {elapsed:.1f}s',flush=True)
    table=pd.DataFrame(cv_rows);save_csv(table,prefix/f'metrics/{name}_cv_results.csv')
    save_csv(pd.DataFrame(all_trials),prefix/f'metrics/{name}_search_trials.csv')
    dump_json(prefix/f'runs/{name}_warnings.json',warnings_log)
    preference=['logistic','gaussian_nb','knn','qda','logistic_numeric','logistic_restricted','logistic_engineered','xgboost','random_forest','logistic_balanced','knn_unscaled']
    eligible=table[table.model!='dummy'];top=eligible.mean_score.max()
    tied=eligible[eligible.mean_score>=top-.002].copy()
    tied['preference']=tied.model.map({k:i for i,k in enumerate(preference)})
    selected=tied.sort_values('preference').iloc[0].model
    joblib.dump(models[selected],prefix/f'models/{name}_selected.joblib',compress=3)
    dump_json(prefix/f'runs/{name}_training_manifest.json',{'selected':selected,'config':config,'primary_metric':primary,
              'fit_rows':len(X),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'elapsed_seconds':time.perf_counter()-t0,'processed_hash':sha256(ROOT/f'data/processed/{name}.csv'),
              'split_hash':sha256(ROOT/f'data/splits/{name}.json'),'selected_without_test':True})
    if not smoke:environment()
    print(f'{name} selected by DEVELOPMENT CV: {selected}',flush=True)

def freeze(name):
    path=ROOT/f'artifacts/models/{name}_freeze.json'
    if path.exists():return
    manifest=load_json(ROOT/f'artifacts/runs/{name}_training_manifest.json')
    models={p.name:sha256(p) for p in (ROOT/'artifacts/models').glob(f'{name}_*.joblib')}
    assert models
    dump_json(path,{'dataset':name,'models':models,'split_hash':manifest['split_hash'],
                    'processed_hash':manifest['processed_hash'],'selected':manifest['selected'],
                    'protocol_hash':sha256(ROOT/'reports/protocol.md'),
                    'pipeline_source_hashes':{f:sha256(ROOT/'src/healthcare_ml'/f) for f in ['models.py','data.py','calibration.py']},
                    'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})

def predict_artifact(artifact,X,calibrated=True):
    p=artifact['pipeline'].predict_proba(X)[:,1]
    if calibrated and artifact['calibrator'] is not None:p=artifact['calibrator'].predict(p)
    assert np.all(np.isfinite(p)) and np.all((p>=0)&(p<=1))
    return p

