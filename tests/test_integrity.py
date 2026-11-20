import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from healthcare_ml.common import ROOT,load_json
from healthcare_ml.common import sha256
from healthcare_ml.data import load_dataset,feature_columns,assert_disjoint
from healthcare_ml.models import pipeline_for,diagnostic_group
from healthcare_ml.evaluation import cluster_indices,select_threshold,metrics
from healthcare_ml.optimization import objective_gradient
from healthcare_ml.training import folds_for,predict_artifact
import joblib

@pytest.mark.parametrize('name',['heart','readmission'])
def test_saved_split_disjoint(name):
    df,s=load_dataset(name)
    assert set(df.y.unique())=={0,1}
    parts=['fit','calibration','threshold','test']
    assert set().union(*(set(s[p]) for p in parts))==set(range(len(df)))
    for i,a in enumerate(parts):
        for b in parts[i+1:]:assert_disjoint(s[a],s[b],df.patient_nbr)
    assert not set(feature_columns(df,name))&{'row_id','patient_nbr','encounter_id','y','readmitted','num'}
    folds_for(df,np.array(s['fit']),name,3)

def test_none_preserved_and_mapping():
    df,_=load_dataset('readmission')
    assert 'None' in set(df.A1Cresult.dropna())
    m=load_json(ROOT/'data/metadata/discharge_mapping.json')
    assert set(m['excluded'])=={11,13,14,19,20,21}
    assert not set(df.discharge_disposition_id)&set(m['excluded'])
    assert diagnostic_group('250.13')=='diabetes'
    assert diagnostic_group('V45')=='supplementary'

@pytest.mark.parametrize('name',['heart','readmission'])
def test_pipeline_train_only_unknown_and_roundtrip(name,tmp_path):
    df,s=load_dataset(name);train=df.iloc[s['fit']].head(1000);test=df.iloc[s['test']].head(10).copy()
    model=pipeline_for(df,name,'logistic');model.fit(train,train.y)
    med=model.named_steps['prepare'].named_transformers_['num'].named_steps['impute'].statistics_.copy()
    if name=='readmission':test['race']='NEVER_SEEN_CATEGORY'
    else:test['cp']=999
    test['patient_nbr']=999999999;test['y']=1
    p=model.predict_proba(test)
    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1),1)
    assert np.array_equal(med,model.named_steps['prepare'].named_transformers_['num'].named_steps['impute'].statistics_)
    joblib.dump(model,tmp_path/'model.joblib');assert np.allclose(p,joblib.load(tmp_path/'model.joblib').predict_proba(test))

def test_bootstrap_groups_and_multiplicity():
    g=np.array([10,10,20,30,30,30]);ids=cluster_indices(g,np.random.default_rng(12));cnt=np.bincount(ids,minlength=len(g))
    assert cnt[0]==cnt[1] and cnt[3]==cnt[4]==cnt[5]
    draws=sum(cnt[np.array([0,2,3])]);assert draws==3
    assert any(np.max(np.bincount(cluster_indices(g,np.random.default_rng(i)),minlength=6))>1 for i in range(10))

def test_threshold_expected_and_metrics():
    y=np.array([0,0,1,1]);p=np.array([.1,.2,.6,.8]);t=select_threshold(y,p)
    assert t==.6 and metrics(y,p,t)['f2']==1
    assert np.isnan(metrics([0,1],[.1,.1],.5)['precision'])
    assert metrics([0,1],[0,1],1)['tp']==1

def test_gradient():
    X=np.array([[1.,2.],[-2.,.1],[.5,-.3]]);y=np.array([1,0,1]);w=np.array([.2,-.1]);eps=1e-6
    _,gw,gb=objective_gradient(X,y,w,.3,.04)
    for j in range(2):
        e=np.zeros(2);e[j]=eps
        fd=(objective_gradient(X,y,w+e,.3,.04)[0]-objective_gradient(X,y,w-e,.3,.04)[0])/(2*eps)
        assert abs(fd-gw[j])<1e-7
    loss,g,b=objective_gradient(np.array([[1e4,1e4]]),np.array([0]),np.array([1e4,1e4]),0)
    assert np.isfinite(loss) and np.isfinite(g).all()

@pytest.mark.parametrize('name',['heart','readmission'])
def test_smoke_predictions(name):
    artifact=joblib.load(ROOT/f'artifacts/smoke/models/{name}_selected.joblib')
    df,s=load_dataset(name);p=predict_artifact(artifact,df.iloc[s['threshold']].head(15))
    assert np.all((p>=0)&(p<=1))

@pytest.mark.parametrize('name',['heart','readmission'])
@pytest.mark.skipif(not (ROOT/'artifacts/models/readmission_freeze.json').exists(),reason='Full run not frozen yet')
def test_freeze_threshold_and_independent_assessment(name):
    frozen=load_json(ROOT/f'artifacts/models/{name}_freeze.json')
    for file,digest in frozen['models'].items():assert sha256(ROOT/'artifacts/models'/file)==digest
    df,s=load_dataset(name);art=joblib.load(ROOT/f'artifacts/models/{name}_selected.joblib')
    th=df.iloc[s['threshold']];p=predict_artifact(art,th)
    assert art['threshold']==select_threshold(th.y,p)
    cv=pd.read_csv(ROOT/f'artifacts/metrics/{name}_cv_results.csv')
    assert frozen['selected'] in set(cv.loc[cv.mean_score>=cv.loc[cv.model!='dummy','mean_score'].max()-.002,'model'])
    for split in ['calibration','threshold','test']:assert not set(df.iloc[s['fit']].patient_nbr)&set(df.iloc[s[split]].patient_nbr)
    evaluated=load_json(ROOT/f'artifacts/runs/{name}_evaluated.json')
    assert evaluated['test_used_for_selection'] is False
    preds=pd.read_csv(ROOT/f'artifacts/predictions/{name}_{frozen["selected"]}_test.csv')
    assert set(preds.row_id)==set(df.iloc[s['test']].row_id)
    assert np.allclose(preds.probability,predict_artifact(art,df.iloc[s['test']])), 'Saved predictions do not match frozen pipeline'


