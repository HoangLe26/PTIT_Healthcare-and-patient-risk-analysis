import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from .data import feature_columns,HEART_NUM,READ_NUM

def diagnostic_group(value):
    if pd.isna(value):return '__MISSING__'
    s=str(value)
    if s.startswith(('V','E')):return 'supplementary'
    try:v=float(s)
    except ValueError:return 'unknown'
    if 390<=v<460 or int(v)==785:return 'circulatory'
    if 460<=v<520 or int(v)==786:return 'respiratory'
    if 520<=v<580 or int(v)==787:return 'digestive'
    if 250<=v<251:return 'diabetes'
    if 800<=v<1000:return 'injury'
    if 710<=v<740:return 'musculoskeletal'
    if 580<=v<630 or int(v)==788:return 'genitourinary'
    if 140<=v<240:return 'neoplasms'
    return 'other'

class Features(BaseEstimator,TransformerMixin):
    def __init__(self,columns,engineered=False):self.columns=columns;self.engineered=engineered
    def fit(self,X,y=None):
        assert not {'patient_nbr','encounter_id','row_id','y','readmitted','num'}&set(self.columns)
        return self
    def transform(self,X):
        out=X[self.columns].copy()
        if self.engineered:
            for c in ['diag_1','diag_2','diag_3']:out[c]=out[c].map(diagnostic_group)
            out['prior_visits']=out[['number_outpatient','number_emergency','number_inpatient']].sum(axis=1)
        return out

def pipeline_for(df,name,model,feature_set='full',scaled=True,weighted=False):
    cols=feature_columns(df,name,feature_set)
    numerical=[c for c in cols if c in (HEART_NUM if name=='heart' else READ_NUM)]
    if feature_set=='engineered':numerical=numerical+['prior_visits']
    categorical=[c for c in cols if c not in numerical]
    num_steps=[('impute',SimpleImputer(strategy='median',keep_empty_features=True))]
    if scaled:num_steps.append(('scale',StandardScaler()))
    branches=[('num',Pipeline(num_steps),numerical)]
    if categorical:branches.append(('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),
        ('onehot',OneHotEncoder(handle_unknown='ignore',min_frequency=10 if name=='readmission' else None,sparse_output=False))]),categorical))
    prep=ColumnTransformer(branches,remainder='drop')
    weight='balanced' if weighted else None
    estimators={
       'dummy':DummyClassifier(strategy='prior'),
       'logistic':LogisticRegression(max_iter=2000,solver='liblinear',class_weight=weight,random_state=42),
       'knn':KNeighborsClassifier(n_jobs=2),
       'gaussian_nb':GaussianNB(),
       'qda':QuadraticDiscriminantAnalysis(reg_param=.1),
       'random_forest':RandomForestClassifier(n_estimators=100,max_depth=12,min_samples_leaf=10,n_jobs=2,random_state=42,class_weight=weight),
       'xgboost':XGBClassifier(n_estimators=160,max_depth=3,learning_rate=.06,subsample=.85,colsample_bytree=.85,
                    reg_lambda=2,n_jobs=2,tree_method='hist',random_state=42,eval_metric='logloss')}
    return Pipeline([('features',Features(cols,feature_set=='engineered')),('prepare',prep),('model',estimators[model])])

def search_space(model):
    return {
      'dummy':{'model__strategy':['prior','most_frequent']},
      'logistic':{'model__C':[.01,.1,1,10,100],'model__penalty':['l1','l2']},
      'knn':{'model__n_neighbors':[3,5,7,11,15,21],'model__weights':['uniform','distance'],'model__p':[1,2]},
      'gaussian_nb':{'model__var_smoothing':[1e-9,1e-7,1e-5,.001,.1]},
      'qda':{'model__reg_param':[.01,.1,.3,.6,.9]},
      'random_forest':{'model__max_depth':[8,12,18],'model__min_samples_leaf':[5,15,30],
                       'model__max_features':['sqrt',.6],'model__n_estimators':[80,120]},
      'xgboost':{'model__max_depth':[2,3,5],'model__learning_rate':[.03,.07,.12],
                 'model__n_estimators':[100,180],'model__reg_lambda':[1,5]}}[model]

def model_specs(name):
    specs=[('dummy','dummy','full',True,False),('logistic','logistic','full',True,False),
           ('logistic_balanced','logistic','full',True,True),('gaussian_nb','gaussian_nb','numeric',True,False),
           ('xgboost','xgboost','full',False,False)]
    if name=='heart':
        specs += [('knn','knn','full',True,False),('knn_unscaled','knn','full',False,False),
          ('qda','qda','numeric',True,False),('logistic_numeric','logistic','numeric',True,False),
          ('logistic_restricted','logistic','restricted',True,False)]
    else:specs += [('random_forest','random_forest','full',False,False),('logistic_engineered','logistic','engineered',True,False)]
    return specs

