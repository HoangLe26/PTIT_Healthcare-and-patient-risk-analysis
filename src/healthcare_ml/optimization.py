import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from .common import ROOT,dump_json,save_csv
from .data import load_dataset,HEART_NUM
import pandas as pd

def objective_gradient(X,y,w,b,l2=0):
    z=X@w+b;p=expit(z)
    loss=np.mean(np.logaddexp(0,z)-y*z)+.5*l2*np.dot(w,w)
    return float(loss),X.T@(p-y)/len(y)+l2*w,float(np.mean(p-y))

def fit_numpy(X,y,rate=.1,steps=2000,l2=0):
    w=np.zeros(X.shape[1]);b=0.;losses=[]
    for _ in range(steps):
        loss,gw,gb=objective_gradient(X,y,w,b,l2)
        losses.append(loss);w-=rate*gw;b-=rate*gb
    return w,b,losses

def run_optimization():
    df,split=load_dataset('heart');ids=split['fit'];X=df.iloc[ids][HEART_NUM];y=df.iloc[ids].y.to_numpy()
    a,b=train_test_split(np.arange(len(X)),test_size=.25,stratify=y,random_state=45)
    imputer=SimpleImputer(strategy='median');scaler=StandardScaler()
    tr=scaler.fit_transform(imputer.fit_transform(X.iloc[a]));va=scaler.transform(imputer.transform(X.iloc[b]))
    rows=[]
    for rate in [.01,.1,1.,10.]:
        w,inter,loss=fit_numpy(tr,y[a],rate,1500)
        for i,l in enumerate(loss):rows.append({'rate':rate,'iteration':i,'train_loss':l})
    save_csv(pd.DataFrame(rows),ROOT/'artifacts/metrics/optimization_loss.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5))
    table=pd.DataFrame(rows)
    for rate,block in table.groupby('rate'):ax.plot(block.iteration,block.train_loss,label=f'learning rate={rate}')
    ax.set(xlabel='Iteration',ylabel='Mean Bernoulli log loss',ylim=(max(0,table.train_loss.min()-.05),min(3,table.train_loss.max()+.05)),title='Logistic Regression NumPy: optimization loss')
    ax.legend();fig.tight_layout();fig.savefig(ROOT/'artifacts/figures/optimization_loss.png',dpi=180);plt.close(fig)
    full=scaler.fit_transform(imputer.fit_transform(X))
    w,inter,loss=fit_numpy(full,y,.3,15000)
    sklearn=LogisticRegression(penalty=None,solver='lbfgs',tol=1e-10,max_iter=10000).fit(full,y)
    delta=np.max(np.abs(expit(full@w+inter)-sklearn.predict_proba(full)[:,1]))
    rng=np.random.default_rng(42);mini=rng.normal(size=(12,3));target=rng.integers(0,2,12);weights=rng.normal(size=3);intercept=.2
    _,g,gb=objective_gradient(mini,target,weights,intercept,.03);eps=1e-6;fd=[]
    for j in range(3):
        e=np.zeros(3);e[j]=eps
        fd.append((objective_gradient(mini,target,weights+e,intercept,.03)[0]-objective_gradient(mini,target,weights-e,intercept,.03)[0])/(2*eps))
    fdb=(objective_gradient(mini,target,weights,intercept+eps,.03)[0]-objective_gradient(mini,target,weights,intercept-eps,.03)[0])/(2*eps)
    error=max(np.max(np.abs(g-fd)),abs(gb-fdb))
    assert error<1e-6 and delta<1e-3
    dump_json(ROOT/'artifacts/metrics/optimization.json',{'gradient_max_error':error,'prediction_max_difference_sklearn':delta,
         'final_train_log_loss':loss[-1],'steps':15000,'features':HEART_NUM,'objective':'unpenalized mean Bernoulli NLL',
         'sklearn_solver':'lbfgs','note':'NumPy demo uses development fit only; not a replacement for final pipeline'})

