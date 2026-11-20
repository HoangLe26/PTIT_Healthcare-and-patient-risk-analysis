"""Post-hoc explanations; fitted background comes only from fit development."""
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .common import ROOT,save_csv,dump_json
from .data import load_dataset
from .analysis import savefig

def explain(name):
    import shap
    df,split=load_dataset(name);art=joblib.load(ROOT/f'artifacts/models/{name}_selected.joblib')
    pipeline=art['pipeline'];background=df.iloc[split['fit']].sample(min(80,len(split['fit'])),random_state=48)
    cases=pd.read_csv(ROOT/f'artifacts/metrics/{name}_case_studies.csv')
    test=df[df.row_id.isin(cases.row_id)].copy()
    preprocess=pipeline[:-1];xb=preprocess.transform(background);xt=preprocess.transform(test)
    features=pipeline.named_steps['prepare'].get_feature_names_out();est=pipeline.named_steps['model'];start=time.perf_counter()
    background_source='explicit fit development sample'
    method='SHAP package'
    try:
        if art['kind']=='logistic':
            explainer=shap.LinearExplainer(est,xb);values=np.asarray(explainer.shap_values(xt));base=float(np.ravel(explainer.expected_value)[0]);units='raw log-odds (before post-hoc calibration)'
        elif art['kind'] in ['xgboost','random_forest']:
            try:
                explainer=shap.TreeExplainer(est,data=xb,feature_perturbation='interventional');values=np.asarray(explainer.shap_values(xt))
                if values.ndim==3:values=values[:,:,1]
                base=float(np.ravel(explainer.expected_value)[-1]);units='raw model output; before post-hoc calibration'
            except ValueError as error:
                if art['kind']!='xgboost':raise
                # XGBoost3.x vector base_score is not parsed by SHAP0.49.
                # Native TreeSHAP preserves the original model and fitted tree cover.
                from xgboost import DMatrix
                contributions=est.get_booster().predict(DMatrix(xt),pred_contribs=True)
                values=contributions[:,:-1];base=float(contributions[0,-1])
                margins=est.predict(xt,output_margin=True)
                assert np.allclose(values.sum(axis=1)+base,margins,atol=1e-4)
                reconstruction_error=float(np.max(np.abs(values.sum(axis=1)+base-margins)))
                method='XGBoost native exact TreeSHAP (pred_contribs=True)'
                background_source='implicit training node cover from trees fitted only on fit development'
                units='raw log-odds; before post-hoc calibration'
        else:
            dump_json(ROOT/f'artifacts/metrics/{name}_shap_status.json',{'status':'skipped','reason':'Selected Gaussian/KNN model; permutation importance supplied without costly generic approximation'});return
        long=[]
        for i,row_id in enumerate(test.row_id):
            for j,f in enumerate(features):long.append({'row_id':row_id,'encoded_feature':f,'shap':float(values[i,j]),'value':float(xt[i,j]),'base_value':base,'units':units})
        detail=pd.DataFrame(long);save_csv(detail,ROOT/f'artifacts/metrics/{name}_local_shap.csv')
        # Recover original variables using known feature selector, longest prefix wins.
        originals=pipeline.named_steps['features'].columns.copy()
        if art['feature_set']=='engineered':originals+=['prior_visits']
        def group(f):
            raw=f.split('__',1)[-1]
            return next((c for c in sorted(originals,key=len,reverse=True) if raw==c or raw.startswith(c+'_')),raw)
        detail['feature']=detail.encoded_feature.map(group)
        aggregated=detail.groupby(['row_id','feature'],as_index=False).shap.sum()
        save_csv(aggregated,ROOT/f'artifacts/metrics/{name}_local_shap_grouped.csv')
        global_imp=aggregated.groupby('feature').shap.apply(lambda x:np.abs(x).mean()).sort_values(ascending=False)
        save_csv(global_imp.rename('mean_abs_local_shap').reset_index(),ROOT/f'artifacts/metrics/{name}_shap_case_importance.csv')
        fig,axes=plt.subplots(2,3,figsize=(14,8));axes=axes.ravel()
        for ax,(row_id,block) in zip(axes,aggregated.groupby('row_id')):
            block=block.iloc[np.argsort(-np.abs(block.shap.values))[:8]].sort_values('shap')
            error=cases.loc[cases.row_id==row_id,'error_type'].iloc[0]
            ax.barh(block.feature,block.shap,color=np.where(block.shap>=0,'#e5a030','#2a788e'));ax.set_title(f'row {row_id}: {error}',fontsize=9);ax.tick_params(labelsize=7);ax.axvline(0,color='gray',lw=.5)
        for ax in axes[len(test):]:ax.axis('off')
        fig.suptitle(f'{name}: local SHAP ({units})',fontsize=12);savefig(f'{name}_local_shap')
        dump_json(ROOT/f'artifacts/metrics/{name}_shap_status.json',{'status':'completed','model':art['model'],
            'background_n':len(background) if background_source.startswith('explicit') else None,
            'training_pool_n':len(split['fit']), 'additivity_max_abs_error':locals().get('reconstruction_error'),
            'case_n':len(test),'background_source':background_source,'method':method,'seed':48,'seconds':time.perf_counter()-start,'units':units,
            'limitation':'Case-selected mean absolute SHAP is not population-level importance. Correlated features and interventions not causal.'})
    except Exception as error:
        dump_json(ROOT/f'artifacts/metrics/{name}_shap_status.json',{'status':'failed','reason':repr(error),'fallback':'Permutation importance and logistic coefficients available'})
        print(f'SHAP {name} failed: {error}',flush=True)

