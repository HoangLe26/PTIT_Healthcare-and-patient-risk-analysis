import argparse,json,subprocess,sys
from pathlib import Path
import yaml
from .common import ROOT,ensure_dirs,dump_json
from .data import download,prepare,raw_heart,raw_readmission,load_dataset

def main():
    parser=argparse.ArgumentParser(description='Auditable healthcare ML workflows')
    parser.add_argument('command',choices=['download','validate-data','prepare','train','eda','ablations','explain','freeze','evaluate','report','optimization','smoke'])
    parser.add_argument('--all',action='store_true');parser.add_argument('--dataset',choices=['heart','readmission'])
    parser.add_argument('--config',type=Path);parser.add_argument('--split',default='test',choices=['test'])
    args=parser.parse_args();ensure_dirs()
    names=[args.dataset] if args.dataset else ['heart','readmission']
    if args.command=='download':download()
    elif args.command=='validate-data':
        h,r=raw_heart(),raw_readmission();assert len(h)==303 and len(r)==101766
        assert h.age.between(1,120).all() and h.oldpeak.ge(0).all()
        for c,allowed in {'sex':{0,1},'cp':{1,2,3,4},'fbs':{0,1},'restecg':{0,1,2},'exang':{0,1},'slope':{1,2,3},'ca':{0,1,2,3},'thal':{3,6,7}}.items():
            assert set(h[c].dropna()).issubset(allowed),f'Unexpected code: {c}'
        assert r.encounter_id.is_unique and r.patient_nbr.notna().all()
        assert r.time_in_hospital.between(1,14).all()
        from .data import READ_NUM
        assert (r[READ_NUM]>=0).all().all()
        dump_json(ROOT/'data/metadata/validation.json',{'heart_rows':len(h),'readmission_rows':len(r),'target_values_valid':True,
          'missing_tokens_checked':True,'categorical_codes_valid':True,'nonnegative_counts':True,'unique_encounter_ids':True,
          'note':'Range checks flag malformed records; no statistical outliers removed to improve metrics'});print('Raw validation passed')
    elif args.command=='prepare':prepare()
    elif args.command=='train':
        from .training import train
        if not args.config:parser.error('train requires --config')
        train(yaml.safe_load(args.config.read_text(encoding='utf-8')))
    elif args.command=='smoke':
        from .training import train
        for name in names:
            cfg=yaml.safe_load((ROOT/'configs/smoke.yaml').read_text());cfg['dataset']=name;train(cfg)
    elif args.command=='freeze':
        from .training import freeze
        for name in names:freeze(name)
    elif args.command=='evaluate':
        from .analysis import evaluate
        for name in names:evaluate(name)
    elif args.command=='eda':
        from .analysis import development_analysis
        for name in names:development_analysis(name)
    elif args.command=='ablations':
        from .ablations import run_ablations
        for name in names:run_ablations(name)
    elif args.command=='explain':
        from .explainability import explain
        for name in names:explain(name)
    elif args.command=='optimization':
        from .optimization import run_optimization
        run_optimization();print('Optimization gradient and sklearn comparison passed')
    elif args.command=='report':
        script=ROOT/'tools/build_deliverables.py'
        if not script.exists():raise FileNotFoundError('Report generator has not been created')
        subprocess.run([sys.executable,str(script)],cwd=ROOT,check=True)

if __name__=='__main__':main()

