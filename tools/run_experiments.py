"""Resumable orchestration; all outputs are real executions."""
import subprocess,sys,os,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.update(PYTHONUTF8='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
steps=[['train','--config','configs/heart.yaml'],['train','--config','configs/readmission.yaml'],
       ['eda','--all'],['ablations','--all'],['optimization'],['freeze','--all'],['evaluate','--all']]
for i,args in enumerate(steps):
    log=ROOT/f'artifacts/runs/stage_{i}_{args[0]}.log';log.parent.mkdir(parents=True,exist_ok=True)
    print('STAGE',i,*args,flush=True)
    with log.open('w',encoding='utf-8') as stream:
        p=subprocess.Popen([sys.executable,'-X','utf8','-u','-m','healthcare_ml.cli',*args],cwd=ROOT,
                           stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        for line in p.stdout:stream.write(line);stream.flush();print(line,end='',flush=True)
        if p.wait():raise SystemExit(f'Stage failed; inspect {log}')
print('All experiments completed',flush=True)

