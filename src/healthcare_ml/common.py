from pathlib import Path
import hashlib, json, platform, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

def ensure_dirs():
    for d in ['data/raw/heart','data/raw/readmission','data/processed','data/splits','data/metadata',
              'artifacts/runs','artifacts/models','artifacts/predictions','artifacts/metrics','artifacts/figures',
              'reports/model_cards','notebooks','scripts','tools','app','tests']:
        (ROOT/d).mkdir(parents=True,exist_ok=True)

def dump_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    def cast(v):
        if isinstance(v,np.generic): return v.item()
        if isinstance(v,np.ndarray): return v.tolist()
        if isinstance(v,Path): return str(v)
        return str(v)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=cast),encoding='utf-8')

def load_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_csv(df,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);df.to_csv(path,index=False,encoding='utf-8-sig')

def environment():
    from importlib.metadata import distributions
    dump_json(ROOT/'artifacts/runs/environment.json',{
        'python':sys.version,'platform':platform.platform(),
        'packages':{d.metadata['Name']:d.version for d in distributions()},
        'seed':42,'threads':2,'hardware':'16 GB RAM; 8 logical CPU; GTX1650 Max-Q 4GB (training on CPU)'})

