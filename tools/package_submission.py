"""Create a portable private submission archive; excludes environments and temp files."""
from pathlib import Path
import zipfile,json
from healthcare_ml.common import ROOT,sha256,dump_json

def package():
    output=ROOT/'output';output.mkdir(exist_ok=True)
    path=output/'healthcare_ml_submission.zip'
    folders=['src','configs','scripts','tools','tests','notebooks','reports','artifacts','data','Slide']
    files=[ROOT/name for name in ['README.md','TASKS.md','AI_AGENT_HEALTHCARE_ML_END_TO_END.md','pyproject.toml','requirements-lock.txt','.gitignore']]
    for folder in folders:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.pyc'))
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in files:z.write(file,file.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert 'reports/final_report.pdf' in z.namelist()
        assert 'reports/presentation.pptx' in z.namelist()
    dump_json(output/'submission_manifest.json',{'file':path.name,'sha256':sha256(path),'bytes':path.stat().st_size,'file_count':len(files),
        'excluded':['.venv','tmp','__pycache__'],'purpose':'Private local handoff; not submitted or published'})
    print(f'Packaged {len(files)} files, {path.stat().st_size/1024**2:.1f} MB')

if __name__=='__main__':package()

