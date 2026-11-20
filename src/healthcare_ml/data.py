import io, zipfile, urllib.request, datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from .common import ROOT,ensure_dirs,dump_json,load_json,sha256,save_csv

HEART_COLUMNS=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']
HEART_NUM=['age','trestbps','chol','thalach','oldpeak']
READ_NUM=['time_in_hospital','num_lab_procedures','num_procedures','num_medications','number_outpatient',
          'number_emergency','number_inpatient','number_diagnoses']
SOURCES={
 'heart':{'url':'https://archive.ics.uci.edu/static/public/45/heart+disease.zip',
          'page':'https://archive.ics.uci.edu/dataset/45/heart+disease','doi':'10.24432/C52P4X'},
 'readmission':{'url':'https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip',
                'page':'https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008','doi':'10.24432/C5230J'}}

def download():
    ensure_dirs();records=[]
    for name,src in SOURCES.items():
        dest=ROOT/f'data/raw/{name}/source.zip'
        if not dest.exists():
            req=urllib.request.Request(src['url'],headers={'User-Agent':'HealthcareCourseResearch/1.0'})
            with urllib.request.urlopen(req,timeout=120) as response:dest.write_bytes(response.read())
        with zipfile.ZipFile(dest) as archive:
            for member in archive.infolist():
                target=(dest.parent/member.filename).resolve()
                if not target.is_relative_to(dest.parent.resolve()):raise ValueError('Unsafe archive path')
            archive.extractall(dest.parent)
        records.append(dict(dataset=name,**src,license='CC BY 4.0',downloaded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            path=str(dest.relative_to(ROOT)),bytes=dest.stat().st_size,sha256=sha256(dest)))
    dump_json(ROOT/'data/metadata/sources.json',records)
    print('Downloaded/verified both UCI archives',flush=True)

def raw_heart():
    file=next((ROOT/'data/raw/heart').rglob('processed.cleveland.data'))
    df=pd.read_csv(file,header=None,names=HEART_COLUMNS,na_values=['?'],keep_default_na=False)
    df=df.apply(pd.to_numeric,errors='raise')
    assert len(df)==303 and set(df['num']).issubset({0,1,2,3,4})
    return df

def raw_readmission():
    file=next((ROOT/'data/raw/readmission').rglob('diabetic_data.csv'))
    # 'None' means no recorded test, not parser NA.
    df=pd.read_csv(file,na_values=['?'],keep_default_na=False,low_memory=False)
    assert len(df)==101766 and set(df['readmitted'])=={'<30','>30','NO'}
    return df

def read_dispositions():
    file=next((ROOT/'data/raw/readmission').rglob('IDS_mapping.csv'))
    lines=file.read_text(encoding='utf-8-sig').splitlines()
    start=next(i for i,l in enumerate(lines) if l.startswith('discharge_disposition_id'))
    rows=[]
    for line in lines[start+1:]:
        if not line.strip().strip(',') or line.startswith('admission_source_id'):break
        k,_,v=line.partition(',')
        if k.isdigit():rows.append((int(k),v.strip('"')))
    return dict(rows)

def assert_disjoint(a,b,groups):
    assert not set(a)&set(b),'row overlap'
    assert not set(groups.iloc[a])&set(groups.iloc[b]),'patient overlap'

def prepare():
    ensure_dirs()
    for name in SOURCES:
        raw=raw_heart() if name=='heart' else raw_readmission()
        df=raw.copy(); flow=[]
        def step(reason,mask):
            nonlocal df
            before=len(df);df=df.loc[mask].copy()
            flow.append({'step':len(flow)+1,'reason':reason,'rows_before':before,'rows_removed':before-len(df),'rows_after':len(df)})
        if name=='heart':
            df['y']=(df['num']>0).astype(int);df=df.drop(columns='num')
            df['row_id']=np.arange(len(df));df['patient_nbr']=df['row_id']
            dev,test=train_test_split(np.arange(len(df)),test_size=.2,stratify=df.y,random_state=42)
        else:
            mapping=read_dispositions()
            excluded=[k for k,v in mapping.items() if 'expired' in v.lower() or 'hospice' in v.lower()]
            assert excluded,'No expired/hospice IDs recognized'
            dump_json(ROOT/'data/metadata/discharge_mapping.json',{'mapping':mapping,'excluded':excluded})
            step('Loại expired/hospice theo IDS_mapping.csv',~df.discharge_disposition_id.isin(excluded))
            step('Loại gender Unknown/Invalid',df.gender!='Unknown/Invalid')
            df['y']=(df.readmitted=='<30').astype(int);df=df.drop(columns='readmitted')
            df['row_id']=df['encounter_id'];df=df.reset_index(drop=True)
            dev,test=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(df,df.y,df.patient_nbr))
        df=df.reset_index(drop=True)
        groups=df.patient_nbr
        assert_disjoint(dev,test,groups)
        # independent calibration and threshold partitions inside development
        if name=='heart':
            fit,hold=train_test_split(dev,test_size=.30,stratify=df.y.iloc[dev],random_state=43)
            cal,threshold=train_test_split(hold,test_size=.5,stratify=df.y.iloc[hold],random_state=44)
        else:
            fit_local,hold_local=next(StratifiedGroupKFold(4,shuffle=True,random_state=43).split(df.iloc[dev],df.y.iloc[dev],groups.iloc[dev]))
            fit=dev[fit_local];hold=dev[hold_local]
            ca,th=next(StratifiedGroupKFold(2,shuffle=True,random_state=44).split(df.iloc[hold],df.y.iloc[hold],groups.iloc[hold]))
            cal=hold[ca];threshold=hold[th]
        split={'fit':sorted(map(int,fit)),'calibration':sorted(map(int,cal)),
               'threshold':sorted(map(int,threshold)),'test':sorted(map(int,test)), 'development':sorted(map(int,dev))}
        for i,a in enumerate(['fit','calibration','threshold','test']):
            for b in ['fit','calibration','threshold','test'][i+1:]:assert_disjoint(split[a],split[b],groups)
        save_csv(df,ROOT/f'data/processed/{name}.csv');dump_json(ROOT/f'data/splits/{name}.json',split)
        summary={'dataset':name,'raw_rows':len(raw),'rows':len(df),'columns':len(df.columns),
                 'patients':int(df.patient_nbr.nunique()),'positive':int(df.y.sum()),'prevalence':float(df.y.mean()),
                 'missing':{c:int(df[c].isna().sum()) for c in df.columns},'flow':flow,
                 'splits':{k:{'rows':len(v),'patients':int(df.iloc[v].patient_nbr.nunique()),'positive':int(df.iloc[v].y.sum()),'prevalence':float(df.iloc[v].y.mean())} for k,v in split.items()},
                 'processed_sha256':sha256(ROOT/f'data/processed/{name}.csv'),'split_sha256':sha256(ROOT/f'data/splits/{name}.json')}
        dump_json(ROOT/f'data/metadata/{name}_summary.json',summary)
        save_csv(pd.DataFrame(flow),ROOT/f'data/metadata/{name}_cohort_flow.csv')
        dictionary=pd.DataFrame([{'feature':c,'dtype':str(raw[c].dtype),'missing':int(raw[c].isna().sum()),
            'examples':'; '.join(map(str,raw[c].dropna().unique()[:12]))} for c in raw.columns])
        save_csv(dictionary,ROOT/f'data/metadata/{name}_data_dictionary.csv')
        included=set(feature_columns(df,name,'full'))
        save_csv(pd.DataFrame([{'dataset':name,'feature':c,'available_at_prediction': c not in ['y','num','readmitted'],
            'include':c in included,'evidence':'UCI dictionary; current encounter available at discharge' if name=='readmission' else 'UCI Cleveland diagnostic variables',
            'reason':'ID/target/excluded by protocol' if c not in included else 'eligible by protocol'} for c in df.columns]),ROOT/f'data/metadata/{name}_feature_availability.csv')
        card=f'''# Data card: {name}

Nguồn: [{name}]({SOURCES[name]['page']}); DOI {SOURCES[name]['doi']}; license CC BY 4.0.

Raw {len(raw):,} hàng; cohort {len(df):,} hàng; {df.patient_nbr.nunique():,} đơn vị group; positive {df.y.sum():,} ({df.y.mean():.2%}).

Target: {'num > 0; bệnh hiện diện tại quá trình chẩn đoán, không tiên lượng tương lai' if name=='heart' else 'readmitted == <30; nguy cơ tại lúc xuất viện; NO không đảm bảo không tái nhập viện ngoài hệ thống'}.

Schema/mã categorical/missing: xem `{name}_data_dictionary.csv`, `{name}_feature_availability.csv` và raw documentation. Numerical Heart: age (years), trestbps (mmHg), chol (mg/dL), thalach (beats/min), oldpeak (ST depression). Numerical readmission: counts hoặc days theo tên biến. Missing ? chuyển NaN; None xét nghiệm giữ category. Không tự coi 0 là missing.

Các splits/counts: `{name}_summary.json`. Group splitting: {'row là đơn vị độc lập do không có patient identifier được xác minh' if name=='heart' else 'patient_nbr; mọi encounter cùng người thuộc một split'}. Train-only transformations, test không dùng chọn mô hình.

Duplicate raw cả hàng: {raw.duplicated().sum()}; không bỏ hàng chỉ vì giống features. Features không chứa ID/nhãn. Nhóm cohort loại có flow CSV. Dữ liệu lịch sử, selection bias, label uncertainty và giới hạn tổng quát hóa; không phải mô hình đã thẩm định lâm sàng.
'''
        (ROOT/f'data/metadata/{name}_data_card.md').write_text(card,encoding='utf-8')
        print(f'{name}: {len(df)} rows; splits {summary["splits"]}',flush=True)

def load_dataset(name):
    df=pd.read_csv(ROOT/f'data/processed/{name}.csv',keep_default_na=False,na_values=[''],low_memory=False)
    return df,load_json(ROOT/f'data/splits/{name}.json')

def feature_columns(df,name,feature_set='full'):
    if name=='heart':
        cols=[c for c in HEART_COLUMNS if c!='num']
        if feature_set=='numeric':return HEART_NUM
        if feature_set=='restricted':return [c for c in cols if c not in ['ca','thal']]
        return cols
    if feature_set=='numeric':return READ_NUM
    excluded={'encounter_id','patient_nbr','row_id','y','weight','payer_code','medical_specialty'}
    return [c for c in df.columns if c not in excluded]

