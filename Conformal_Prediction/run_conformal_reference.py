"""Run with python run_conformal_reference.py from any directory.

Reference models with fixed parameters. Members can pass probabilities from
their final models to conformal_prediction.summarize using the same subsets.
"""
import json
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from conformal_prediction import summarize

ROOT = Path(__file__).resolve().parent
NUMERIC = {
    "heart": "age trestbps chol thalach oldpeak",
    "readmission": "time_in_hospital num_lab_procedures num_procedures num_medications number_outpatient number_emergency number_inpatient number_diagnoses",
}


def run(dataset):
    df = pd.read_csv(ROOT / f"data/processed/{dataset}.csv", keep_default_na=False,
                     na_values=[""], low_memory=False)
    split = json.loads((ROOT / f"data/splits/{dataset}.json").read_text())
    meta = pd.read_csv(ROOT / f"data/metadata/{dataset}_feature_availability.csv")
    cols = meta.loc[meta.include, "feature"].tolist()
    numeric = NUMERIC[dataset].split()
    categorical = [c for c in cols if c not in numeric]
    df[categorical] = df[categorical].fillna("__missing__").astype(str)
    train = df.iloc[split["fit"]]
    cal, test = [df.iloc[split[key]] for key in ("calibration", "test")]
    if dataset == "readmission":
        cal, test = [s.sort_values("encounter_id").drop_duplicates("patient_nbr")
                     for s in (cal, test)]
        assert set(cal.patient_nbr).isdisjoint(test.patient_nbr)
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    x = prep.fit_transform(train[cols])
    xc, xt = prep.transform(cal[cols]), prep.transform(test[cols])
    rows = []
    for name, model in [
        ("KNN_k15_uniform", KNeighborsClassifier(n_neighbors=15)),
        ("Logistic_L2_C0.1", LogisticRegression(C=0.1, max_iter=2000)),
        ("Tree_depth5_leaf20", DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42)),
    ]:
        model.fit(x, train.y)
        pc, pt = [model.predict_proba(z)[:, list(model.classes_).index(1)] for z in (xc, xt)]
        for row in summarize(cal.y, pc, test.y, pt):
            rows.append(dict(dataset=dataset, model=name, **row))
    return rows


if __name__ == "__main__":
    results = pd.DataFrame(run("heart") + run("readmission"))
    results.to_csv(ROOT / "conformal_reference_results.csv", index=False)
    print(results.to_string(index=False))
