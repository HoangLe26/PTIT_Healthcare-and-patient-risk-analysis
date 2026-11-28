"""Run from the project root: python -m algorithm.Logistic.run_logistic"""
from pathlib import Path
from time import perf_counter
import warnings

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from benchmark.cohort import cohort_paths, load_cohort, write_json
from benchmark.evaluation import build_test_prediction_table, compute_binary_metrics
from benchmark.thresholds import find_optimal_f2_threshold
from Conformal_Prediction.conformal_prediction import calibrate, summarize

OUTPUT = Path(__file__).resolve().parent / "artifacts"


def run(dataset):
    start = perf_counter()
    out = OUTPUT / dataset
    out.mkdir(parents=True, exist_ok=True)
    df, split, features, cats = load_cohort(dataset)
    df[cats] = df[cats].astype("string").fillna("__missing__").astype(str)
    numeric = [c for c in features if c not in cats]
    prep = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), cats),
    ])
    pipe = Pipeline([("prep", prep), ("lr", LogisticRegression(
        solver="liblinear", random_state=42, max_iter=2000))])
    fit = df.iloc[split["fit"]]
    paths = cohort_paths(dataset)
    fold = pd.read_csv(paths["cv"]).set_index("row_position").loc[fit.index, "fold"].to_numpy()
    folds = [((fold != k).nonzero()[0], (fold == k).nonzero()[0]) for k in sorted(set(fold))]
    metric = "roc_auc" if dataset == "heart" else "average_precision"
    search = GridSearchCV(pipe, {"lr__C": [0.01, 0.1, 1], "lr__penalty": ["l2", "l1"]},
                          scoring=metric, cv=folds, error_score="raise")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        search.fit(fit[features], fit.y)
    pd.DataFrame(search.cv_results_).to_csv(out / "cv_results.csv", index=False)
    model = search.best_estimator_
    joblib.dump(model, out / "model.joblib")
    def probability(frame):
        return model.predict_proba(frame[features])[:, list(model.classes_).index(1)]

    threshold_data = df.iloc[split["threshold"]]
    threshold = float(find_optimal_f2_threshold(threshold_data.y, probability(threshold_data))["best_threshold"])
    selected = pd.read_csv(paths["conformal"])
    cal = df.iloc[selected.loc[selected.split.eq("calibration"), "row_position"]]
    pc = probability(cal)
    q = {str(level): float(calibrate(cal.y, pc, level)) for level in (0.90, 0.95)}
    frozen = dict(params=search.best_params_, cv_metric=metric, cv_mean=search.best_score_,
                  threshold=threshold, conformal_q=q, development_seconds=perf_counter()-start)
    write_json(out / "pretest_freeze.json", frozen)
    test = df.iloc[split["test"]]
    pt = probability(test)
    pred = build_test_prediction_table(test.row_id.to_numpy(), test.y, pt, threshold)
    pred.to_csv(out / "test_predictions.csv", index=False)
    ct = selected.loc[selected.split.eq("test"), "row_position"]
    pos = test.index.get_indexer(ct)
    assert (pos >= 0).all()
    rows = pd.DataFrame(summarize(cal.y, pc, test.y.iloc[pos], pt[pos])).assign(
        dataset=dataset, model="Logistic Regression", C=model["lr"].C, penalty=model["lr"].penalty)
    rows.to_csv(out / "conformal_results.csv", index=False)
    metrics = compute_binary_metrics(test.y, pt, threshold=threshold)
    write_json(out / "run_summary.json", dict(**frozen, test_metrics=metrics))
    print(dataset, search.best_params_, metric, metrics[metric], flush=True)
    return rows


if __name__ == "__main__":
    pd.concat([run("heart"), run("readmission")]).to_csv(
        OUTPUT / "conformal_4rows_for_hoang.csv", index=False)
