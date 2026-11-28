"""Run Tùng's two KNN experiments on the group's frozen splits."""

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

from algorithm.KNN.neighbors import (
    BATCH_SIZE, BLAS_THREADS, K_VALUES, WEIGHTS, EuclideanNeighbors, positive_probability, vote_weights,
)
from benchmark.cohort import DATA, EXPECTED, ROOT, cohort_paths, load_cohort, require, sha256, write_json
from benchmark.conformal import assign_conformal_prediction_labels
from benchmark.evaluation import build_test_prediction_table, compute_binary_metrics, extract_extreme_errors
from benchmark.run_benchmark import verify_sample_alignment
from benchmark.thresholds import find_optimal_f2_threshold
from Conformal_Prediction.conformal_prediction import calibrate, summarize

OUTPUT = Path(__file__).resolve().parent / "artifacts"
LEVELS = (0.90, 0.95)
METRICS = {"heart": ("roc_auc", roc_auc_score),
           "readmission": ("average_precision", average_precision_score)}


def prepare(dataset, audit, output):
    start = perf_counter()
    paths = cohort_paths(dataset)
    require({name: sha256(path) for name, path in paths.items()} == audit["sha256"],
            f"{dataset}: cohort, metadata or split manifest hash mismatch")
    frame, splits, features, categorical = load_cohort(dataset)
    numeric = [col for col in features if col not in categorical]
    fit = frame.iloc[splits["fit"]]
    folds = pd.read_csv(paths["cv"]).set_index("row_position").loc[fit.index, "fold"]
    selected = pd.read_csv(paths["conformal"])
    cal = frame.iloc[selected.loc[selected["split"].eq("calibration"), "row_position"]]

    scores = {(k, weighting): {} for k in K_VALUES for weighting in WEIGHTS}
    metric_name, score_func = METRICS[dataset]
    fold_seconds = {}
    for fold in sorted(folds.unique()):
        tick = perf_counter()
        train = fit.loc[folds.ne(fold)]
        val = fit.loc[folds.eq(fold)]
        search = EuclideanNeighbors(train, numeric, categorical)
        distances, indices = search.neighbors(val)
        for (k, weighting), fold_scores in scores.items():
            p = positive_probability(search.y, distances, indices, k, weighting)
            fold_scores[f"fold_{fold}"] = float(score_func(val["y"], p))
        fold_seconds[f"fold_{fold}"] = perf_counter() - tick
        print(f"{dataset} fold {fold}/{EXPECTED[dataset][-1]} "
              f"({fold_seconds[f'fold_{fold}']:.1f} s)", flush=True)

    cv = pd.DataFrame([
        dict(k=k, weights=weighting, metric=metric_name, **fold_scores,
             mean_score=float(np.mean(list(fold_scores.values()))),
             std_score=float(np.std(list(fold_scores.values()))))
        for (k, weighting), fold_scores in scores.items()
    ]).sort_values(["mean_score", "k", "weights"], ascending=[False, False, False], ignore_index=True)
    # For identical mean scores, larger k wins. "uniform" sorts ahead of "distance".
    cv.insert(0, "selected", cv.index == 0)
    output.mkdir(parents=True, exist_ok=True)
    cv.to_csv(output / "cv_results.csv", index=False)
    k, weighting = int(cv.loc[0, "k"]), str(cv.loc[0, "weights"])

    tick = perf_counter()
    model = EuclideanNeighbors(fit, numeric, categorical)
    refit_seconds = perf_counter() - tick
    threshold_data = frame.iloc[splits["threshold"]]
    tick = perf_counter()
    p_threshold = positive_probability(model.y, *model.neighbors(threshold_data), k, weighting)
    threshold = find_optimal_f2_threshold(threshold_data["y"], p_threshold, return_curve=True)
    threshold.pop("curve_df").to_csv(output / "threshold_curve.csv", index=False)
    threshold_seconds = perf_counter() - tick
    tick = perf_counter()
    p_cal = positive_probability(model.y, *model.neighbors(cal), k, weighting)
    q = {str(int(level * 100)): float(calibrate(cal["y"], p_cal, level)) for level in LEVELS}
    calibration_seconds = perf_counter() - tick
    frozen = {
        "dataset": dataset,
        "model": f"KNN_k{k}_{weighting}",
        "k": k, "weights": weighting, "distance": "euclidean", "metric": metric_name,
        "cv_mean": float(cv.loc[0, "mean_score"]), "cv_seconds_by_fold": fold_seconds,
        "refit_seconds": refit_seconds, "threshold_seconds": threshold_seconds,
        "calibration_seconds": calibration_seconds, "development_seconds": perf_counter() - start,
        "threshold": threshold, "conformal_q": q, "n_cal": len(cal),
        "batch_size": BATCH_SIZE, "blas_threads": BLAS_THREADS,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "sha256": audit["sha256"],
    }
    write_json(output / "pretest_freeze.json", frozen)
    print(f"{dataset} frozen {frozen['model']}, {metric_name}={frozen['cv_mean']:.6f}, "
          f"threshold={threshold['best_threshold']:.6f}, q={q}", flush=True)
    return frame, splits, features, selected, model, cal, p_cal, frozen


def save_errors(test, predictions, features, distances, indices, model, frozen, output):
    # For equally extreme errors, keep the smallest row ID.
    unique = predictions.sort_values("sample_id").drop_duplicates(["result_type", "y_prob"])
    extremes = extract_extreme_errors(unique)
    cases, neighbors = [], []
    for error_type in ("FN", "FP"):
        row = extremes[f"worst_{error_type.lower()}"]
        if row is None:
            continue
        i = row.name
        sample = test.iloc[[i]]
        d, idx = distances[i, :frozen["k"]], indices[i, :frozen["k"]]
        labels = model.y[idx]
        weights = vote_weights(d[None, :], frozen["weights"])[0]
        contributions = model.mean_squared_components(sample, idx)
        cases.append({
            "error_type": error_type, **row.to_dict(),
            "delta_from_threshold": float(row["y_prob"] - row["threshold"]),
            "tied_error_count": int((predictions["result_type"].eq(error_type)
                                     & predictions["y_prob"].eq(row["y_prob"])).sum()),
            "positive_neighbors": int(labels.sum()), "neighbors": len(idx),
            "positive_vote_share": float(weights[labels == 1].sum() / weights.sum()),
            "top_mean_squared_components": json.dumps(contributions.head(5).to_dict()),
            **sample.iloc[0][features].to_dict(),
        })
        neighbors.append(model.train.iloc[idx][features].assign(
            error_type=error_type, sample_id=row["sample_id"], rank=np.arange(1, len(idx) + 1),
            neighbor_id=model.train.iloc[idx]["row_id"].to_numpy(),
            neighbor_y=labels, distance=d, vote_weight=weights,
        ))
    pd.DataFrame(cases).to_csv(output / "extreme_errors.csv", index=False)
    neighbor_keys = ["error_type", "sample_id", "rank", "neighbor_id", "neighbor_y", "distance", "vote_weight"]
    table = pd.concat(neighbors)[neighbor_keys + features] if neighbors else pd.DataFrame(columns=neighbor_keys + features)
    table.to_csv(output / "error_neighbors.csv", index=False)


def evaluate(frame, splits, features, selected, model, cal, p_cal, frozen, output):
    start = perf_counter()
    test = frame.iloc[splits["test"]]
    test_positions = selected.loc[selected["split"].eq("test"), "row_position"]
    conformal_idx = test.index.get_indexer(test_positions)
    require((conformal_idx >= 0).all(), "Conformal test samples must belong to test")
    distances, indices = model.neighbors(test)
    p_test = positive_probability(model.y, distances, indices, frozen["k"], frozen["weights"])
    threshold = frozen["threshold"]["best_threshold"]
    metrics = compute_binary_metrics(test["y"], p_test, threshold=threshold)
    pred = build_test_prediction_table(test["row_id"].to_numpy(), test["y"], p_test, threshold)

    rows = summarize(cal["y"], p_cal, test["y"].iloc[conformal_idx], p_test[conformal_idx])
    pred["conformal_selected"] = False
    pred.loc[conformal_idx, "conformal_selected"] = True
    for row in rows:
        level = int(row["target"] * 100)
        require(row["q"] == frozen["conformal_q"][str(level)], "Conformal q differs from frozen value")
        sets, labels = assign_conformal_prediction_labels(p_test[conformal_idx], row["q"])
        set_column, decision_column = f"conformal_set_{level}", f"conformal_decision_{level}"
        pred[set_column] = None
        pred[decision_column] = None
        pred.loc[conformal_idx, set_column] = sets
        pred.loc[conformal_idx, decision_column] = labels
    conformal = pd.DataFrame([
        dict(dataset=frozen["dataset"], model=frozen["model"], k=frozen["k"], weights=frozen["weights"], **row)
        for row in rows
    ])
    conformal.to_csv(output / "conformal_results.csv", index=False)
    pred.to_csv(output / "test_predictions.csv", index=False)
    save_errors(test, pred, features, distances, indices, model, frozen, output)
    evaluation_seconds = perf_counter() - start
    write_json(output / "run_summary.json", {
        **frozen, "test_metrics": metrics, "evaluation_seconds": evaluation_seconds,
        "runtime_seconds": frozen["development_seconds"] + evaluation_seconds,
    })
    print(f"{frozen['dataset']} test {frozen['metric']}={metrics[frozen['metric']]:.6f}, "
          f"F2={metrics['f2']:.6f} ({evaluation_seconds:.1f} s)", flush=True)
    return conformal


def check_alignment(output):
    paths = {
        "Logistic Regression": ROOT / "algorithm" / "Logistic" / "artifacts",
        "CatBoost": ROOT / "algorithm" / "CatBoost" / "artifacts",
    }
    rows = []
    for dataset in EXPECTED:
        own = pd.read_csv(output / dataset / "test_predictions.csv")
        require(own["sample_id"].is_unique, f"{dataset}: duplicate KNN sample IDs")
        frame, splits, *_ = load_cohort(dataset)
        frozen_test = frame.iloc[splits["test"]][["row_id", "y"]].rename(
            columns={"row_id": "sample_id", "y": "y_true"},
        )
        verify_sample_alignment({"KNN": own, "Frozen test": frozen_test}, dataset, verbose=False)
        rows.append(dict(dataset=dataset, comparison="Frozen test", status="matched", n=len(own)))
        for name, folder in paths.items():
            path = folder / dataset / "test_predictions.csv"
            if not path.is_file():
                rows.append(dict(dataset=dataset, comparison=name, status="missing", n=""))
                continue
            peer = pd.read_csv(path, usecols=["sample_id", "y_true"])
            verify_sample_alignment({"KNN": own, name: peer}, dataset, verbose=False)
            rows.append(dict(dataset=dataset, comparison=name, status="matched", n=len(peer)))
    pd.DataFrame(rows).to_csv(output / "test_alignment.csv", index=False)
    print("Test sample alignment saved to test_alignment.csv", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--check-alignment", action="store_true", help="Only compare existing test predictions")
    args = parser.parse_args()
    output = args.output_dir
    if not args.check_alignment:
        audits = json.loads((DATA / "metadata" / "experiment_cohort_audit.json").read_text(encoding="utf-8"))
        prepared = {name: prepare(name, audits[name], output / name) for name in EXPECTED}
        results = [evaluate(*state, output / name) for name, state in prepared.items()]
        pd.concat(results, ignore_index=True).to_csv(output / "conformal_4rows_for_hoang.csv", index=False)
    check_alignment(output)


if __name__ == "__main__":
    main()
