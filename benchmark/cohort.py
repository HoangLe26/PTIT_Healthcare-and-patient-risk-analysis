"""Audit frozen cohorts and publish the row assignments shared by all models."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from benchmark.data_loader import NUMERIC_COLS, load_heart_data, load_readmission_data

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 42
PARTS = ("fit", "threshold", "calibration", "test")
EXPECTED = {
    "heart": ((169, 37, 36, 61), 303, 139, 13, 5),
    "readmission": ((59372, 9854, 10083, 20031), 69987, 11314, 44, 3),
}
LOADERS = {"heart": load_heart_data, "readmission": load_readmission_data}


def load_cohort(dataset):
    return LOADERS[dataset](
        data_path=DATA / "processed" / f"{dataset}.csv",
        splits_path=DATA / "splits" / f"{dataset}.json",
        meta_path=DATA / "metadata" / f"{dataset}_feature_availability.csv",
    )


def cohort_paths(dataset):
    return {
        "processed": DATA / "processed" / f"{dataset}.csv",
        "split": DATA / "splits" / f"{dataset}.json",
        "features": DATA / "metadata" / f"{dataset}_feature_availability.csv",
        "cv": DATA / "splits" / f"{dataset}_cv.csv",
        "conformal": DATA / "splits" / f"{dataset}_conformal.csv",
    }


def sha256(path):
    # Git can check out the same CSV/JSON with CRLF or LF.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def counts(frame):
    return dict(rows=len(frame), patients=int(frame["patient_nbr"].nunique()),
                positive=int(frame["y"].sum()), negative=int(frame["y"].eq(0).sum()),
                prevalence=float(frame["y"].mean()))


def matching_rows(frame, columns):
    repeated = frame.loc[frame.duplicated(columns, keep=False), columns]
    return {"rows_in_matching_groups": len(repeated),
            "matching_groups": len(repeated.drop_duplicates())}


def verify_raw(dataset, frame, features):
    if dataset == "heart":
        path = DATA / "raw" / "heart" / "processed.cleveland.data"
        raw = pd.read_csv(path, header=None, names=[*features, "num"], na_values=["?"])
        raw["y"] = raw["num"].gt(0).astype(int)
        require(len(raw) == 303 and raw["num"].isin(range(5)).all(), "Unexpected Cleveland labels or size")
        require(np.array_equal(frame["row_id"], raw.index), "Cleveland row IDs must follow raw order")
        pd.testing.assert_frame_equal(frame[[*features, "y"]], raw[[*features, "y"]], check_dtype=False)
        return {"raw_rows": len(raw), "matching_raw_rows": matching_rows(raw, [*features, "num"]),
                "raw_sha256": sha256(path)}

    path = DATA / "raw" / "readmission" / "diabetic_data.csv"
    raw = pd.read_csv(path, keep_default_na=False, na_values=["?"], low_memory=False)
    require(raw["readmitted"].isin(["<30", ">30", "NO"]).all(), "Unexpected readmitted labels")
    excluded = json.loads((DATA / "metadata" / "discharge_mapping.json").read_text())["excluded"]
    after_discharge = raw.loc[~raw["discharge_disposition_id"].isin(excluded)]
    eligible = after_discharge.loc[after_discharge["gender"].ne("Unknown/Invalid")].reset_index(drop=True)
    raw_rows = len(raw)
    excluded_discharge = raw_rows - len(after_discharge)
    excluded_gender = len(after_discharge) - len(eligible)
    require((raw_rows, excluded_discharge, excluded_gender) == (101766, 2423, 3), "Unexpected cohort flow")
    require(set(features) == set(raw.columns) - {
        "readmitted", "encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty",
    }, "Unexpected Diabetes features")
    require(np.array_equal(frame["row_id"], frame["encounter_id"]), "Diabetes row ID differs from encounter")
    eligible["y"] = eligible["readmitted"].eq("<30").astype(int)
    cols = ["encounter_id", "patient_nbr", *features, "y"]
    pd.testing.assert_frame_equal(frame[cols], eligible[cols], check_dtype=False)
    duplicate_ids = int(raw["encounter_id"].duplicated().sum())
    require(duplicate_ids == 0, "Duplicate Diabetes encounter ID in raw data")
    return {"raw_rows": raw_rows, "excluded_discharge_codes": excluded,
            "excluded_discharge": excluded_discharge, "excluded_gender_after_discharge": excluded_gender,
            "duplicate_raw_encounter_ids": duplicate_ids,
            "matching_raw_rows": matching_rows(raw, list(raw.columns)), "raw_sha256": sha256(path)}


def audit_cohort(dataset, frame, splits, features, categorical):
    sizes, patients, positives, n_features, n_folds = EXPECTED[dataset]
    require(set(PARTS).issubset(splits), f"{dataset}: incomplete split manifest")
    require(tuple(len(splits[part]) for part in PARTS) == sizes, f"{dataset}: unexpected sizes")
    positions = [pos for part in PARTS for pos in splits[part]]
    require(sorted(positions) == list(range(len(frame))), f"{dataset}: splits must partition cohort")
    require(frame["row_id"].is_unique and frame["row_id"].notna().all(), f"{dataset}: invalid row IDs")
    require(frame["patient_nbr"].notna().all(), f"{dataset}: missing patient IDs")
    require(frame["y"].isin([0, 1]).all() and frame["y"].sum() == positives, f"{dataset}: unexpected labels")
    require(frame["patient_nbr"].nunique() == patients, f"{dataset}: unexpected patient count")
    require(len(set(features)) == n_features and set(NUMERIC_COLS[dataset]).issubset(features),
            f"{dataset}: incorrect number or types of features")
    require(not set(features).intersection({
        "row_id", "patient_nbr", "encounter_id", "y", "num", "readmitted",
        "weight", "payer_code", "medical_specialty",
    }), f"{dataset}: excluded input")
    paths = cohort_paths(dataset)
    metadata = pd.read_csv(paths["features"])
    require(metadata.loc[metadata["include"], "available_at_prediction"].all(),
            f"{dataset}: future features marked for inclusion")

    split_frames = {part: frame.iloc[splits[part]] for part in PARTS}
    if "development" in splits:
        require(sorted(splits["development"]) == sorted(positions[:-sizes[-1]]),
                f"{dataset}: incorrect development membership")
    groups = {part: set(block["patient_nbr"]) for part, block in split_frames.items()}
    overlaps = {f"{a}/{b}": len(groups[a] & groups[b]) for a, b in itertools.combinations(PARTS, 2)}
    require(not any(overlaps.values()), f"{dataset}: patient overlap between holdouts")

    fit = split_frames["fit"]
    if dataset == "heart":
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        split_ids = cv.split(fit, fit["y"])
    else:
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        split_ids = cv.split(fit, fit["y"], fit["patient_nbr"])
    validation_folds = []
    fold_counts = []
    for fold, (train_idx, val_idx) in enumerate(split_ids, start=1):
        train, val = fit.iloc[train_idx], fit.iloc[val_idx]
        overlap = len(set(train["patient_nbr"]) & set(val["patient_nbr"]))
        require(overlap == 0, f"{dataset}: patient overlap in fold {fold}")
        require(train["y"].nunique() == val["y"].nunique() == 2, f"{dataset}: single class in fold {fold}")
        validation_folds.append(val[["row_id"]].rename_axis("row_position").reset_index().assign(fold=fold))
        fold_counts.append(dict(fold=fold, train=counts(train), validation=counts(val), patient_overlap=overlap))
    cv_table = pd.concat(validation_folds, ignore_index=True)
    require(sorted(cv_table["row_position"]) == sorted(splits["fit"]), f"{dataset}: incomplete validation folds")

    selected_parts = []
    conformal_counts = {}
    for part in ("calibration", "test"):
        block = split_frames[part]
        if dataset == "readmission":
            block = block.sort_values("encounter_id").drop_duplicates("patient_nbr")
        selected_parts.append(block[["row_id"]].rename_axis("row_position").reset_index().assign(split=part))
        conformal_counts[part] = counts(block)

    # Validate against the raw data before publishing assignments for use by other models.
    raw_audit = verify_raw(dataset, frame, features)
    audit = {
        **raw_audit,
        "cohort": counts(frame),
        "matching_features": matching_rows(frame, features),
        "matching_features_and_label": matching_rows(frame, [*features, "y"]),
        "numeric_features": [col for col in features if col not in categorical],
        "categorical_features": categorical,
        "splits": {part: counts(block) for part, block in split_frames.items()},
        "holdout_patient_overlaps": overlaps,
        "cv": {"splitter": type(cv).__name__, "seed": SEED, "shuffle": True,
               "scikit_learn_version": sklearn.__version__, "folds": fold_counts},
        "conformal": conformal_counts,
    }
    cv_table.to_csv(paths["cv"], index=False)
    pd.concat(selected_parts, ignore_index=True).to_csv(paths["conformal"], index=False)
    audit["sha256"] = {name: sha256(path) for name, path in paths.items()}
    return audit


def main():
    audits = {dataset: audit_cohort(dataset, *load_cohort(dataset)) for dataset in EXPECTED}
    write_json(DATA / "metadata" / "experiment_cohort_audit.json", audits)
    rows = [dict(dataset=dataset, split=part, **counts_)
            for dataset, audit in audits.items() for part, counts_ in audit["splits"].items()]
    table = pd.DataFrame(rows)
    table.to_csv(DATA / "metadata" / "experiment_split_counts.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
