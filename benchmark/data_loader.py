import json
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List


NUMERIC_COLS = {
    "heart": ["age", "trestbps", "chol", "thalach", "oldpeak"],
    "readmission": [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ],
}


def load_heart_data(
    data_path: str = "data/processed/heart.csv",
    splits_path: str = "data/splits/heart.json",
    meta_path: str = "data/metadata/heart_feature_availability.csv",
) -> Tuple[pd.DataFrame, Dict[str, List[int]], List[str], List[str]]:
    """
    Tải dữ liệu Heart Disease Cleveland và danh sách splits.
    Dùng đúng quy tắc đọc dữ liệu của nhóm (keep_default_na=False, na_values=[''])
    và lấy danh sách biến từ metadata/heart_feature_availability.csv.
    """
    df = pd.read_csv(data_path, keep_default_na=False, na_values=[""], low_memory=False)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    meta = pd.read_csv(meta_path)
    feature_cols = meta.loc[meta["include"], "feature"].tolist()
    num_cols = set(NUMERIC_COLS["heart"])
    cat_cols = [c for c in feature_cols if c not in num_cols]
    
    # Đảm bảo cột y là số nguyên 0/1
    df["y"] = pd.to_numeric(df["y"]).astype(int)
    return df, splits, feature_cols, cat_cols


def load_readmission_data(
    data_path: str = "data/processed/readmission.csv",
    splits_path: str = "data/splits/readmission.json",
    meta_path: str = "data/metadata/readmission_feature_availability.csv",
) -> Tuple[pd.DataFrame, Dict[str, List[int]], List[str], List[str]]:
    """
    Tải dữ liệu Diabetes Readmission và danh sách splits.
    Dùng keep_default_na=False, na_values=[''] để bảo toàn chuỗi 'None' có ý nghĩa
    trong max_glu_serum và A1Cresult (khớp 100% với Conformal_Prediction/run_conformal_reference.py).
    """
    df = pd.read_csv(data_path, keep_default_na=False, na_values=[""], low_memory=False)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    meta = pd.read_csv(meta_path)
    feature_cols = meta.loc[meta["include"], "feature"].tolist()
    num_cols = set(NUMERIC_COLS["readmission"])
    cat_cols = [c for c in feature_cols if c not in num_cols]
    
    # Đảm bảo cột y và các ID là số nguyên
    df["y"] = pd.to_numeric(df["y"]).astype(int)
    df["encounter_id"] = pd.to_numeric(df["encounter_id"]).astype(int)
    df["patient_nbr"] = pd.to_numeric(df["patient_nbr"]).astype(int)
    return df, splits, feature_cols, cat_cols
