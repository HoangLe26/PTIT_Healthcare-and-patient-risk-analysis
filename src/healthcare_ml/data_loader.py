import json
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List


def load_heart_data(
    data_path: str = "data/processed/heart.csv",
    splits_path: str = "data/splits/heart.json",
) -> Tuple[pd.DataFrame, Dict[str, List[int]], List[str], List[str]]:
    """
    Tải dữ liệu Heart Disease Cleveland và danh sách splits.
    
    Returns:
    --------
    df : pd.DataFrame
    splits : dict of {split_name: list of row indices}
    feature_cols : list of feature column names (13 features)
    cat_cols : list of categorical feature names
    """
    df = pd.read_csv(data_path)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
        
    feature_cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    # Các cột phân loại theo data dictionary
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    
    return df, splits, feature_cols, cat_cols


def load_readmission_data(
    data_path: str = "data/processed/readmission.csv",
    splits_path: str = "data/splits/readmission.json",
) -> Tuple[pd.DataFrame, Dict[str, List[int]], List[str], List[str]]:
    """
    Tải dữ liệu Diabetes Readmission và danh sách splits.
    Loại bỏ các cột theo đúng quy ước:
    - encounter_id, patient_nbr, weight, payer_code, medical_specialty, y, row_id.
    
    Returns:
    --------
    df : pd.DataFrame
    splits : dict of {split_name: list of row indices}
    feature_cols : list of feature column names
    cat_cols : list of categorical feature names
    """
    df = pd.read_csv(data_path)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
        
    cols_to_exclude = {
        "encounter_id", "patient_nbr", "weight", "payer_code",
        "medical_specialty", "y", "row_id"
    }
    feature_cols = [c for c in df.columns if c not in cols_to_exclude]
    
    # Xác định categorical columns: kiểu object hoặc ID phân loại
    id_categoricals = {"admission_type_id", "discharge_disposition_id", "admission_source_id"}
    cat_cols = [
        c for c in feature_cols
        if df[c].dtype == "object" or c in id_categoricals
    ]
    
    return df, splits, feature_cols, cat_cols
