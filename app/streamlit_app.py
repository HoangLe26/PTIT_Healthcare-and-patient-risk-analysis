"""Local academic demo. Run: python -X utf8 -m streamlit run app/streamlit_app.py."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from healthcare_ml.training import predict_artifact

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = {
    "heart": {"age", "trestbps", "chol", "thalach", "oldpeak"},
    "readmission": {"time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
                    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"},
}
RANGES = {
    "age": (0, 120), "trestbps": (0, 400), "chol": (0, 1500), "thalach": (0, 300),
    "oldpeak": (-10, 20), "time_in_hospital": (1, 14), "num_lab_procedures": (0, 1000),
    "num_procedures": (0, 1000), "num_medications": (0, 1000), "number_outpatient": (0, 10000),
    "number_emergency": (0, 10000), "number_inpatient": (0, 10000), "number_diagnoses": (0, 1000),
}


@st.cache_resource
def load_model(dataset: str, modified: int):
    # Only project-produced local model artifacts, never arbitrary uploaded pickle.
    return joblib.load(ROOT / f"artifacts/models/{dataset}_selected.joblib")


@st.cache_data
def example_and_categories(dataset: str, modified: int, columns: tuple):
    split = json.loads((ROOT / f"data/splits/{dataset}.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(ROOT / f"data/processed/{dataset}.csv", keep_default_na=False, na_values=[""], low_memory=False)
    fit = frame.iloc[split["fit"]]
    sample = json.loads(fit.loc[:, list(columns)].iloc[0].to_json(force_ascii=False))
    categories = {c: fit[c].dropna().unique().tolist() for c in columns if c not in NUMERIC[dataset]}
    return json.dumps(sample, ensure_ascii=False, indent=2, allow_nan=False), categories


def validate_payload(raw: str, columns: list, dataset: str, categories: dict):
    def reject_constant(value):
        raise ValueError(f"Giá trị JSON không hợp lệ: {value}; dùng null cho missing.")
    values = json.loads(raw, parse_constant=reject_constant)
    if not isinstance(values, dict):
        raise ValueError("Đầu vào phải là một JSON object cho một quan sát.")
    unknown = set(values) - set(columns)
    missing = set(columns) - set(values)
    if unknown:
        raise ValueError("Cột không thuộc schema (không nhập ID hoặc nhãn): " + ", ".join(sorted(unknown)))
    if missing:
        raise ValueError("Thiếu cột: " + ", ".join(sorted(missing)) + ". Dùng null nếu chưa có thông tin.")
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, (dict, list, bool)):
            raise ValueError(f"{key}: cần số, category hoặc null.")
        if key in NUMERIC[dataset]:
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{key}: cần số hữu hạn hoặc null.")
            lower, upper = RANGES[key]
            if not lower <= value <= upper:
                raise ValueError(f"{key}: ngoài khoảng kiểm tra đầu vào [{lower}, {upper}].")
            if dataset == "readmission" and value != int(value):
                raise ValueError(f"{key}: cần số đếm nguyên.")
        elif value not in categories.get(key, []):
            allowed = categories.get(key, [])
            preview = ", ".join(str(x) for x in allowed[:12])
            raise ValueError(f"{key}: category chưa thuộc schema fit ({preview}{', ...' if len(allowed)>12 else ''}).")
    return pd.DataFrame([{c: np.nan if values[c] is None else values[c] for c in columns}])


def dataset_tab(dataset: str):
    model_path = ROOT / f"artifacts/models/{dataset}_selected.joblib"
    data_path = ROOT / f"data/processed/{dataset}.csv"
    split_path = ROOT / f"data/splits/{dataset}.json"
    if not all(p.exists() for p in (model_path, data_path, split_path)):
        st.info("Chưa có đủ artifact đã huấn luyện và split cho bài toán này.")
        return
    artifact = load_model(dataset, model_path.stat().st_mtime_ns)
    columns = list(artifact["pipeline"].named_steps["features"].columns)
    example, categories = example_and_categories(dataset, data_path.stat().st_mtime_ns, tuple(columns))
    title = "Bệnh tim hiện diện sau khi có xét nghiệm" if dataset == "heart" else "Tái nhập viện dưới 30 ngày tại xuất viện"
    st.subheader(title)
    st.caption(f"Recipe đóng băng: {artifact['model']}, feature set {artifact['feature_set']}, "
               f"threshold {artifact['threshold']:.6f}. Sigmoid calibration: {'có' if artifact['calibrator'] is not None else 'không'}.")
    st.write("Đầu vào mẫu lấy từ phần fit của dữ liệu nghiên cứu, không phải ca kiểm định mới. "
             "Đổi giá trị JSON để minh họa hành vi mô hình. `null` biểu thị missing; chuỗi `None` là category xét nghiệm khi schema có giá trị đó.")
    with st.form(f"{dataset}_form"):
        raw = st.text_area("Đầu vào một quan sát (JSON)", example, height=360, key=f"{dataset}_payload")
        submitted = st.form_submit_button("Tính xác suất minh họa", key=f"{dataset}_submit")
    if submitted:
        try:
            frame = validate_payload(raw, columns, dataset, categories)
            probability = float(predict_artifact(artifact, frame)[0])
            threshold = float(artifact["threshold"])
            prediction = int(probability >= threshold)
            st.metric("Xác suất ước lượng của lớp dương", f"{probability:.2%}")
            st.write(f"Threshold vận hành: **{threshold:.6f}**. Nhãn mô hình: **{prediction}** "
                     f"({'đạt ngưỡng gắn cờ' if prediction else 'dưới ngưỡng gắn cờ'}).")
            st.caption("Kết quả theo recipe đã lưu, không huấn luyện hoặc điều chỉnh threshold trong demo. "
                       "Đây là đầu ra thống kê của prototype, không phải chẩn đoán hoặc khuyến nghị điều trị.")
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            st.error(str(error))
    with st.expander("Schema và cách xử lý đầu vào"):
        st.write("Pipeline đã lưu chọn cột, impute, encode và dự đoán. JSON phải đủ cột; giá trị chưa biết dùng null. "
                 "Category phải thuộc schema fit để tránh diễn giải nhầm mã. Các khoảng số chỉ là kiểm tra giao diện, "
                 "không phải ngưỡng chẩn đoán. ID và nhãn không nằm trong đầu vào.")
        st.write(pd.DataFrame({"feature": columns, "kind": ["numerical" if c in NUMERIC[dataset] else "categorical" for c in columns]}))
    with st.expander("Kết quả test và giới hạn"):
        metric_path = ROOT / f"artifacts/metrics/{dataset}_test_metrics.csv"
        if metric_path.exists():
            metrics = pd.read_csv(metric_path)
            selected = metrics[metrics.model == artifact["model"]]
            st.dataframe(selected[[c for c in ["model", "feature_set", "roc_auc", "ap", "precision", "recall", "specificity", "threshold", "tp", "tn", "fp", "fn"] if c in selected]], hide_index=True)
        if dataset == "heart":
            st.write("Cleveland có 303 hàng lịch sử, test 61 hàng. Không xác minh được repeated patient ID. "
                     "Nhãn mô tả bệnh hiện diện khi đã có xét nghiệm, không phải nguy cơ mắc bệnh tương lai. "
                     "Không có FN trên test này không chứng minh sensitivity quần thể bằng 100%.")
        else:
            st.write("Dữ liệu Mỹ năm 1999–2008, đánh giá theo lượt điều trị trên bệnh nhân tách nhóm. "
                     "Chưa có temporal hoặc external hospital validation. Precision thấp và nhiều false alarms "
                     "ở threshold F2 hạn chế giá trị vận hành thực tế.")


st.set_page_config(page_title="Healthcare ML · Demo học thuật", page_icon="📊", layout="wide")
st.title("Phân tích nguy cơ bệnh nhân bằng học máy")
st.write("Prototype học thuật trên hai bộ dữ liệu UCI. Hai nhiệm vụ có schema và nhãn riêng.")
heart_tab, readmission_tab = st.tabs(["Heart Cleveland", "Readmission <30 ngày"])
with heart_tab:
    dataset_tab("heart")
with readmission_tab:
    dataset_tab("readmission")
st.caption("Nguồn: UCI Heart Disease (DOI 10.24432/C52P4X) và Diabetes 130-US Hospitals (DOI 10.24432/C5230J), CC BY 4.0. Chỉ sử dụng cục bộ cho minh họa học thuật.")
