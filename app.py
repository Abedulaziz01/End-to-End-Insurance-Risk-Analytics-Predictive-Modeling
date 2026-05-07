from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from insurance_risk.data import DEFAULT_TARGET, REAL_DATA_PATH, load_dataset
from insurance_risk.modeling import DEFAULT_METRICS_PATH, DEFAULT_MODEL_PATH, load_model_artifact, predict_premium, train_model

st.set_page_config(page_title="Insurance Risk Analytics", page_icon=":bar_chart:", layout="wide")


@st.cache_data(show_spinner=False)
def get_dataset() -> tuple[pd.DataFrame, str]:
    return load_dataset(path=REAL_DATA_PATH, target=DEFAULT_TARGET, demo_rows=5000, max_rows=20000)


@st.cache_resource(show_spinner=False)
def get_artifact(df: pd.DataFrame, mode: str):
    if DEFAULT_MODEL_PATH.exists():
        return load_model_artifact(DEFAULT_MODEL_PATH)

    model, metrics, feature_columns = train_model(df, target=DEFAULT_TARGET, model_path=DEFAULT_MODEL_PATH)
    DEFAULT_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_METRICS_PATH.write_text(
        json.dumps({"dataset_mode": mode, "metrics": metrics, "feature_count": len(feature_columns)}, indent=2),
        encoding="utf-8",
    )
    return {"model": model, "feature_columns": feature_columns, "target": DEFAULT_TARGET}


def risk_band(predicted_premium: float) -> str:
    if predicted_premium < 70:
        return "Low"
    if predicted_premium < 110:
        return "Medium"
    return "High"


def make_prediction_input(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    st.subheader("Premium Prediction")
    columns = st.columns(2)
    values: dict[str, object] = {}

    for index, feature in enumerate(feature_columns):
        container = columns[index % 2]
        series = df[feature]
        if pd.api.types.is_numeric_dtype(series):
            min_value = float(series.min())
            max_value = float(series.max())
            default_value = float(series.median())
            if pd.api.types.is_integer_dtype(series):
                values[feature] = container.number_input(
                    feature,
                    min_value=int(min_value),
                    max_value=int(max_value),
                    value=int(default_value),
                    step=1,
                )
            else:
                values[feature] = container.number_input(
                    feature,
                    min_value=min_value,
                    max_value=max_value,
                    value=default_value,
                )
        else:
            options = sorted(series.dropna().astype(str).unique().tolist())
            default_index = 0
            if "Toyota" in options and feature == "make":
                default_index = options.index("Toyota")
            values[feature] = container.selectbox(feature, options=options, index=default_index)

    return pd.DataFrame([values])


st.title("End-to-End Insurance Risk Analytics")
st.caption("A fast, runnable version of the original notebook project with modeling and Streamlit.")

DATAFRAME, MODE = get_dataset()
ARTIFACT = get_artifact(DATAFRAME, MODE)

metrics_payload = {}
if DEFAULT_METRICS_PATH.exists():
    metrics_payload = json.loads(DEFAULT_METRICS_PATH.read_text(encoding="utf-8"))

with st.sidebar:
    st.header("Project Mode")
    if MODE == "real":
        st.success("Using real dataset from `Data/cleaned1.csv`")
    else:
        st.info("Using synthetic demo dataset because `Data/cleaned1.csv` is not available.")

    st.write(f"Rows: `{len(DATAFRAME):,}`")
    st.write(f"Features: `{len(ARTIFACT['feature_columns'])}`")
    if metrics_payload.get("metrics"):
        st.write("Latest model metrics")
        st.json(metrics_payload["metrics"])

tab_overview, tab_analytics, tab_predict = st.tabs(["Overview", "Analytics", "Predict"])

with tab_overview:
    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows", f"{len(DATAFRAME):,}")
    metric_columns[1].metric("Avg Premium", f"{DATAFRAME[DEFAULT_TARGET].mean():.2f}")
    metric_columns[2].metric("Avg Claims", f"{DATAFRAME['TotalClaims'].mean():.2f}")
    metric_columns[3].metric("Top Province", DATAFRAME["Province"].mode().iloc[0])

    st.subheader("Dataset Preview")
    st.dataframe(DATAFRAME.head(20), use_container_width=True)

with tab_analytics:
    left, right = st.columns(2)

    with left:
        st.subheader("Average Premium by Province")
        premium_by_province = (
            DATAFRAME.groupby("Province")[DEFAULT_TARGET]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(premium_by_province)

        st.subheader("Cover Type Mix")
        cover_mix = DATAFRAME["CoverType"].value_counts()
        st.bar_chart(cover_mix)

    with right:
        st.subheader("Top Vehicle Makes")
        top_makes = DATAFRAME["make"].value_counts().head(10)
        st.bar_chart(top_makes)

        st.subheader("Claims vs Premium Sample")
        sample = DATAFRAME[[DEFAULT_TARGET, "TotalClaims", "Province"]].sample(
            n=min(500, len(DATAFRAME)),
            random_state=42,
        )
        st.scatter_chart(sample, x="TotalClaims", y=DEFAULT_TARGET, color="Province")

with tab_predict:
    input_df = make_prediction_input(DATAFRAME, ARTIFACT["feature_columns"])
    if st.button("Predict Premium", type="primary"):
        predicted_premium = float(predict_premium(ARTIFACT, input_df)[0])
        band = risk_band(predicted_premium)

        result_columns = st.columns(3)
        result_columns[0].metric("Predicted Premium", f"{predicted_premium:.2f}")
        result_columns[1].metric("Risk Band", band)
        result_columns[2].metric("Mode", MODE.title())

        st.write("Input used for prediction")
        st.dataframe(input_df, use_container_width=True)
