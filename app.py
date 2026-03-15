import os
from pathlib import Path
from typing import Dict, Optional

import joblib
import mlflow.sklearn
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

FEATURES = [
    "age",
    "gpa",
    "study_time",
    "absences",
    "parent_support",
    "extracurricular",
    "financial_stress",
]
ARTIFACT_PIPELINE = Path("artifacts/model_pipeline.pkl")
ARTIFACT_META = Path("artifacts/meta.pkl")


def generate_synthetic(n_rows: int = 400, random_state: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    age = rng.integers(16, 23, size=n_rows)
    gpa = rng.normal(2.8, 0.5, size=n_rows).clip(0.0, 4.0)
    study_time = rng.normal(2.5, 1.0, size=n_rows).clip(0.0, 8.0)
    absences = rng.poisson(5, size=n_rows).clip(0, 40)
    parent_support = rng.integers(1, 6, size=n_rows)
    extracurricular = rng.integers(0, 2, size=n_rows)
    financial_stress = rng.integers(0, 2, size=n_rows)

    return pd.DataFrame(
        {
            "age": age,
            "gpa": gpa,
            "study_time": study_time,
            "absences": absences,
            "parent_support": parent_support,
            "extracurricular": extracurricular,
            "financial_stress": financial_stress,
        }
    )


@st.cache_resource
def load_model(model_uri: Optional[str] = None):
    if ARTIFACT_PIPELINE.exists():
        return joblib.load(ARTIFACT_PIPELINE)
    resolved_uri = model_uri or os.getenv("MODEL_URI")
    if not resolved_uri:
        raise FileNotFoundError("No local model artifact found and MODEL_URI not set.")
    return mlflow.sklearn.load_model(resolved_uri)


@st.cache_resource
def load_meta() -> Dict[str, object]:
    if ARTIFACT_META.exists():
        return joblib.load(ARTIFACT_META)
    raise FileNotFoundError("Meta artifact not found; run training first.")


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    return generate_synthetic()


def predict_proba(model, features: Dict[str, float]) -> float:
    ordered = [features[col] for col in FEATURES]
    df = pd.DataFrame([ordered], columns=FEATURES)
    return float(model.predict_proba(df)[0][1])


def risk_band(p: float) -> str:
    if p < 0.3:
        return "Low"
    if p < 0.6:
        return "Moderate"
    return "High"


def render_plots(df: pd.DataFrame) -> None:
    st.subheader("Training-like distributions")
    cols = st.columns(3)
    targets = [
        ("gpa", "GPA"),
        ("study_time", "Study hours"),
        ("absences", "Absences"),
        ("parent_support", "Parent support"),
        ("age", "Age"),
        ("financial_stress", "Financial stress"),
    ]
    for idx, (col, label) in enumerate(targets):
        fig = px.histogram(df, x=col, nbins=25, opacity=0.85, title=f"{label} distribution")
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
        cols[idx % 3].plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Student Dropout Risk", layout="wide")
    st.title("🎓 Student Dropout Risk")
    st.write("Predict dropout likelihood with an MLflow-tracked model.")

    meta = load_meta()
    ranges = meta.get("ranges", {})
    model = load_model()
    ref_df = load_reference_data()

    st.sidebar.header("Student profile")
    st.sidebar.caption("Provide current metrics; model handles scaling.")

    def slider(name: str, fmt: str, step: float = 0.1):
        r = ranges.get(name, {})
        return st.sidebar.slider(
            fmt,
            r.get("min", float(ref_df[name].min())),
            r.get("max", float(ref_df[name].max())),
            r.get("mean", float(ref_df[name].mean())),
            step=step,
        )

    age = slider("age", "Age", 1.0)
    gpa = slider("gpa", "GPA", 0.05)
    study_time = slider("study_time", "Daily study hours", 0.25)
    absences = slider("absences", "Unexcused absences", 1.0)
    parent_support = slider("parent_support", "Parent support (1-5)", 1.0)
    extracurricular = st.sidebar.selectbox("Extracurriculars", [0, 1], format_func=lambda x: "Yes" if x else "No")
    financial_stress = st.sidebar.selectbox("Financial stress", [0, 1], format_func=lambda x: "Yes" if x else "No")

    if st.sidebar.button("Predict", type="primary"):
        features = {
            "age": age,
            "gpa": gpa,
            "study_time": study_time,
            "absences": absences,
            "parent_support": parent_support,
            "extracurricular": int(extracurricular),
            "financial_stress": int(financial_stress),
        }
        p = predict_proba(model, features)
        band = risk_band(p)
        st.success(f"Estimated dropout probability: {p:.2%}")
        st.info(f"Risk band: {band}")

    with st.expander("What the model sees", expanded=False):
        st.write("LogisticRegression over standardized features; class weights balanced.")
        st.write("Tracked with MLflow for metrics, params, and registry.")

    render_plots(ref_df)

    st.caption("For planning support only; combine with advisor review.")


if __name__ == "__main__":
    main()
