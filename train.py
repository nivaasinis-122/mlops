import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "age",
    "gpa",
    "study_time",
    "absences",
    "parent_support",
    "extracurricular",
    "financial_stress",
]
TARGET = "dropout"
ARTIFACT_DIR = Path("artifacts")
DEFAULT_EXPERIMENT = "student-dropout"
DEFAULT_REGISTERED_MODEL = "student_dropout_lr"


def generate_synthetic(n_rows: int = 1200, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    age = rng.integers(16, 23, size=n_rows)
    gpa = rng.normal(2.8, 0.5, size=n_rows).clip(0.0, 4.0)
    study_time = rng.normal(2.5, 1.0, size=n_rows).clip(0.0, 8.0)
    absences = rng.poisson(5, size=n_rows).clip(0, 40)
    parent_support = rng.integers(1, 6, size=n_rows)
    extracurricular = rng.integers(0, 2, size=n_rows)
    financial_stress = rng.integers(0, 2, size=n_rows)

    base = (
        0.6
        - 0.9 * gpa
        - 0.08 * study_time
        + 0.05 * absences
        - 0.18 * parent_support
        + 0.3 * financial_stress
        - 0.1 * extracurricular
        + 0.02 * (age - 18)
    )
    prob = 1.0 / (1.0 + np.exp(-base))
    dropout = rng.binomial(1, prob)

    return pd.DataFrame(
        {
            "age": age,
            "gpa": gpa,
            "study_time": study_time,
            "absences": absences,
            "parent_support": parent_support,
            "extracurricular": extracurricular,
            "financial_stress": financial_stress,
            TARGET: dropout,
        }
    )


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    csv_path = os.getenv("STUDENT_DATA_PATH")
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        df = generate_synthetic()

    missing = [col for col in [*FEATURES, TARGET] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[FEATURES]
    y = df[TARGET]
    return X, y


def build_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=1000, C=1.2, class_weight="balanced", random_state=random_state),
            ),
        ]
    )


def summarize_ranges(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    return {
        col: {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
        }
        for col in FEATURES
    }


def save_artifacts(pipeline: Pipeline, meta: Dict[str, object]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ARTIFACT_DIR / "model_pipeline.pkl")
    joblib.dump(meta, ARTIFACT_DIR / "meta.pkl")
    mlflow.log_artifact(ARTIFACT_DIR / "model_pipeline.pkl", artifact_path="artifacts")
    mlflow.log_artifact(ARTIFACT_DIR / "meta.pkl", artifact_path="artifacts")


def main() -> None:
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT)
    registered_model = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", DEFAULT_REGISTERED_MODEL)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()

    with mlflow.start_run() as run:
        fitted = pipeline.fit(X_train, y_train)
        preds = fitted.predict(X_test)
        proba = fitted.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        mlflow.log_metrics(metrics)

        mlflow.log_params(
            {
                "model": "LogisticRegression",
                "features": ",".join(FEATURES),
                "class_weight": "balanced",
                "data_source": os.getenv("STUDENT_DATA_PATH", "synthetic"),
            }
        )

        signature = infer_signature(X_train, fitted.predict_proba(X_train))
        mlflow.sklearn.log_model(
            sk_model=fitted,
            artifact_path="model",
            registered_model_name=registered_model,
            signature=signature,
            input_example=X_train.head(3),
        )

        meta = {"feature_names": FEATURES, "target": TARGET, "ranges": summarize_ranges(X)}
        mlflow.log_dict(meta, "feature_spec.json")
        save_artifacts(fitted, meta)

        print("Run ID:", run.info.run_id)
        print(
            f"Accuracy: {metrics['accuracy']:.3f}  F1: {metrics['f1']:.3f}  ROC-AUC: {metrics['roc_auc']:.3f}"
        )
        print("Artifacts saved to", ARTIFACT_DIR.resolve())


if __name__ == "__main__":
    main()
