# Student Dropout Risk (MLOps demo)

Predicts student dropout likelihood with MLflow tracking, Docker packaging, and a Streamlit risk dashboard.

## Quickstart
1. Install deps: `pip install -r requirements.txt`
2. Train and log: `python train.py`
3. Run app: `streamlit run app.py`

## Data
- Default: synthetic generator with realistic signals (age, GPA, study time, absences, parent support, extracurricular, financial stress).
- Custom: point `STUDENT_DATA_PATH` to a CSV containing all feature columns plus `dropout` (0/1).

## Environment switches
- `MLFLOW_TRACKING_URI`: remote MLflow server; otherwise local `mlruns` is used.
- `MLFLOW_EXPERIMENT_NAME`: experiment name (default `student-dropout`).
- `MLFLOW_REGISTERED_MODEL_NAME`: registry entry (default `student_dropout_lr`).
- `MODEL_URI`: optional MLflow model URI for the app when local artifacts are absent.
- `STUDENT_DATA_PATH`: optional CSV path for training with your data.

## Docker
Build and run locally:
```
docker build -t student-dropout .
docker run -p 8080:8080 student-dropout
```

## Railway
`Procfile` runs Streamlit on `$PORT` for Railway deployments. Push the repo and deploy.

## Notes
- Model: StandardScaler + LogisticRegression (balanced classes).
- Metrics logged: accuracy, F1, ROC-AUC; artifacts saved under `artifacts/` and to MLflow.
- The app exposes sliders/selectors for raw features; scaling is handled by the pipeline.
