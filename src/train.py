# src/train.py (fixed: compute RMSE without using squared= kwarg)
import os
import argparse
import joblib
import math
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from preprocessing import load_data, basic_cleaning, build_preprocessor, train_test_val_split, get_feature_lists
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Optional: try importing xgboost
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def classification_metrics(y_true, y_pred, y_proba=None):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba, multi_class="ovr") if y_proba is not None else None
    }

def regression_metrics(y_true, y_pred):
    # compute RMSE in a version-compatible way
    try:
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(math.sqrt(mse))
    except Exception:
        # fallback: convert to numpy and compute
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        mse = float(np.mean((y_true_arr - y_pred_arr) ** 2))
        rmse = float(math.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

def _encode_labels(y_train, y_val=None):
    le = LabelEncoder()
    if y_val is not None:
        combined = pd.concat([y_train.astype(str), y_val.astype(str)], ignore_index=True)
        le.fit(combined)
    else:
        le.fit(y_train.astype(str))
    y_train_enc = le.transform(y_train.astype(str))
    y_val_enc = le.transform(y_val.astype(str)) if y_val is not None else None
    return y_train_enc, y_val_enc, le

def train_and_log_classification(X_train, y_train, X_val, y_val, preprocessor, run_name="classification_run"):
    models = {}
    pipelines = {}

    pipelines["logistic"] = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipelines["rf"] = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42))
    ])

    if HAS_XGB:
        pipelines["xgb"] = Pipeline([
            ("pre", preprocessor),
            ("clf", xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42))
        ])

    # Encode labels to numeric (important for xgboost)
    y_train_enc, y_val_enc, le = _encode_labels(y_train, y_val)

    # Log mapping to MLflow so we know what 0/1 means
    mapping = {str(i): str(label) for i, label in enumerate(le.classes_)}
    try:
        mlflow.log_param("label_mapping", str(mapping))
    except Exception:
        pass

    best_model = None
    best_score = -np.inf
    for name, pipe in pipelines.items():
        mlflow.start_run(run_name=f"{run_name}_{name}", nested=True)
        pipe.fit(X_train, y_train_enc)
        # Predict on validation set (must pass processed X_val)
        y_pred = pipe.predict(X_val)
        y_true = y_val_enc

        y_proba = None
        try:
            if hasattr(pipe.named_steps["clf"], "predict_proba"):
                y_proba = pipe.predict_proba(X_val)
        except Exception:
            y_proba = None

        metrics = classification_metrics(y_true, y_pred, y_proba)
        mlflow.log_params({"model": name})
        for k, v in metrics.items():
            if v is not None:
                try:
                    mlflow.log_metric(k, float(v))
                except Exception:
                    pass

        # save model artifact
        model_path = os.path.join("models", f"classification_{name}.pkl")
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipe, model_path)
        try:
            mlflow.log_artifact(model_path, artifact_path="models")
        except Exception:
            pass
        try:
            mlflow.sklearn.log_model(pipe, artifact_path=f"sklearn_models/{name}")
        except Exception:
            pass

        mlflow.end_run()

        score = metrics.get("accuracy", 0)
        if score > best_score:
            best_score = score
            best_model = (name, model_path, metrics)

    return best_model, le

def train_and_log_regression(X_train, y_train, X_val, y_val, preprocessor, run_name="regression_run"):
    pipelines = {}
    pipelines["linear"] = Pipeline([("pre", preprocessor), ("reg", LinearRegression())])
    pipelines["rf"] = Pipeline([("pre", preprocessor), ("reg", RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42))])
    if HAS_XGB:
        pipelines["xgb"] = Pipeline([("pre", preprocessor), ("reg", xgb.XGBRegressor(random_state=42))])

    best_model = None
    best_score = np.inf  # lower is better for RMSE
    for name, pipe in pipelines.items():
        mlflow.start_run(run_name=f"{run_name}_{name}", nested=True)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
        metrics = regression_metrics(y_val, y_pred)
        mlflow.log_params({"model": name})
        for k, v in metrics.items():
            try:
                mlflow.log_metric(k, float(v))
            except Exception:
                pass
        model_path = os.path.join("models", f"regression_{name}.pkl")
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipe, model_path)
        try:
            mlflow.log_artifact(model_path, artifact_path="models")
        except Exception:
            pass
        try:
            mlflow.sklearn.log_model(pipe, artifact_path=f"sklearn_models/{name}")
        except Exception:
            pass
        mlflow.end_run()

        score = metrics.get("rmse", np.inf)
        if score < best_score:
            best_score = score
            best_model = (name, model_path, metrics)

    return best_model

def main(args):
    df = load_data(args.data)
    df = basic_cleaning(df)

    CLASS_COL = "emi_eligibility"
    REG_COL = "max_monthly_emi"

    if REG_COL in df.columns:
        df[REG_COL] = pd.to_numeric(df[REG_COL], errors="coerce")

    train_df, val_df, test_df = train_test_val_split(df, target_class_col=CLASS_COL)

    preprocessor, numeric_features, categorical_features = build_preprocessor(train_df)

    X_train = train_df[numeric_features + categorical_features]
    y_train_clf = train_df[CLASS_COL] if CLASS_COL in train_df.columns else None

    X_val = val_df[numeric_features + categorical_features]
    y_val_clf = val_df[CLASS_COL] if CLASS_COL in val_df.columns else None

    mlflow.set_experiment(args.experiment)

    label_encoder = None
    if y_train_clf is not None:
        mlflow.start_run(run_name="classification_experiment")
        best_clf, label_encoder = train_and_log_classification(X_train, y_train_clf, X_val, y_val_clf, preprocessor)
        if best_clf:
            name, path, metrics = best_clf
            mlflow.log_param("best_classification", name)
            for k, v in (metrics or {}).items():
                try:
                    mlflow.log_metric(f"classification_{k}", float(v))
                except Exception:
                    pass
        mlflow.end_run()

    if REG_COL in train_df.columns:
        y_train_reg = train_df[REG_COL]
        y_val_reg = val_df[REG_COL]
        mlflow.start_run(run_name="regression_experiment")
        best_reg = train_and_log_regression(X_train, y_train_reg, X_val, y_val_reg, preprocessor)
        if best_reg:
            name, path, metrics = best_reg
            mlflow.log_param("best_regression", name)
            for k, v in (metrics or {}).items():
                try:
                    mlflow.log_metric(f"regression_{k}", float(v))
                except Exception:
                    pass
        mlflow.end_run()

    # Optionally save the label encoder mapping to models/ so Streamlit can load and decode predictions
    if label_encoder is not None:
        try:
            joblib.dump(label_encoder, os.path.join("models", "label_encoder.pkl"))
        except Exception:
            pass

    print("Training complete. Check MLflow UI for results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EMIPredict models and log to MLflow")
    parser.add_argument("--data", type=str, default="../data/sample_EMI_dataset.csv", help="Path to CSV dataset")
    parser.add_argument("--experiment", type=str, default="EMIPredict", help="MLflow experiment name")
    args = parser.parse_args()
    main(args)
