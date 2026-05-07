from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DEFAULT_MODEL_PATH = Path("artifacts/premium_model.pkl")
DEFAULT_METRICS_PATH = Path("artifacts/metrics.json")


def infer_feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    excluded = {target}
    return [column for column in df.columns if column not in excluded]


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = [column for column in X.columns if column not in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=14,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_model(
    df: pd.DataFrame,
    target: str,
    model_path: Path | None = None,
) -> tuple[Pipeline, dict[str, float], list[str]]:
    feature_columns = infer_feature_columns(df, target)
    X = df[feature_columns]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    metrics = evaluate_model(y_test, predictions)

    artifact_path = model_path or DEFAULT_MODEL_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("wb") as file:
        pickle.dump(
            {
                "model": pipeline,
                "feature_columns": feature_columns,
                "target": target,
            },
            file,
        )

    return pipeline, metrics, feature_columns


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def load_model_artifact(model_path: Path | None = None) -> dict[str, object]:
    artifact_path = model_path or DEFAULT_MODEL_PATH
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {artifact_path}.")

    with artifact_path.open("rb") as file:
        return pickle.load(file)


def predict_premium(artifact: dict[str, object], input_frame: pd.DataFrame) -> np.ndarray:
    feature_columns = artifact["feature_columns"]
    model = artifact["model"]
    scored = input_frame.reindex(columns=feature_columns)
    return model.predict(scored)
