"""Model construction, training and inference helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, validate_input


def build_pipeline() -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )


def evaluate(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    probability = model.predict_proba(x_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_test, probability)), 4),
        "accuracy": round(float(accuracy_score(y_test, prediction)), 4),
        "precision": round(float(precision_score(y_test, prediction, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, prediction, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, prediction, zero_division=0)), 4),
    }


def save_artifacts(model: Pipeline, metrics: dict[str, Any], model_path: Path, metrics_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def predict_churn(model: Pipeline, customers: pd.DataFrame) -> pd.DataFrame:
    validate_input(customers)
    result = customers.copy()
    result["churn_probability"] = model.predict_proba(customers[ALL_FEATURES])[:, 1].round(4)
    result["risk_level"] = pd.cut(
        result["churn_probability"],
        bins=[-0.01, 0.35, 0.65, 1.0],
        labels=["low", "medium", "high"],
    ).astype(str)
    return result

