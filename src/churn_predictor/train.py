"""Command-line entry point for model training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .features import ALL_FEATURES, TARGET, validate_input
from .model import build_pipeline, evaluate, save_artifacts


def train(data_path: Path, model_path: Path, metrics_path: Path) -> dict[str, float]:
    data = pd.read_csv(data_path)
    validate_input(data)
    if TARGET not in data:
        raise ValueError(f"Training data must contain '{TARGET}'")
    x_train, x_test, y_train, y_test = train_test_split(
        data[ALL_FEATURES], data[TARGET], test_size=0.25, random_state=42, stratify=data[TARGET]
    )
    model = build_pipeline()
    model.fit(x_train, y_train)
    metrics = evaluate(model, x_test, y_test)
    metrics["train_rows"] = len(x_train)
    metrics["test_rows"] = len(x_test)
    save_artifacts(model, metrics, model_path, metrics_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the churn model")
    parser.add_argument("--data", type=Path, default=Path("data/customers.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/churn_model.joblib"))
    parser.add_argument("--metrics", type=Path, default=Path("reports/metrics.json"))
    args = parser.parse_args()
    metrics = train(args.data, args.model, args.metrics)
    print("Training complete")
    for name, value in metrics.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()

