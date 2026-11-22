"""Shared feature definitions and input validation."""

from __future__ import annotations

import pandas as pd

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_spend",
    "support_tickets",
    "weekly_usage_hours",
    "late_payments",
]

CATEGORICAL_FEATURES = ["plan", "contract_type", "autopay"]
TARGET = "churned"
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def validate_input(frame: pd.DataFrame) -> None:
    """Raise a helpful error when prediction data has missing columns."""
    missing = sorted(set(ALL_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

