"""Generate a small, reproducible synthetic customer dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_customers(n_rows: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """Return realistic-looking customer records with a binary churn target."""
    if n_rows < 10:
        raise ValueError("n_rows must be at least 10")

    rng = np.random.default_rng(random_state)
    plan = rng.choice(["basic", "standard", "premium"], n_rows, p=[0.45, 0.38, 0.17])
    contract = rng.choice(["monthly", "annual"], n_rows, p=[0.68, 0.32])
    autopay = rng.choice(["yes", "no"], n_rows, p=[0.58, 0.42])
    tenure = np.clip(rng.gamma(2.2, 11, n_rows).round(), 1, 72).astype(int)
    base_spend = np.select(
        [plan == "basic", plan == "standard", plan == "premium"],
        [12, 25, 49],
    )
    monthly_spend = np.maximum(5, base_spend + rng.normal(0, 4, n_rows)).round(2)
    weekly_usage = np.maximum(0.2, rng.normal(6.5, 3, n_rows)).round(1)
    tickets = np.clip(rng.poisson(1.2, n_rows), 0, 8)
    late_payments = np.clip(rng.poisson(np.where(autopay == "yes", 0.25, 0.9)), 0, 6)

    # Hidden relationship used only to create labels. Noise keeps the task realistic.
    score = (
        -1.65
        + 1.05 * (contract == "monthly")
        + 0.62 * (autopay == "no")
        + 0.38 * tickets
        + 0.42 * late_payments
        - 0.035 * tenure
        - 0.11 * weekly_usage
        + 0.35 * (plan == "basic")
        + rng.normal(0, 0.65, n_rows)
    )
    probability = 1 / (1 + np.exp(-score))
    churned = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(1, n_rows + 1)],
            "tenure_months": tenure,
            "monthly_spend": monthly_spend,
            "support_tickets": tickets,
            "weekly_usage_hours": weekly_usage,
            "late_payments": late_payments,
            "plan": plan,
            "contract_type": contract,
            "autopay": autopay,
            "churned": churned,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo churn data")
    parser.add_argument("--rows", type=int, default=2500)
    parser.add_argument("--output", type=Path, default=Path("data/customers.csv"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = generate_customers(args.rows)
    data.to_csv(args.output, index=False)
    print(f"Saved {len(data)} rows to {args.output}")


if __name__ == "__main__":
    main()

