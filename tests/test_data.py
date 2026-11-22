import pandas as pd

from churn_predictor.data import generate_customers


def test_generated_data_has_expected_shape_and_target() -> None:
    data = generate_customers(n_rows=100, random_state=1)
    assert len(data) == 100
    assert set(data["churned"].unique()).issubset({0, 1})
    assert data["customer_id"].is_unique


def test_generation_is_reproducible() -> None:
    first = generate_customers(n_rows=30, random_state=7)
    second = generate_customers(n_rows=30, random_state=7)
    pd.testing.assert_frame_equal(first, second)

