import pandas as pd

from churn_predictor.data import generate_customers
from churn_predictor.features import ALL_FEATURES
from churn_predictor.model import build_pipeline, predict_churn


def test_model_returns_valid_probabilities() -> None:
    data = generate_customers(n_rows=200, random_state=10)
    model = build_pipeline().fit(data[ALL_FEATURES], data["churned"])
    result = predict_churn(model, data.head(5)[ALL_FEATURES])
    assert result["churn_probability"].between(0, 1).all()
    assert set(result["risk_level"]).issubset({"low", "medium", "high"})


def test_prediction_rejects_missing_columns() -> None:
    data = generate_customers(n_rows=100, random_state=11)
    model = build_pipeline().fit(data[ALL_FEATURES], data["churned"])
    broken = pd.DataFrame({"tenure_months": [12]})
    try:
        predict_churn(model, broken)
    except ValueError as error:
        assert "Missing columns" in str(error)
    else:
        raise AssertionError("Expected ValueError")

