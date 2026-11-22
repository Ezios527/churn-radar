"""Streamlit interface for the trained churn model."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from churn_predictor.model import predict_churn

MODEL_PATH = Path("models/churn_model.joblib")

st.set_page_config(page_title="Churn Radar", page_icon="📡", layout="wide")
st.title("📡 Churn Radar")
st.caption("Учебный ML-проект: оценка риска оттока клиента")

if not MODEL_PATH.exists():
    st.error("Модель не найдена. Сначала выполните команды генерации данных и обучения из README.")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.sidebar:
    st.header("Параметры клиента")
    tenure = st.slider("Сколько месяцев с нами", 1, 72, 12)
    spend = st.number_input("Расходы в месяц, $", 5.0, 150.0, 25.0, step=1.0)
    tickets = st.slider("Обращения в поддержку", 0, 8, 1)
    usage = st.slider("Часов использования в неделю", 0.0, 30.0, 6.0, step=0.5)
    late = st.slider("Просроченных платежей", 0, 6, 0)
    plan = st.selectbox("Тариф", ["basic", "standard", "premium"])
    contract = st.selectbox("Тип контракта", ["monthly", "annual"])
    autopay = st.selectbox("Автоплатёж", ["yes", "no"])

customer = pd.DataFrame(
    [
        {
            "tenure_months": tenure,
            "monthly_spend": spend,
            "support_tickets": tickets,
            "weekly_usage_hours": usage,
            "late_payments": late,
            "plan": plan,
            "contract_type": contract,
            "autopay": autopay,
        }
    ]
)
prediction = predict_churn(model, customer).iloc[0]
probability = float(prediction["churn_probability"])

left, right = st.columns([1, 2])
with left:
    st.metric("Вероятность оттока", f"{probability:.1%}")
    if probability < 0.35:
        st.success("Низкий риск")
    elif probability < 0.65:
        st.warning("Средний риск")
    else:
        st.error("Высокий риск")
with right:
    st.subheader("Как читать результат")
    st.write(
        "Модель ищет закономерности в истории похожих клиентов. Результат — не точный прогноз, "
        "а оценка, кому стоит уделить внимание: предложить помощь, скидку или годовой контракт."
    )
    st.progress(probability)

st.divider()
st.caption("Logistic Regression · scikit-learn · синтетические данные")

