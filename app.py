import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ---------------- CONFIG ---------------- #
st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="🏦",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

# ---------------- HEADER ---------------- #
st.markdown(
    """
    <h1 style='text-align: center;'>🏦 Credit Risk Intelligence Dashboard</h1>
    <p style='text-align: center; font-size:18px;'>
    AI-powered loan risk assessment using Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("🔧 Applicant Information")

age = st.sidebar.slider("Age", 18, 75, 30)
credit_amount = st.sidebar.number_input("Credit Amount", 500, 100000, 5000, step=500)
duration = st.sidebar.slider("Loan Duration (months)", 6, 72, 24)
installment_rate = st.sidebar.selectbox("Installment Rate", [1, 2, 3, 4])
existing_credits = st.sidebar.selectbox("Existing Credits", [0, 1, 2, 3])

risk_threshold = st.sidebar.slider(
    "⚖️ Risk Decision Threshold",
    0.1, 0.9, 0.5, 0.05
)

# ---------------- INPUT DATA ---------------- #
input_df = pd.DataFrame({
    "Age": [age],
    "CreditAmount": [credit_amount],
    "Duration": [duration],
    "InstallmentRate": [installment_rate],
    "ExistingCredits": [existing_credits]
})

input_scaled = scaler.transform(input_df)

# ---------------- PREDICTION ---------------- #
probability = model.predict_proba(input_scaled)[0][1]
prediction = int(probability >= risk_threshold)

# ---------------- LAYOUT ---------------- #
col1, col2 = st.columns([1, 1.3])

# ---------------- RESULT CARD ---------------- #
with col1:
    st.subheader("📌 Credit Decision")

    if prediction == 1:
        st.error("❌ HIGH CREDIT RISK")
    else:
        st.success("✅ LOW CREDIT RISK")

    st.metric(
        label="Risk Probability",
        value=f"{probability:.2%}"
    )

    st.caption(
        f"Decision threshold set at **{risk_threshold:.2f}**"
    )

# ---------------- GAUGE CHART ---------------- #
with col2:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%"},
        title={"text": "Credit Risk Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#1f77b4"},
            "steps": [
                {"range": [0, 40], "color": "#2ecc71"},
                {"range": [40, 70], "color": "#f1c40f"},
                {"range": [70, 100], "color": "#e74c3c"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": risk_threshold * 100
            }
        }
    ))

    st.plotly_chart(fig_gauge, use_container_width=True)

st.divider()

# ---------------- FEATURE CONTRIBUTION (SIMULATED) ---------------- #
st.subheader("📊 Input Feature Overview")

feature_df = input_df.T.reset_index()
feature_df.columns = ["Feature", "Value"]

fig_bar = px.bar(
    feature_df,
    x="Value",
    y="Feature",
    orientation="h",
    color="Feature",
    title="Applicant Profile Snapshot"
)

st.plotly_chart(fig_bar, use_container_width=True)

# ---------------- INSIGHTS ---------------- #
st.subheader("🧠 Model Insights")

st.info(
    """
    **How to interpret this dashboard:**
    - The model estimates the probability of credit default.
    - Threshold can be adjusted based on risk appetite.
    - Lower thresholds increase approvals but raise risk.
    - Higher thresholds reduce defaults but reject more applicants.
    """
)

st.success(
    "This application simulates how banks deploy ML models for real-time credit decisions."
)
