import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="🏦",
    layout="wide"
)

# =========================================================
# SAFE FILE LOADING (FIXES YOUR ERROR)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

if not os.path.exists(MODEL_PATH):
    st.error("❌ model.pkl not found. Please upload it to the GitHub repo.")
    st.stop()

if not os.path.exists(SCALER_PATH):
    st.error("❌ scaler.pkl not found. Please upload it to the GitHub repo.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <h1 style='text-align:center;'>🏦 Credit Risk Intelligence Dashboard</h1>
    <p style='text-align:center;font-size:18px;'>
    Machine Learning powered real-time credit risk assessment
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# =========================================================
# SIDEBAR INPUTS
# =========================================================
st.sidebar.header("🧾 Applicant Details")

age = st.sidebar.slider("Age", 18, 75, 30)
credit_amount = st.sidebar.number_input("Credit Amount", 500, 100000, 5000, step=500)
duration = st.sidebar.slider("Loan Duration (Months)", 6, 72, 24)
installment_rate = st.sidebar.selectbox("Installment Rate", [1, 2, 3, 4])
existing_credits = st.sidebar.selectbox("Existing Credits", [0, 1, 2, 3])

threshold = st.sidebar.slider(
    "⚖️ Risk Decision Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

# =========================================================
# INPUT DATA
# =========================================================
input_df = pd.DataFrame({
    "Age": [age],
    "CreditAmount": [credit_amount],
    "Duration": [duration],
    "InstallmentRate": [installment_rate],
    "ExistingCredits": [existing_credits]
})

input_scaled = scaler.transform(input_df)

# =========================================================
# PREDICTION
# =========================================================
risk_probability = model.predict_proba(input_scaled)[0][1]
risk_prediction = int(risk_probability >= threshold)

# =========================================================
# MAIN LAYOUT
# =========================================================
col1, col2 = st.columns([1, 1.3])

# ---------------- RESULT CARD ----------------
with col1:
    st.subheader("📌 Credit Decision")

    if risk_prediction == 1:
        st.error("❌ HIGH CREDIT RISK")
    else:
        st.success("✅ LOW CREDIT RISK")

    st.metric(
        label="Default Risk Probability",
        value=f"{risk_probability:.2%}"
    )

    st.caption(f"Decision threshold = **{threshold:.2f}**")

# ---------------- GAUGE CHART ----------------
with col2:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_probability * 100,
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
                "value": threshold * 100
            }
        }
    ))

    st.plotly_chart(fig_gauge, use_container_width=True)

st.divider()

# =========================================================
# FEATURE OVERVIEW
# =========================================================
st.subheader("📊 Applicant Profile Snapshot")

feature_df = input_df.T.reset_index()
feature_df.columns = ["Feature", "Value"]

fig_bar = px.bar(
    feature_df,
    x="Value",
    y="Feature",
    orientation="h",
    color="Feature",
    title="Input Feature Distribution"
)

st.plotly_chart(fig_bar, use_container_width=True)

# =========================================================
# INSIGHTS SECTION
# =========================================================
st.subheader("🧠 Model Interpretation")

st.info(
    """
    - This model estimates the probability of credit default.
    - Threshold can be adjusted based on risk appetite.
    - Lower threshold → more approvals, higher risk.
    - Higher threshold → fewer approvals, lower risk.
    """
)

st.success(
    "This dashboard demonstrates how ML models are deployed in real banking decision systems."
)
