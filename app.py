import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------------------------
# BASIC PAGE CHECK (CONFIRMS THIS FILE IS RUNNING)
# -------------------------------------------------
st.title("✅ Credit Risk App – Debug Mode")
st.write("If you can see this text, Streamlit is running THIS app.py")

# -------------------------------------------------
# CHECK CURRENT DIRECTORY
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
st.write("📂 Current directory:", BASE_DIR)
st.write("📁 Files available:", os.listdir(BASE_DIR))

# -------------------------------------------------
# LOAD MODEL SAFELY
# -------------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

if not os.path.exists(MODEL_PATH):
    st.error("❌ model.pkl NOT FOUND in this directory")
    st.stop()

if not os.path.exists(SCALER_PATH):
    st.error("❌ scaler.pkl NOT FOUND in this directory")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

st.success("✅ Model and scaler loaded successfully")

# -------------------------------------------------
# SIMPLE INPUT & PREDICTION
# -------------------------------------------------
st.subheader("🔍 Test Prediction")

age = st.slider("Age", 18, 75, 30)
credit_amount = st.number_input("Credit Amount", 500, 100000, 5000)
duration = st.slider("Loan Duration (months)", 6, 72, 24)
installment_rate = st.selectbox("Installment Rate", [1, 2, 3, 4])
existing_credits = st.selectbox("Existing Credits", [0, 1, 2, 3])

input_df = pd.DataFrame({
    "Age": [age],
    "CreditAmount": [credit_amount],
    "Duration": [duration],
    "InstallmentRate": [installment_rate],
    "ExistingCredits": [existing_credits]
})

input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    prob = model.predict_proba(input_scaled)[0][1]
    st.write("📊 Risk Probability:", round(prob, 3))
