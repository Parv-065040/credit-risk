import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np

# ==================================================
# BASIC CHECK
# ==================================================
st.title("✅ Credit Risk App – Working Version")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
st.write("📁 Files:", os.listdir(BASE_DIR))

# ==================================================
# LOAD MODEL & SCALER
# ==================================================
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

st.success("Model & Scaler Loaded")

# ==================================================
# FEATURE NAMES USED DURING TRAINING
# ==================================================
FEATURES = list(scaler.feature_names_in_)

st.write("📊 Model expects features:", FEATURES)

# ==================================================
# USER INPUT (ONLY CORE NUMERIC ONES)
# ==================================================
st.sidebar.header("Applicant Details")

age = st.sidebar.slider("Age", 18, 75, 30)
credit_amount = st.sidebar.number_input("Credit Amount", 500, 100000, 5000)
duration = st.sidebar.slider("Loan Duration", 6, 72, 24)
installment_rate = st.sidebar.selectbox("Installment Rate", [1, 2, 3, 4])
existing_credits = st.sidebar.selectbox("Existing Credits", [0, 1, 2, 3])

# ==================================================
# BUILD INPUT ROW SAFELY
# ==================================================
input_data = {
    "Age": age,
    "CreditAmount": credit_amount,
    "Duration": duration,
    "InstallmentRate": installment_rate,
    "ExistingCredits": existing_credits
}

# Fill missing features with 0
final_input = {col: input_data.get(col, 0) for col in FEATURES}

input_df = pd.DataFrame([final_input])

# ==================================================
# SCALE & PREDICT
# ==================================================
input_scaled = scaler.transform(input_df)
probability = model.predict_proba(input_scaled)[0][1]

st.subheader("📊 Prediction Result")
st.write("Risk Probability:", round(probability, 3))

if probability >= 0.5:
    st.error("❌ HIGH CREDIT RISK")
else:
    st.success("✅ LOW CREDIT RISK")
