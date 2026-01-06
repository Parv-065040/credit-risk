import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Credit Risk Prediction", layout="centered")

st.title("🏦 Credit Risk Prediction App")
st.write("Predict whether a loan applicant is **High Risk or Low Risk**")

st.divider()

# 🔹 USER INPUTS
age = st.number_input("Age", min_value=18, max_value=75, value=30)
credit_amount = st.number_input("Credit Amount", min_value=100, max_value=100000, value=5000)
duration = st.number_input("Loan Duration (months)", min_value=6, max_value=72, value=24)
installment_rate = st.slider("Installment Rate (%)", 1, 4, 2)
existing_credits = st.selectbox("Existing Credits", [0, 1, 2, 3])

st.divider()

# 🔹 CREATE INPUT DATAFRAME
input_data = pd.DataFrame({
    "Age": [age],
    "CreditAmount": [credit_amount],
    "Duration": [duration],
    "InstallmentRate": [installment_rate],
    "ExistingCredits": [existing_credits]
})

# Scale input
input_scaled = scaler.transform(input_data)

# 🔹 PREDICTION
if st.button("🔍 Predict Credit Risk"):
    probability = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Credit Risk")
    else:
        st.success("✅ Low Credit Risk")

    st.write(f"**Risk Probability:** {probability:.2f}")
