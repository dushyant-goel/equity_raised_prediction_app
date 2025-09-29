import streamlit as st

import pandas as pd
import numpy as np

import statsmodels.api as sm
from sklearn.linear_model import Ridge

import pickle

st.set_page_config(page_title="Startup Fundraising Analysis", layout="centered")

# ---- Title ----
st.title("Predicting Startup Amount Raised")
st.markdown("""
This app predicts the amount raised by startups based on 
            factors such as documentation valuation, monthly revenue, target raised
            location, and more
""")

# ---- Slider for User Input ----

# Input for Numerical Features
monthly_revenue = st.slider("Monthly Revenue (£)", min_value=1_000, max_value=150_000, value=25_000, step=1_000)
pre_valuation = st.slider("Pre-Money Valuation (£)", min_value=100_000, max_value=7_000_000, value=2_000_000, step=50_000)
target_raise = st.slider("Target Raise (£)", min_value=50_000, max_value=7_000_000, value=1_600_000, step=10_000)

# ---- Checklist for document flags ----

# Input for Boolean Features
bool_columns = [
    "eis",
    "seis",
    "advanceAssurance",
    "pitchDeck",
    "financialModel",
    "businessPlan",
    "dataRoom",
    "keyPersonInsurance",
    "cyberInsurance",
    "directorsOfficersInsurance",
]

# Mapping for column names -> checkbox list labels
column_titles = {
    "eis": "EIS",
    "seis": "SEIS",
    "advanceAssurance": "Advance Assurance",
    "pitchDeck": "Pitch Deck",
    "financialModel": "Financial Model",
    "businessPlan": "Business Plan",
    "dataRoom": "Data Room",
    "keyPersonInsurance": "Key Person Insurance",
    "cyberInsurance": "Cyber Insurance",
    "directorsOfficersInsurance": "Directors & Officers Insurance"
}

st.markdown("### Select Completed Documents:")
selected_docs = []
cols = st.columns(2)
for i, col in enumerate(bool_columns):
    with cols[i % 2]:
        if st.checkbox(column_titles[col], value=False):
            selected_docs.append(col)

# ---- Load the Trained Model Weights ----

with open("trained_bayes_model.pkl", "rb") as f:
    model_params = pickle.load(f)

beta = model_params['beta']           
intercept = model_params['intercept']    
sigma = model_params['sigma']

# --- Prediction Results ---

x = pd.DataFrame({
    'monthlyRevenue': [np.log10(monthly_revenue)],
    'targetRaise': [np.log10(target_raise)],
    'preMoneyValuation': [np.log10(pre_valuation)]
})

for col in bool_columns:
    if col not in x.columns:
        x[col] = 0

# Document Boost Features

x_boosted = pd.DataFrame()
x_boosted = x.copy()

for col in selected_docs:
    x_boosted[col] = 1

# Base Prediction with 95% confidence interval based on sigma
y_pred_mean = intercept + np.dot(x, beta)
y_pred_low = y_pred_mean - 1.96 * sigma
y_pred_high = y_pred_mean + 1.96 * sigma

# Document Boosted Prediction with 95% confidence interval based on sigma
y_pred_boosted_mean = intercept + np.dot(x_boosted, beta)
y_pred_boosted_low = y_pred_boosted_mean - 1.96 * sigma
y_pred_boosted_high = y_pred_boosted_mean + 1.96 * sigma

# Convert to original scale
pred_mean = 10 ** y_pred_mean[0]
pred_low = 10 ** y_pred_low[0]
pred_high = 10 ** y_pred_high[0]

pred_boosted_mean = 10 ** y_pred_boosted_mean[0]
pred_boosted_low = 10 ** y_pred_boosted_low[0]
pred_boosted_high = 10 ** y_pred_boosted_high[0]

# ---- Display Results ----
st.subheader("Predicted Amount Raised")
st.write(f"Predicted Amount Raised: £{pred_mean:,.2f}")

st.markdown(f"""
- 95% Confidence Interval: **£{pred_low:,.2f} - £{pred_high:,.2f}**
- Based on your inputs:
    - Monthly Revenue: £{monthly_revenue:,}
    - Target Raise: £{target_raise:,}
    - Pre-Money Valuation: £{pre_valuation:,}
""")

st.subheader("Impact of Selected Documents")

boost = pred_boosted_mean - pred_mean

if boost > 0:
    st.success(f"📈 **Boost:** +£{boost:,.0f} (~{(boost / pred_mean) * 100:,.1f}%)")
    st.write(f"Predicted Amount Raised with Selected Documents: £{pred_boosted_mean:,.2f}")
    st.markdown(f"95% Confidence Interval with Selected Documents: **£{pred_boosted_low:,.2f} - £{pred_boosted_high:,.2f}**")
else:
    st.info("No boost (or no documents selected).")

st.caption("Predictions are log-linear regression estimates and are indicative only.")

