import streamlit as st

import pandas as pd
import numpy as np

import statsmodels.api as sm
from sklearn.linear_model import Ridge

import pickle

st.set_page_config(page_title="Startup Fundraising Decision Aiding Tool", layout="centered")

# ---- Title ----
st.title("Predicting Startup Amount Raised")
st.markdown("""
This app predicts the amount raised by startups based on a mixed predictors
such as documentation completion, valuation, monthly revenue and target raised            
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

with open("ridge_model.pkl", "rb") as f:
    model_params = pickle.load(f)

beta = model_params['coefficients'][1:]           
intercept = model_params['intercept']
sigma = model_params['sigma']    

# --- Prediction Results ---

x = pd.DataFrame({
    'monthlyRevenue': [np.log10(monthly_revenue)],
    'targetRaise': [np.log10(target_raise)],
    'preMoneyValuation': [np.log10(pre_valuation)]
})

# Base Prediction with 95% confidence interval based on sigma
y_pred = intercept + np.dot(x, beta)

y_pred_low = y_pred - 1.96 * sigma
y_pred_high = y_pred + 1.96 * sigma

# Convert to original scale
pred_mean = np.power(y_pred[0], 10)
pred_low = np.power(y_pred_low[0], 10)
pred_high = np.power(y_pred_high[0], 10)


### Document Boost Features ###
document_row = []

# NOTE: currently this is dependent on order of fields
# being same across files. Can be improved.
for col in bool_columns:
    if col in selected_docs:
        document_row.append(1)
    else:    
        document_row.append(0)

# Load the trained decision tree #
with open("decision_tree.pkl", "rb") as f:
    tree = pickle.load(f)

def predict_row(row, tree):
    
    # Leaf node
    if 'prediction' in tree:
        return tree['prediction']
    
    if row[tree['feature']] == 1:
        return predict_row(row, tree['right'])
    else:
        return predict_row(row, tree['left'])


# Document Boost Prediction 
pred_boost = predict_row(document_row ,tree)

# ---- Display Results ----
st.subheader("Predicted Amount Raised")
st.write(f"Predicted Amount Raised: £{pred_mean:,.2f}")

st.markdown(f"""
- 95% Confidence Interval: **£{pred_low:.2f} - £{pred_high:.2f}**
- Based on your inputs:
    - Monthly Revenue: £{monthly_revenue:,}
    - Target Raise: £{target_raise:,}
    - Pre-Money Valuation: £{pre_valuation:,}
""")

st.subheader("Impact of Selected Documents")


if pred_boost > 0:
    st.success(f"📈 **Boost:** ~({(pred_boost) * 100:.2f}%)")
    st.write(f"Predicted Amount Raised with Selected Documents: £{pred_mean * pred_boost:,.2f}")
    st.markdown(f"95% Confidence Interval with Selected Documents: **£{pred_low * pred_boost:.2f} - £{pred_high * pred_boost:,.2f}**")
else:
    st.info("No boost (or no documents selected).")

st.caption("Predictions are log-linear regression estimates and are indicative only.")

