# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import time
import json
from utils import get_rag_explanation, create_audit_log_entry

# Load components
@st.cache_resource
def load_components():
    xgb_model = joblib.load('xgb_model.joblib')
    class FraudNet(nn.Module):
        def __init__(self, input_dim):
            super(FraudNet, self).__init__()
            self.fc1 = nn.Linear(input_dim, 256)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(256, 128)
            self.dropout = nn.Dropout(0.3)
            self.fc3 = nn.Linear(128, 64)
            self.fc4 = nn.Linear(64, 2)
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(self.relu(self.fc2(x)))
            x = self.relu(self.fc3(x))
            x = self.fc4(x)
            return x
    nn_model = FraudNet(30)
    nn_model.load_state_dict(torch.load('nn_model.pth'))
    nn_model.eval()
    return xgb_model, nn_model

xgb_model, nn_model = load_components()

# --- Dashboard UI ---
st.set_page_config(layout="wide")
st.title("Local Fraud Detection Dashboard")
st.sidebar.header("System Controls")
start_button = st.sidebar.button("Start Real-Time Simulation")
if start_button:
    st.session_state.run_simulation = True

if 'run_simulation' in st.session_state and st.session_state.run_simulation:
    st.subheader("Real-Time Transaction Stream")
    alert_placeholder = st.empty()
    df = pd.read_csv('creditcard.csv')
    df_test = df.iloc[len(df)//2:len(df)//2+100]
    for index, row in df_test.iterrows():
        transaction = pd.DataFrame(row).T
        transaction_id = index
        xgb_score = xgb_model.predict_proba(transaction.drop('Class', axis=1))[:, 1][0]
        status = 'APPROVED'
        explanation = "Transaction is low risk."
        risk_score = xgb_score
        models_used = ["XGBoost"]
        if xgb_score > 0.9:
            status = 'FLAGGED'
            transaction_tensor = torch.tensor(transaction.drop('Class', axis=1).values, dtype=torch.float32)
            with torch.no_grad():
                nn_output = nn_model(transaction_tensor)
                nn_score = torch.softmax(nn_output, dim=1)[:, 1].item()
            models_used.append("Neural Network")
            risk_score = (xgb_score + nn_score) / 2
            llm_explanation = get_rag_explanation(transaction_id, xgb_score, nn_score, transaction)
            explanation = f"**{llm_explanation}**"
            alert_placeholder.error(f"🚨 **ALERT!** High-risk transaction detected: {transaction_id}")
        st.write(f"**Transaction ID {transaction_id}:** Status -> **{status}**")
        st.markdown(explanation)
        create_audit_log_entry(transaction_id, status, explanation, models_used, risk_score)
        time.sleep(2)
    st.sidebar.success("Simulation Complete.")

st.markdown("---")
st.markdown("### Transaction Audit Log")
if st.button("Refresh Audit Log"):
    try:
        with open('audit_log.json', 'r') as f:
            log_entries = [json.loads(line) for line in f.readlines()]
            st.dataframe(pd.DataFrame(log_entries).style.set_properties(**{'font-size': '10pt'}))
    except FileNotFoundError:
        st.warning("No audit log found. Run the simulation first.")