import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Define paths to the model files
logistic_regression_model_path = 'logistic_model (1).pkl'
decision_tree_model_path = 'decision_tree_model (3).pkl'
naive_bayes_model_path = 'naive_bayes_model (2).pkl'

# Load pre-trained models with error handling
def load_model(path):
    try:
        if os.path.isfile(path):
            return joblib.load(path)
        else:
            st.error(f"Model file not found: {path}")
            return None
    except EOFError:
        st.error(f"EOFError: The model file '{path}' appears to be corrupted or incomplete.")
        return None
    except Exception as e:
        st.error(f"Error loading model from '{path}': {e}")
        return None

# Load models
logistic_regression_model = load_model(logistic_regression_model_path)
decision_tree_model = load_model(decision_tree_model_path)
naive_bayes_model = load_model(naive_bayes_model_path)

# Define feature columns used for training the models
feature_cols = [
    'total_payment', 'CreditScore', 'OrigInterestRate', 'MonthlyIncome', 'MIP',
    'OCLTV', 'MonthlyRate', 'MSA', 'OrigLoanTerm', 'interest_amt', 'EMI', 'cur_principal',
    'MonthsDelinquent', 'DTI', 'OrigUPB', 'MonthsInRepayment'
]

# Sample values for testing
sample_values = {
    'total_payment': 500.0,
    'CreditScore': 700.0,
    'OrigInterestRate': 3.5,
    'MonthlyIncome': 5000.0,
    'MIP': 0.5,
    'OCLTV': 80.0,
    'MonthlyRate': 0.03,
    'MSA': 100.0,
    'OrigLoanTerm': 30.0,
    'interest_amt': 1000.0,
    'EMI': 1500.0,
    'cur_principal': 200000.0,
    'MonthsDelinquent': 3.0,
    'DTI': 0.35,
    'OrigUPB': 180000.0,
    'MonthsInRepayment': 24.0
}

# Define Streamlit UI
st.set_page_config(page_title='Mortgage Model Prediction', layout='wide')
st.title('Mortgage Model Prediction Web Application')

st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f0f5;
    }
    .css-18e3th9 {
        padding: 2rem;
    }
    .css-1r6slb0 {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# User input
st.sidebar.header('Input Features')
st.sidebar.write("Please enter the values for the features below:")

input_data = {col: st.sidebar.number_input(f'{col}', value=float(sample_values[col]), format="%.2f", step=0.01, min_value=-1e10, max_value=1e10) for col in feature_cols}

input_df = pd.DataFrame([input_data])

# Ensure the DataFrame has the correct columns and order
input_df = input_df[feature_cols]

# Display input data for debugging
st.write("### Input Data")
st.write(input_df)

# Model selection
model_choice = st.sidebar.selectbox(
    'Select Model for Prediction',
    ['Logistic Regression', 'Decision Tree', 'Naive Bayes']
)

# Make predictions
if st.sidebar.button('Predict'):
    if model_choice == 'Logistic Regression' and logistic_regression_model:
        try:
            # Make predictions with Logistic Regression
            lr_pred = logistic_regression_model.predict(input_df)
            result_lr = 'Accepted for Credit' if lr_pred[0] == 1 else 'Rejected for Credit'
            st.write(f"### Logistic Regression Result: **{result_lr}**")
        except Exception as e:
            st.write(f"**Error in Logistic Regression prediction:** {e}")

    elif model_choice == 'Decision Tree' and decision_tree_model:
        try:
            # Make predictions with Decision Tree
            dt_pred = decision_tree_model.predict(input_df)
            result_dt = 'Accepted for Credit' if dt_pred[0] == 1 else 'Rejected for Credit'
            st.write(f"### Decision Tree Result: **{result_dt}**")
        except Exception as e:
            st.write(f"**Error in Decision Tree prediction:** {e}")

    elif model_choice == 'Naive Bayes' and naive_bayes_model:
        try:
            # Make predictions with Naive Bayes
            nb_pred = naive_bayes_model.predict(input_df)
            result_nb = 'Accepted for Credit' if nb_pred[0] == 1 else 'Rejected for Credit'
            st.write(f"### Naive Bayes Result: **{result_nb}**")
        except Exception as e:
            st.write(f"**Error in Naive Bayes prediction:** {e}")

    else:
        st.write("**Error:** The selected model failed to load or is not available.")
