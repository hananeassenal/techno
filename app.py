import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Paths to the model files
logistic_regression_model_path = 'logistic_model (1).pkl'
decision_tree_model_path = 'decision_tree_model (3).pkl'
naive_bayes_model_path = 'naive_bayes_model (2).pkl'

# Load models with error handling
def load_model(path):
    try:
        if os.path.isfile(path):
            return joblib.load(path)
        else:
            st.error(f"Model file not found: {path}")
            return None
    except EOFError:
        st.error(f"EOFError: The model file '{path}' appears to be corrupted.")
        return None
    except Exception as e:
        st.error(f"Error loading model from '{path}': {e}")
        return None

# Load models
logistic_regression_model = load_model(logistic_regression_model_path)
decision_tree_model = load_model(decision_tree_model_path)
naive_bayes_model = load_model(naive_bayes_model_path)

# Define the features expected by the model
training_feature_cols = [
    'CreditScore', 'MIP', 'DTI', 'EverDelinquent', 'MonthsDelinquent', 'MonthsInRepayment'
]

# Preprocess input data
def preprocess_data(data):
    df = pd.DataFrame([data])
    df = df[training_feature_cols]
    df = df.fillna(0)
    return df

# Streamlit UI setup
st.set_page_config(page_title='Mortgage Model Prediction', layout='wide')
st.title('Mortgage Model Prediction Web Application')

st.sidebar.header('Input Features')
st.sidebar.write("Enter the feature values below:")

# Example input data
input_data = {
    'CreditScore': st.sidebar.number_input('CreditScore', value=550.0, format="%.2f"),
    'MIP': st.sidebar.number_input('MIP', value=0.5, format="%.2f"),
    'DTI': st.sidebar.number_input('DTI', value=0.55, format="%.2f"),
    'EverDelinquent': st.sidebar.selectbox('EverDelinquent', ['1', '0']),
    'MonthsDelinquent': st.sidebar.number_input('MonthsDelinquent', value=12.0, format="%.2f"),
    'MonthsInRepayment': st.sidebar.number_input('MonthsInRepayment', value=6.0, format="%.2f"),
}

input_data['EverDelinquent'] = int(input_data['EverDelinquent'])
processed_data = preprocess_data(input_data)

st.write("### Input Data")
st.write(processed_data)

model_choice = st.sidebar.selectbox(
    'Select Model for Prediction',
    ['Logistic Regression', 'Decision Tree', 'Naive Bayes']
)

if st.sidebar.button('Predict'):
    if model_choice == 'Logistic Regression' and logistic_regression_model:
        try:
            # Predict probabilities for adjusting threshold
            lr_probs = logistic_regression_model.predict_proba(processed_data)
            threshold = 0.4
            lr_pred = (lr_probs[:, 1] > threshold).astype(int)
            st.write(f"**Logistic Regression Raw Prediction:** {lr_pred[0]}")
            result_lr = 'Accepted for Credit' if lr_pred[0] == 1 else 'Rejected for Credit'
            st.write(f"### Logistic Regression Result: **{result_lr}**")
        except Exception as e:
            st.write(f"**Error in Logistic Regression prediction:** {e}")

    elif model_choice == 'Decision Tree' and decision_tree_model:
        try:
            dt_pred = decision_tree_model.predict(processed_data)
            st.write(f"**Decision Tree Raw Prediction:** {dt_pred[0]}")
            result_dt = 'Accepted for Credit' if dt_pred[0] == 1 else 'Rejected for Credit'
            st.write(f"### Decision Tree Result: **{result_dt}**")
        except Exception as e:
            st.write(f"**Error in Decision Tree prediction:** {e}")

    elif model_choice == 'Naive Bayes' and naive_bayes_model:
        try:
            nb_pred = naive_bayes_model.predict(processed_data)
            st.write(f"**Naive Bayes Raw Prediction:** {nb_pred[0]}")
            result_nb = 'Accepted for Credit' if nb_pred[0] == 1 else 'Rejected for Credit'
            st.write(f"### Naive Bayes Result: **{result_nb}**")
        except Exception as e:
            st.write(f"**Error in Naive Bayes prediction:** {e}")

    else:
        st.write("**Error:** The selected model failed to load or is not available.")
