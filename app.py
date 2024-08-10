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
    
    # Ensure the DataFrame has the correct columns
    df = df[training_feature_cols]
    
    # Handle missing values
    df = df.fillna(0)  # Example: fill missing values with 0
    
    return df

# Streamlit UI setup
st.set_page_config(page_title='Mortgage Model Prediction', layout='wide')
st.title('Mortgage Model Prediction Web Application')

st.sidebar.header('Input Features')
st.sidebar.write("Enter the feature values below:")

# Input data from user
input_data = {
    'CreditScore': st.sidebar.number_input('CreditScore', value=700.0, format="%.2f"),
    'MIP': st.sidebar.number_input('MIP', value=0.0, format="%.2f"),
    'DTI': st.sidebar.number_input('DTI', value=0.35, format="%.2f"),
    'EverDelinquent': st.sidebar.selectbox('EverDelinquent', ['0', '1']),
    'MonthsDelinquent': st.sidebar.number_input('MonthsDelinquent', value=3.0, format="%.2f"),
    'MonthsInRepayment': st.sidebar.number_input('MonthsInRepayment', value=24.0, format="%.2f"),
}

# Convert categorical inputs
input_data['EverDelinquent'] = int(input_data['EverDelinquent'])

# Preprocess input data
processed_data = preprocess_data(input_data)

# Display preprocessed data for debugging
st.write("### Input Data")
st.write(processed_data)

# Model selection
model_choice = st.sidebar.selectbox(
    'Select Model for Prediction',
    ['Logistic Regression', 'Decision Tree', 'Naive Bayes']
)

# Make predictions
if st.sidebar.button('Predict'):
    if model_choice == 'Logistic Regression' and logistic_regression_model:
        try:
            # Logistic Regression prediction
            lr_pred = logistic_regression_model.predict(processed_data)
            st.write(f"**Logistic Regression Raw Prediction:** {lr_pred[0]}")
            result_lr = 'Accepted for Credit' if lr_pred[0] == 1 else 'Rejected for Credit'
            st.write(f"### Logistic Regression Result: **{result_lr}**")
        except Exception as e:
            st.write(f"**Error in Logistic Regression prediction:** {e}")

    elif model_choice == 'Decision Tree' and decision_tree_model:
        try:
            # Decision Tree prediction
            dt_pred = decision_tree_model.predict(processed_data)
            st.write(f"**Decision Tree Raw Prediction:** {dt_pred[0]}")
            result_dt = 'Accepted for Credit' if dt_pred[0] == 1 else 'Rejected for Credit'
            st.write(f"### Decision Tree Result: **{result_dt}**")
        except Exception as e:
            st.write(f"**Error in Decision Tree prediction:** {e}")

    elif model_choice == 'Naive Bayes' and naive_bayes_model:
        try:
            # Naive Bayes prediction
            nb_pred = naive_bayes_model.predict(processed_data)
            st.write(f"**Naive Bayes Raw Prediction:** {nb_pred[0]}")
            result_nb = 'Accepted for Credit' if nb_pred[0] == 1 else 'Rejected for Credit'
            st.write(f"### Naive Bayes Result: **{result_nb}**")
        except Exception as e:
            st.write(f"**Error in Naive Bayes prediction:** {e}")

    else:
        st.write("**Error:** The selected model failed to load or is not available.")
