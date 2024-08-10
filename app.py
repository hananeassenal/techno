import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Paths to the model files
logistic_regression_model_path = 'logistic_regression_model.pkl'
decision_tree_model_path = 'decision_tree_model.pkl'
random_forest_model_path = 'random_forest_model.pkl'
naive_bayes_model_path = 'naive_bayes_model.pkl'

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
random_forest_model = load_model(random_forest_model_path)
naive_bayes_model = load_model(naive_bayes_model_path)

# Define the features expected by the models
# Example feature sets: you should verify the exact feature sets used for each model
expected_feature_cols_lr_dt_rf = [
    'CreditScore', 'MIP', 'DTI', 'EverDelinquent', 'MonthsDelinquent', 'MonthsInRepayment'
]
expected_feature_cols_nb = [
    'CreditScore', 'DTI', 'EverDelinquent', 'MonthsDelinquent'
]

# Preprocess input data
def preprocess_data(data, feature_cols):
    df = pd.DataFrame([data])
    # Ensure the DataFrame has the correct columns
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df

# Streamlit UI setup
st.set_page_config(page_title='Loan Credit Prediction', layout='wide')
st.title('Loan Credit Prediction Web Application')

st.sidebar.header('Input Features')
st.sidebar.write("Enter the feature values below:")

# Example input data
input_data = {
    'CreditScore': st.sidebar.number_input('CreditScore', value=650.0, format="%.2f"),
    'MIP': st.sidebar.number_input('MIP', value=0.5, format="%.2f"),
    'DTI': st.sidebar.number_input('DTI', value=0.3, format="%.2f"),
    'EverDelinquent': st.sidebar.selectbox('EverDelinquent', ['0', '1']),
    'MonthsDelinquent': st.sidebar.number_input('MonthsDelinquent', value=6.0, format="%.2f"),
    'MonthsInRepayment': st.sidebar.number_input('MonthsInRepayment', value=12.0, format="%.2f"),
}

# Convert categorical inputs
input_data['EverDelinquent'] = int(input_data['EverDelinquent'])

# Model selection
model_choice = st.sidebar.selectbox(
    'Select Model for Prediction',
    ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes']
)

# Make predictions
if st.sidebar.button('Predict'):
    if model_choice == 'Logistic Regression' and logistic_regression_model:
        try:
            processed_data = preprocess_data(input_data, expected_feature_cols_lr_dt_rf)
            # Ensure the input data for Logistic Regression is of correct shape
            if processed_data.shape[1] == len(expected_feature_cols_lr_dt_rf):
                lr_pred = logistic_regression_model.predict(processed_data)
                st.write(f"**Logistic Regression Raw Prediction:** {lr_pred[0]}")
                result_lr = 'Accepted for Credit' if lr_pred[0] == 1 else 'Rejected for Credit'
                st.write(f"### Logistic Regression Result: **{result_lr}**")
            else:
                st.write("**Error:** The input data does not match the expected feature set for Logistic Regression.")
        except Exception as e:
            st.write(f"**Error in Logistic Regression prediction:** {e}")

    elif model_choice == 'Decision Tree' and decision_tree_model:
        try:
            processed_data = preprocess_data(input_data, expected_feature_cols_lr_dt_rf)
            # Ensure the input data for Decision Tree is of correct shape
            if processed_data.shape[1] == len(expected_feature_cols_lr_dt_rf):
                dt_pred = decision_tree_model.predict(processed_data)
                st.write(f"**Decision Tree Raw Prediction:** {dt_pred[0]}")
                result_dt = 'Accepted for Credit' if dt_pred[0] == 1 else 'Rejected for Credit'
                st.write(f"### Decision Tree Result: **{result_dt}**")
            else:
                st.write("**Error:** The input data does not match the expected feature set for Decision Tree.")
        except Exception as e:
            st.write(f"**Error in Decision Tree prediction:** {e}")

    elif model_choice == 'Random Forest' and random_forest_model:
        try:
            processed_data = preprocess_data(input_data, expected_feature_cols_lr_dt_rf)
            # Ensure the input data for Random Forest is of correct shape
            if processed_data.shape[1] == len(expected_feature_cols_lr_dt_rf):
                rf_pred = random_forest_model.predict(processed_data)
                st.write(f"**Random Forest Raw Prediction:** {rf_pred[0]}")
                result_rf = 'Accepted for Credit' if rf_pred[0] == 1 else 'Rejected for Credit'
                st.write(f"### Random Forest Result: **{result_rf}**")
            else:
                st.write("**Error:** The input data does not match the expected feature set for Random Forest.")
        except Exception as e:
            st.write(f"**Error in Random Forest prediction:** {e}")

    elif model_choice == 'Naive Bayes' and naive_bayes_model:
        try:
            processed_data = preprocess_data(input_data, expected_feature_cols_nb)
            # Ensure the input data for Naive Bayes is of correct shape
            if processed_data.shape[1] == len(expected_feature_cols_nb):
                nb_pred = naive_bayes_model.predict(processed_data)
                st.write(f"**Naive Bayes Raw Prediction:** {nb_pred[0]}")
                result_nb = 'Accepted for Credit' if nb_pred[0] == 1 else 'Rejected for Credit'
                st.write(f"### Naive Bayes Result: **{result_nb}**")
            else:
                st.write("**Error:** The input data does not match the expected feature set for Naive Bayes.")
        except Exception as e:
            st.write(f"**Error in Naive Bayes prediction:** {e}")

    else:
        st.write("**Error:** The selected model failed to load or is not available.")
