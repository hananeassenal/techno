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
lasso_regression_model_path = 'lasso_regression_model.pkl'

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
lasso_regression_model = load_model(lasso_regression_model_path)

# Define the features expected by each model
expected_feature_cols = {
    'logistic_regression': ['CreditScore', 'MIP', 'DTI', 'MonthsDelinquent', 'MonthsInRepayment'],
    'decision_tree': ['CreditScore', 'MIP', 'DTI', 'MonthsDelinquent', 'MonthsInRepayment'],
    'random_forest': ['CreditScore', 'MIP', 'DTI', 'MonthsDelinquent', 'MonthsInRepayment'],
    'naive_bayes': ['CreditScore', 'MIP', 'DTI', 'MonthsDelinquent', 'MonthsInRepayment'],
    'lasso_regression': ['CreditScore', 'Units', 'DTI', 'OrigUPB', 'OrigInterestRate', 'MIP',
                         'MonthsDelinquent', 'OrigLoanTerm', 'MonthsInRepayment', 'EMI',
                         'interest_amt', 'MonthlyIncome', 'cur_principal']
}

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

# Input fields
input_data = {
    'CreditScore': st.sidebar.number_input('CreditScore', value=550.0, format="%.2f"),
    'MIP': st.sidebar.number_input('MIP', value=0.5, format="%.2f"),
    'DTI': st.sidebar.number_input('DTI', value=0.55, format="%.2f"),
    'EverDelinquent': st.sidebar.selectbox('EverDelinquent', ['1', '0']),
    'MonthsDelinquent': st.sidebar.number_input('MonthsDelinquent', value=12.0, format="%.2f"),
    'MonthsInRepayment': st.sidebar.number_input('MonthsInRepayment', value=6.0, format="%.2f"),
    'Units': st.sidebar.number_input('Units', value=1.0, format="%.2f"),
    'OrigUPB': st.sidebar.number_input('OrigUPB', value=100000.0, format="%.2f"),
    'OrigInterestRate': st.sidebar.number_input('OrigInterestRate', value=0.04, format="%.2f"),
    'OrigLoanTerm': st.sidebar.number_input('OrigLoanTerm', value=360.0, format="%.2f"),
    'EMI': st.sidebar.number_input('EMI', value=500.0, format="%.2f"),
    'interest_amt': st.sidebar.number_input('interest_amt', value=200.0, format="%.2f"),
    'MonthlyIncome': st.sidebar.number_input('MonthlyIncome', value=3000.0, format="%.2f"),
    'cur_principal': st.sidebar.number_input('cur_principal', value=50000.0, format="%.2f")
}

# Convert categorical inputs
input_data['EverDelinquent'] = int(input_data['EverDelinquent'])

# Model selection
model_choice = st.sidebar.selectbox(
    'Select Model for Prediction',
    ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes', 'Lasso Regression']
)

# Preprocess input data for the selected model
feature_cols = expected_feature_cols.get(model_choice.lower().replace(' ', '_'), [])
processed_data = preprocess_data(input_data, feature_cols)

# Display preprocessed data for debugging
st.write("### Input Data")
st.write(processed_data)

# Make predictions
if st.sidebar.button('Predict'):
    if model_choice == 'Logistic Regression' and logistic_regression_model:
        try:
            lr_pred = logistic_regression_model.predict(processed_data)
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

    elif model_choice == 'Random Forest' and random_forest_model:
        try:
            rf_pred = random_forest_model.predict(processed_data)
            st.write(f"**Random Forest Raw Prediction:** {rf_pred[0]}")
            result_rf = 'Accepted for Credit' if rf_pred[0] == 1 else 'Rejected for Credit'
            st.write(f"### Random Forest Result: **{result_rf}**")
        except Exception as e:
            st.write(f"**Error in Random Forest prediction:** {e}")

    elif model_choice == 'Naive Bayes' and naive_bayes_model:
        try:
            nb_pred = naive_bayes_model.predict(processed_data)
            st.write(f"**Naive Bayes Raw Prediction:** {nb_pred[0]}")
            result_nb = 'Accepted for Credit' if nb_pred[0] == 1 else 'Rejected for Credit'
            st.write(f"### Naive Bayes Result: **{result_nb}**")
        except Exception as e:
            st.write(f"**Error in Naive Bayes prediction:** {e}")

    elif model_choice == 'Lasso Regression' and lasso_regression_model:
        try:
            lasso_pred = lasso_regression_model.predict(processed_data)
            st.write(f"**Lasso Regression Raw Prediction:** {lasso_pred[0]}")
            result_lasso = 'Accepted for Credit' if lasso_pred[0] > 0 else 'Rejected for Credit'
            st.write(f"### Lasso Regression Result: **{result_lasso}**")
        except Exception as e:
            st.write(f"**Error in Lasso Regression prediction:** {e}")

    else:
        st.write("**Error:** The selected model failed to load or is not available.")
