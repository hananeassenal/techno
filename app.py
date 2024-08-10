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

# Define the correct feature columns as used during training
training_feature_cols = [
    'CreditScore', 'FirstPaymentDate', 'FirstTimeHomebuyer', 'MaturityDate', 'MSA', 'MIP', 'Units', 'Occupancy', 
    'OCLTV', 'DTI', 'OrigUPB', 'LTV', 'OrigInterestRate', 'Channel', 'PPM', 'ProductType', 'PropertyState', 
    'PropertyType', 'LoanPurpose', 'OrigLoanTerm', 'NumBorrowers', 'SellerName', 'ServicerName', 
    'EverDelinquent', 'MonthsDelinquent', 'MonthsInRepayment', 'CreditScoreGroup', 'RepaymentYearsGroup', 'LTVGroup'
]

# Define preprocessing function
def preprocess_data(data):
    df = pd.DataFrame([data])
    
    # Ensure all feature columns are present
    df = df.reindex(columns=training_feature_cols, fill_value=0)
    
    # Convert categorical columns to appropriate data types
    categorical_cols = ['FirstTimeHomebuyer', 'PPM', 'EverDelinquent']
    df[categorical_cols] = df[categorical_cols].astype(int)
    
    # Handle date columns if needed
    if 'FirstPaymentDate' in df.columns:
        df['FirstPaymentDate'] = pd.to_datetime(df['FirstPaymentDate'], format='%Y-%m-%d', errors='coerce')
    if 'MaturityDate' in df.columns:
        df['MaturityDate'] = pd.to_datetime(df['MaturityDate'], format='%Y-%m-%d', errors='coerce')
    
    # Additional preprocessing steps if required

    return df

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

# Sample values for testing
input_data = {
    'CreditScore': st.sidebar.number_input('CreditScore', value=700.0, format="%.2f"),
    'FirstPaymentDate': st.sidebar.date_input('FirstPaymentDate'),
    'FirstTimeHomebuyer': st.sidebar.selectbox('FirstTimeHomebuyer', ['Yes', 'No']),
    'MaturityDate': st.sidebar.date_input('MaturityDate'),
    'MSA': st.sidebar.number_input('MSA', value=100.0, format="%.2f"),
    'MIP': st.sidebar.number_input('MIP', value=0.0, format="%.2f"),
    'Units': st.sidebar.number_input('Units', value=1, format="%d"),
    'Occupancy': st.sidebar.text_input('Occupancy'),
    'OCLTV': st.sidebar.number_input('OCLTV', value=80.0, format="%.2f"),
    'DTI': st.sidebar.number_input('DTI', value=0.35, format="%.2f"),
    'OrigUPB': st.sidebar.number_input('OrigUPB', value=180000.0, format="%.2f"),
    'LTV': st.sidebar.number_input('LTV', value=80.0, format="%.2f"),
    'OrigInterestRate': st.sidebar.number_input('OrigInterestRate', value=3.5, format="%.2f"),
    'Channel': st.sidebar.text_input('Channel'),
    'PPM': st.sidebar.selectbox('PPM', ['Yes', 'No']),
    'ProductType': st.sidebar.text_input('ProductType'),
    'PropertyState': st.sidebar.text_input('PropertyState'),
    'PropertyType': st.sidebar.text_input('PropertyType'),
    'LoanPurpose': st.sidebar.text_input('LoanPurpose'),
    'OrigLoanTerm': st.sidebar.number_input('OrigLoanTerm', value=30, format="%d"),
    'NumBorrowers': st.sidebar.number_input('NumBorrowers', value=1, format="%d"),
    'SellerName': st.sidebar.text_input('SellerName'),
    'ServicerName': st.sidebar.text_input('ServicerName'),
    'EverDelinquent': st.sidebar.selectbox('EverDelinquent', ['0', '1']),
    'MonthsDelinquent': st.sidebar.number_input('MonthsDelinquent', value=3.0, format="%.2f"),
    'MonthsInRepayment': st.sidebar.number_input('MonthsInRepayment', value=24.0, format="%.2f"),
    'CreditScoreGroup': st.sidebar.text_input('CreditScoreGroup'),
    'RepaymentYearsGroup': st.sidebar.text_input('RepaymentYearsGroup'),
    'LTVGroup': st.sidebar.text_input('LTVGroup')
}

# Convert categorical inputs
input_data['FirstTimeHomebuyer'] = 1 if input_data['FirstTimeHomebuyer'] == 'Yes' else 0
input_data['PPM'] = 1 if input_data['PPM'] == 'Yes' else 0
input_data['EverDelinquent'] = int(input_data['EverDelinquent'])

input_df = preprocess_data(input_data)

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
