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
    'CreditScore', 'FirstPaymentDate', 'FirstTimeHomebuyer', 'MaturityDate', 'MSA', 'MIP', 'Units', 
    'Occupancy', 'OCLTV', 'DTI', 'OrigUPB', 'LTV', 'OrigInterestRate', 'Channel', 'PPM', 'ProductType', 
    'PropertyState', 'PropertyType', 'LoanPurpose', 'OrigLoanTerm', 'NumBorrowers', 'SellerName', 
    'ServicerName', 'EverDelinquent', 'MonthsDelinquent', 'MonthsInRepayment', 'CreditScoreGroup', 
    'RepaymentYearsGroup', 'LTVGroup'
]

# Define preprocessing function
def preprocess_data(data):
    # Ensure input data has the correct feature columns
    df = pd.DataFrame([data])
    
    # Align features
    df = df.reindex(columns=training_feature_cols, fill_value=0)  # Ensure all features are present
    
    # Apply any additional preprocessing if needed (e.g., scaling, encoding)
    
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
    'Units': st.sidebar.number_input('Units', value=1.0, format="%.2f"),
    'Occupancy': st.sidebar.selectbox('Occupancy', ['Owner Occupied', 'Investment', 'Second Home']),
    'OCLTV': st.sidebar.number_input('OCLTV', value=80.0, format="%.2f"),
    'DTI': st.sidebar.number_input('DTI', value=0.35, format="%.2f"),
    'OrigUPB': st.sidebar.number_input('OrigUPB', value=180000.0, format="%.2f"),
    'LTV': st.sidebar.number_input('LTV', value=80.0, format="%.2f"),
    'OrigInterestRate': st.sidebar.number_input('OrigInterestRate', value=3.5, format="%.2f"),
    'Channel': st.sidebar.selectbox('Channel', ['Retail', 'Wholesale']),
    'PPM': st.sidebar.selectbox('PPM', ['Yes', 'No']),
    'ProductType': st.sidebar.selectbox('ProductType', ['Fixed', 'ARM']),
    'PropertyState': st.sidebar.text_input('PropertyState', value='CA'),
    'PropertyType': st.sidebar.selectbox('PropertyType', ['Single Family', 'Condo']),
    'LoanPurpose': st.sidebar.selectbox('LoanPurpose', ['Purchase', 'Refinance']),
    'OrigLoanTerm': st.sidebar.number_input('OrigLoanTerm', value=360.0, format="%.2f"),
    'NumBorrowers': st.sidebar.number_input('NumBorrowers', value=1.0, format="%.2f"),
    'SellerName': st.sidebar.text_input('SellerName', value='ABC Corp'),
    'ServicerName': st.sidebar.text_input('ServicerName', value='XYZ Servicer'),
    'EverDelinquent': st.sidebar.selectbox('EverDelinquent', ['0', '1']),
    'MonthsDelinquent': st.sidebar.number_input('MonthsDelinquent', value=3.0, format="%.2f"),
    'MonthsInRepayment': st.sidebar.number_input('MonthsInRepayment', value=24.0, format="%.2f"),
    'CreditScoreGroup': st.sidebar.selectbox('CreditScoreGroup', ['Low', 'Medium', 'High']),
    'RepaymentYearsGroup': st.sidebar.selectbox('RepaymentYearsGroup', ['Short', 'Medium', 'Long']),
    'LTVGroup': st.sidebar.selectbox('LTVGroup', ['Low', 'Medium', 'High'])
}

# Convert categorical inputs
input_data['FirstTimeHomebuyer'] = 1 if input_data['FirstTimeHomebuyer'] == 'Yes' else 0
input_data['Occupancy'] = 1 if input_data['Occupancy'] == 'Owner Occupied' else (2 if input_data['Occupancy'] == 'Investment' else 3)
input_data['Channel'] = 1 if input_data['Channel'] == 'Retail' else 2
input_data['PPM'] = 1 if input_data['PPM'] == 'Yes' else 0
input_data['ProductType'] = 1 if input_data['ProductType'] == 'Fixed' else 2
input_data['PropertyType'] = 1 if input_data['PropertyType'] == 'Single Family' else 2
input_data['LoanPurpose'] = 1 if input_data['LoanPurpose'] == 'Purchase' else 2
input_data['CreditScoreGroup'] = 1 if input_data['CreditScoreGroup'] == 'Low' else (2 if input_data['CreditScoreGroup'] == 'Medium' else 3)
input_data['RepaymentYearsGroup'] = 1 if input_data['RepaymentYearsGroup'] == 'Short' else (2 if input_data['RepaymentYearsGroup'] == 'Medium' else 3)
input_data['LTVGroup'] = 1 if input_data['LTVGroup'] == 'Low' else (2 if input_data['LTVGroup'] == 'Medium' else 3)
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
