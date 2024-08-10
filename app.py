import streamlit as st
import joblib
import pandas as pd
import os

# Define paths to the model files
logistic_regression_model_path = 'logistic_regression_model.pkl'
decision_tree_model_path = 'decision_tree_model.pkl'
naive_bayes_model_path = 'naive_bayes_model.pkl'
lasso_regression_model_path = 'lasso_regression_pipeline.joblib'

# Function to load models
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
lasso_regression_model = load_model(lasso_regression_model_path)

# Define the features expected by each model
features = {
    'CreditScore': float,
    'Units': float,
    'DTI': float,
    'OrigUPB': float,
    'OrigInterestRate': float,
    'MIP': float,
    'MonthsDelinquent': float,
    'OrigLoanTerm': float,
    'MonthsInRepayment': float,
    'EMI': float,
    'interest_amt': float,
    'MonthlyIncome': float,
    'cur_principal': float
}

# Streamlit UI setup
st.set_page_config(page_title='Loan Credit Prediction', layout='wide')
st.title('Loan Credit Prediction Web Application')

st.sidebar.header('Input Features')
st.sidebar.write("Enter the feature values below:")

# Input form
input_data = {key: st.sidebar.number_input(key, format="%.2f") for key in features.keys()}

# Convert input data to DataFrame
def preprocess_data(data):
    df = pd.DataFrame([data])
    # Ensure the DataFrame has the correct columns
    df = df[sorted(data.keys())]
    # Handle missing values
    df = df.fillna(0)
    return df

processed_data = preprocess_data(input_data)

# Display preprocessed data for debugging
st.write("### Input Data")
st.write(processed_data)

# Model selection
model_choice = st.sidebar.selectbox(
    'Select Model for Prediction',
    ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'Lasso Regression']
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

    elif model_choice == 'Lasso Regression' and lasso_regression_model:
        try:
            # Lasso Regression prediction
            lr_pred = lasso_regression_model.predict(processed_data)
            st.write(f"**Lasso Regression Raw Prediction:** {lr_pred[0]}")
            result_lr = 'Accepted for Credit' if lr_pred[0] >= 0.5 else 'Rejected for Credit'
            st.write(f"### Lasso Regression Result: **{result_lr}**")
        except Exception as e:
            st.write(f"**Error in Lasso Regression prediction:** {e}")

    else:
        st.write("**Error:** The selected model failed to load or is not available.")
