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
scaler_path = 'scaler.pkl'

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
scaler = load_model(scaler_path)

# Define the features expected by the models and scaler
expected_feature_cols = [
    'CreditScore', 'MIP', 'DTI', 'EverDelinquent', 'MonthsDelinquent', 'MonthsInRepayment'
]

# Preprocess input data
def preprocess_data(data):
    df = pd.DataFrame([data])
    
    # Ensure the DataFrame has the correct columns
    df = df.reindex(columns=expected_feature_cols, fill_value=0)
    
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

# Preprocess input data
processed_data = preprocess_data(input_data)

# Display preprocessed data for debugging
st.write("### Input Data")
st.write(processed_data)

# Model selection
model_choice = st.sidebar.selectbox(
    'Select Model for Prediction',
    ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes']
)

# Make predictions
if st.sidebar.button('Predict'):
    if scaler:
        try:
            # Check the number of features in the input data
            if processed_data.shape[1] == 4:
                st.warning("The input data has 4 features, but the scaler expects 6 features.")
                scaled_data = None
            else:
                # Scale the data
                scaled_data = scaler.transform(processed_data)
                st.write("### Scaled Data")
                st.write(scaled_data)
        except Exception as e:
            st.write(f"**Error during scaling:** {e}")
            scaled_data = None
        
        if scaled_data is not None:
            if model_choice == 'Logistic Regression' and logistic_regression_model:
                try:
                    # Logistic Regression prediction
                    lr_pred = logistic_regression_model.predict(scaled_data)
                    st.write(f"**Logistic Regression Raw Prediction:** {lr_pred[0]}")
                    result_lr = 'Accepted for Credit' if lr_pred[0] == 1 else 'Rejected for Credit'
                    st.write(f"### Logistic Regression Result: **{result_lr}**")
                except Exception as e:
                    st.write(f"**Error in Logistic Regression prediction:** {e}")

            elif model_choice == 'Decision Tree' and decision_tree_model:
                try:
                    # Decision Tree prediction
                    dt_pred = decision_tree_model.predict(scaled_data)
                    st.write(f"**Decision Tree Raw Prediction:** {dt_pred[0]}")
                    result_dt = 'Accepted for Credit' if dt_pred[0] == 1 else 'Rejected for Credit'
                    st.write(f"### Decision Tree Result: **{result_dt}**")
                except Exception as e:
                    st.write(f"**Error in Decision Tree prediction:** {e}")

            elif model_choice == 'Random Forest' and random_forest_model:
                try:
                    # Random Forest prediction
                    rf_pred = random_forest_model.predict(scaled_data)
                    st.write(f"**Random Forest Raw Prediction:** {rf_pred[0]}")
                    result_rf = 'Accepted for Credit' if rf_pred[0] == 1 else 'Rejected for Credit'
                    st.write(f"### Random Forest Result: **{result_rf}**")
                except Exception as e:
                    st.write(f"**Error in Random Forest prediction:** {e}")

            elif model_choice == 'Naive Bayes' and naive_bayes_model:
                try:
                    # Naive Bayes prediction
                    nb_pred = naive_bayes_model.predict(scaled_data)
                    st.write(f"**Naive Bayes Raw Prediction:** {nb_pred[0]}")
                    result_nb = 'Accepted for Credit' if nb_pred[0] == 1 else 'Rejected for Credit'
                    st.write(f"### Naive Bayes Result: **{result_nb}**")
                except Exception as e:
                    st.write(f"**Error in Naive Bayes prediction:** {e}")

            else:
                st.write("**Error:** The selected model failed to load or is not available.")
    else:
        st.write("**Error:** Scaler not loaded or not available.")
