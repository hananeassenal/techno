import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the models and scaler
logistic_regression_model = joblib.load('logistic_regression_model.pkl')
decision_tree_model = joblib.load('decision_tree_model.pkl')
random_forest_model = joblib.load('random_forest_model.pkl')
naive_bayes_model = joblib.load('naive_bayes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app
st.title('Loan Credit Prediction')

# Input features
st.sidebar.header('User Input Features')

def user_input_features():
    credit_score = st.sidebar.slider('Credit Score', min_value=300, max_value=850, value=650)
    mip = st.sidebar.slider('MIP', min_value=0.0, max_value=1.0, value=0.5)
    dti = st.sidebar.slider('DTI', min_value=0.0, max_value=1.0, value=0.3)
    ever_delinquent = st.sidebar.slider('Ever Delinquent', min_value=0, max_value=1, value=0)
    months_delinquent = st.sidebar.slider('Months Delinquent', min_value=0, max_value=12, value=0)
    months_in_repayment = st.sidebar.slider('Months In Repayment', min_value=0, max_value=360, value=60)
    
    data = {
        'CreditScore': credit_score,
        'MIP': mip,
        'DTI': dti,
        'EverDelinquent': ever_delinquent,
        'MonthsDelinquent': months_delinquent,
        'MonthsInRepayment': months_in_repayment
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_data = user_input_features()

# Scale the data
scaled_input = scaler.transform(input_data)

# Predictions
logistic_regression_prediction = logistic_regression_model.predict(scaled_input)
decision_tree_prediction = decision_tree_model.predict(scaled_input)
random_forest_prediction = random_forest_model.predict(scaled_input)
naive_bayes_prediction = naive_bayes_model.predict(scaled_input)

# Display results
st.subheader('Model Predictions')

st.write("Logistic Regression Prediction: ", "Accepted for Credit" if logistic_regression_prediction[0] == 1 else "Not Accepted for Credit")
st.write("Decision Tree Prediction: ", "Accepted for Credit" if decision_tree_prediction[0] == 1 else "Not Accepted for Credit")
st.write("Random Forest Prediction: ", "Accepted for Credit" if random_forest_prediction[0] == 1 else "Not Accepted for Credit")
st.write("Naive Bayes Prediction: ", "Accepted for Credit" if naive_bayes_prediction[0] == 1 else "Not Accepted for Credit")

# If you want to show the feature importance from the models
st.subheader('Model Feature Importances')

if hasattr(decision_tree_model, 'feature_importances_'):
    feature_importances = decision_tree_model.feature_importances_
    features = input_data.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    st.write(importance_df)
