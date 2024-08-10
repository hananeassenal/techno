import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the scaler and models
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
    
    # Adjust to match the features used during model training
    data = {
        'CreditScore': credit_score,
        'MIP': mip,
        'DTI': dti,
        'EverDelinquent': ever_delinquent
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_data = user_input_features()

# Display input data
st.write("Input Data:")
st.write(input_data)

# Verify if the features match
expected_features = ['CreditScore', 'MIP', 'DTI', 'EverDelinquent']
if list(input_data.columns) != expected_features:
    st.write(f"Feature columns mismatch: Expected {expected_features}, but got {list(input_data.columns)}")
else:
    try:
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
    except Exception as e:
        st.write(f"Error during scaling or prediction: {e}")

    # Feature Importance
    st.subheader('Model Feature Importances')

    feature_importances = {}
    if hasattr(decision_tree_model, 'feature_importances_'):
        feature_importances['Decision Tree'] = decision_tree_model.feature_importances_
    if hasattr(random_forest_model, 'feature_importances_'):
        feature_importances['Random Forest'] = random_forest_model.feature_importances_

    if feature_importances:
        for model_name, importances in feature_importances.items():
            features = expected_features
            if len(features) == len(importances):
                importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
                st.write(f"{model_name} Feature Importances")
                st.write(importance_df)
            else:
                st.write(f"Feature importances length mismatch for {model_name}.")
    else:
        st.write("Feature importance is not available for the loaded models.")
