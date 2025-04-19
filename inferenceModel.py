import streamlit as st
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

# Set page configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="✅",
    layout="wide"
)

# Constants
FEATURE_RANGES = {
    'person_age': (20, 144),
    'person_income': (8000, 150000),
    'person_emp_exp': (0, 125),
    'loan_amnt': (500, 35000),
    'loan_int_rate': (5.42, 20.00),
    'loan_percent_income': (0.0, 0.66),
    'cb_person_cred_hist_length': (2, 30),
    'credit_score': (390, 850)
}

EDUCATION_ORDER = {'High School': 1, 'Associate': 2, 'Bachelor': 3, 'Master': 4, 'Doctorate': 5}

# Model loading
def load_model():
    try:
        with open('xgboost_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Input collection from UI
def collect_user_input():
    features = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        
        features['person_age'] = st.slider("Age", 
                                          min_value=FEATURE_RANGES['person_age'][0], 
                                          max_value=FEATURE_RANGES['person_age'][1], 
                                          value=FEATURE_RANGES['person_age'][0])
        
        features['person_income'] = st.slider("Annual Income ($)", 
                                             min_value=FEATURE_RANGES['person_income'][0], 
                                             max_value=FEATURE_RANGES['person_income'][1], 
                                             value=FEATURE_RANGES['person_income'][0],
                                             step=1000)
        
        features['person_emp_exp'] = st.slider("Employment Experience (years)", 
                                             min_value=FEATURE_RANGES['person_emp_exp'][0], 
                                             max_value=FEATURE_RANGES['person_emp_exp'][1], 
                                             value=FEATURE_RANGES['person_emp_exp'][0])
        
        features['person_gender'] = st.radio("Gender", ['female', 'male'])
        
        education = st.selectbox("Education", list(EDUCATION_ORDER.keys()))
        features['education_level'] = EDUCATION_ORDER[education]
        
        features['person_home_ownership'] = st.selectbox("Home Ownership", 
                                                       ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
        
        features['cb_person_cred_hist_length'] = st.slider("Credit History Length (years)", 
                                                         min_value=FEATURE_RANGES['cb_person_cred_hist_length'][0], 
                                                         max_value=FEATURE_RANGES['cb_person_cred_hist_length'][1], 
                                                         value=FEATURE_RANGES['cb_person_cred_hist_length'][0])
        
        features['credit_score'] = st.slider("Credit Score", 
                                           min_value=FEATURE_RANGES['credit_score'][0], 
                                           max_value=FEATURE_RANGES['credit_score'][1], 
                                           value=FEATURE_RANGES['credit_score'][0],
                                           step=10)

    with col2:
        st.subheader("Loan Information")
        
        features['loan_amnt'] = st.slider("Loan Amount ($)", 
                                        min_value=FEATURE_RANGES['loan_amnt'][0], 
                                        max_value=FEATURE_RANGES['loan_amnt'][1], 
                                        value=FEATURE_RANGES['loan_amnt'][0],
                                        step=500)
        
        features['loan_int_rate'] = st.slider("Interest Rate (%)", 
                                           min_value=FEATURE_RANGES['loan_int_rate'][0], 
                                           max_value=FEATURE_RANGES['loan_int_rate'][1], 
                                           value=FEATURE_RANGES['loan_int_rate'][0],
                                           step=0.05)
        
        features['loan_percent_income'] = st.slider("Loan Percent of Income", 
                                                 min_value=FEATURE_RANGES['loan_percent_income'][0], 
                                                 max_value=FEATURE_RANGES['loan_percent_income'][1], 
                                                 value=FEATURE_RANGES['loan_percent_income'][0],
                                                 step=0.01)
        
        features['loan_intent'] = st.selectbox("Loan Intent", 
                                            ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'])
        
        features['previous_loan_defaults_on_file'] = st.radio("Previous Loan Defaults", ['No', 'Yes'])
        
    return features

# Preprocessing function
def preprocess_input(features_dict):
    df = pd.DataFrame([features_dict])
    
    # Use get_dummies for one-hot encoding
    df = pd.get_dummies(df)
    
    # Expected columns in the exact order
    expected_columns = [
        'person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
        'credit_score', 'education_level', 'person_gender_female',
        'person_gender_male', 'person_home_ownership_MORTGAGE',
        'person_home_ownership_OTHER', 'person_home_ownership_OWN',
        'person_home_ownership_RENT', 'loan_intent_DEBTCONSOLIDATION',
        'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT',
        'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE',
        'previous_loan_defaults_on_file_No',
        'previous_loan_defaults_on_file_Yes'
    ]
    
    # df = df[expected_columns]
    
    return df

# Prediction function
def make_prediction(model, features):
    if model is None:
        st.error("Model not loaded. Please check your model file.")
        return None
        
    try:
        prediction = model.predict(features)[0]
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.error(f"Features shape: {features.shape}")
        st.error(f"Features columns: {features.columns.tolist()}")
        return None

# Display results function
def display_results(prediction, features):
    st.markdown("## Prediction Result")
    if prediction == 1:
        st.success("**Loan Status: APPROVED**")
        st.write("Based on the provided information, this applicant is likely to be approved for the loan.")
    else:
        st.error("**Loan Status: NOT APPROVED**")
        st.write("Based on the provided information, this applicant is unlikely to be approved for the loan.")
    
    # Show key decision factors
    st.markdown("### Key Decision Factors")
    
    debt_to_income = features['loan_amnt'] / features['person_income']
    income_per_exp = features['person_income'] / (features['person_emp_exp'] + 1)
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Debt-to-Income Ratio", f"{debt_to_income:.2f}")
    with metric_col2:
        st.metric("Income per Year of Experience", f"${income_per_exp:.2f}")
    with metric_col3:
        st.metric("Credit Score", features['credit_score'])

# Display sidebar information
def show_info():
    st.sidebar.header("About")
    st.sidebar.info("""
    This app uses an XGBoost machine learning model to predict loan approval status based on applicant information.
    
    The model was trained on historical loan data with features including age, income, education level, credit history, and loan details.
    """)
    
    st.sidebar.header("Key Features")
    st.sidebar.markdown("""
    - **Personal**: Age, Income, Gender, Education, Employment Experience, Home Ownership
    - **Credit**: Credit Score, Credit History Length, Previous Defaults
    - **Loan**: Amount, Interest Rate, Purpose, Percent of Income
    """)

# Main application
def main():
    st.title('✅ Loan Approval Prediction')
    st.write("""
    This application predicts whether a loan will be approved based on various applicant characteristics.
    Fill in the information below to get a prediction.
    """)
    
    # Load model once when app starts
    model = load_model()
    
    # Collect user inputs
    features = collect_user_input()
    
    # Make prediction when button is clicked
    if st.button('Predict Loan Approval'):
        # Preprocess the input
        processed_features = preprocess_input(features)
        
        # Make prediction
        prediction = make_prediction(model, processed_features)
        
        # Display results
        if prediction is not None:
            display_results(prediction, features)

if __name__ == '__main__':
    main()
    show_info()
