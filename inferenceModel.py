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

# Load the XGBoost model
def load_model():
    with open('xgboost_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Define numerical ranges based on dataset statistics
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

# Define the exact education mapping used during training
EDUCATION_ORDER = {'High School': 1, 'Associate': 2, 'Bachelor': 3, 'Master': 4, 'Doctorate': 5}

def main():
    st.title('✅ Loan Approval Prediction')
    st.write("""
    This application predicts whether a loan will be approved based on various applicant characteristics.
    Fill in the information below to get a prediction.
    """)
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    # Store all input features
    features = {}
    
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
        
        # Gender as categorical for one-hot encoding
        features['person_gender'] = st.radio("Gender", ['female', 'male'])
        
        # Education directly mapped to education_level numeric value
        education = st.selectbox("Education", 
                               list(EDUCATION_ORDER.keys()))
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
        
        # Previous defaults as Yes/No for one-hot encoding
        has_defaults = st.radio("Previous Loan Defaults", ['No', 'Yes'])
        features['previous_loan_defaults'] = has_defaults  # Will be one-hot encoded later

    # Button to make prediction
    if st.button('Predict Loan Approval'):
        if model is not None:
            result = make_prediction(features)
            
            # Display result with color
            st.markdown("## Prediction Result")
            if result == 1:
                st.success("**Loan Status: APPROVED**")
                st.write("Based on the provided information, this applicant is likely to be approved for the loan.")
            else:
                st.error("**Loan Status: NOT APPROVED**")
                st.write("Based on the provided information, this applicant is unlikely to be approved for the loan.")
                
            # Add some explanation about key factors
            st.markdown("### Key Decision Factors")
            
            # Engineering key ratios for display
            debt_to_income = features['loan_amnt'] / features['person_income']
            income_per_exp = features['person_income'] / (features['person_emp_exp'] + 1)
            
            # Show visualizations or key metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Debt-to-Income Ratio", f"{debt_to_income:.2f}")
            with metric_col2:
                st.metric("Income per Year of Experience", f"${income_per_exp:.2f}")
            with metric_col3:
                st.metric("Credit Score", features['credit_score'])
                
        else:
            st.error("Model not loaded. Please check your model file.")

def preprocess_input(features_dict):
    """
    Preprocess the input features to match the model's expected format exactly
    """
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame([features_dict])
    
    # One-hot encode categorical variables
    # Handle gender
    df['person_gender_female'] = (df['person_gender'] == 'female').astype(int)
    df['person_gender_male'] = (df['person_gender'] == 'male').astype(int)
    
    # Handle home ownership
    for category in ['MORTGAGE', 'OTHER', 'OWN', 'RENT']:
        df[f'person_home_ownership_{category}'] = (df['person_home_ownership'] == category).astype(int)
        
    # Handle loan intent
    for category in ['DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE']:
        df[f'loan_intent_{category}'] = (df['loan_intent'] == category).astype(int)
        
    # Handle previous loan defaults
    df['previous_loan_defaults_on_file_No'] = (df['previous_loan_defaults'] == 'No').astype(int)
    df['previous_loan_defaults_on_file_Yes'] = (df['previous_loan_defaults'] == 'Yes').astype(int)
    
    # Define the exact column order expected by the model
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

    # Drop the original categorical columns and select only expected columns
    df_final = df.drop(['person_gender', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults'], axis=1)
    
    # Ensure all expected columns exist
    for col in expected_columns:
        if col not in df_final.columns:
            df_final[col] = 0
    
    # Return only the columns needed by the model in the exact order
    return df_final[expected_columns]

def make_prediction(features):
    """
    Use the loaded model to make a prediction
    """
    # Preprocess the features
    processed_features = preprocess_input(features)
    
    # Make prediction
    try:
        prediction = model.predict(processed_features)[0]
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.error(f"Features shape: {processed_features.shape}")
        st.error(f"Features columns: {processed_features.columns.tolist()}")
        return None

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

if __name__ == '__main__':
    main()
    show_info()
