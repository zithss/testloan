import streamlit as st
import pickle
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="✅",
    layout="wide"
)

# Load the XGBoost model
@st.cache_resource
def load_model():
    with open('xgboost_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Define numerical ranges
FEATURE_RANGES = {
    'person_age': (20, 144, 28),
    'person_income': (8000, 150000, 80000),
    'person_emp_exp': (0, 125, 5),
    'loan_amnt': (500, 35000, 10000),
    'loan_int_rate': (5.42, 20.00, 11.00),
    'loan_percent_income': (0.0, 0.66, 0.14),
    'cb_person_cred_hist_length': (2, 30, 6),
    'credit_score': (390, 850, 630)
}

def main():
    st.title('✅ Loan Approval Prediction')
    st.write("This application predicts loan approval based on applicant characteristics.")
    
    col1, col2 = st.columns(2)
    features = {}
    
    with col1:
        st.subheader("Personal Information")
        features['person_age'] = st.slider("Age", *FEATURE_RANGES['person_age'])
        features['person_income'] = st.slider("Annual Income ($)", *FEATURE_RANGES['person_income'], step=1000)
        features['person_emp_exp'] = st.slider("Employment Experience (years)", *FEATURE_RANGES['person_emp_exp'])
        features['person_gender'] = st.radio("Gender", ['male', 'female'])
        features['person_education'] = st.selectbox("Education", ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'])
        features['person_home_ownership'] = st.selectbox("Home Ownership", ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
        features['cb_person_cred_hist_length'] = st.slider("Credit History Length (years)", *FEATURE_RANGES['cb_person_cred_hist_length'])
        features['credit_score'] = st.slider("Credit Score", *FEATURE_RANGES['credit_score'], step=10)

    with col2:
        st.subheader("Loan Information")
        features['loan_amnt'] = st.slider("Loan Amount ($)", *FEATURE_RANGES['loan_amnt'], step=500)
        features['loan_int_rate'] = st.slider("Interest Rate (%)", *FEATURE_RANGES['loan_int_rate'], step=0.05)
        features['loan_percent_income'] = st.slider("Loan Percent of Income", *FEATURE_RANGES['loan_percent_income'], step=0.01)
        features['loan_intent'] = st.selectbox("Loan Purpose", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'])
        features['previous_loan_defaults_on_file'] = st.radio("Previous Defaults", ['No', 'Yes'])

    if st.button('Predict Loan Approval'):
        if model:
            try:
                processed = preprocess_input(features)
                prediction = model.predict(processed)[0]
                display_result(prediction, features)
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.error("Model not loaded")

def preprocess_input(features):
    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    # Handle education as ordinal feature
    edu_map = {'High School':0, 'Associate':1, 'Bachelor':2, 'Master':3, 'Doctorate':4}
    df['education_level'] = df['person_education'].map(edu_map)
    
    # Convert gender to binary
    df['person_gender_female'] = (df['person_gender'] == 'female').astype(int)
    
    # Handle previous defaults
    df['previous_loan_defaults_on_file_Yes'] = (df['previous_loan_defaults_on_file'] == 'Yes').astype(int)
    df['previous_loan_defaults_on_file_No'] = (df['previous_loan_defaults_on_file'] == 'No').astype(int)
    
    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent'])
    
    # Ensure all expected columns are present
    expected_columns = [
        'person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
        'credit_score', 'education_level', 'person_gender_female',
        'person_home_ownership_MORTGAGE', 'person_home_ownership_OTHER',
        'person_home_ownership_OWN', 'person_home_ownership_RENT',
        'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION',
        'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
        'loan_intent_PERSONAL', 'loan_intent_VENTURE',
        'previous_loan_defaults_on_file_Yes', 'previous_loan_defaults_on_file_No'
    ]
    
    # Add missing columns with 0 values
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    return df[expected_columns]

def display_result(prediction, features):
    st.markdown("## Prediction Result")
    if prediction == 1:
        st.success("**Loan Status: APPROVED**")
    else:
        st.error("**Loan Status: NOT APPROVED**")
    
    st.markdown("### Key Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Debt-to-Income", f"{(features['loan_amnt']/features['person_income']):.2f}")
    with col2:
        st.metric("Credit Score", features['credit_score'])
    with col3:
        st.metric("Interest Rate", f"{features['loan_int_rate']}%")

def show_info():
    st.sidebar.header("About")
    st.sidebar.info("""
    This app predicts loan approvals using an XGBoost model trained on historical data.
    Key features include credit score, income, loan details, and financial history.
    """)

if __name__ == '__main__':
    main()
    show_info()
