import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")
st.title("🏦 Loan Approval Predictor")
st.write("Enter applicant details to predict credit/loan risk evaluation.")

# Path to the serialized model
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(current_dir, "..", "models", "final_classifier.pkl"))

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}: {e}")
        return None

model = load_model()

with st.form("loan_prediction_form"):
    st.subheader("Demographics & Employment")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        annual_income = st.number_input("Annual Income ($)", min_value=1000, value=65000, step=1000)
        occupation_status = st.selectbox(
            "Occupation Status", 
            ["EMPLOYED", "SELF_EMPLOYED", "UNEMPLOYED", "OTHER"]
        )
    with col2:
        years_employed = st.number_input("Years Employed", min_value=0, max_value=60, value=5, step=1)
        savings_assets = st.number_input("Savings / Assets ($)", min_value=0, value=15000, step=1000)
        current_debt = st.number_input("Current Outstanding Debt ($)", min_value=0, value=5000, step=500)

    st.subheader("Loan Details")
    col3, col4 = st.columns(2)
    with col3:
        loan_amount = st.number_input("Requested Loan Amount ($)", min_value=500, value=15000, step=500)
        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=40.0, value=11.5, step=0.1)
    with col4:
        loan_intent = st.selectbox(
            "Loan Intent", 
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
        )
        product_type = st.selectbox(
            "Product Type", 
            ["UNSECURED", "SECURED", "REVOLVING", "INSTALLMENT"]
        )

    st.subheader("Credit Bureau History")
    col5, col6 = st.columns(2)
    with col5:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=710, step=5)
        credit_history_years = st.number_input("Credit History (Years)", min_value=0, max_value=50, value=8, step=1)
        delinquencies_last_2yrs = st.number_input("Delinquencies (Last 2 Yrs)", min_value=0, value=0, step=1)
    with col6:
        defaults_on_file = st.selectbox("Defaults on File?", ["N", "Y"])
        derogatory_marks = st.number_input("Derogatory Marks", min_value=0, value=0, step=1)

    submitted = st.form_submit_button("Predict Approval Status")

if submitted and model is not None:
    # Compute the engineered ratios expected by the feature pipeline
    income_val = max(annual_income, 1.0)
    loan_to_income = loan_amount / income_val
    debt_to_income = (current_debt + loan_amount) / income_val
    estimated_payment = (loan_amount * (1 + (interest_rate / 100))) / 12
    payment_to_income = estimated_payment / (income_val / 12)
    disposable_income = max(0.0, 1.0 - debt_to_income)

    # DataFrame with the exact 19 feature names matching the pipeline
    input_df = pd.DataFrame([{
        'age': age,
        'loan_intent': loan_intent,
        'derogatory_marks': derogatory_marks,
        'delinquencies_last_2yrs': delinquencies_last_2yrs,
        'defaults_on_file': defaults_on_file,
        'savings_assets': savings_assets,
        'current_debt': current_debt,
        'product_type': product_type,
        'payment_to_income_ratio': payment_to_income,
        'interest_rate': interest_rate,
        'loan_amount': loan_amount,
        'occupation_status': occupation_status,
        'credit_score': credit_score,
        'annual_income': annual_income,
        'debt_to_income_ratio': debt_to_income,
        'credit_history_years': credit_history_years,
        'disposable_income_ratio': disposable_income,
        'loan_to_income_ratio': loan_to_income,
        'years_employed': years_employed
    }])

    with st.spinner("Evaluating credit risk..."):
        try:
            pred = model.predict(input_df)[0]
            
            # Predict probabilities if supported by the champion model
            conf = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                conf = max(probs) * 100

            st.divider()
            
            # Check for binary 0/1 or 'Approved'/'Rejected' label formats
            is_approved = (pred == 1 or str(pred).lower() in ['1', 'approved', 'y', 'yes'])

            if is_approved:
                score_str = f" (Confidence: {conf:.1f}%)" if conf else ""
                st.success(f"### 🎉 Application Approved{score_str}")
                st.write("The applicant's debt profile and credit metrics meet the approval threshold.")
            else:
                score_str = f" (Confidence: {conf:.1f}%)" if conf else ""
                st.error(f"### ❌ Application Rejected{score_str}")
                st.write("The applicant's risk factors exceed the approval criteria.")

        except Exception as e:
            st.error(f"Prediction Pipeline Error: {e}")