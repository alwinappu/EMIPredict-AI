import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = "models/emi_eligibility_model.pkl"

def run():
    st.header("EMI Eligibility Prediction")
    st.info("This page loads a saved classifier (default: Random Forest).")

    st.sidebar.header("Applicant info")

    # Basic applicant inputs
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30, key="elig_age")
    monthly_salary = st.sidebar.number_input("Monthly salary", value=50000, key="elig_monthly_salary")
    years_of_employment = st.sidebar.number_input("Years of employment", value=3, key="elig_years_of_employment")
    monthly_rent = st.sidebar.number_input("Monthly rent", value=0, key="elig_monthly_rent")
    family_size = st.sidebar.number_input("Family size", value=4, min_value=1, key="elig_family_size")
    dependents = st.sidebar.number_input("Dependents", value=1, min_value=0, key="elig_dependents")
    credit_score = st.sidebar.number_input("Credit score", value=700, min_value=0, key="elig_credit_score")
    requested_amount = st.sidebar.number_input("Requested loan amount", value=200000, key="elig_requested_amount")
    requested_tenure = st.sidebar.number_input("Requested tenure (months)", value=36, key="elig_requested_tenure")

    # Extra financial factors (new fields your model expects)
    current_emi_amount = st.sidebar.number_input("Current EMI amount", value=0, key="elig_current_emi_amount")
    college_fees = st.sidebar.number_input("College fees", value=0, key="elig_college_fees")
    school_fees = st.sidebar.number_input("School fees", value=0, key="elig_school_fees")
    emergency_fund = st.sidebar.number_input("Emergency fund (₹)", value=0, key="elig_emergency_fund")
    groceries_utilities = st.sidebar.number_input("Groceries & Utilities", value=0, key="elig_groceries_utilities")
    bank_balance = st.sidebar.number_input("Bank balance", value=0, key="elig_bank_balance")
    travel_expenses = st.sidebar.number_input("Travel expenses", value=0, key="elig_travel_expenses")
    other_monthly_expenses = st.sidebar.number_input("Other monthly expenses", value=0, key="elig_other_monthly_expenses")

    # Categorical inputs
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"], key="elig_gender")
    marital_status = st.sidebar.selectbox("Marital status", ["Single", "Married"], key="elig_marital_status")
    education = st.sidebar.selectbox("Education", ["Graduate", "Bachelor", "Diploma", "Master", "PhD"], key="elig_education")
    employment_type = st.sidebar.selectbox("Employment type", ["Private", "Government", "Self-employed"], key="elig_employment_type")
    company_type = st.sidebar.selectbox("Company type", ["Private", "Public", "Medium", "Small"], key="elig_company_type")
    house_type = st.sidebar.selectbox("House type", ["Own", "Rented"], key="elig_house_type")
    existing_loans = st.sidebar.selectbox("Existing loans", ["Yes", "No"], key="elig_existing_loans")
    emi_scenario = st.sidebar.selectbox("EMI scenario", ["Normal", "E-commerce", "Other"], key="elig_emi_scenario")

    # Build input DataFrame
    X = pd.DataFrame({
        "age": [age],
        "monthly_salary": [monthly_salary],
        "years_of_employment": [years_of_employment],
        "monthly_rent": [monthly_rent],
        "family_size": [family_size],
        "dependents": [dependents],
        "credit_score": [credit_score],
        "requested_amount": [requested_amount],
        "requested_tenure": [requested_tenure],
        "current_emi_amount": [current_emi_amount],
        "college_fees": [college_fees],
        "school_fees": [school_fees],
        "emergency_fund": [emergency_fund],
        "groceries_utilities": [groceries_utilities],
        "bank_balance": [bank_balance],
        "travel_expenses": [travel_expenses],
        "other_monthly_expenses": [other_monthly_expenses],
        "gender": [gender],
        "marital_status": [marital_status],
        "education": [education],
        "employment_type": [employment_type],
        "company_type": [company_type],
        "house_type": [house_type],
        "existing_loans": [existing_loans],
        "emi_scenario": [emi_scenario]
    })

    st.write("Input preview:")
    st.dataframe(X)

    if not os.path.exists(MODEL_PATH):
        st.warning(f"No model found at {MODEL_PATH}. Place a saved model in models/")
        return

    if st.button("Predict eligibility", key="elig_predict_button"):
        try:
            model = joblib.load(MODEL_PATH)

            # Fill missing columns automatically (safe fallback)
            if hasattr(model, "feature_names_in_"):
                missing = [c for c in model.feature_names_in_ if c not in X.columns]
                if missing:
                    st.warning(f"Auto-filling missing columns: {missing}")
                    for col in missing:
                        X[col] = 0
                X = X.reindex(columns=model.feature_names_in_, fill_value=0)

            pred = model.predict(X)[0]
            st.success(f"Predicted EMI eligibility: **{pred}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
# --- Auto-added alignment helper ---
def align_features_for_model(X, model):
    import pandas as pd
    required = list(getattr(model, "feature_names_in_", []))
    if not required:
        return X, []
    categorical = {"gender","marital_status","education","employment_type",
                   "company_type","house_type","existing_loans","emi_scenario"}
    X2 = X.copy()
    added = []
    for col in required:
        if col not in X2.columns:
            X2[col] = "Unknown" if col in categorical else 0
            added.append(col)
    X2 = X2.loc[:, required]
    return X2, added

# --- Safe prediction wrapper ---
try:
    X_aligned, added_cols = align_features_for_model(X, model)
    if added_cols:
        st.warning(f"Added default values for missing columns: {added_cols}")
    pred = model.predict(X_aligned)
    st.success(f"Predicted EMI eligibility: **{pred[0]}**")
except Exception as e:
    st.error(f"Prediction failed after alignment: {e}")
