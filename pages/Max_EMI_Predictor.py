# pages/Max_EMI_Predictor.py
import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

MODEL_PATH = os.path.join(os.getcwd(), "models", "regression_rf.pkl")
DATA_PATH = os.path.join(os.getcwd(), "data", "sample_EMI_dataset_large.csv")

def train_and_save_model():
    """Train a new regression model and save it"""
    try:
        if not os.path.exists(DATA_PATH):
            st.error(f"Training data not found at {DATA_PATH}")
            return None
        
        # Load training data
        df = pd.read_csv(DATA_PATH)
        
        # Prepare features and target
        target_col = 'max_emi'  # or 'max_emi_amount' - check your CSV
        if target_col not in df.columns:
            target_col = 'max_emi_amount'
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        # Save model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        
        st.success("✅ Model trained successfully with current environment!")
        return model
    except Exception as e:
        st.error(f"Training failed: {e}")
        return None

def load_model(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"⚠️ Incompatible model detected. Retraining with current environment...")
            return train_and_save_model()
    return train_and_save_model()

def build_input_df():
    st.sidebar.header("Applicant Info")
    
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30, key="max_age")
    monthly_salary = st.sidebar.number_input("Monthly salary", value=50000, key="max_monthly_salary")
    years_of_employment = st.sidebar.number_input("Years of employment", value=3, key="max_years_employment")
    monthly_rent = st.sidebar.number_input("Monthly rent", value=0, key="max_monthly_rent")
    family_size = st.sidebar.number_input("Family size", value=4, min_value=1, key="max_family_size")
    dependents = st.sidebar.number_input("Dependents", value=1, min_value=0, key="max_dependents")
    credit_score = st.sidebar.number_input("Credit score", value=700, min_value=0, key="max_credit_score")
    requested_amount = st.sidebar.number_input("Requested loan amount", value=200000, key="max_requested_amount")
    requested_tenure = st.sidebar.number_input("Requested tenure (months)", value=36, key="max_requested_tenure")
    
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"], key="max_gender")
    marital_status = st.sidebar.selectbox("Marital status", ["Single", "Married"], key="max_marital_status")
    education = st.sidebar.selectbox("Education", ["Graduate", "Bachelor", "Diploma", "Master", "Doctorate"], key="max_education")
    employment_type = st.sidebar.selectbox("Employment type", ["Private", "Government", "Self-employed", "Unemployed"], key="max_employment_type")
    company_type = st.sidebar.selectbox("Company type", ["Private", "Public", "Medium", "Small", "Large", "Startup"], key="max_company_type")
    house_type = st.sidebar.selectbox("House type", ["Own", "Rented"], key="max_house_type")
    existing_loans = st.sidebar.selectbox("Existing loans", ["Yes", "No"], key="max_existing_loans")
    emi_scenario = st.sidebar.selectbox("EMI scenario", ["Normal", "E-commerce", "Other"], key="max_emi_scenario")
    
    row = {
        "age": age,
        "monthly_salary": monthly_salary,
        "years_of_employment": years_of_employment,
        "monthly_rent": monthly_rent,
        "family_size": family_size,
        "dependents": dependents,
        "credit_score": credit_score,
        "school_fees": 0,
        "college_fees": 0,
        "travel_expenses": 0,
        "groceries_utilities": 0,
        "other_monthly_expenses": 0,
        "current_emi_amount": 0,
        "bank_balance": 0,
        "emergency_fund": 0,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "employment_type": employment_type,
        "company_type": company_type,
        "house_type": house_type,
        "existing_loans": existing_loans,
        "emi_scenario": emi_scenario
    }
    
    return pd.DataFrame([row])

def run():
    st.title("Maximum EMI Predictor")
    st.info("Predicts a recommended 'max safe monthly EMI' for the applicant using a regression model.")
    
    model = load_model(MODEL_PATH)
    X = build_input_df()
    
    st.write("Input preview:")
    st.dataframe(X)
    
    MODEL_FEATURES = [
        "age", "monthly_salary", "years_of_employment", "monthly_rent", "family_size", "dependents",
        "school_fees", "college_fees", "travel_expenses", "groceries_utilities", "other_monthly_expenses",
        "current_emi_amount", "credit_score", "bank_balance", "emergency_fund", "requested_amount",
        "requested_tenure", "gender", "marital_status", "education", "employment_type", "company_type",
        "house_type", "existing_loans", "emi_scenario"
    ]
    
    categorical_defaults = {
        "gender": "Unknown", "marital_status": "Unknown", "education": "Unknown",
        "employment_type": "Unknown", "company_type": "Unknown", "house_type": "Unknown",
        "existing_loans": "No", "emi_scenario": "Normal",
    }
    
    numeric_defaults = {
        "school_fees": 0, "college_fees": 0, "travel_expenses": 0, "groceries_utilities": 0,
        "other_monthly_expenses": 0, "current_emi_amount": 0, "bank_balance": 0, "emergency_fund": 0
    }
    
    for col in MODEL_FEATURES:
        if col not in X.columns:
            if col in categorical_defaults:
                X[col] = categorical_defaults[col]
            elif col in numeric_defaults:
                X[col] = numeric_defaults[col]
            else:
                X[col] = 0
    
    X = X[MODEL_FEATURES]
    X_encoded = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    
    if model is None:
        st.error("❌ Model unavailable. Please ensure training data exists in data/ folder.")
    else:
        if st.button("Predict max EMI", key="max_predict_button"):
            try:
                pred = model.predict(X_encoded)[0]
                st.success(f"✅ Predicted max monthly EMI: **₹{float(pred):,.2f}**")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

if __name__ == "__main__":
    run()
