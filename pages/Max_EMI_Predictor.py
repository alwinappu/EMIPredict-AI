# pages/Max_EMI_Predictor.py
import streamlit as st
import pandas as pd

def calculate_max_emi(monthly_salary, current_emi_amount, monthly_rent, 
                      school_fees, college_fees, travel_expenses, 
                      groceries_utilities, other_monthly_expenses, 
                      credit_score, age, dependents, emergency_fund, bank_balance):
    """
    Calculate maximum safe EMI using standard lending criteria:
    - Max debt-to-income ratio: 40-50% of monthly income
    - Adjusted based on credit score, age, dependents
    """
    
    # Calculate total fixed monthly expenses
    total_expenses = (
        monthly_rent + school_fees + college_fees + 
        travel_expenses + groceries_utilities + 
        other_monthly_expenses + current_emi_amount
    )
    
    # Calculate disposable income
    disposable_income = monthly_salary - total_expenses
    
    # Base EMI capacity: 40-50% of monthly income
    base_emi_ratio = 0.45  # 45% baseline
    
    # Adjust based on credit score
    if credit_score >= 750:
        credit_multiplier = 1.1  # Can afford 10% more
    elif credit_score >= 650:
        credit_multiplier = 1.0
    elif credit_score >= 550:
        credit_multiplier = 0.85  # Reduce by 15%
    else:
        credit_multiplier = 0.7  # Reduce by 30%
    
    # Adjust based on age (younger = longer loan tenure possible)
    if age < 30:
        age_multiplier = 1.05
    elif age < 45:
        age_multiplier = 1.0
    elif age < 55:
        age_multiplier = 0.9
    else:
        age_multiplier = 0.75
    
    # Adjust based on dependents
    if dependents == 0:
        dependent_multiplier = 1.05
    elif dependents <= 2:
        dependent_multiplier = 1.0
    else:
        dependent_multiplier = 0.9
    
    # Emergency fund factor (should have 6 months expenses saved)
    required_emergency = total_expenses * 6
    if emergency_fund >= required_emergency:
        safety_multiplier = 1.0
    elif emergency_fund >= required_emergency * 0.5:
        safety_multiplier = 0.95
    else:
        safety_multiplier = 0.85
    
    # Calculate maximum safe EMI
    max_emi = (
        monthly_salary * base_emi_ratio * 
        credit_multiplier * age_multiplier * 
        dependent_multiplier * safety_multiplier
    ) - current_emi_amount
    
    # Ensure minimum safety: EMI should not exceed 50% of disposable income
    safe_limit = disposable_income * 0.5
    max_emi = min(max_emi, safe_limit)
    
    # Cannot be negative
    max_emi = max(0, max_emi)
    
    return max_emi

def build_input_df():
    st.sidebar.header("Applicant Info")
    
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30, key="max_age")
    monthly_salary = st.sidebar.number_input("Monthly salary", value=50000, key="max_monthly_salary")
    years_of_employment = st.sidebar.number_input("Years of employment", value=3, key="max_years_employment")
    monthly_rent = st.sidebar.number_input("Monthly rent", value=0, key="max_monthly_rent")
    family_size = st.sidebar.number_input("Family size", value=4, min_value=1, key="max_family_size")
    dependents = st.sidebar.number_input("Dependents", value=1, min_value=0, key="max_dependents")
    credit_score = st.sidebar.number_input("Credit score", value=700, min_value=300, max_value=900, key="max_credit_score")
    
    with st.sidebar.expander("📊 Additional Financial Details"):
        school_fees = st.number_input("School fees (monthly)", value=0, key="school_fees")
        college_fees = st.number_input("College fees (monthly)", value=0, key="college_fees")
        travel_expenses = st.number_input("Travel expenses (monthly)", value=2000, key="travel_exp")
        groceries_utilities = st.number_input("Groceries & utilities (monthly)", value=5000, key="groceries")
        other_monthly_expenses = st.number_input("Other monthly expenses", value=1000, key="other_exp")
        current_emi_amount = st.number_input("Current EMI amount", value=0, key="current_emi")
        bank_balance = st.number_input("Bank balance", value=20000, key="bank_bal")
        emergency_fund = st.number_input("Emergency fund", value=0, key="emerg_fund")
    
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
    
    return {
        "age": age,
        "monthly_salary": monthly_salary,
        "years_of_employment": years_of_employment,
        "monthly_rent": monthly_rent,
        "family_size": family_size,
        "dependents": dependents,
        "credit_score": credit_score,
        "school_fees": school_fees,
        "college_fees": college_fees,
        "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "current_emi_amount": current_emi_amount,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
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

def run():
    st.title("Maximum EMI Predictor")
    st.info("✨ Calculates a recommended 'max safe monthly EMI' based on proven lending criteria and your financial profile.")
    
    data = build_input_df()
    
    # Display input preview
    st.write("📋 **Input preview:**")
    preview_df = pd.DataFrame([{
        "Age": data["age"],
        "Monthly Salary": f"₹{data['monthly_salary']:,}",
        "Credit Score": data["credit_score"],
        "Dependents": data["dependents"],
        "Current EMI": f"₹{data['current_emi_amount']:,}",
        "Monthly Rent": f"₹{data['monthly_rent']:,}"
    }])
    st.dataframe(preview_df, use_container_width=True)
    
    if st.button("💰 Calculate Maximum EMI", key="max_predict_button", type="primary"):
        try:
            max_emi = calculate_max_emi(
                monthly_salary=data["monthly_salary"],
                current_emi_amount=data["current_emi_amount"],
                monthly_rent=data["monthly_rent"],
                school_fees=data["school_fees"],
                college_fees=data["college_fees"],
                travel_expenses=data["travel_expenses"],
                groceries_utilities=data["groceries_utilities"],
                other_monthly_expenses=data["other_monthly_expenses"],
                credit_score=data["credit_score"],
                age=data["age"],
                dependents=data["dependents"],
                emergency_fund=data["emergency_fund"],
                bank_balance=data["bank_balance"]
            )
            
            st.success(f"### ✅ Recommended Maximum Monthly EMI: **₹{max_emi:,.2f}**")
            
            # Additional insights
            col1, col2, col3 = st.columns(3)
            
            total_expenses = (data["monthly_rent"] + data["school_fees"] + 
                            data["college_fees"] + data["travel_expenses"] + 
                            data["groceries_utilities"] + data["other_monthly_expenses"] + 
                            data["current_emi_amount"])
            
            disposable = data["monthly_salary"] - total_expenses
            emi_percentage = (max_emi / data["monthly_salary"] * 100) if data["monthly_salary"] > 0 else 0
            
            with col1:
                st.metric("Total Monthly Expenses", f"₹{total_expenses:,.0f}")
            with col2:
                st.metric("Disposable Income", f"₹{disposable:,.0f}")
            with col3:
                st.metric("EMI as % of Income", f"{emi_percentage:.1f}%")
            
            # Loan affordability
            if max_emi > 0:
                # Assuming 10% annual interest rate and requested tenure
                rate_monthly = 0.10 / 12
                tenure = data["requested_tenure"]
                
                if rate_monthly > 0:
                    max_loan_amount = max_emi * ((1 - (1 + rate_monthly)**(-tenure)) / rate_monthly)
                else:
                    max_loan_amount = max_emi * tenure
                
                st.info(f"📊 **Loan Affordability:** With this EMI capacity, you can afford a loan of approximately **₹{max_loan_amount:,.0f}** over {tenure} months at 10% interest rate.")
                
                if data["requested_amount"] <= max_loan_amount:
                    st.success(f"✅ Your requested loan amount (₹{data['requested_amount']:,}) is within your safe borrowing capacity!")
                else:
                    st.warning(f"⚠️ Your requested loan amount (₹{data['requested_amount']:,}) exceeds the recommended safe limit. Consider a longer tenure or lower amount.")
            
        except Exception as e:
            st.error(f"❌ Calculation failed: {e}")

if __name__ == "__main__":
    run()
