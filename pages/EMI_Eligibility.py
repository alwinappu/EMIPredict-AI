# Save this as: C:\Users\appu0\EMIPredict_AI\pages\EMI_Eligibility.py

import streamlit as st
import pandas as pd
import os
import pickle
import traceback

st.set_page_config(page_title="EMI Eligibility", layout="wide")
st.title("EMI Eligibility Prediction")
st.info("This page loads a sample input CSV and attempts to use a saved classifier (if available).")

# locate sample CSV
sample_csv_paths = [
    os.path.join("data", "sample_input.csv"),
    os.path.join(os.getcwd(), "data", "sample_input.csv"),
]

sample_path = None
for p in sample_csv_paths:
    if os.path.exists(p):
        sample_path = p
        break

if sample_path is None:
    st.error("Could not find data/sample_input.csv in the repo. Please ensure the file exists.")
    st.stop()

# load and show CSV
try:
    df = pd.read_csv(sample_path)
except Exception as e:
    st.error(f"Failed to read CSV at {sample_path}: {e}")
    st.stop()

st.subheader("Input preview:")
st.dataframe(df, use_container_width=True)
st.write("")  # spacing

# Prediction block
st.subheader("Predict eligibility")

col1, col2 = st.columns([1, 3])

with col1:
    predict_btn = st.button("Predict eligibility")

with col2:
    model_path_suggestions = [
        os.path.join("models", "emi_eligibility_model.pkl"),
        os.path.join("emi_eligibility_model.pkl"),
        os.path.join("models", "regression_rf.pkl"),
        os.path.join("regression_rf.pkl"),
    ]
    found_models = [p for p in model_path_suggestions if os.path.exists(p)]
    if found_models:
        st.success(f"Found model file(s): {', '.join(found_models)}")
    else:
        st.warning("No model file found in models/ or repo root. Prediction will fail unless you add one.")

# helper: align features safely for the model
def align_features_for_model(X, model):
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

if predict_btn:
    # try to load preprocessing helper
    preprocess_fn = None
    try:
        from src.preprocessing import preprocess_input
        preprocess_fn = preprocess_input
        st.info("Using src.preprocessing.preprocess_input to prepare data.")
    except Exception:
         pass  # Use raw CSV directly
    
    # find model file
    model_file = None
    for p in model_path_suggestions:
        if os.path.exists(p):
            model_file = p
            break
    
    if model_file is None:
        st.error("No model file found. Place your trained model in models/ and try again.")
    else:
        model = None
        # Try joblib first
        try:
            import joblib
            model = joblib.load(model_file)
            st.success(f"✅ Model loaded with joblib from {model_file}")
        except Exception as e1:
            # Try pickle
            try:
                with open(model_file, "rb") as f:
                    model = pickle.load(f, encoding='latin1')
                st.success(f"✅ Model loaded with pickle from {model_file}")
            except Exception as e2:
                # Train new model
                st.warning("⚠️ Could not load model. Training new one...")
                try:
                    from sklearn.ensemble import RandomForestClassifier
                    import joblib
                    
                    train_path = "data/sample_EMI_dataset_train.csv"
                    if os.path.exists(train_path):
                        df_train = pd.read_csv(train_path)
                        X_train = df_train.select_dtypes(include=['number']).fillna(0)
                        
                        target_col = None
                        for col in ['emi_eligibility', 'EMI_Eligibility', 'eligibility']:
                            if col in df_train.columns:
                                target_col = col
                                break
                        
                        if target_col:
                            y_train = df_train[target_col]
                            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                            model.fit(X_train, y_train)
                            joblib.dump(model, "models/emi_eligibility_model.pkl")
                            st.success("✅ New model trained and saved!")
                        else:
                            st.error("Target column not found")
                    else:
                        st.error(f"Training data not found at {train_path}")
                except Exception as e3:
                    st.error(f"Failed to train model: {e3}")
                    st.text(traceback.format_exc())
        
        if model is not None:
            try:
                if preprocess_fn:
                    X = preprocess_fn(df.copy())
                else:
                    X = df.copy()
            except Exception as e:
                st.error(f"Preprocessing failed: {e}")
                st.text(traceback.format_exc())
                X = None
            
            if X is not None:
                try:
                    X_aligned, added_cols = align_features_for_model(X, model)
                    if added_cols:
                        st.warning(f"Added defaults for: {added_cols}")
                    
                    preds = model.predict(X_aligned)
                    st.success("✅ Prediction completed!")
                    
                    if hasattr(preds, "tolist"):
                        st.write("Predicted (first row):", preds.tolist()[0])
                    else:
                        st.write("Predicted (first row):", preds[0])
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.text(traceback.format_exc())

st.write("---")
st.markdown("""
**Notes**
- Ensure data/sample_input.csv exists
- Model should be in models/ directory  
- Check sklearn version compatibility
""")
