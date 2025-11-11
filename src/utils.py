# src/utils.py
import joblib
import pandas as pd

def load_model(path):
    return joblib.load(path)

def predict(model, X_df):
    # Accept a DataFrame and return predictions
    return model.predict(X_df)
