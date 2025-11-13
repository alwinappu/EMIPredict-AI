# src/preprocessing.py (compat OneHotEncoder for sklearn versions)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from collections import Counter

def _make_onehot():
    """
    Create a OneHotEncoder that's compatible across sklearn versions.
    Newer sklearn uses 'sparse_output', older uses 'sparse'.
    """
    try:
        # try older arg first (some environments still accept it)
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
        return enc
    except TypeError:
        # fall back to newer arg name
        try:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            return enc
        except Exception as e:
            # last resort: plain OneHotEncoder and we'll handle sparse matrices later
            return OneHotEncoder(handle_unknown="ignore")

def load_data(path):
    df = pd.read_csv(path)
    return df

def basic_cleaning(df):
    df = df.drop_duplicates().reset_index(drop=True)
    df.columns = [c.strip() for c in df.columns]
    return df

def get_feature_lists(df):
    numeric_features = [
        "age","monthly_salary","years_of_employment","monthly_rent","family_size",
        "dependents","school_fees","college_fees","travel_expenses",
        "groceries_utilities","other_monthly_expenses","current_emi_amount",
        "credit_score","bank_balance","emergency_fund","requested_amount","requested_tenure"
    ]
    categorical_features = [
        "gender","marital_status","education","employment_type","company_type",
        "house_type","existing_loans","emi_scenario"
    ]
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]
    return numeric_features, categorical_features

def build_preprocessor(df):
    numeric_features, categorical_features = get_feature_lists(df)

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _make_onehot())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ], remainder="drop")

    return preprocessor, numeric_features, categorical_features

def _safe_stratify_column(series, min_count=2):
    if series is None:
        return None
    counts = Counter(series.dropna().tolist())
    if not counts:
        return None
    if any(c < min_count for c in counts.values()):
        return None
    return series

def train_test_val_split(df, target_class_col, target_reg_col=None, test_size=0.2, val_size=0.1, random_state=42):
    df = df.copy()

    stratify_col = None
    if target_class_col in df.columns:
        stratify_candidate = df[target_class_col]
        stratify_col = _safe_stratify_column(stratify_candidate, min_count=2)

    try:
        if stratify_col is not None:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify_col)
        else:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    except Exception:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    stratify_for_val = None
    if target_class_col in train_df.columns:
        stratify_for_val = _safe_stratify_column(train_df[target_class_col], min_count=2)
    if stratify_for_val is not None:
        train_df, val_df = train_test_split(train_df, test_size=val_size/(1-test_size), random_state=random_state, stratify=stratify_for_val)
    else:
        train_df, val_df = train_test_split(train_df, test_size=val_size/(1-test_size), random_state=random_state)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
