import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import pickle

# Load your training data
df = pd.read_csv('data/sample_EMI_dataset_train.csv')

# Prepare data - using only numeric features for simplicity
X = df.select_dtypes(include=['number'])

# Target variable - adjust column name if different
if 'emi_eligibility' in df.columns:
    y = df['emi_eligibility']
elif 'EMI_Eligibility' in df.columns:
    y = df['EMI_Eligibility']
else:
    print("Available columns:", df.columns.tolist())
    y = df.iloc[:, -2]  # second to last column

print(f"Training with {len(X)} samples and {len(X.columns)} features")
print(f"Target classes: {y.unique()}")

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X.fillna(0), y)

# Save with BOTH pickle and joblib to be safe
joblib.dump(model, 'models/emi_eligibility_model.pkl')
joblib.dump(model, 'emi_eligibility_model.pkl')

# Also save with pickle protocol 4 for better compatibility
with open('models/emi_eligibility_model_v2.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=4)

print("✅ Model retrained and saved successfully!")
print(f"Model accuracy on training data: {model.score(X.fillna(0), y):.2%}")
