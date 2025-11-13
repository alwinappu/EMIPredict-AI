import pandas as pd, traceback, os, sys
try:
    import src.preprocessing as preprocessing
except Exception as e:
    print("Failed importing src.preprocessing:", repr(e))
    traceback.print_exc()
    sys.exit(1)

p = r"C:\Users\appu0\EMIPredict_AI\data\sample_EMI_dataset_large.csv"
print("FILE PATH:", p)
print("Exists on disk:", os.path.exists(p))

# RAW CSV
try:
    df_raw = pd.read_csv(p)
    print("\nRAW CSV shape:", df_raw.shape)
    print("RAW CSV head:\n", df_raw.head().to_string())
except Exception as e:
    print("pandas read_csv failed:", repr(e))
    traceback.print_exc()
    sys.exit(1)

# PREPROCESSING RESULT
try:
    df_loaded = preprocessing.load_data(p)
    print("\npreprocessing.load_data returned shape:", getattr(df_loaded, "shape", None))
    print("Loaded head:\n", df_loaded.head().to_string())
    print("Loaded columns:", list(df_loaded.columns))
    if "emi_scenario" in df_loaded.columns:
        print("\nemi_scenario value counts:")
        print(df_loaded["emi_scenario"].value_counts(dropna=False))
except Exception as e:
    print("\npreprocessing.load_data raised exception:", repr(e))
    traceback.print_exc()
