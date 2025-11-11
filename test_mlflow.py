import mlflow
import random
mlflow.set_experiment("Demo_Experiment")

with mlflow.start_run(run_name="test_run"):
    mlflow.log_param("parameter1", 42)
    mlflow.log_metric("metric1", random.random())
    print("Logged demo experiment successfully!")
