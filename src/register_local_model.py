import mlflow
import os

model_dir = "data/artifact/mlp_model/"
abs_path = os.path.abspath(model_dir)
model_uri = f"file://{abs_path}"

# Register model
model_version = mlflow.register_model(model_uri=model_uri, name="mlp_adaptive-mnist")
