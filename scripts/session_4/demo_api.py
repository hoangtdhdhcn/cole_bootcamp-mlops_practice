from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
from datetime import datetime
import os

# ======================
# Configuration
# ======================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MODEL_NAME = "iris_model"
MODEL_STAGE = "Production"  # could also load "Staging" version during testing
DATA_LOG_PATH = "data/new_iris_data.csv"

# ======================
# Initialize FastAPI
# ======================
app = FastAPI(title="Iris Model API (MLOps Edition)")

# ======================
# Load Model from MLflow Model Registry
# ======================
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

try:
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Loaded model from MLflow Registry: {model_uri}")
except Exception as e:
    print(f"Could not load model from MLflow Registry: {e}")
    model = None

# ======================
# Input schema
# ======================
class IrisFeatures(BaseModel):
    petal_length: float
    petal_width: float


# ======================
# Prediction endpoint
# ======================
@app.post("/predict")
async def predict_iris_species(features: list[IrisFeatures]):
    if model is None:
        return {"error": "Model not loaded."}

    # Prepare input data
    data = {
        "petal length (cm)": [f.petal_length for f in features],
        "petal width (cm)": [f.petal_width for f in features],
    }
    df = pd.DataFrame(data)

    # Make predictions
    preds = model.predict(df)

    # Map numerical predictions to species
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    species = [species_map.get(int(p), "unknown") for p in preds]

    # Log requests for monitoring
    log_df = df.copy()
    log_df["prediction"] = species
    log_df["timestamp"] = datetime.now()

    os.makedirs(os.path.dirname(DATA_LOG_PATH), exist_ok=True)
    log_df.to_csv(DATA_LOG_PATH, mode="a", header=not os.path.exists(DATA_LOG_PATH), index=False)

    return {"predictions": species}


# ======================
# Health check endpoint
# ======================
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}
