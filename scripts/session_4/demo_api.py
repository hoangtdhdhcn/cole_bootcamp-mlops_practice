from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
from datetime import datetime
import os
from typing import Union, List
from mlflow.exceptions import MlflowException
import logging

# ======================
# Configuration
# ======================
MODEL_NAME = "iris_model"
# MODEL_STAGE = "Production"
MODEL_VERSION = "1"
DATA_LOG_PATH = "data/new_iris_data.csv"

logging.basicConfig(level=logging.INFO)

# ======================
# Helper to determine MLflow tracking URI
# ======================
def get_tracking_uri():
    """Return correct MLflow tracking URI for Docker, CI, or local."""
    if os.getenv("RUNNING_IN_DOCKER") == "1":
        return "http://mlflow-server:5000"
    # Default to local file-based tracking if no MLflow server
    return os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")


# ======================
# Initialize FastAPI
# ======================
app = FastAPI(title="Iris Model API (MLOps Edition)")
mlflow.set_tracking_uri(get_tracking_uri())

# ======================
# Load model from MLflow (Registry or local)
# ======================
model = None
try:
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    model = mlflow.pyfunc.load_model(model_uri)
    logging.info(f"Loaded model from MLflow Registry: {model_uri}")
except Exception as e:
    logging.warning(f"Could not load model from MLflow Registry: {e}")
    # fallback to local artifact path
    local_model_path = "./mlruns/0/iris_model"
    if os.path.exists(local_model_path):
        model = mlflow.pyfunc.load_model(local_model_path)
        logging.info(f"Loaded model from local mlruns path: {local_model_path}")
    else:
        logging.error("No model found locally — predictions will not work.")


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
async def predict_iris_species(features: Union[IrisFeatures, List[IrisFeatures]]):
    if model is None:
        return {"error": "Model not loaded."}

    if isinstance(features, IrisFeatures):
        features = [features]

    df = pd.DataFrame({
        "petal length (cm)": [f.petal_length for f in features],
        "petal width (cm)": [f.petal_width for f in features]
    })

    preds = model.predict(df)
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    species = [species_map.get(int(p), "unknown") for p in preds]

    # Log data locally
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


# ======================
# Startup logging
# ======================
@app.on_event("startup")
async def startup_event():
    if model is None:
        logging.warning("No model loaded — check MLflow tracking or local model path.")
    else:
        logging.info(f"Model '{MODEL_NAME}' ({MODEL_VERSION}) is ready for inference.")
