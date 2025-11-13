from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import pandas as pd
import mlflow
import mlflow.pyfunc
import os
from datetime import datetime

# ==========================================================
# Configuration
# ==========================================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MODEL_NAME = "iris_model"
MODEL_STAGE = "Production"   # or "Staging" for testing
DATA_LOG_PATH = "data/new_iris_data.csv"

# ==========================================================
# Initialize FastAPI and assets
# ==========================================================
app = FastAPI(title="Iris Classifier (MLOps UI Edition)")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ==========================================================
# Load Model from MLflow Registry
# ==========================================================
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

try:
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Loaded model from MLflow Registry: {model_uri}")
except Exception as e:
    model = None
    print(f"Could not load model from MLflow Registry: {e}")

# ==========================================================
# Schema for API input (JSON endpoint)
# ==========================================================
class IrisFeatures(BaseModel):
    petal_length: float
    petal_width: float

# ==========================================================
# API Prediction Endpoint
# ==========================================================
@app.post("/predict")
async def predict_iris_species(features: List[IrisFeatures]):
    if model is None:
        return {"error": "Model not loaded"}

    df = pd.DataFrame({
        "petal length (cm)": [f.petal_length for f in features],
        "petal width (cm)": [f.petal_width for f in features]
    })

    preds = model.predict(df)

    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    species = [species_map.get(int(p), "Unknown") for p in preds]

    # Log inference data
    log_df = df.copy()
    log_df["prediction"] = species
    log_df["timestamp"] = datetime.now()
    os.makedirs(os.path.dirname(DATA_LOG_PATH), exist_ok=True)
    log_df.to_csv(DATA_LOG_PATH, mode="a", header=not os.path.exists(DATA_LOG_PATH), index=False)

    return {"predictions": species}

# ==========================================================
# Web UI Routes
# ==========================================================
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/submit", response_class=HTMLResponse)
async def submit_form(
    request: Request,
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    if model is None:
        result = "Model not loaded"
    else:
        df = pd.DataFrame({
            "petal length (cm)": [petal_length],
            "petal width (cm)": [petal_width]
        })

        preds = model.predict(df)
        species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        result = species_map.get(int(preds[0]), "Unknown")

        # Log data for drift monitoring
        log_df = df.copy()
        log_df["prediction"] = [result]
        log_df["timestamp"] = datetime.now()
        os.makedirs(os.path.dirname(DATA_LOG_PATH), exist_ok=True)
        log_df.to_csv(DATA_LOG_PATH, mode="a", header=not os.path.exists(DATA_LOG_PATH), index=False)

    # Render HTML with result displayed
    return templates.TemplateResponse("index.html", {"request": request, "result": result})

# ==========================================================
# Health Check Endpoint
# ==========================================================
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}
