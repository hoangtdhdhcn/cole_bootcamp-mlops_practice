from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pickle
import pandas as pd
from pydantic import BaseModel
from typing import List
import os

# Initialize FastAPI app
app = FastAPI()

# Load the trained model from the .pkl file (cross-platform approach)
script_dir = os.path.dirname(__file__)

# Define possible model paths (relative to the script directory)
model_paths = [
    os.path.join(script_dir, "iris_model.pkl"),                # main path
    os.path.join(script_dir, "scripts", "session_3", "iris_model.pkl"),  # fallback for tests/CI
    os.path.join(script_dir, "model", "iris_model.pkl"),                # main path
    os.path.join(script_dir, "scripts", "session_3", "model", "iris_model.pkl")  # fallback for tests/CI
]

# Try loading the model from any existing path
model = None
for path in model_paths:
    if os.path.exists(path):
        with open(path, "rb") as file:
            model = pickle.load(file)
        print(f"Loaded model from: {path}")
        break

# Serving static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serving HTML templates
templates = Jinja2Templates(directory="templates")

# Define Pydantic model for input validation
class IrisFeatures(BaseModel):
    petal_length: float
    petal_width: float

# Prediction endpoint
@app.post("/predict")
async def predict_iris_species(features: List[IrisFeatures]):
    # Convert incoming data to a DataFrame
    data = {
        "petal length (cm)": [feature.petal_length for feature in features],
        "petal width (cm)": [feature.petal_width for feature in features]
    }
    df = pd.DataFrame(data)

    # Make predictions using the loaded model
    predictions = model.predict(df)

    # Convert numerical predictions (0, 1, 2) to species names
    species = []
    for pred in predictions:
        if pred == 0:
            species.append("setosa")
        elif pred == 1:
            species.append("versicolor")
        elif pred == 2:
            species.append("virginica")
        else:
            species.append("Unknown")

    return {"predictions": species}

# Root endpoint to serve the HTML form
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint for handling the form submission
@app.post("/submit")
async def submit_form(petal_length: float = Form(...), petal_width: float = Form(...)):
    # Create an IrisFeatures list from the form data
    features = [IrisFeatures(petal_length=petal_length, petal_width=petal_width)]
    
    # Call the prediction endpoint 
    data = {
        "petal length (cm)": [petal_length],
        "petal width (cm)": [petal_width]
    }
    df = pd.DataFrame(data)
    predictions = model.predict(df)
    
    species = []
    for pred in predictions:
        if pred == 0:
            species.append("setosa")
        elif pred == 1:
            species.append("versicolor")
        elif pred == 2:
            species.append("virginica")
        else:
            species.append("Unknown")
    
    # Return the prediction result to the UI
    return {"species": species[0]}

