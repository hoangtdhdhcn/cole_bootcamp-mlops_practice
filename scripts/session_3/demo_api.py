from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import os

# Initialize FastAPI app
app = FastAPI()

# Load the trained model from the .pkl file (cross-platform approach)
script_dir = os.path.dirname(__file__)

# Define possible model paths (relative to the script directory)
model_paths = [
    os.path.join(script_dir, "iris_model.pkl"),                # main path
    os.path.join(script_dir, "scripts", "session_3", "iris_model.pkl")  # fallback for tests/CI
]

# Try loading the model from any existing path
model = None
for path in model_paths:
    if os.path.exists(path):
        with open(path, "rb") as file:
            model = pickle.load(file)
        print(f"Loaded model from: {path}")
        break

# Define Pydantic model for input validation
class IrisFeatures(BaseModel):
    petal_length: float
    petal_width: float

# Prediction endpoint
@app.post("/predict")
async def predict_iris_species(features: list[IrisFeatures]):
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
        else:
            species.append("virginica")

    return {"predictions": species}

