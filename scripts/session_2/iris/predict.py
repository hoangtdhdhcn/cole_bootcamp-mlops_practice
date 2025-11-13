import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# Function to load the model from MLflow and make predictions
def load_model_and_predict():
    # Load the model from MLflow using the registered model name
    model_name = "iris_model"
    model_version = "1"
    alias = ""
    logged_model = f"models:/{model_name}/{model_version}"  # Here 1 is the version of the model. Update accordingly.
    # logged_model = f"models:/{model_name}/{model_version}@{alias}"
    
    # Load the model
    model = mlflow.sklearn.load_model(logged_model)

    # Load the Iris dataset for testing
    iris = load_iris()
    iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)
    
    # Select petal length and petal width (similar to what was used in the training)
    X_test = iris_data[["petal length (cm)", "petal width (cm)"]].to_numpy()

    # Predict using the loaded model
    predictions = model.predict(X_test)
    
    # Print some of the predictions alongside the true labels
    print(f"Predictions: {predictions}")
    print(f"True Labels: {iris.target[:len(predictions)]}")

    # Optionally, map the numeric predictions back to species names for better readability
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    predicted_species = [species_map[pred] for pred in predictions]
    print(f"Predicted Species: {predicted_species}")

if __name__ == "__main__":
    load_model_and_predict()
