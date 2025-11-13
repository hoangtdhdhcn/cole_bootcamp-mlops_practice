import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mlflow.exceptions import MlflowException
import logging

logging.basicConfig(level=logging.INFO)


def get_tracking_uri():
    """
    Dynamically select MLflow tracking URI based on environment.
    - In CI or Docker: use remote MLflow tracking server.
    - Otherwise: default to local file-based tracking.
    """
    if os.environ.get("RUNNING_IN_DOCKER") == "1":
        uri = "http://mlflow-server:5000"
    elif os.environ.get("USE_REMOTE_MLFLOW", "false").lower() == "true":
        uri = "http://localhost:5000"
    else:
        # Local MLflow fallback
        uri = "file:./mlruns"
    return uri


def train():
    tracking_uri = get_tracking_uri()
    logging.info(f"Using MLflow tracking URI: {tracking_uri}")

    # Initialize MLflow experiment
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("iris_training")

    # Load dataset
    iris = load_iris(as_frame=True)
    df = iris.frame
    df["species"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

    # Use only petal features
    X = df[["petal length (cm)", "petal width (cm)"]].to_numpy()
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=200)

    with mlflow.start_run(run_name="iris_training_v1"):
        mlflow.log_param("random_state", 42)
        mlflow.log_param("features", ["petal length", "petal width"])

        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)

        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="iris_model",
                registered_model_name="iris_model"
            )
            logging.info("Model registered in MLflow Model Registry.")
        except MlflowException as e:
            logging.warning(f"Model registration failed: {e}")
            mlflow.sklearn.log_model(sk_model=model, artifact_path="iris_model")

        logging.info(f"Training done (test acc = {test_acc:.3f})")

    return test_acc


if __name__ == "__main__":
    train()
