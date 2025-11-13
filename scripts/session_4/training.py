import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train():
    # Connect to MLflow server (inside Docker use service name)
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    mlflow.set_experiment("iris_training")

    iris = load_iris()
    iris = pd.DataFrame(
        np.c_[iris["data"], iris["target"]],
        columns=iris.feature_names + ["target"]
    )

    species = ["setosa" if t == 0 else "versicolor" if t == 1 else "virginica"
               for t in iris["target"]]
    iris["species"] = species

    X = iris.drop(["target", "species"], axis=1).to_numpy()[:, (2, 3)]
    y = iris["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    log_reg = LogisticRegression()

    with mlflow.start_run(run_name="iris_training_v1"):
        mlflow.log_param("random_state", 42)
        mlflow.log_param("target", y.name)

        log_reg.fit(X_train, y_train)

        mlflow.log_metric("train_accuracy", log_reg.score(X_train, y_train))
        mlflow.log_metric("test_accuracy", log_reg.score(X_test, y_test))

        mlflow.sklearn.log_model(
            sk_model=log_reg,
            artifact_path="iris_model",
            registered_model_name="iris_model"
        )

        print("Model logged to MLflow!")


if __name__ == "__main__":
    train()
