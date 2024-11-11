import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from pathlib import Path
from mlflow.models import infer_signature

from .mlp import MLPWrapper

import yaml

train_file = Path("data/mnist_train.csv")
test_file = Path("data/mnist_test.csv")


def load_hyper_param(path):
    with open(path) as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params["hyper_best_parmas"]


# Data Preparation
def load_data():
    train = np.genfromtxt(train_file, delimiter=",")
    test = np.genfromtxt(test_file, delimiter=",")
    x_train = train[:, 1:]
    y_train = train[:, 0]
    x_test = test[:, 1:]
    y_test = test[:, 0]

    x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255
    x_test_p = x_test.reshape(-1, 28 * 28).astype("float32") / 255
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_p = encoder.transform(y_test.reshape(-1, 1))
    p = len(x_test_p) // 2

    x_test = x_test_p[:p]
    x_val = x_test_p[p:]

    y_test = y_test_p[:p]
    y_val = y_test_p[p:]
    return x_train, y_train, x_test, y_test, x_val, y_val


def train_model(
    hidden_size=256, epochs=100, batch_size=32, initial_lr=0.01, decay=0.001, patience=5
):
    # Save the model
    with mlflow.start_run():
        mlflow.autolog()

        print("starting training")

        # Trening sieci
        x_train, y_train, x_test, y_test, x_val, y_val = load_data()
        input_size = 784  # 28*28
        output_size = 10  # 0-9

        mlp = MLPWrapper(input_size, hidden_size, output_size)
        mlp.train(
            x_train,
            y_train,
            x_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            initial_lr=initial_lr,
            decay=decay,
            patience=patience,
        )

        mlflow_pyfunc_model_path = "mlp_model"
        artifacts = {"model_path": "src/app/ds/mlp.py"}

        # Define the signature associated with the model
        signature = infer_signature(x_train, params={"predict_method": "predict"})

        model_info = mlflow.pyfunc.log_model(
            artifact_path=mlflow_pyfunc_model_path,
            python_model=mlp,
            code_path=["src/app/ds/mlp.py"],
            input_example=x_train[:5],
            signature=signature,
            artifacts=artifacts,
        )

        mlp = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)

        # Ewaluacja modelu
        y_pred = mlp.predict(x_test)
        # proba = mlp.predict(x_test, params={"predict_method": "predict_proba"})

        y_test_labels = np.argmax(y_test, axis=1)

        accuracy = accuracy_score(y_test_labels, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Test accuracy: {accuracy * 100:.2f}%")

        # Macierz pomyłek
        cm = confusion_matrix(y_test_labels, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=range(10),
            yticklabels=range(10),
        )
        plt.xlabel("Predykcja")
        plt.ylabel("Rzeczywistość")
        plt.title("Macierz Pomyłek")
        # %%
        mlflow.log_figure(plt.gcf(), "plots/con_matrix.png")


def main():
    conf_path = Path("src/app/config/mlp_hiperparameters.yaml")
    hyperparams = load_hyper_param(conf_path)

    train_model(**hyperparams)


if __name__ == "__main__":
    main()
