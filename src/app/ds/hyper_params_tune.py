# %%
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from pathlib import Path
from mlp import MLPWrapper
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from typing import Tuple, Dict, Any

train_file = Path("data/mnist_train.csv")
test_file = Path("data/mnist_test.csv")


# Przygotowanie danych
def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ładuje i przetwarza dane treningowe, testowe i walidacyjne.

    Returns:
        Tuple zawierający x_train, y_train, x_test, y_test, x_val, y_val.
    """
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


# Funkcja celu dla Hyperopt
def objective(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Funkcja celu dla optymalizacji Hyperopt. Używana do trenowania modelu z wybranymi parametrami
    i oceny jego dokładności na danych testowych.

    Args:
        params (Dict[str, Any]): Słownik parametrów hiperparametrów do optymalizacji.

    Returns:
        Dict[str, Any]: Wynik zawierający stratę (ujemną dokładność) i status.
    """
    with mlflow.start_run():
        x_train, y_train, x_test, y_test, x_val, y_val = load_data()

        # Rozpakowanie hiperparametrów
        hidden_size1 = int(params["hidden_size1"])
        initial_lr = params["initial_lr"]
        decay = params["decay"]
        batch_size = int(params["batch_size"])
        patience = int(params["patience"])
        epochs = 100

        # Inicjalizacja i trening modelu
        mlp = MLPWrapper(
            input_size=784,
            hidden_size1=hidden_size1,
            output_size=10,
        )
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

        # Ewaluacja modelu
        y_pred = mlp.predict(None, x_test)
        y_test_labels = np.argmax(y_test, axis=1)
        accuracy = accuracy_score(y_test_labels, y_pred)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)

        # Zwraca ujemną dokładność do minimalizacji w Hyperopt
        return {"loss": -accuracy, "status": STATUS_OK}


# Definicja przestrzeni przeszukiwań Hyperopt
search_space = {
    "hidden_size": hp.choice("hidden_size", [128, 256]),
    "initial_lr": hp.loguniform("initial_lr", np.log(0.001), np.log(0.1)),
    "decay": hp.loguniform("decay", np.log(0.00001), np.log(0.001)),
    "batch_size": hp.choice("batch_size", [16, 32, 64, 128]),
    "dropout_rate": hp.uniform("dropout_rate", 0.0, 0.5),
    "patience": hp.choice("patience", [3, 5, 10, 15]),
    "activation": hp.choice("activation", ["relu", "tanh", "sigmoid"]),
    "optimizer": hp.choice("optimizer", ["adam", "sgd", "rmsprop"]),
}

# Uruchomienie optymalizacji Hyperopt
if __name__ == "__main__":
    trials = Trials()
    best_params = fmin(
        fn=objective, space=search_space, algo=tpe.suggest, max_evals=50, trials=trials
    )

    print("Najlepsze parametry:", best_params)

    # Trening końcowego modelu przy użyciu najlepszych parametrów
    with mlflow.start_run():
        mlflow.autolog()

        x_train, y_train, x_test, y_test, x_val, y_val = load_data()
        best_hidden_size = [128, 256, 512][best_params["hidden_size"]]
        best_batch_size = [16, 32, 64][best_params["batch_size"]]
        best_patience = [3, 5, 10][best_params["patience"]]

        mlp = MLPWrapper(
            input_size=784,
            hidden_size1=best_hidden_size,
            output_size=10,
        )
        mlp.train(
            x_train,
            y_train,
            x_val,
            y_val,
            epochs=100,
            batch_size=best_batch_size,
            initial_lr=best_params["initial_lr"],
            decay=best_params["decay"],
            patience=best_patience,
        )

        y_pred = mlp.predict(x_test)
        y_test_labels = np.argmax(y_test, axis=1)
        accuracy = accuracy_score(y_test_labels, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Dokładność testowa: {accuracy * 100:.2f}%")

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
        plt.xlabel("Przewidywane")
        plt.ylabel("Prawdziwe")
        plt.title("Macierz pomyłek")
        mlflow.log_figure(plt.gcf(), "plots/conf_matrix.png")
