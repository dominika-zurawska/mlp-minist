from typing import Optional, Tuple, Union, List
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import mlflow.pyfunc


class MLP:
    """Implementacja perceptronu wielowarstwowego (MLP) z dropout i momentum.
    
    Ta implementacja zawiera:
    - Aktywację ReLU w warstwie ukrytej
    - Aktywację softmax w warstwie wyjściowej
    - Regularyzację dropout
    - Regularyzację L2
    - Optymalizację opartą na momentum
    - Mini-batch gradient descent
    - Harmonogram współczynnika uczenia
    - Wczesne zatrzymywanie
    
    Atrybuty:
        W1 (np.ndarray): Wagi pierwszej warstwy
        b1 (np.ndarray): Biasy pierwszej warstwy
        W2 (np.ndarray): Wagi drugiej warstwy
        b2 (np.ndarray): Biasy drugiej warstwy
        dropout_rate (float): Prawdopodobieństwo wyłączenia neuronu
        reg_lambda (float): Siła regularyzacji L2
        learning_rate (float): Współczynnik uczenia
        vW1 (np.ndarray): Momentum dla W1
        vb1 (np.ndarray): Momentum dla b1
        vW2 (np.ndarray): Momentum dla W2
        vb2 (np.ndarray): Momentum dla b2
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout_rate: float = 0.5,
        reg_lambda: float = 0.01,
        learning_rate: float = 0.1,
    ) -> None:
        """Inicjalizuje MLP z podaną architekturą i hiperparametrami.
        
        Argumenty:
            input_size: Liczba cech wejściowych
            hidden_size: Liczba neuronów w warstwie ukrytej
            output_size: Liczba klas wyjściowych
            dropout_rate: Prawdopodobieństwo wyłączenia neuronu (domyślnie: 0.5)
            reg_lambda: Siła regularyzacji L2 (domyślnie: 0.01)
            learning_rate: Początkowy współczynnik uczenia (domyślnie: 0.1)
        """
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        self.dropout_rate = dropout_rate
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate

        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

    def relu(self, Z: np.ndarray) -> np.ndarray:
        """Stosuje funkcję aktywacji ReLU.
        
        Argumenty:
            Z: Tablica wejściowa
            
        Zwraca:
            Tablica z zastosowaną aktywacją ReLU
        """
        return np.maximum(0, Z)

    def softmax(self, Z: np.ndarray) -> np.ndarray:
        """Stosuje funkcję aktywacji softmax ze stabilnością numeryczną.
        
        Argumenty:
            Z: Tablica wejściowa
            
        Zwraca:
            Tablica z zastosowaną aktywacją softmax
        """
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Wykonuje propagację w przód przez sieć.
        
        Argumenty:
            X: Cechy wejściowe o kształcie (batch_size, input_size)
            training: Czy stosować dropout (True podczas treningu, False podczas wnioskowania)
            
        Zwraca:
            Prawdopodobieństwa wyjściowe o kształcie (batch_size, output_size)
        """
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)

        if training:
            self.dropout_mask = np.random.rand(*self.A1.shape) > self.dropout_rate
            self.A1 *= self.dropout_mask
        else:
            self.A1 *= 1 - self.dropout_rate

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def compute_loss(self, Y_hat: np.ndarray, Y: np.ndarray) -> float:
        """Oblicza funkcję straty cross-entropy z regularyzacją L2.
        
        Argumenty:
            Y_hat: Przewidywane prawdopodobieństwa
            Y: Prawdziwe etykiety (one-hot encoded)
            
        Zwraca:
            Całkowita strata (cross-entropy + regularyzacja L2)
        """
        cross_entropy_loss = -np.mean(np.sum(Y * np.log(Y_hat + 1e-10), axis=1))
        l2_reg = (self.reg_lambda / (2 * Y.shape[0])) * (
            np.sum(np.square(self.W1)) + np.sum(np.square(self.W2))
        )
        return cross_entropy_loss + l2_reg

    def backward(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Wykonuje propagację wsteczną i aktualizuje wagi używając momentum.
        
        Argumenty:
            X: Cechy wejściowe
            Y: Prawdziwe etykiety (one-hot encoded)
        """
        m = X.shape[0]

        dZ2 = self.A2 - Y
        dW2 = np.dot(self.A1.T, dZ2) / m + (self.reg_lambda * self.W2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0)
        dW1 = np.dot(X.T, dZ1) / m + (self.reg_lambda * self.W1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        beta = 0.9
        self.vW1 = beta * self.vW1 + (1 - beta) * dW1
        self.vb1 = beta * self.vb1 + (1 - beta) * db1
        self.vW2 = beta * self.vW2 + (1 - beta) * dW2
        self.vb2 = beta * self.vb2 + (1 - beta) * db2

        self.W1 -= self.learning_rate * self.vW1
        self.b1 -= self.learning_rate * self.vb1
        self.W2 -= self.learning_rate * self.vW2
        self.b2 -= self.learning_rate * self.vb2

    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        epochs: int,
        batch_size: int = 64,
        initial_lr: float = 0.1,
        decay: float = 0.001,
        patience: int = 5,
    ) -> None:
        """Trenuje model używając mini-batch gradient descent.
        
        Argumenty:
            X_train: Cechy treningowe
            Y_train: Etykiety treningowe (one-hot encoded)
            X_val: Cechy walidacyjne
            Y_val: Etykiety walidacyjne (one-hot encoded)
            epochs: Liczba epok treningu
            batch_size: Rozmiar mini-batchy (domyślnie: 64)
            initial_lr: Początkowy współczynnik uczenia (domyślnie: 0.1)
            decay: Współczynnik zaniku learning rate (domyślnie: 0.001)
            patience: Liczba epok do oczekiwania przed early stopping (domyślnie: 5)
        """
        num_samples = X_train.shape[0]
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self.learning_rate = initial_lr / (1 + decay * epoch)

            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            Y_train = Y_train[indices]

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                X_batch, Y_batch = X_train[start:end], Y_train[start:end]

                Y_hat = self.forward(X_batch, training=True)
                loss = self.compute_loss(Y_hat, Y_batch)
                self.backward(X_batch, Y_batch)

            Y_val_hat = self.forward(X_val, training=False)
            val_loss = self.compute_loss(Y_val_hat, Y_val)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Wczesne zatrzymanie")
                break

            print(
                f"Epoka {epoch}, Strata treningowa: {loss:.4f}, "
                f"Strata walidacyjna: {val_loss:.4f}, "
                f"Współczynnik uczenia: {self.learning_rate:.4f}"
            )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Przewiduje prawdopodobieństwa klas.
        
        Argumenty:
            X: Cechy wejściowe
            
        Zwraca:
            Tablica prawdopodobieństw klas
        """
        return self.forward(X, training=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Przewiduje klasy.
        
        Argumenty:
            X: Cechy wejściowe
            
        Zwraca:
            Tablica przewidywanych klas
        """
        Y_hat = self.predict_proba(X)
        return np.argmax(Y_hat, axis=1)


class MLPWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper dla MLP do integracji z MLflow.
    
    Opakowuje model MLP, dodając funkcjonalność śledzenia eksperymentów MLflow
    i obsługę przewidywań w formacie zgodnym z MLflow.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """Inicjalizuje wrapper MLP.
        
        Argumenty:
            input_size: Liczba cech wejściowych
            hidden_size: Liczba neuronów w warstwie ukrytej
            output_size: Liczba klas wyjściowych
        """
        self.model = MLP(input_size, hidden_size, output_size)
        self.encoder = OneHotEncoder(sparse_output=False)
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("output_size", output_size)

    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        epochs: int,
        batch_size: int = 64,
        initial_lr: float = 0.1,
        decay: float = 0.001,
        patience: int = 5,
    ) -> None:
        """Trenuje model i loguje parametry do MLflow.
        
        Argumenty:
            X_train: Cechy treningowe
            Y_train: Etykiety treningowe
            X_val: Cechy walidacyjne
            Y_val: Etykiety walidacyjne
            epochs: Liczba epok
            batch_size: Rozmiar mini-batchy
            initial_lr: Początkowy współczynnik uczenia
            decay: Współczynnik zaniku learning rate
            patience: Liczba epok do early stopping
        """
        self.model.train(
            X_train,
            Y_train,
            X_val,
            Y_val,
            epochs,
            batch_size,
            initial_lr,
            decay,
            patience,
        )
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("initial_lr", initial_lr)
        mlflow.log_param("decay", decay)
        mlflow.log_param("patience", patience)

    def predict(
        self, 
        context: mlflow.pyfunc.PythonModelContext, 
        model_input: np.ndarray, 
        params: Optional[dict] = None
    ) -> np.ndarray:
        """Wykonuje przewidywania używając modelu.
        
        Argumenty:
            context: Kontekst modelu MLflow
            model_input: Dane wejściowe do przewidywania
            params: Parametry przewidywania (domyślnie: {"predict_method": "predict"})
            
        Zwraca:
            Przewidywane wartości
            
        Raises:
            ValueError: Jeśli metoda przewidywania nie jest wspierana
        """
        params = params or {"predict_method": "predict"}
        predict_method = params.get("predict_method")
        if predict_method == "predict":
            return self.model.predict(model_input)
        elif predict_method == "predict_proba":
            return self.model.predict_proba(model_input)
        else:
            raise ValueError(
                f"Metoda przewidywania '{predict_method}' nie jest wspierana."
            )