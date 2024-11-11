import numpy as np
from sklearn.preprocessing import OneHotEncoder
import mlflow.pyfunc


class MLP:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout_rate=0.5,
        reg_lambda=0.01,
        learning_rate=0.1,
    ):
        # Weight Initialization with He initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        # Hyperparameters
        self.dropout_rate = dropout_rate
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate

        # Momentum terms initialization
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

    def relu(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def forward(self, X, training=True):
        # Forward propagation with dropout
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)

        if training:
            self.dropout_mask = np.random.rand(*self.A1.shape) > self.dropout_rate
            self.A1 *= self.dropout_mask  # Apply dropout mask
        else:
            self.A1 *= 1 - self.dropout_rate  # Scale during inference

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def compute_loss(self, Y_hat, Y):
        # Cross-entropy loss with L2 regularization
        cross_entropy_loss = -np.mean(np.sum(Y * np.log(Y_hat + 1e-10), axis=1))
        l2_reg = (self.reg_lambda / (2 * Y.shape[0])) * (
            np.sum(np.square(self.W1)) + np.sum(np.square(self.W2))
        )
        return cross_entropy_loss + l2_reg

    def backward(self, X, Y):
        m = X.shape[0]

        # Backpropagation
        dZ2 = self.A2 - Y
        dW2 = np.dot(self.A1.T, dZ2) / m + (self.reg_lambda * self.W2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0)  # Derivative of ReLU
        dW1 = np.dot(X.T, dZ1) / m + (self.reg_lambda * self.W1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Momentum update
        beta = 0.9
        self.vW1 = beta * self.vW1 + (1 - beta) * dW1
        self.vb1 = beta * self.vb1 + (1 - beta) * db1
        self.vW2 = beta * self.vW2 + (1 - beta) * dW2
        self.vb2 = beta * self.vb2 + (1 - beta) * db2

        # Gradient descent step with momentum
        self.W1 -= self.learning_rate * self.vW1
        self.b1 -= self.learning_rate * self.vb1
        self.W2 -= self.learning_rate * self.vW2
        self.b2 -= self.learning_rate * self.vb2

    def train(
        self,
        X_train,
        Y_train,
        X_val,
        Y_val,
        epochs,
        batch_size=64,
        initial_lr=0.1,
        decay=0.001,
        patience=5,
    ):
        num_samples = X_train.shape[0]
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Learning rate schedule
            self.learning_rate = initial_lr / (1 + decay * epoch)

            # Shuffle the data
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            Y_train = Y_train[indices]

            # Mini-batch training
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                X_batch, Y_batch = X_train[start:end], Y_train[start:end]

                # Forward and backward pass
                Y_hat = self.forward(X_batch, training=True)
                loss = self.compute_loss(Y_hat, Y_batch)
                self.backward(X_batch, Y_batch)

            # Validation loss and early stopping
            Y_val_hat = self.forward(X_val, training=False)
            val_loss = self.compute_loss(Y_val_hat, Y_val)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping")
                break

            print(
                f"Epoch {epoch}, Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}, Learning Rate: {self.learning_rate:.4f}"
            )

    def predict_proba(self, X):
        return self.forward(X, training=False)

    def predict(self, X):
        Y_hat = self.predict_proba(X)
        return np.argmax(Y_hat, axis=1)


class MLPWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, input_size, hidden_size, output_size):
        self.model = MLP(input_size, hidden_size, output_size)
        self.encoder = OneHotEncoder(sparse_output=False)
        mlflow.log_param("input_size", input_size)
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("output_size", output_size)

    def train(
        self,
        X_train,
        Y_train,
        X_val,
        Y_val,
        epochs,
        batch_size=64,
        initial_lr=0.1,
        decay=0.001,
        patience=5,
    ):
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

    def predict(self, context, model_input, params=None):
        params = params or {"predict_method": "predict"}
        predict_method = params.get("predict_method")
        if predict_method == "predict":
            return self.model.predict(model_input)
        elif predict_method == "predict_proba":
            return self.model.predict_proba(model_input)
        else:
            raise ValueError(
                f"The prediction method '{predict_method}' is not supported."
            )
