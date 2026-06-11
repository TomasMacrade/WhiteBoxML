from __future__ import annotations
import numpy as np
from typing import Tuple
from whiteboxml.base_linear_model import BaseLinearModel


class LogisticRegression(BaseLinearModel):
    """
    Implementación de regresión logística usando descenso por gradiente.
    """

    def __init__(self, lr: float = 0.01, n_iter: int = 1000) -> None:
        """
        Inicializa el modelo.

        :param lr: Learning rate.
        :param n_iter: Número de iteraciones.
        :return: None
        :authors: Cristian
        :date: 2026-04-19
        """
        super().__init__()
        self.lr = lr
        self.n_iter = n_iter

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Función sigmoide.

        :param z: Entrada lineal.
        :return: Probabilidad.
        :authors: Cristian
        :date: 2026-04-19
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Entrena el modelo usando descenso por gradiente.

        :param X: Matriz de características.
        :param y: Vector de etiquetas.
        :return: None
        :authors: Cristian
        :date: 2026-04-19
        """
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        self.gradient_descent(X, y, self.lr, self.n_iter)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones binarias.

        :param X: Matriz de características.
        :return: Vector de clases (0 o 1).
        :authors: Pablo
        :date: 2026-04-19
        """
        if self.weights is None or self.bias is None:
            raise ValueError("El modelo debe entrenarse antes de predecir.")

        z = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(z)

        return (y_pred >= 0.5).astype(int)

    def compute_gradient(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Calcula el gradiente de la función de costo logística.

        :param X: Matriz de características.
        :param y: Vector de etiquetas.
        :return: Gradientes (dw, db).
        :authors: Pablo
        :date: 2026-04-19
        """
        n_samples = X.shape[0]

        z = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(z)

        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)

        return dw, db

    def compute_hessian(
        self, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        Calcula el hessiano de la función de costo logística.

        :param X: Matriz de características.
        :param y: Vector de etiquetas.
        :return: Matriz Hessiana.
        :authors: Cristian
        :date: 2026-04-19
        """
        n_samples = X.shape[0]

        z = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(z)

        S = y_pred * (1 - y_pred)  # vector diagonal implícito

        # Construcción del Hessiano
        H = (1 / n_samples) * X.T @ (X * S[:, np.newaxis])

        return H