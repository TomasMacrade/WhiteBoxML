"""
Módulo de modelos lineales.

Este módulo implementa modelos predictivos lineales desde cero.

:authors: Tomás Macrade
:date: 24/04/2026
"""

# pylint: disable=invalid-name

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from whiteboxml.base_linear_model import BaseLinearModel


class LogisticRegression(BaseLinearModel):
    """
    Modelo de Regresión Logística para clasificación binaria.

    Implementa el algoritmo de clasificación basado en la función sigmoide y
    la minimización de la entropía cruzada mediante descenso de gradiente.

    :authors: Tomás Macrade
    :date: 24/04/2026
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000) -> None:
        """
        Inicializa los parámetros del modelo.

        :param learning_rate: Tasa de aprendizaje para el descenso de gradiente.
        :param n_iterations: Número de iteraciones de optimización.
        :return: None
        :authors: Tomás Macrade
        :date: 24/04/2026
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.coef_: Optional[np.ndarray] = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Calcula la función logística (sigmoide).

        :param z: Combinación lineal de los inputs y los pesos.
        :return: Probabilidades estimadas entre 0 y 1.
        :authors: Tomás Macrade
        :date: 24/04/2026
        """
        z_clipped = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Ajusta el modelo de regresión logística a los datos de entrenamiento.

        :param X: Matriz de características de entrenamiento.
        :param y: Vector de targets binarios reales (0 o 1).
        :return: La instancia del modelo ajustado.
        :authors: Tomás Macrade
        :date: 24/04/2026
        """
        n_samples, n_features = X.shape
        X_with_bias = np.c_[np.ones((n_samples, 1)), X]

        self.coef_ = np.zeros(n_features + 1)

        for _ in range(self.n_iterations):
            linear_model = X_with_bias @ self.coef_
            y_predicted = self._sigmoid(linear_model)
            gradient = (X_with_bias.T @ (y_predicted - y)) / n_samples
            self.coef_ -= self.learning_rate * gradient

        self.weights = self.coef_[1:]
        self.bias = float(self.coef_[0])

    def compute_gradient(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Calcula el gradiente de la función de costo (entropía cruzada).

        :param X: Matriz de características.
        :param y: Vector de etiquetas binarias.
        :return: Tupla (gradiente_pesos, gradiente_bias).
        :authors: Julian Carro, Lorena Bernal, Valentin Prina Cerai, Ivan Ledesma
        :date: 14/06/2026
        """
        n_samples = X.shape[0]
        z = X @ self.weights + self.bias
        y_pred = self._sigmoid(z)
        dw = (X.T @ (y_pred - y)) / n_samples
        db = float(np.sum(y_pred - y) / n_samples)
        return dw, db

    def compute_hessian(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calcula el hessiano de la función de costo logística.

        :param X: Matriz de características.
        :param y: Vector de etiquetas binarias.
        :return: Matriz Hessiana de shape (n_features, n_features).
        :authors: Julian Carro, Lorena Bernal, Valentin Prina Cerai, Ivan Ledesma
        :date: 14/06/2026
        """
        n_samples = X.shape[0]
        z = X @ self.weights + self.bias
        s = self._sigmoid(z)
        S = s * (1 - s)
        return (X.T @ (X * S[:, np.newaxis])) / n_samples

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Estima la probabilidad de pertenecer a la clase positiva (1).

        :param X: Matriz de características a predecir.
        :return: Vector de probabilidades estimadas.
        :authors: Tomás Macrade
        :date: 24/04/2026
        """
        if self.coef_ is None:
            raise ValueError(
                "El modelo debe ser entrenado con 'fit' antes de predecir."
            )

        X_with_bias = np.c_[np.ones((X.shape[0], 1)), X]
        linear_model = X_with_bias @ self.coef_
        return self._sigmoid(linear_model)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predice la etiqueta de clase utilizando un umbral de decisión.

        :param X: Matriz de características a predecir.
        :param threshold: Umbral de decisión (por defecto 0.5 según la teoría).
        :return: Vector de clases predichas (0 o 1).
        :authors: Tomás Macrade
        :date: 24/04/2026
        """
        y_predicted_cls = [1 if i >= threshold else 0 for i in self.predict_proba(X)]
        return np.array(y_predicted_cls)
