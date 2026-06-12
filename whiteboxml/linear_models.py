"""
Módulo de modelos lineales.

Este módulo implementa modelos predictivos lineales desde cero.

:authors: Tomás Macrade
:date: 24/04/2026
"""

# pylint: disable=invalid-name

from __future__ import annotations

from typing import Optional

import numpy as np


class LogisticRegression:
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
        # Se recorta z para evitar warnings por overflow en np.exp
        z_clipped = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        """
        Ajusta el modelo de regresión logística a los datos de entrenamiento.

        :param X: Matriz de características de entrenamiento.
        :param y: Vector de targets binarios reales (0 o 1).
        :return: La instancia del modelo ajustado.
        :authors: Tomás Macrade
        :date: 24/04/2026
        """
        n_samples, n_features = X.shape
        # Añadimos una columna de 1s para el término independiente (x0 = 1)
        X_with_bias = np.c_[np.ones((n_samples, 1)), X]

        # Inicializamos los pesos en cero
        self.coef_ = np.zeros(n_features + 1)

        for _ in range(self.n_iterations):
            linear_model = X_with_bias @ self.coef_
            y_predicted = self._sigmoid(linear_model)

            # Gradiente de la función de costo (entropía cruzada)
            # Promediamos dividiendo por n_samples
            gradient = (X_with_bias.T @ (y_predicted - y)) / n_samples

            # Actualización de pesos usando Descenso de Gradiente
            self.coef_ -= self.learning_rate * gradient

        return self

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
