"""
Clase base abstracta para modelos lineales.

:authors: Pablo, Cristian, Matias
:date: 2026-04-19
"""

# pylint: disable=invalid-name, unnecessary-pass

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class BaseLinearModel(ABC):
    """
    Clase base abstracta para modelos lineales.

    Define la interfaz común para todos los modelos del paquete.

    :authors: Pablo, Cristian, Matias
    :date: 2026-04-19
    """

    def __init__(self) -> None:
        """
        Inicializa los parámetros del modelo.

        :authors: Pablo
        :date: 2026-04-19
        """
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None

    def __repr__(self) -> str:
        """
        Representación del modelo.

        :return: Representación en string del modelo.
        :authors: Pablo, Cristian
        :date: 2026-04-19
        """
        return f"{self.__class__.__name__}(weights={self.weights}, bias={self.bias})"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Entrena el modelo.

        :param X: Matriz de características.
        :param y: Vector de etiquetas.
        :return: None
        :authors: Pablo, Cristian
        :date: 2026-04-19
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones.

        :param X: Matriz de características.
        :return: Vector de predicciones.
        :authors: Carla, Cristian, Pablo
        :date: 2026-04-19
        """
        pass

    def get_params(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Obtiene los parámetros del modelo.

        :return: Tupla (weights, bias).
        :authors: Carla, Cristian, Pablo
        :date: 2026-04-19
        """
        return self.weights, self.bias

    def set_params(self, weights: np.ndarray, bias: float) -> None:
        """
        Asigna los parámetros del modelo.

        :param weights: Vector de pesos.
        :param bias: Término independiente.
        :return: None
        :authors: Cristian, Matias, Pablo
        :date: 2026-04-19
        """
        self.weights = weights
        self.bias = bias

    @abstractmethod
    def compute_gradient(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Calcula el gradiente de la función de costo.

        :param X: Matriz de características.
        :param y: Vector de etiquetas.
        :return: Gradientes (dw, db).
        :authors: Cristian
        :date: 2026-04-19
        """
        pass

    @abstractmethod
    def compute_hessian(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calcula el hessiano de la función de costo.

        :param X: Matriz de características.
        :param y: Vector de etiquetas.
        :return: Matriz Hessiana.
        :authors: Pablo, Cristian
        :date: 2026-04-19
        """
        pass

    def gradient_descent(
        self, X: np.ndarray, y: np.ndarray, lr: float, n_iter: int
    ) -> None:
        """
        Ejecuta descenso por gradiente.

        :param X: Matriz de características.
        :param y: Vector de etiquetas.
        :param lr: Learning rate.
        :param n_iter: Número de iteraciones.
        :return: None
        :authors: Carla, Cristian, Mati, Pablo
        :date: 2026-04-19
        """
        if self.weights is None or self.bias is None:
            raise ValueError(
                "Los parámetros deben inicializarse antes del descenso de gradiente."
            )

        for _ in range(n_iter):
            dw, db = self.compute_gradient(X, y)
            self.weights -= lr * dw
            self.bias -= lr * db

    def newton_method(self, X: np.ndarray, y: np.ndarray, n_iter: int) -> None:
        """
        Ejecuta optimización usando método de Newton.

        :param X: Matriz de características.
        :param y: Vector de etiquetas.
        :param n_iter: Número de iteraciones.
        :return: None
        :authors: Carla, Mati, Pablo
        :date: 2026-04-19
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Inicializar parámetros antes.")

        n_samples = X.shape[0]

        # Agregamos columna de 1s para el bias
        X_ext = np.hstack([X, np.ones((n_samples, 1))])

        # Vector completo de parámetros
        theta = np.hstack([self.weights, self.bias])

        for _ in range(n_iter):
            # Predicción
            y_pred = X_ext @ theta

            # Gradiente completo
            grad = (1 / n_samples) * (X_ext.T @ (y_pred - y))

            # Hessiano completo
            H = (1 / n_samples) * (X_ext.T @ X_ext)

            # Inversa
            H_inv = np.linalg.inv(H)

            # Update
            theta = theta - H_inv @ grad

        # Separar nuevamente
        self.weights = theta[:-1]
        self.bias = theta[-1]
