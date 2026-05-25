# -*- coding: utf-8 -*-
import numpy as np
from typing import List, Tuple

class LinearRegression():
    """
    Modelo de regresión lineal con soporte para:
    - solución analítica
    - GD batch, SGD y mini-batch
    :author: Equipo Regresión Lineal
    :date: 10/05/26
    """

    def __init__(
        self,
        method: str = "analytic",
        learning_rate: float = 1e-3,
        iterations: int = 10000,
        tol: float = 1e-6,
        normalize: bool = False,
        mode: str = "batch",
        batch_size: int = 32
    ) -> None:
        """
        Inicializa el modelo.
        Ajusta los parámetros del modelo mediante una solución analítica o métodos iterativos
        (descenso por gradiente por lote, minilotes o estocástico). Opcionalmente, puede aplicar una
        estandarización previa de las variables de entrada.

        :param method: Método de entrenamiento ("analytic" o "gd").
        :type method: str
        :param learning_rate: Tasa de aprendizaje (solo GD).
        :type learning_rate: float
        :param iterations: Máximo número de iteraciones (solo GD).
        :type iterations: int
        :param tol: Tolerancia de convergencia (solo GD).
        :type tol: float
        :param normalize: Si True, las variables de entrada se normalizan.
        :type normalize: bool
        :param mode: Modo de GD: "batch", "sgd" o "mbgd"
        :type mode: str
        :param batch_size: Tamaño del lote para el modo minibatch
        :type batch_size: int
        :return: None
        :rtype: None
        :author: Equipo Regresión Lineal
        :date: 10/05/26
        """

        self.method = method
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.tol = tol
        self.normalize = normalize
        self.mode = mode
        self.batch_size = batch_size

        self._theta = np.array([])
        self._theta_scaled = np.array([])
        self._trained = False

        self._mean = np.array([])
        self._std = np.array([])

        self.cost_history: List[float] = []

    def check_X_y(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Valida que X e y tengan dimensiones compatibles para regresión.

        :param X: Matriz de características.
        :type X: np.ndarray
        :param y: Vector de valores objetivo.
        :type y: np.ndarray
        :return: Tupla (X, y) validada.
        :rtype: Tuple[np.ndarray, np.ndarray]
        :raises ValueError: Si las dimensiones son incompatibles.
        :author: Equipo Regresión Lineal
        :date: 10/05/26
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X e y deben ser np.ndarray")

        if X.ndim != 2:
            raise ValueError("X debe ser una matriz 2D")

        if y.ndim != 1:
            raise ValueError("y debe ser un vector 1D")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y deben tener la misma cantidad de filas")

        return X, y

    def fit_standardize(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Ajusta y transforma X (media 0, varianza 1).

        :param X: Matriz de entrada.
        :type X: np.ndarray
        :return: X transformado, media y std.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        :author: Equipo Regresión Lineal
        :date: 10/05/26
        """
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1.0

        return (X - mean) / std, mean, std

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """
        Agrega una columna de unos a la matriz X para el intercepto.

        :param X: Matriz de características.
        :type X: np.ndarray
        :return: Matriz con columna de unos agregada.
        :rtype: np.ndarray
        :author: Equipo Regresión Lineal
        :date: 10/05/26
        """
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def compute_cost(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """
        Calcula el costo cuadrático medio.

        :param X: Matriz de entrada.
        :type X: np.ndarray
        :param y: Vector objetivo.
        :type y: np.ndarray
        :param theta: Parámetros.
        :type theta: np.ndarray
        :return: Valor del costo.
        :rtype: float
        :author: Equipo Regresión Lineal
        :date: 10/05/26
        """
        n = X.shape[0]
        residuals = X @ theta - y
        return (1.0 / (2 * n)) * (residuals @ residuals)

    def gradient_descent(
        self,
        X: np.ndarray,
        y: np.ndarray,
        theta: np.ndarray,
        learning_rate: float,
        iterations: int,
        tol: float,
        mode: str = "batch",
        batch_size: int = 32,
        random_state: int = 42
    ) -> Tuple[np.ndarray, int, List[float]]:
        """
        Descenso por gradiente: batch, SGD o mini-batch.

        :param X: Matriz con intercepto.
        :type X: np.ndarray
        :param y: Vector objetivo.
        :type y: np.ndarray
        :param theta: Inicialización.
        :type theta: np.ndarray
        :param learning_rate: Tasa de aprendizaje.
        :type learning_rate: float
        :param iterations: Iteraciones máximas.
        :type iterations: int
        :param tol: Tolerancia.
        :type tol: float
        :param mode: "batch", "sgd" o "mbgd".
        :type mode: str
        :param batch_size: Tamaño del batch.
        :type batch_size: int
        :param random_state: Semilla.
        :type random_state: int
        :return: theta final, iteraciones, historial de costo.
        :rtype: Tuple[np.ndarray, int, List[float]]
        :author: Equipo Regresión Lineal
        :date: 10/05/26
        """
        n = X.shape[0]
        rng = np.random.default_rng(random_state)
        cost_history: List[float] = []

        for i in range(iterations):
            old_theta = theta.copy()

            if mode == "batch":
                grad = (1/n) * (X.T @ (X @ theta - y))
            elif mode == "sgd":
                idx = rng.integers(0, n)
                xi = X[idx]
                yi = y[idx]
                grad = xi * (xi @ theta - yi)
            elif mode == "mbgd":
                idx = rng.choice(n, size=min(batch_size, n), replace=False)
                Xb = X[idx]
                yb = y[idx]
                grad = (1/len(idx)) * (Xb.T @ (Xb @ theta - yb))
            else:
                raise ValueError("Modo inválido")

            theta = theta - learning_rate * grad
            cost_history.append(self.compute_cost(X, y, theta))

            if np.linalg.norm(theta - old_theta) <= tol:
                return theta, i + 1, cost_history

        return theta, iterations, cost_history

    def _recover_original_theta(self) -> None:
        """
        Recupera los parámetros del modelo en la escala original de las variables.

        :return: None
        :rtype: None
        :author: Equipo Regresión Lineal
        :date: 10/05/26
        """
        if not self.normalize:
            self._theta = self._theta_scaled.copy()
            return

        theta0 = self._theta_scaled[0]
        rest = self._theta_scaled[1:]

        theta_rest = rest / self._std
        theta0 = theta0 - np.sum((rest * self._mean) / self._std)

        self._theta = np.concatenate(([theta0], theta_rest))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Entrena el modelo de regresión lineal utilizando el método especificado.

        :param X: Matriz de características.
        :type X: np.ndarray
        :param y: Vector objetivo.
        :type y: np.ndarray
        :return: None
        :rtype: None
        :raises ValueError: Si el método no es válido.
        :author: Equipo Regresión Lineal
        :date: 10/05/26
        """
        X, y = self.check_X_y(X, y)

        if self.normalize:
            X, self._mean, self._std = self.fit_standardize(X)

        X = self._add_intercept(X)

        if self.method == "analytic":
            self._theta_scaled = np.linalg.pinv(X) @ y
            self.cost_history = []
        elif self.method == "gd":
            theta0 = np.zeros(X.shape[1])
            self._theta_scaled, _, self.cost_history = self.gradient_descent(
                X, y,
                theta0,
                self.learning_rate,
                self.iterations,
                self.tol,
                mode=self.mode,
                batch_size=self.batch_size
            )
        else:
            raise ValueError("Método inválido")

        self._recover_original_theta()
        self._trained = True

    @property
    def theta(self) -> np.ndarray:
        """
        Devuelve los parámetros del modelo.

        :return: Vector de parámetros.
        :rtype: np.ndarray
        :raises RuntimeError: Si el modelo no fue entrenado.
        :author: Equipo Regresión Lineal
        :date: 10/05/26
        """
        if not self._trained:
            raise RuntimeError("Modelo no entrenado")
        return self._theta

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones.

        :param X: Matriz de entrada.
        :type X: np.ndarray
        :return: Predicciones.
        :rtype: np.ndarray
        :raises RuntimeError: Si el modelo no fue entrenado.
        :author: Equipo Regresión Lineal
        :date: 10/05/26
        """
        if not self._trained:
            raise RuntimeError("Modelo no entrenado")

        X = self._add_intercept(X)
        return X @ self._theta
