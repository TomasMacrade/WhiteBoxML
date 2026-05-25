import numpy as np

class LinearRegression:
    """
    Clase para realizar una Regresión Lineal usando el método de 
    Descenso por Gradiente.
    """

    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000):
        """
        Constructor del modelo.

        :param learning_rate: Tasa de aprendizaje (tamaño del paso).
        :type learning_rate: float
        :param iterations: Número de iteraciones del algoritmo.
        :type iterations: int
        """
        self.lr = learning_rate
        self.iterations = iterations
        self.theta = None  # Pesos del modelo
        self.cost_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Entrena el modelo ajustando los parámetros para minimizar el error.

        :param X: Matriz de entrenamiento (muestras, características).
        :type X: np.ndarray
        :param y: Vector de etiquetas reales.
        :type y: np.ndarray
        :return: La instancia del modelo entrenada.
        :rtype: LinearRegression
        """
        m, n = X.shape
        # Añadimos columna de 1s para el intercepto (ordenada al origen)
        X_b = np.c_[np.ones((m, 1)), X]
        self.theta = np.zeros(n + 1)

        for _ in range(self.iterations):
            # Cálculo de la hipótesis (predicción)
            predictions = X_b.dot(self.theta)
            errors = predictions - y
            # Cálculo del gradiente
            gradient = (1 / m) * X_b.T.dot(errors)
            # Actualización de pesos
            self.theta -= self.lr * gradient
            
            # Guardamos el costo (MSE)
            cost = (1 / (2 * m)) * np.sum(np.square(errors))
            self.cost_history.append(cost)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice valores para nuevos datos.

        :param X: Datos de entrada.
        :type X: np.ndarray
        :return: Vector de predicciones.
        :rtype: np.ndarray
        """
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]
        return X_b.dot(self.theta)