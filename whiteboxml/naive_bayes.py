"""
Clasificador Naive Bayes híbrido para variables continuas y binarias.

:authors: Ivana Zurdo, Nicolas Barreto, Tomas Muino, Seba Sanchez Bentolila
:date: 11/06/2026
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


class NaiveBayes:
    """
    Clasificador Naive Bayes híbrido.

    Soporta:
    - Variables binarias mediante Bernoulli.
    - Variables continuas mediante distribución normal.

    El tipo de cada feature se detecta automáticamente.

    :authors: Ivana Zurdo, Nicolas Barreto, Tomas Muino, Seba Sanchez Bentolila
    :date: 11/06/2026
    """

    def __init__(self) -> None:
        """
        Inicializa el modelo.

        :return: None.
        :authors: Ivana Zurdo, Nicolas Barreto, Tomas Muino, Seba Sanchez Bentolila
        :date: 11/06/2026
        """

        # Clases posibles
        self.classes: np.ndarray | None = None

        # Probabilidades P(C)
        self.class_priors: Dict = {}

        # Parámetros de cada feature
        self.parameters: Dict = {}

        # Tipo de cada feature
        self.feature_types: List[str] = []

    def _detect_feature_type(self, column: np.ndarray) -> str:
        """
        Detecta automáticamente el tipo de variable.

        Si la variable contiene únicamente:
        - 0 y 1
        - True y False (python los toma como 1 y 0 respectivamente)

        entonces se considera Bernoulli.

        En cualquier otro caso se considera Gaussiana.

        :param column: Columna de datos.
        :return: Tipo de feature detectado.
        :authors: Ivana Zurdo, Nicolas Barreto, Tomas Muino, Seba Sanchez Bentolila
        :date: 11/06/2026
        """

        unique_values = set(np.unique(column))

        if unique_values.issubset({0, 1}):
            return "bernoulli"

        return "gaussian"

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NaiveBayes":
        """
        Entrena el modelo Naive Bayes.
        Para cada clase estima:
        - El prior P(C) como frecuencia relativa en el dataset.
        - Para features Bernoulli: P(x=1|C) con suavizado de Laplace.
        - Para features Gaussianas: media mu y varianza sigma^2.

        :param x: Matriz de features.
        :param y: Vector de etiquetas.
        :return: El modelo entrenado.
        :authors: Ivana Zurdo, Nicolas Barreto, Tomas Muino, Seba Sanchez Bentolila
        :date: 11/06/2026
        """

        x = np.array(x)
        y = np.array(y)

        # Obtener clases únicas
        self.classes = np.unique(y)

        # Detectar tipo de cada feature
        self.feature_types = []

        for i in range(x.shape[1]):

            column = x[:, i]

            feature_type = self._detect_feature_type(column)

            self.feature_types.append(feature_type)

        # Entrenar parámetros para cada clase
        for c in self.classes:

            # Filtrar muestras de la clase
            x_c = x[y == c]

            # Prior de clase
            self.class_priors[c] = len(x_c) / len(x)

            self.parameters[c] = []

            # Recorrer features
            for i, feature_type in enumerate(self.feature_types):

                column = x_c[:, i]

                # -------------------------
                # FEATURE BERNOULLI
                # -------------------------
                if feature_type == "bernoulli":

                    column = column.astype(int)

                    # Suavizado de Laplace para evitar probabilidades cero
                    p = (np.sum(column) + 1) / (len(column) + 2)

                    self.parameters[c].append({"type": "bernoulli", "p": p})

                # -------------------------
                # FEATURE GAUSSIANA
                # -------------------------
                elif feature_type == "gaussian":

                    mean = np.mean(column)
                    var = np.var(column)

                    # Evitar división por cero
                    if var == 0:
                        var = 1e-9

                    self.parameters[c].append(
                        {"type": "gaussian", "mean": mean, "var": var}
                    )
        return self

    def _bernoulli_log_probability(self, x: int, p: float) -> float:
        """
        Calcula log(P(x|C)) para Bernoulli.

        :param x: Valor observado.
        :param p: Probabilidad de éxito.
        :return: Log probabilidad.
        :authors: Ivana Zurdo, Nicolas Barreto, Tomas Muino, Seba Sanchez Bentolila
        :date: 11/06/2026
        """

        if x == 1:
            return np.log(p)

        return np.log(1 - p)

    def _gaussian_log_probability(self, x: float, mean: float, var: float) -> float:
        """
        Calcula log(P(x|C)) usando distribución normal.

        :param x: Valor observado.
        :param mean: Media de la distribución.
        :param var: Varianza de la distribución.
        :return: Log probabilidad.
        :authors: Ivana Zurdo, Nicolas Barreto, Tomas Muino, Seba Sanchez Bentolila
        :date: 11/06/2026
        """

        return -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)

    def _best_class(self, class_scores: dict) -> str:
        """
        Retorna la clase con mayor log-probabilidad.

        :param class_scores: Diccionario con log-probabilidades por clase.
        :return: Clase con mayor score.
        :authors: Ivana Zurdo, Nicolas Barreto, Tomas Muino, Seba Sanchez Bentolila
        :date: 11/06/2026
        """
        return max(class_scores, key=lambda cls: class_scores[cls])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predice las clases para nuevas muestras.

        :param x: Matriz de features.
        :return: Predicciones.
        :authors: Ivana Zurdo, Nicolas Barreto, Tomas Muino, Seba Sanchez Bentolila
        :date: 11/06/2026
        """
        if self.classes is None:
            raise ValueError(
                "El modelo no ha sido entrenado. Llama a fit() antes de predecir."
            )
        x = np.array(x)

        predictions = []

        # Recorrer muestras
        for sample in x:

            class_scores: dict = {}

            # Calcular score para cada clase
            for c in self.classes:

                # Comenzar con log(P(C))
                log_prob = np.log(self.class_priors[c])

                # Recorrer features
                for i, value in enumerate(sample):

                    params = self.parameters[c][i]

                    # -------------------------
                    # BERNOULLI
                    # -------------------------
                    if params["type"] == "bernoulli":

                        log_prob += self._bernoulli_log_probability(value, params["p"])

                    # -------------------------
                    # GAUSSIAN
                    # -------------------------
                    elif params["type"] == "gaussian":

                        log_prob += self._gaussian_log_probability(
                            value, params["mean"], params["var"]
                        )

                class_scores[c] = log_prob

            # Elegir clase con mayor probabilidad
            predicted_class = self._best_class(class_scores)

            predictions.append(predicted_class)

        return np.array(predictions)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Devuelve las probabilidades calibradas por clase para cada muestra.

        Aplica log-sum-exp sobre los log-posteriors para normalizar.

        :param x: Matriz de features a clasificar.
        :return: Probabilidades en [0, 1] de forma (n_samples, n_classes).
        :raises ValueError: Si el modelo no fue entrenado.
        :authors: Ivana Zurdo, Nicolas Barreto, Tomas Muino, Seba Sanchez Bentolila
        :date: 11/06/2026
        """
        if self.classes is None:
            raise ValueError(
                "El modelo no fue entrenado. Llama a fit() antes de predecir."
            )

        x = np.array(x)
        all_probs = []

        for sample in x:
            log_scores = []

            for c in self.classes:
                log_prob = np.log(self.class_priors[c])

                for i, value in enumerate(sample):
                    params = self.parameters[c][i]
                    if params["type"] == "bernoulli":
                        log_prob += self._bernoulli_log_probability(value, params["p"])
                    elif params["type"] == "gaussian":
                        log_prob += self._gaussian_log_probability(
                            value, params["mean"], params["var"]
                        )

                log_scores.append(log_prob)

            log_scores = np.array(log_scores)
            log_sum = np.log(np.sum(np.exp(log_scores)))
            all_probs.append(np.exp(log_scores - log_sum))

        return np.array(all_probs)

    def __repr__(self) -> str:
        """
        Retorna si el modelo esta entrenado.

        :return: String con el estado del modelo.
        :authors: Ivana Zurdo, Nicolas Barreto, Tomas Muino, Seba Sanchez Bentolila
        :date: 11/06/2026
        """
        estado = "entrenado" if self.classes is not None else "sin entrenar"
        return f"NaiveBayes({estado})"
