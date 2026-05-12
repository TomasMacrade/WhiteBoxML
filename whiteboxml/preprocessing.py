"""
Módulo de preprocessing.

:authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
:date: 07/05/2026
"""

# pylint: disable=invalid-name, missing-module-docstring


from __future__ import annotations

import numpy as np


class StandardScaler:
    """
    Escala los datos usando normalización Z-score.
    :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
    :date: 07/05/2026
    """

    def __init__(self) -> None:
        """
        Inicializa el scaler.

        :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
        :date: 07/05/2026
        """

        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> StandardScaler:
        """
        Calcula la media y desviación estándar de los datos.

        :param X: Datos de entrada.
        :type X: np.ndarray
        :return: Instancia del scaler entrenado.
        :rtype: StandardScaler
        :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
        :date: 07/05/2026
        """

        X = np.array(X)

        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        self.std_[self.std_ == 0] = 1

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Aplica la normalización Z-score.

        :param X: Datos a transformar.
        :type X: np.ndarray
        :return: Datos escalados.
        :rtype: np.ndarray
        :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
        :date: 07/05/2026
        """

        if self.mean_ is None or self.std_ is None:
            raise ValueError("Debes llamar a fit antes de transform")

        X = np.array(X)

        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Ajusta y transforma los datos.

        :param X: Datos de entrada.
        :type X: np.ndarray
        :return: Datos transformados.
        :rtype: np.ndarray
        :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
        :date: 07/05/2026
        """

        return self.fit(X).transform(X)


class MinMaxScaler:
    """
    Escala los datos al rango [0, 1].
    :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
    :date: 07/05/2026
    """

    def __init__(self) -> None:
        """
        Inicializa el scaler.

        :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
        :date: 07/05/2026
        """
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> MinMaxScaler:
        """
        Calcula mínimos y máximos.

        :param X: Datos de entrada.
        :type X: np.ndarray
        :return: Instancia entrenada.
        :rtype: MinMaxScaler
        :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
        :date: 07/05/2026
        """

        X = np.array(X)

        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Aplica escalado Min-Max.

        :param X: Datos a transformar.
        :type X: np.ndarray
        :return: Datos escalados.
        :rtype: np.ndarray
        :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
        :date: 07/05/2026
        """

        if self.min_ is None or self.max_ is None:
            raise ValueError("Debes llamar a fit antes de transform")

        X = np.array(X)

        range_ = self.max_ - self.min_
        range_[range_ == 0] = 1

        return (X - self.min_) / range_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Ajusta y transforma los datos.

        :param X: Datos de entrada.
        :type X: np.ndarray
        :return: Datos transformados.
        :rtype: np.ndarray
        :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
        :date: 07/05/2026
        """

        return self.fit(X).transform(X)


# -------------------------------#


class LabelEncoder:
    """
    Codifica etiquetas categóricas como enteros.
    :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
    :date: 07/05/2026
    """

    def __init__(self) -> None:
        """
        Inicializa el encoder.

        :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
        :date: 07/05/2026
        """
        self.classes_: dict | None = None

    def fit(self, y: np.ndarray) -> LabelEncoder:
        """
        Aprende las categorías únicas.

        :param y: Etiquetas.
        :type y: np.ndarray
        :return: Instancia entrenada.
        :rtype: LabelEncoder
        :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
        :date: 07/05/2026
        """

        y = np.array(y)

        unique = np.unique(y)
        self.classes_ = {label: idx for idx, label in enumerate(unique)}

        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Transforma etiquetas a números.

        :param y: Etiquetas.
        :type y: np.ndarray
        :return: Etiquetas codificadas.
        :rtype: np.ndarray
        :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
        :date: 07/05/2026
        """

        if self.classes_ is None:
            raise ValueError("Debes llamar a fit antes de transform")

        y = np.array(y)

        return np.array([self.classes_[label] for label in y])

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Ajusta y transforma los datos.

        :param X: Datos de entrada.
        :type X: np.ndarray
        :return: Datos transformados.
        :rtype: np.ndarray
        :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
        :date: 07/05/2026
        """

        return self.fit(X).transform(X)


class OneHotEncoder:
    """
    Codifica variables categóricas en formato one-hot.
    :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
    :date: 07/05/2026
    """

    def __init__(self) -> None:
        """
        Inicializa el encoder.

        :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
        :date: 07/05/2026
        """
        self.categories_: np.ndarray | None = None
        self.mapping_: dict | None = None

    def fit(self, y: np.ndarray) -> OneHotEncoder:
        """
        Aprende categorías únicas.

        :param y: Etiquetas.
        :type y: np.ndarray
        :return: Instancia entrenada.
        :rtype: OneHotEncoder
        :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
        :date: 07/05/2026
        """

        y = np.array(y)

        self.categories_ = np.unique(y)
        self.mapping_ = {cat: i for i, cat in enumerate(self.categories_)}

        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Aplica codificación one-hot.

        :param y: Etiquetas.
        :type y: np.ndarray
        :return: Matriz one-hot.
        :rtype: np.ndarray
        :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
        :date: 07/05/2026
        """

        if self.categories_ is None or self.mapping_ is None:
            raise ValueError("Debes llamar a fit antes de transform")

        y = np.array(y)

        one_hot = np.zeros((len(y), len(self.categories_)))

        for i, label in enumerate(y):
            idx = self.mapping_[label]
            one_hot[i, idx] = 1

        return one_hot

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Ajusta y transforma los datos.

        :param X: Datos de entrada.
        :type X: np.ndarray
        :return: Datos transformados.
        :rtype: np.ndarray
        :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
        :date: 07/05/2026
        """

        return self.fit(X).transform(X)
