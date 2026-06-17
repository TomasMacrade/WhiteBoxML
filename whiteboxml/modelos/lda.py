"""
Implementación del Análisis Discriminante Lineal (LDA).

:authors: Gonzalo Ramirez
:date: 13/06/2026
"""

from typing import Optional, cast

import numpy as np
from numpy.typing import ArrayLike

from whiteboxml.modelos.base import BaseDiscriminantAnalysis


class LDA(BaseDiscriminantAnalysis):
    """
    Clasificador por Análisis Discriminante Lineal (LDA).

    Instancia el modelo asumiendo covarianza compartida.

    :authors: Gonzalo Ramirez
    :date: 13/06/2026
    """

    def __init__(self) -> None:
        """
        Inicializa el modelo LDA configurando la covarianza como nula.

        :authors: Gonzalo Ramirez
        :date: 13/06/2026
        """
        super().__init__()
        self.covariance_: Optional[np.ndarray] = None

    def fit(self, features: ArrayLike, targets: ArrayLike) -> "LDA":
        """
        Ajusta el modelo LDA calculando la covarianza compartida.

        :param features: Matriz de características de entrenamiento.
        :type features: ArrayLike
        :param targets: Vector de etiquetas reales.
        :type targets: ArrayLike
        :return: La propia instancia entrenada.
        :rtype: LDA
        :authors: Gonzalo Ramirez
        :date: 13/06/2026
        """
        self._compute_priors_and_means(features, targets)

        if self.classes_ is None or self.means_ is None:
            raise ValueError("Falló el cálculo de estadísticas base.")

        valid_classes = cast(np.ndarray, self.classes_)
        valid_means = cast(np.ndarray, self.means_)

        features_array = np.asarray(features)
        targets_array = np.asarray(targets)

        n_samples = features_array.shape[0]
        n_features = features_array.shape[1]

        self.covariance_ = np.zeros((n_features, n_features))

        for idx, cls in enumerate(valid_classes):
            cls_features = features_array[targets_array == cls]
            cls_centroid = valid_means[idx]

            deviations = cls_features - cls_centroid
            self.covariance_ += deviations.T @ deviations

        self.covariance_ /= n_samples

        return self

    def predict(self, features: ArrayLike) -> np.ndarray:
        """
        Predice las etiquetas utilizando la frontera de decisión lineal.

        :param features: Matriz de características a predecir.
        :type features: ArrayLike
        :return: Vector de etiquetas predichas.
        :rtype: np.ndarray
        :authors: Gonzalo Ramirez
        :date: 13/06/2026
        """
        if (
            self.classes_ is None
            or self.means_ is None
            or self.priors_ is None
            or self.covariance_ is None
        ):
            raise ValueError(
                "El modelo no está ajustado. Llame a 'fit' antes de 'predict'."
            )

        valid_classes = cast(np.ndarray, self.classes_)
        valid_means = cast(np.ndarray, self.means_)
        valid_priors = cast(np.ndarray, self.priors_)
        valid_covariance = cast(np.ndarray, self.covariance_)

        features_array = np.asarray(features)
        inv_covariance = np.linalg.inv(valid_covariance)

        # Reducimos las variables locales calculando las dimensiones directamente
        scores = np.zeros((features_array.shape[0], len(valid_classes)))

        for idx, _ in enumerate(valid_classes):
            cls_centroid = valid_means[idx]
            cls_prior = valid_priors[idx]

            # Unificamos los cálculos matemáticos para cumplir
            # con la regla de variables máximas de Pylint
            scores[:, idx] = (
                (features_array @ inv_covariance @ cls_centroid)
                - 0.5 * (cls_centroid.T @ inv_covariance @ cls_centroid)
                + np.log(cls_prior)
            )

        # Devolvemos directamente las predicciones
        # sin guardar índices en una variable extra
        return valid_classes[np.argmax(scores, axis=1)]
