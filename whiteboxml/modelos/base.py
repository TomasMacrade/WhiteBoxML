"""
Módulo con la abstracción base para los modelos discriminantes.

:authors: Gonzalo Ramirez
:date: 09/06/2026
"""

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike


# pylint: disable=too-few-public-methods
# Esta clase es una abstracción base para LDA y QDA,
# por lo que no se necesitan muchos métodos públicos.
class BaseDiscriminantAnalysis:
    """
    Clase base para Análisis Discriminante (LDA y QDA).

    :authors: Gonzalo Ramirez
    :date: 09/06/2026
    """

    def __init__(self) -> None:
        """
        Inicializa los atributos internos del modelo discriminante.

        :authors: Gonzalo Ramirez
        :date: 09/06/2026
        """
        self.classes_: Optional[np.ndarray] = None
        self.priors_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None

    def _compute_priors_and_means(
        self, features: ArrayLike, targets: ArrayLike
    ) -> None:
        """
        Calcula las probabilidades previas y los vectores de medias.

        :param features: Matriz de características.
        :param targets: Vector de etiquetas objetivo.
        :return: None
        :authors: Gonzalo Ramirez
        :date: 09/06/2026
        """
        matriz_features = np.asarray(features)
        vector_targets = np.asarray(targets)

        self.classes_ = np.unique(vector_targets)

        n_samples = matriz_features.shape[0]
        n_classes = len(self.classes_)
        n_features = matriz_features.shape[1]

        self.priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes_):
            features_c = matriz_features[vector_targets == c]

            self.priors_[idx] = float(features_c.shape[0] / n_samples)
            self.means_[idx, :] = np.mean(features_c, axis=0)
