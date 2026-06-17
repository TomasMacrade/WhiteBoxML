"""
Implementación del Análisis Discriminante Cuadrático (QDA).

"""

from typing import Optional, cast
import numpy as np
from numpy.typing import ArrayLike
from whiteboxml.modelos.base import BaseDiscriminantAnalysis


class QDA(BaseDiscriminantAnalysis):
    """
    Clasificador por Análisis Discriminante Cuadrático (QDA).

    Instancia el modelo asumiendo covarianza distinta por clase.

    """

    def __init__(self) -> None:
        """
        Inicializa el modelo QDA configurando las covarianzas como nulas.
        """
        super().__init__()
        self.covariances_: Optional[np.ndarray] = None  # Una matriz de covarianza por clase

    def fit(self, features: ArrayLike, targets: ArrayLike) -> "QDA":
        """
        Ajusta el modelo QDA calculando la covarianza para cada clase.
        """
        self._compute_priors_and_means(features, targets)

        if self.classes_ is None or self.means_ is None:
            raise ValueError("Falló el cálculo de estadísticas base.")

        valid_classes = cast(np.ndarray, self.classes_)
        valid_means = cast(np.ndarray, self.means_)

        features_array = np.asarray(features)
        targets_array = np.asarray(targets)

        n_features = features_array.shape[1]

        # Inicializamos una matriz de covarianza por clase
        self.covariances_ = np.zeros((len(valid_classes), n_features, n_features))

        for idx, cls in enumerate(valid_classes):
            class_features = features_array[targets_array == cls]

            # Edge case: si hay menos de 2 muestras no se puede estimar covarianza
            if class_features.shape[0] < 2:
                raise ValueError("No hay suficientes muestras para estimar covarianza")  # Validación de robustez
             
            centered_features = class_features - valid_means[idx]

            # Estimador de covarianza por clase
            self.covariances_[idx] = (
                centered_features.T @ centered_features
            ) / (class_features.shape[0] - 1)  # Uso de n_k - 1 (estimador insesgado)

        return self

    def predict(self, features: ArrayLike) -> np.ndarray:
        """
        Predice las etiquetas utilizando la función de score cuadrático.
        """
        if (
            self.classes_ is None
            or self.means_ is None
            or self.covariances_ is None
            or self.priors_ is None  # Validación faltante: se usan priors en el cálculo de scores, deben estar presentes
        ):
            raise ValueError("El modelo no ha sido entrenado correctamente.")

        valid_classes = cast(np.ndarray, self.classes_)# Consistencia con LDA: validación de clases, medias, covarianzas y priors antes de predecir
        valid_means = cast(np.ndarray, self.means_)# Consistencia con LDA: validación de clases, medias, covarianzas y priors antes de predecir
        valid_covariances = cast(np.ndarray, self.covariances_)# Consistencia con LDA: validación de clases, medias, covarianzas y priors antes de predecir
        valid_priors = cast(np.ndarray, self.priors_)  # Consistencia con LDA

        features_array = np.asarray(features)
        n_samples = features_array.shape[0]

        scores = np.zeros((n_samples, len(valid_classes)))# Inicialización de matriz de scores para cada muestra y clase (consistencia con LDA)

        epsilon = 1e-6  # Regularización para estabilidad numérica

        for idx, _ in enumerate(valid_classes):# Iteración por clase para calcular los scores (consistencia con LDA, pero con covarianza específica por clase)
            mean_vec = valid_means[idx]# Media de la clase actual (consistencia con LDA)
            cov_matrix = valid_covariances[idx]# Covarianza de la clase actual (consistencia con LDA, pero con covarianza específica por clase)

            # Regularización: evita problemas con matrices singulares o mal condicionadas
            cov_matrix_reg = cov_matrix + epsilon * np.eye(cov_matrix.shape[0])  # sigma + epsilon * I (regularización)#

            # Se calcula una vez por clase (evita ineficiencia de recalcular por muestra)
            inv_cov_matrix = np.linalg.inv(cov_matrix_reg)  # Puede fallar sin regularización
            sign, log_det_cov = np.linalg.slogdet(cov_matrix_reg)  # Más estable que log(det)

            # Chequeo de validez de la matriz de covarianza
            if sign <= 0:
                raise ValueError("Covarianza no válida")  # Evita log de determinantes no positivos

            for i in range(n_samples):
                diff = features_array[i] - mean_vec

                scores[i, idx] = (
                    -0.5 * diff.T @ inv_cov_matrix @ diff  # Término cuadrático
                    - 0.5 * log_det_cov                  # Penalización por volumen (usa slogdet)
                    + np.log(valid_priors[idx])          # Uso de priors (requiere validación previa)
                )

        predicted_indices = np.argmax(scores, axis=1)
        return valid_classes[predicted_indices]