# nosec B101
# ruff: noqa
# bandit: disable=B101
# pylint: skip-file

"""
Tests para los modelos de Análisis Discriminante (LDA y QDA).

Incluye tests básicos de fit/predict, verificación de propiedades de
probabilidad, validación de la clase BaseDiscriminantAnalysis y pruebas con
datos sintéticos generados con NumPy (Escenarios LDA, QDA y Equivalencia).

:authors: Mariana Battistini & Santiago Gabriel Vallejo
:date: 15/06/2026
"""

import numpy as np
import pytest

from whiteboxml.modelos.lda import LDA
from whiteboxml.modelos.qda import QDA


# ---------------------------------------------------------------------------
# Tests básicos de fit/predict y restricciones probabilísticas
# ---------------------------------------------------------------------------


def test_lda_fit_and_predict_basic():
    """
    Test de predicción básica y axiomas de probabilidad para el modelo LDA.

    :authors: Mariana Battistini & Santiago Gabriel Vallejo
    :date: 15/06/2026
    """
    features = np.array([[1.0, 1.0], [1.5, 2.0], [5.0, 5.0], [6.0, 5.5]])
    targets = np.array([0, 0, 1, 1])

    modelo = LDA()
    modelo.fit(features, targets)

    punto_clase_0 = np.array([[1.2, 1.3]])
    punto_clase_1 = np.array([[5.5, 5.2]])

    # Verificación de clasificación estándar
    assert modelo.predict(punto_clase_0)[0] == 0  # nosec
    assert modelo.predict(punto_clase_1)[0] == 1  # nosec

    # Verificación matemática: la suma de las probabilidades a posteriori debe ser 1.0
    probabilities = modelo.predict_proba(punto_clase_0)[0]
    assert np.isclose(np.sum(probabilities), 1.0)  # nosec


def test_qda_fit_and_predict_basic():
    """
    Test de predicción básica y axiomas de probabilidad para el modelo QDA.

    :authors: Mariana Battistini & Santiago Gabriel Vallejo
    :date: 15/06/2026
    """
    features = np.array([[1.0, 1.0], [1.5, 2.0], [5.0, 5.0], [6.0, 5.5]])
    targets = np.array([0, 0, 1, 1])

    modelo = QDA()
    modelo.fit(features, targets)

    punto_clase_0 = np.array([[1.1, 1.2]])
    punto_clase_1 = np.array([[5.8, 5.4]])

    # Verificación de clasificación estándar
    assert modelo.predict(punto_clase_0)[0] == 0  # nosec
    assert modelo.predict(punto_clase_1)[0] == 1  # nosec

    # Verificación matemática: la suma de las probabilidades a posteriori debe ser 1.0
    probabilities = modelo.predict_proba(punto_clase_1)[0]
    assert np.isclose(np.sum(probabilities), 1.0)  # nosec


# ---------------------------------------------------------------------------
# Tests unitarios sobre BaseDiscriminantAnalysis (vía LDA)
# ---------------------------------------------------------------------------


def test_compute_priors_and_means():
    """
    Test de BaseDiscriminantAnalysis: verifica el cálculo balanceado de
    las probabilidades a priori (priors_) y los centroides (means_).

    :authors: Mariana Battistini & Santiago Gabriel Vallejo
    :date: 15/06/2026
    """
    features = np.array(
        [
            [0.0, 0.0],
            [2.0, 2.0],
            [10.0, 10.0],
            [12.0, 12.0],
        ]
    )
    targets = np.array([0, 0, 1, 1])

    modelo = LDA()
    modelo.fit(features, targets)

    assert modelo.classes_ is not None  # nosec
    assert modelo.priors_ is not None  # nosec
    assert modelo.means_ is not None  # nosec

    # Comprobación matemática de los estimadores por máxima verosimilitud
    assert np.array_equal(modelo.classes_, np.array([0, 1]))  # nosec
    assert np.allclose(modelo.priors_, np.array([0.5, 0.5]))  # nosec
    assert np.allclose(modelo.means_[0], np.array([1.0, 1.0]))  # nosec
    assert np.allclose(modelo.means_[1], np.array([11.0, 11.0]))  # nosec


def test_compute_priors_and_means_clases_desbalanceadas():
    """
    Test de BaseDiscriminantAnalysis con desbalanceo: verifica que priors_
    refleje la proporción frecuentista real de la muestra.

    :authors: Mariana Battistini & Santiago Gabriel Vallejo
    :date: 15/06/2026
    """
    features = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [10.0, 10.0],
        ]
    )
    targets = np.array([0, 0, 0, 1])

    modelo = LDA()
    modelo.fit(features, targets)

    assert np.allclose(modelo.priors_, np.array([0.75, 0.25]))  # nosec
    assert np.allclose(modelo.means_[0], np.array([1.0, 1.0]))  # nosec
    assert np.allclose(modelo.means_[1], np.array([10.0, 10.0]))  # nosec


# ---------------------------------------------------------------------------
# Helpers de generación de datos sintéticos
# ---------------------------------------------------------------------------


def _generar_datos_misma_covarianza(n_por_clase: int = 200, seed: int = 42):
    """
    Genera dos clases con medias distintas pero la MISMA matriz de
    covarianza (escenario teórico favorable para LDA).
    :authors: Mariana Battistini & Santiago Gabriel Vallejo
    :date: 15/06/2026
    """
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])

    mean_0 = np.array([0.0, 0.0])

    mean_1 = np.array([4.0, 4.0])

    clase_0 = rng.multivariate_normal(mean_0, cov, size=n_por_clase)
    clase_1 = rng.multivariate_normal(mean_1, cov, size=n_por_clase)

    features = np.vstack([clase_0, clase_1])
    targets = np.concatenate([np.zeros(n_por_clase), np.ones(n_por_clase)])
    return features, targets


def _generar_datos_covarianzas_distintas(n_por_clase: int = 200, seed: int = 42):
    """
    Genera clases con misma media pero matrices de covarianza muy distintas
    (escenario teórico favorable para QDA).
    :authors: Mariana Battistini & Santiago Gabriel Vallejo
    :date: 15/06/2026
    """
    rng = np.random.default_rng(seed)

    mean_0 = np.array([0.0, 0.0])
    mean_1 = np.array([0.0, 0.0])

    cov_0 = np.array([[1.0, 0.0], [0.0, 1.0]])  # Nube circular
    cov_1 = np.array([[8.0, 0.0], [0.0, 0.5]])  # Nube alargada

    clase_0 = rng.multivariate_normal(mean_0, cov_0, size=n_por_clase)
    clase_1 = rng.multivariate_normal(mean_1, cov_1, size=n_por_clase)

    features = np.vstack([clase_0, clase_1])
    targets = np.concatenate([np.zeros(n_por_clase), np.ones(n_por_clase)])
    return features, targets


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el ratio de exactitud (accuracy).
    :authors: Mariana Battistini & Santiago Gabriel Vallejo
    :date: 15/06/2026"""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


# ---------------------------------------------------------------------------
# Test A: LDA con misma covarianza (escenario favorable)
# ---------------------------------------------------------------------------


def test_lda_misma_covarianza_accuracy_alto():
    """
    Test A: Al compartir matriz de covarianza, la frontera lineal de LDA
    debe alcanzar un desempeño óptimo (Acc > 0.9).
    :authors: Mariana Battistini & Santiago Gabriel Vallejo
    :date: 15/06/2026
    """
    features, targets = _generar_datos_misma_covarianza()

    modelo = LDA()
    modelo.fit(features, targets)
    predicciones = modelo.predict(features)

    acc = _accuracy(targets, predicciones)
    assert acc > 0.9  # nosec


def test_lda_predict_shape_y_clases_validas():
    """Verifica consistencia en las dimensiones y etiquetas de salida de LDA.
    :authors: Mariana Battistini & Santiago Gabriel Vallejo
    :date: 15/06/2026
    """
    features, targets = _generar_datos_misma_covarianza(n_por_clase=10)

    modelo = LDA()
    modelo.fit(features, targets)
    predicciones = modelo.predict(features)

    assert predicciones.shape == (features.shape[0],)  # nosec
    assert set(np.unique(predicciones)).issubset(set(np.unique(targets)))  # nosec


def test_lda_predict_sin_fit_lanza_error():
    """
    Garantiza la imposibilidad de predecir con un modelo LDA no entrenado.
    :authors: Mariana Battistini & Santiago Gabriel Vallejo
    :date: 15/06/2026
    """
    modelo = LDA()
    with pytest.raises(ValueError):
        modelo.predict(np.array([[0.0, 0.0]]))


# ---------------------------------------------------------------------------
# Test B: QDA con covarianzas distintas (escenario favorable a QDA)
# ---------------------------------------------------------------------------


def test_qda_covarianzas_distintas_supera_a_lda():
    """
    Test B: Con estructuras de variabilidad distintas, la flexibilidad
    cuadrática de QDA debe superar la rigidez lineal de LDA.
    :authors: Mariana Battistini & Santiago Gabriel Vallejo
    :date: 15/06/2026
    """
    features, targets = _generar_datos_covarianzas_distintas()

    lda = LDA()
    lda.fit(features, targets)
    acc_lda = _accuracy(targets, lda.predict(features))

    qda = QDA()
    qda.fit(features, targets)
    acc_qda = _accuracy(targets, qda.predict(features))

    assert acc_qda > acc_lda  # nosec


def test_qda_predict_shape_y_clases_validas():
    """
    Verifica consistencia en las dimensiones y etiquetas de salida de QDA.
    :authors: Mariana Battistini & Santiago Gabriel Vallejo
    :date: 15/06/2026
    """
    features, targets = _generar_datos_covarianzas_distintas(n_por_clase=10)

    modelo = QDA()
    modelo.fit(features, targets)
    predicciones = modelo.predict(features)

    assert predicciones.shape == (features.shape[0],)  # nosec
    assert set(np.unique(predicciones)).issubset(set(np.unique(targets)))  # nosec


def test_qda_predict_sin_fit_lanza_error():
    """
    Garantiza la imposibilidad de predecir con un modelo QDA no entrenado.
    :authors: Mariana Battistini & Santiago Gabriel Vallejo
    :date: 15/06/2026
    """
    modelo = QDA()
    with pytest.raises(ValueError):
        modelo.predict(np.array([[0.0, 0.0]]))


# ---------------------------------------------------------------------------
# Test C: Teorema de Equivalencia Matemática
# ---------------------------------------------------------------------------


def test_qda_equivale_a_lda_con_covarianzas_iguales():
    """
    Test C: Demuestra la convergencia de modelos. Si las matrices de
    covarianza son idénticas, la frontera cuadrática se linealiza,
    haciendo que QDA equivalga matemáticamente a LDA.
    :authors: Mariana Battistini & Santiago Gabriel Vallejo
    :date: 15/06/2026
    """
    features, targets = _generar_datos_misma_covarianza()

    lda = LDA()
    lda.fit(features, targets)
    pred_lda = lda.predict(features)

    qda = QDA()
    qda.fit(features, targets)
    pred_qda = qda.predict(features)

    coincidencia = _accuracy(pred_lda, pred_qda)
    assert coincidencia > 0.95  # nosec
