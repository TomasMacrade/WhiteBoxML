"""
Tests para OneVsRest.

:authors: Julian Carro, Lorena Bernal, Valentin Prina Cerai, Ivan Ledesma
:date: 14/06/2026
"""

# pylint: disable=invalid-name

import numpy as np
import pytest

from whiteboxml.base_linear_model import BaseLinearModel
from whiteboxml.linear_models import LogisticRegression
from whiteboxml.modelos import OneVsRest


class DummyBinaryClassifier(BaseLinearModel):
    """Clasificador binario mínimo para tests."""

    def __init__(self, noise: float = 0.0) -> None:
        """Inicializa con ruido opcional."""
        super().__init__()
        self.noise = noise
        self._positive_mean: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Entrena guardando la media de la clase positiva."""
        self._positive_mean = (
            X[y == 1].mean(axis=0) if np.any(y == 1) else X.mean(axis=0)
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice clase binaria con umbral 0.5."""
        return (self.predict_proba(X) >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Devuelve probabilidad basada en distancia a la media positiva."""
        dists = np.linalg.norm(X - self._positive_mean, axis=1)
        max_dist = dists.max() + 1e-8
        return 1.0 - (dists / max_dist) + self.noise

    def compute_gradient(self, X, y):
        """Gradiente trivial para tests."""
        return np.zeros(X.shape[1]), 0.0

    def compute_hessian(self, X, y):
        """Hessiano trivial para tests."""
        return np.zeros((X.shape[1], X.shape[1]))


class ModelNotFromBase:
    """Modelo que NO hereda de BaseLinearModel."""

    def fit(self, X, y):
        """Fit vacío."""

    def predict_proba(self, X):
        """Predict proba vacío."""


def _make_multiclass_data():
    rng = np.random.default_rng(42)
    centers = np.array([[0, 0], [5, 0], [0, 5]])
    X = np.vstack([rng.normal(c, 0.5, (30, 2)) for c in centers])
    y = np.repeat([0, 1, 2], 30)
    return X, y


def test_fit_creates_k_models():
    """
    Verifica que fit cree un modelo por cada clase en y.

    :authors: Julian Carro, Lorena Bernal, Valentin Prina Cerai, Ivan Ledesma
    :date: 14/06/2026
    """
    X, y = _make_multiclass_data()
    ovr = OneVsRest(DummyBinaryClassifier)
    ovr.fit(X, y)
    assert len(ovr.models_) == 3
    assert list(ovr.classes_) == [0, 1, 2]


def test_predict_proba_shape():
    """
    Verifica que predict_proba devuelva shape (n_samples, n_clases).

    :authors: Julian Carro, Lorena Bernal, Valentin Prina Cerai, Ivan Ledesma
    :date: 14/06/2026
    """
    X, y = _make_multiclass_data()
    ovr = OneVsRest(DummyBinaryClassifier)
    ovr.fit(X, y)
    probas = ovr.predict_proba(X)
    assert probas.shape == (len(y), 3)


def test_predict_returns_known_classes():
    """
    Verifica que predict devuelva solo clases vistas en entrenamiento.

    :authors: Julian Carro, Lorena Bernal, Valentin Prina Cerai, Ivan Ledesma
    :date: 14/06/2026
    """
    X, y = _make_multiclass_data()
    ovr = OneVsRest(DummyBinaryClassifier)
    ovr.fit(X, y)
    preds = ovr.predict(X)
    assert set(preds).issubset({0, 1, 2})


def test_predict_accuracy_on_separable_data():
    """
    Verifica accuracy mayor a 90% en datos separables.

    :authors: Julian Carro, Lorena Bernal, Valentin Prina Cerai, Ivan Ledesma
    :date: 14/06/2026
    """
    X, y = _make_multiclass_data()
    ovr = OneVsRest(DummyBinaryClassifier)
    ovr.fit(X, y)
    preds = ovr.predict(X)
    assert np.mean(preds == y) > 0.9


def test_predict_before_fit_raises():
    """
    Verifica que predict lance RuntimeError si no se llamó fit antes.

    :authors: Julian Carro, Lorena Bernal, Valentin Prina Cerai, Ivan Ledesma
    :date: 14/06/2026
    """
    ovr = OneVsRest(DummyBinaryClassifier)
    X = np.array([[1, 2], [3, 4]])
    with pytest.raises(RuntimeError, match="fit"):
        ovr.predict(X)


def test_model_kwargs_passed_to_instances():
    """
    Verifica que los kwargs se propaguen a cada instancia del modelo.

    :authors: Julian Carro, Lorena Bernal, Valentin Prina Cerai, Ivan Ledesma
    :date: 14/06/2026
    """
    X, y = _make_multiclass_data()
    ovr = OneVsRest(DummyBinaryClassifier, noise=0.1)
    ovr.fit(X, y)
    for m in ovr.models_:
        assert m.noise == 0.1


def test_model_not_from_base_raises():
    """
    Verifica que un modelo que no herede de BaseLinearModel sea rechazado.

    :authors: Julian Carro, Lorena Bernal, Valentin Prina Cerai, Ivan Ledesma
    :date: 14/06/2026
    """
    with pytest.raises(TypeError, match="BaseLinearModel"):
        OneVsRest(ModelNotFromBase)


def test_instance_instead_of_class_raises():
    """
    Verifica que pasar una instancia en vez de una clase lance TypeError.

    :authors: Julian Carro, Lorena Bernal, Valentin Prina Cerai, Ivan Ledesma
    :date: 14/06/2026
    """
    with pytest.raises(TypeError, match="clase"):
        OneVsRest(DummyBinaryClassifier())


def test_fit_returns_self():
    """
    Verifica que fit devuelva la instancia para encadenamiento.

    :authors: Julian Carro, Lorena Bernal, Valentin Prina Cerai, Ivan Ledesma
    :date: 14/06/2026
    """
    X, y = _make_multiclass_data()
    ovr = OneVsRest(DummyBinaryClassifier)
    result = ovr.fit(X, y)
    assert result is ovr


def test_logistic_regression_como_clasificador_binario():
    """
    Integración: OvR con LogisticRegression del repo sobre 3 clases separables.

    :authors: Julian Carro, Lorena Bernal, Valentin Prina Cerai, Ivan Ledesma
    :date: 14/06/2026
    """
    rng = np.random.default_rng(7)
    centers = np.array([[0, 0], [6, 0], [0, 6]])
    X = np.vstack([rng.normal(c, 0.8, (40, 2)) for c in centers])
    y = np.repeat([0, 1, 2], 40)

    ovr = OneVsRest(LogisticRegression, learning_rate=0.5, n_iterations=1000)
    ovr.fit(X, y)
    preds = ovr.predict(X)

    assert np.mean(preds == y) > 0.95


def test_string_labels():
    """
    Verifica que OvR funcione con etiquetas de tipo string.

    :authors: Julian Carro, Lorena Bernal, Valentin Prina Cerai, Ivan Ledesma
    :date: 14/06/2026
    """
    rng = np.random.default_rng(0)
    centers = {"perro": [0, 0], "gato": [5, 0], "pajaro": [0, 5]}
    X = np.vstack([rng.normal(c, 0.3, (20, 2)) for c in centers.values()])
    y = np.repeat(list(centers.keys()), 20)
    ovr = OneVsRest(DummyBinaryClassifier)
    ovr.fit(X, y)
    preds = ovr.predict(X)
    assert set(preds).issubset(set(centers.keys()))
    assert np.mean(preds == y) > 0.9
