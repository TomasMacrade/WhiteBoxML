# tests/test_base_linear_model.py

import numpy as np
from whiteboxml.base_linear_model import BaseLinearModel


class DummyModel(BaseLinearModel):
    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.zeros(X.shape[0])

    def compute_gradient(self, X, y):
        return np.zeros(X.shape[1]), 0.0

    def compute_hessian(self, X, y):
        return np.eye(X.shape[1])


def test_base_model_params():
    """
    Verifica que get_params y set_params funcionan correctamente.

    :authors: Matias
    :date: 2026-04-26
    """
    model = DummyModel()

    weights = np.array([1.0, 2.0])
    bias = 0.5

    model.set_params(weights, bias)
    w, b = model.get_params()

    assert np.allclose(w, weights)
    assert b == bias