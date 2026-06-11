import numpy as np
from numpy.testing import assert_allclose
from whiteboxml.examples.models.linear_regression import LinearRegression


def test_newton_method_converge_rapido():
    """
    Verifica que el método de Newton converge rápidamente.

    :authors: Cristian, Matias, Pablo
    :date: 2026-04-19
    """
    X = np.array([[1], [2], [3], [4]])
    y = np.array([3, 6, 9, 12])  # y = 3x

    model = LinearRegression()

    model.weights = np.zeros(X.shape[1])
    model.bias = 0.0

    model.newton_method(X, y, n_iter=5)

    y_pred = model.predict(X)

    assert_allclose(y_pred, y, rtol=1e-2)