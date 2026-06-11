import numpy as np
from numpy.testing import assert_allclose
from whiteboxml.examples.models.linear_regression import LinearRegression


def test_linear_regression_aprende_relacion():
    """
    Verifica que el modelo aprende una relación lineal simple.

    :authors: Matias
    :date: 2026-04-19
    """
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])  # y = 2x

    model = LinearRegression(lr=0.1, n_iter=1000)
    model.fit(X, y)

    y_pred = model.predict(X)

    assert_allclose(y_pred, y, rtol=1e-1)