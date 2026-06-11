# tests/test_logistic_regression.py

import numpy as np
from whiteboxml.examples.models.logistic_regression import LogisticRegression


def test_logistic_regression_clasifica():
    """
    Verifica que el modelo clasifica correctamente datos simples.

    :authors: Cristian
    :date: 2026-04-19
    """
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(lr=0.1, n_iter=1000)
    model.fit(X, y)

    y_pred = model.predict(X)

    assert (y_pred == y).all()