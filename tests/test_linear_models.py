"""
Tests para los modelos lineales.

:authors: Tomás Macrade
:date: 24/04/2026
"""

# pylint: disable=invalid-name, no-member

import numpy as np

from whiteboxml.linear_models import LogisticRegression
from whiteboxml.metricas.clasificacion.clasificacion import accuracy


def test_logistic_regression_fit_predict():
    """
    Testea que el modelo de regresión logística converja y prediga
    correctamente un dataset sintético linealmente separable.

    :authors: Tomás Macrade
    :date: 24/04/2026
    """
    rng = np.random.RandomState(42)
    # Generamos 100 muestras con 2 variables
    X = rng.randn(100, 2)
    # Condición lineal para crear clases sintéticas (si x1 + x2 > 0 es clase 1)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = LogisticRegression(learning_rate=0.5, n_iterations=1000)
    model.fit(X, y)
    preds = model.predict(X)

    # Usamos la métrica provista internamente por WhiteBoxML
    acc = accuracy(y, preds)

    # Verificamos que alcance más de un 95% de accuracy
    assert acc > 0.95


def test_logistic_regression_coef_shape():
    """
    Testea que las dimensiones de los coeficientes sean correctas
    (incluyendo el término independiente o bias).

    :authors: Tomás Macrade
    :date: 24/04/2026
    """
    rng = np.random.RandomState(0)
    X = rng.randn(10, 3)
    y = np.random.randint(0, 2, 10)

    model = LogisticRegression(learning_rate=0.1, n_iterations=10)
    model.fit(X, y)

    assert model.coef_ is not None
    assert model.coef_.shape == (4,)


def test_predict_proba_rango():
    """
    Testea que las probabilidades predichas estén contenidas entre 0 y 1.

    :authors: Tomás Macrade
    :date: 24/04/2026
    """
    rng = np.random.RandomState(1)
    X = rng.randn(20, 2)
    y = np.random.randint(0, 2, 20)

    model = LogisticRegression(learning_rate=0.1, n_iterations=50)
    model.fit(X, y)
    probas = model.predict_proba(X)

    assert np.all(probas >= 0.0)
    assert np.all(probas <= 1.0)
