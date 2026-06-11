"""
Tests para el clasificador Naive Bayes híbrido.

:authors: Ivana Zurdo, Nicolas Barreto, Tomas Muino, Seba Sanchez Bentolila
:date: 11/06/2026
"""

import numpy as np
import pytest

from whiteboxml.naive_bayes import NaiveBayes


def test_detecta_bernoulli() -> None:
    """
    Verifica que la columna con solo 0s y 1s se detecte como Bernoulli.

    :return: None.
    :authors: Ivana Zurdo, Nicolas Barreto, Tomas Muino, Seba Sanchez Bentolila
    :date: 11/06/2026
    """
    x = np.array([[1], [0], [1], [0]])
    y = np.array(["a", "b", "a", "b"])
    model = NaiveBayes()
    model.fit(x, y)
    assert model.feature_types[0] == "bernoulli"


def test_detecta_gaussiana() -> None:
    """
    Verifica que la columna con valores continuos se detecte como Gaussiana.

    :return: None.
    :authors: Ivana Zurdo, Nicolas Barreto, Tomas Muino, Seba Sanchez Bentolila
    :date: 11/06/2026
    """
    x = np.array([[180], [175], [160], [155]])
    y = np.array(["hombre", "hombre", "mujer", "mujer"])
    model = NaiveBayes()
    model.fit(x, y)
    assert model.feature_types[0] == "gaussian"


def test_predict_without_fit() -> None:
    """
    Verifica que el modelo lance una excepción
    si se intenta predecir antes de entrenarlo.

    :return: None.
    :authors: Ivana Zurdo, Nicolas Barreto, Tomas Muino, Seba Sanchez Bentolila
    :date: 11/06/2026
    """

    model = NaiveBayes()

    with pytest.raises(ValueError):
        model.predict([[1, 0]])


def test_naive_bayes_bernoulli() -> None:
    """
    Verifica que Naive Bayes pueda clasificar
    correctamente utilizando únicamente
    variables Bernoulli.

    :return: None.
    :authors: Ivana Zurdo, Nicolas Barreto, Tomas Muino, Seba Sanchez Bentolila
    :date: 11/06/2026
    """

    x = np.array([[1], [1], [0], [0]])

    y = np.array(["spam", "spam", "no_spam", "no_spam"])

    model = NaiveBayes()

    model.fit(x, y)

    predictions = model.predict([[1], [0]])

    expected = np.array(["spam", "no_spam"])

    assert np.array_equal(predictions, expected)


def test_naive_bayes_gaussian() -> None:
    """
    Verifica que Naive Bayes pueda clasificar
    correctamente utilizando únicamente
    variables gaussianas.

    :return: None.
    :authors: Ivana Zurdo, Nicolas Barreto, Tomas Muino, Seba Sanchez Bentolila
    :date: 11/06/2026
    """

    x = np.array([[180], [175], [160], [155]])

    y = np.array(["hombre", "hombre", "mujer", "mujer"])

    model = NaiveBayes()

    model.fit(x, y)

    predictions = model.predict([[178], [158]])

    expected = np.array(["hombre", "mujer"])

    assert np.array_equal(predictions, expected)


def test_naive_bayes_predice_correctamente() -> None:
    """
    Verifica que Naive Bayes pueda clasificar
    correctamente un conjunto simple de datos
    con variables gaussianas y Bernoulli.

    :return: None.
    :authors: Ivana Zurdo, Nicolas Barreto, Tomas Muino, Seba Sanchez Bentolila
    :date: 11/06/2026
    """
    # Altura (gaussiana) y usa_barba (Bernoulli)
    x = np.array([[180, 1], [175, 1], [160, 0], [155, 0]])
    # Etiquetas de clase (género)
    y = np.array(["hombre", "hombre", "mujer", "mujer"])

    model = NaiveBayes()

    model.fit(x, y)

    predictions = model.predict([[178, 1], [158, 0]])

    expected = np.array(["hombre", "mujer"])

    assert np.array_equal(predictions, expected)


def test_laplace_evita_cero() -> None:
    """
    Verifica que con suavizado de Laplace P(x=1|C) nunca sea 0 ni 1.

    :return: None.
    :authors: Ivana Zurdo, Nicolas Barreto, Tomas Muino, Seba Sanchez Bentolila
    :date: 11/06/2026
    """
    x = np.array([[0], [0], [0], [1]], dtype=float)
    y = np.array([0, 0, 0, 1])
    model = NaiveBayes()
    model.fit(x, y)
    p = model.parameters[0][0]["p"]
    assert 0 < p < 1


def test_varianza_cero_suavizada() -> None:
    """
    Verifica que si todos los valores de una feature son iguales
    la varianza sea mayor a 0.

    :return: None.
    :authors: Ivana Zurdo, Nicolas Barreto, Tomas Muino, Seba Sanchez Bentolila
    :date: 11/06/2026
    """
    x = np.array([[5.0, 1], [5.0, 1], [10.0, 0], [10.0, 0]])
    y = np.array([0, 0, 1, 1])
    model = NaiveBayes()
    model.fit(x, y)
    assert model.parameters[0][0]["var"] > 0


def test_repr_entrenado() -> None:
    """
    Verifica que __repr__ indique "entrenado" si el modelo fue entrenado.
    :return: None.
    :authors: Ivana Zurdo, Nicolas Barreto, Tomas Muino, Seba Sanchez Bentolila
    :date: 11/06/2026
    """
    x = np.array([[1], [0]])
    y = np.array(["a", "b"])
    model = NaiveBayes()
    model.fit(x, y)
    assert "entrenado" in repr(model)


def test_repr_sin_entrenar() -> None:
    """
    Verifica que __repr__ indique "sin entrenar" si el modelo no fue entrenado.
    :return: None.
    :authors: Ivana Zurdo, Nicolas Barreto, Tomas Muino, Seba Sanchez Bentolila
    :date: 11/06/2026
    """
    model = NaiveBayes()
    assert "sin entrenar" in repr(model)


def test_predict_proba_suma_uno() -> None:
    """
    Verifica que predict_proba() retorne probabilidades que sumen 1 por fila.
    :return: None.
    :authors: Ivana Zurdo, Nicolas Barreto, Tomas Muino, Seba Sanchez Bentolila
    :date: 11/06/2026
    """
    x = np.array([[180, 1], [175, 1], [160, 0], [155, 0]])
    y = np.array(["hombre", "hombre", "mujer", "mujer"])
    model = NaiveBayes()
    model.fit(x, y)
    proba = model.predict_proba(x)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_priors_suman_uno() -> None:
    """
    Verifica que los priors P(C) sumen exactamente 1.
    :return: None.
    :authors: Ivana Zurdo, Nicolas Barreto, Tomas Muino, Seba Sanchez Bentolila
    :date: 11/06/2026
    """
    x = np.array([[180, 1], [175, 1], [160, 0], [155, 0]])
    y = np.array(["hombre", "hombre", "mujer", "mujer"])
    model = NaiveBayes()
    model.fit(x, y)
    assert np.isclose(sum(model.class_priors.values()), 1.0)
