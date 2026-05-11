"""
Tests del módulo de métricas
"""

import numpy as np
import pytest

from whiteboxml import metricas


def test_accuracy_perfect():
    """
    Test Accuracy perfecto
    :authors: Tomás Macrade
    :date: 28/02/2026
    """

    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 1, 1]
    assert metricas.accuracy(y_true, y_pred) == 1.0


def test_accuracy_partial():
    """
    Test Accuracy parcial
    :authors: Tomás Macrade
    :date: 28/02/2026
    """

    y_true = [1, 0, 1, 1]
    y_pred = [1, 1, 0, 1]
    expected = 0.5
    assert metricas.accuracy(y_true, y_pred) == pytest.approx(float(expected))


def test_precision_binary_basic():
    """
    Test Precision perfecto
    :authors: Tomás Macrade
    :date: 28/02/2026
    """

    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]
    assert metricas.precision(y_true, y_pred, average="binary", pos_label=1) == 1.0


def test_precision_binary_no_pred_positives():
    """
    Test Precision sin TP
    :authors: Tomás Macrade
    :date: 28/02/2026
    """

    y_true = [1, 1, 0]
    y_pred = [0, 0, 0]
    assert metricas.precision(y_true, y_pred, average="binary", pos_label=1) == 0.0


def test_precision_micro_equals_accuracy():
    """
    Test Precision micro vs accuracy
    :authors: Tomás Macrade
    :date: 28/02/2026
    """

    y_true = [0, 1, 2, 2]
    y_pred = [0, 2, 2, 1]
    expected = float(np.mean(np.array(y_true) == np.array(y_pred)))
    assert metricas.precision(y_true, y_pred, average="micro") == pytest.approx(
        expected
    )


def test_precision_macro_and_weighted_and_none():
    """
    Test Precision con distintos average
    :authors: Tomás Macrade
    :date: 28/02/2026
    """

    y_true = [0, 1, 2, 0]
    y_pred = [0, 2, 1, 0]
    per_class = np.array([1.0, 0.0, 0.0])

    assert metricas.precision(y_true, y_pred, average="macro") == pytest.approx(
        np.mean(per_class)
    )
    assert metricas.precision(y_true, y_pred, average="weighted") == pytest.approx(0.5)
    arr = metricas.precision(y_true, y_pred, average=None)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)
    assert np.allclose(arr, per_class)


def test_recall_micro_equals_accuracy_default():
    """
    Test recall micro vs accuracy
    :authors: Tomás Macrade
    :date: 28/02/2026
    """

    y_true = [0, 1, 1, 2]
    y_pred = [0, 1, 0, 2]
    expected = float(np.mean(np.array(y_true) == np.array(y_pred)))
    assert metricas.recall(y_true, y_pred) == pytest.approx(expected)


def test_recall_binary_basic():
    """
    Test recall básico
    :authors: Tomás Macrade
    :date: 28/02/2026
    """

    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]
    assert metricas.recall(
        y_true, y_pred, average="binary", pos_label=1
    ) == pytest.approx(2 / 3)


def test_recall_macro_and_weighted_and_none():
    """
    Test recall con distintos average
    :authors: Tomás Macrade
    :date: 28/02/2026
    """

    y_true = [0, 1, 2, 0]
    y_pred = [0, 2, 1, 0]
    per_class = np.array([1.0, 0.0, 0.0])

    assert metricas.recall(y_true, y_pred, average="macro") == pytest.approx(
        np.mean(per_class)
    )
    assert metricas.recall(y_true, y_pred, average="weighted") == pytest.approx(0.5)
    arr = metricas.recall(y_true, y_pred, average=None)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)
    assert np.allclose(arr, per_class)


def test_f1_binary_basic():
    """
    Test F1 básico en binario
    :authors: Nahuel Nicolas Alvarez
    :date: 21/04/2026
    """

    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]
    assert metricas.f1_score(
        y_true, y_pred, average="binary", pos_label=1
    ) == pytest.approx(0.8)


def test_f1_binary_no_tp():
    """
    Test F1 sin verdaderos positivos
    :authors: Nahuel Nicolas Alvarez
    :date: 21/04/2026
    """

    y_true = [1, 1, 1]
    y_pred = [0, 0, 0]
    assert metricas.f1_score(y_true, y_pred, average="binary", pos_label=1) == 0.0


def test_f1_micro_equals_accuracy():
    """
    Test F1 micro vs accuracy
    :authors: Nahuel Nicolas Alvarez
    :date: 21/04/2026
    """

    y_true = [0, 1, 2, 2]
    y_pred = [0, 2, 2, 1]
    expected = float(np.mean(np.array(y_true) == np.array(y_pred)))
    assert metricas.f1_score(y_true, y_pred, average="micro") == pytest.approx(expected)


def test_f1_macro_weighted_and_none():
    """
    Test F1 con distintos average
    :authors: Nahuel Nicolas Alvarez
    :date: 21/04/2026
    """

    y_true = [0, 1, 2, 0]
    y_pred = [0, 2, 1, 0]
    per_class = np.array([1.0, 0.0, 0.0])

    assert metricas.f1_score(y_true, y_pred, average="macro") == pytest.approx(
        np.mean(per_class)
    )
    assert metricas.f1_score(y_true, y_pred, average="weighted") == pytest.approx(0.5)
    arr = metricas.f1_score(y_true, y_pred, average=None)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)
    assert np.allclose(arr, per_class)


def test_mean_squared_error():
    """
    Test MSE
    :authors: Tomás Macrade
    :date: 28/02/2026
    """
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 4.0]
    assert metricas.mean_squared_error(y_true, y_pred) == pytest.approx(1 / 3)


def test_mean_absolute_error():
    """
    Test MAE
    :authors: Tomás Macrade
    :date: 28/02/2026
    """
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 4.0]
    assert metricas.mean_absolute_error(y_true, y_pred) == pytest.approx(1 / 3)


def test_r2_basic():
    """
    Test r2
    :authors: Tomás Macrade
    :date: 28/02/2026
    """
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 4.0]
    assert metricas.r2(y_true, y_pred) == pytest.approx(0.5)


def test_r2_raises_on_zero_variance():
    """
    Test r2 con varianza 0
    :authors: Tomás Macrade
    :date: 28/02/2026
    """
    y_true = [1.0, 1.0, 1.0]
    y_pred = [1.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        metricas.r2(y_true, y_pred)
