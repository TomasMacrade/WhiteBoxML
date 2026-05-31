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
    assert metricas.recall(y_true, y_pred, average="micro") == pytest.approx(expected)


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
    Test F1 score básico
    :authors: Cecilia Gómez
    :date: 24/04/2026
    """

    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]
    expected = 0.8
    assert metricas.f1_score(
        y_true, y_pred, average="binary", pos_label=1
    ) == pytest.approx(expected)


def test_f1_binary_no_pred_positives():
    """
    Test F1 sin positivos predichos
    :authors: Cecilia Gómez
    :date: 24/04/2026
    """

    y_true = [1, 1, 0]
    y_pred = [0, 0, 0]
    assert metricas.f1_score(y_true, y_pred, average="binary", pos_label=1) == 0.0


def test_f1_micro_equals_accuracy():
    """
    Test F1 micro vs accuracy
    :authors: Cecilia Gómez
    :date: 24/04/2026
    """

    y_true = [0, 1, 2, 2]
    y_pred = [0, 2, 2, 1]
    expected = float(np.mean(np.array(y_true) == np.array(y_pred)))
    assert metricas.f1_score(y_true, y_pred, average="micro") == pytest.approx(expected)


def test_f1_macro_and_weighted_and_none():
    """
    Test F1 con distintos average
    :authors: Cecilia Gómez
    :date: 24/04/2026
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


def test_auc_perfecto():
    """
    Test auc perfecto
    :authors: Emiliano David Santis
    :date: 4/05/2026


    """
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]

    res = metricas.auc_roc(y_true, scores, show_plot=False)

    assert res["AUC"] == pytest.approx(1.0)
    assert 0.2 < res["Mejor Umbral"] <= 0.8


def test_auc_aleatorio():
    """
    Test_auc_aleatorio
    :authors: Emiliano David Santis
    :date: 4/05/2026

    """
    y_true = [0, 1, 0, 1]
    scores = [0.5, 0.5, 0.5, 0.5]

    res = metricas.auc_roc(y_true, scores, show_plot=False)

    assert res["AUC"] == pytest.approx(0.5)


def test_error_dimensiones():
    """
    Test error de dimensión
    :authors: Emiliano David Santis
    :date: 4/05/2026

    """
    with pytest.raises(ValueError):
        metricas.auc_roc([0, 1], [0.5], show_plot=False)


def test_estructura_diccionario():
    """
    Test estructura del diccionario
    :authors: Emiliano David Santis
    :date: 4/05/2026

    """
    y_true = [0, 1]
    scores = [0.2, 0.8]
    res = metricas.auc_roc(y_true, scores, show_plot=False)

    llaves_esperadas = {"TVP", "TFP", "AUC", "Mejor Umbral"}
    assert llaves_esperadas.issubset(res.keys())


def test_mejor_umbral_logica():
    """
    Test mejor umbral
    :authors: Emiliano David Santis
    :date: 4/05/2026

    """
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.3, 0.6, 0.9]
    res = metricas.auc_roc(y_true, scores, show_plot=False)

    assert 0.0 <= res["Mejor Umbral"] <= 1.0


def test_auc_pr_perfecto():
    """
    Test auc pr con un clasificador perfecto

    :authors: Emiliano David Santis
    :date: 14/05/2026
    """
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]

    res = metricas.auc_pr(y_true, scores, show_plot=False)

    assert res["AUC_PR"] == pytest.approx(1.0)

    assert 0.2 < res["Mejor Umbral (F1)"] <= 0.8


def test_auc_pr_estructura_diccionario():
    """
    Test las llaves y tipos devueltos por el diccionario de salida

    :authors: Emiliano David Santis
    :date: 14/05/2026
    """
    y_true = [0, 1]
    scores = [0.3, 0.7]
    res = metricas.auc_pr(y_true, scores, show_plot=False)

    llaves_esperadas = {"Precision", "Recall", "AUC_PR", "Mejor Umbral (F1)"}

    assert llaves_esperadas.issubset(res.keys())

    assert isinstance(res["Precision"], np.ndarray)
    assert isinstance(res["Recall"], np.ndarray)
    assert isinstance(res["AUC_PR"], float)
    assert isinstance(res["Mejor Umbral (F1)"], float)


def test_auc_pr_sin_positivos():
    """Test de estabilidad cuando el dataset no contiene la clase positiva (1)

    Garantiza que la división por cero devuelva un comportamiento controlado.

    :authors: Emiliano David Santis
    :date: 14/05/2026
    """
    y_true = [0, 0, 0, 0]
    scores = [0.2, 0.4, 0.1, 0.5]

    res = metricas.auc_pr(y_true, scores, show_plot=False)

    assert res["AUC_PR"] == pytest.approx(0.0)
    assert np.all(res["Recall"] == 0.0)


def test_auc_pr_error_dimensiones():
    """Test que levanta ValueError si las dimensiones de inputs difieren

    :authors: Emiliano David Santis
    :date: 14/05/2026
    """
    y_true = [0, 1, 1]
    scores = [0.2, 0.8]  # Falta un score

    with pytest.raises(ValueError):
        metricas.auc_pr(y_true, scores, show_plot=False)


def test_auc_pr_thresholds_param():
    """Test que comprueba que el número de umbrales modifique el tamaño de los arrays

    :authors: Emiliano David Santis
    :date: 14/05/2026
    """
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.6, 0.9]
    N = 20

    res = metricas.auc_pr(y_true, scores, thresholds=N, show_plot=False)

    assert len(res["Precision"]) == N
    assert len(res["Recall"]) == N
