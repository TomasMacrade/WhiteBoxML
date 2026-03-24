"""
Tests de funciones utils
"""

import numpy as np
import pytest

from whiteboxml.utils import (
    _compute_metric_components,
    _validacion_average,
    _validacion_inputs,
)


def test_conversion_exitosa():
    """
    Test trivial inputs
    :authors: Tomás Macrade
    :date: 23/03/2026
    """
    a1, _ = _validacion_inputs([1, 2, 3], [4, 5, 6])
    assert isinstance(a1, np.ndarray)
    assert a1.shape == (3,)
    assert np.array_equal(a1, [1, 2, 3])

def test_exception_conversion_array():
    """
    Test exception en conversión array
    :authors: Tomás Macrade
    :date: 24/03/2026
    """
    class ObjetoRoto:
        def __len__(self):
            return 10
        def __getitem__(self):
            raise RuntimeError("Fallo forzado")

    with pytest.raises(ValueError, match="Los inputs deben ser convertibles a arrays de NumPy."):
        _validacion_inputs(ObjetoRoto(), [1, 2, 3])


def test_reshape_columna_fila():
    """
    Test reshape inputs
    :authors: Tomás Macrade
    :date: 23/03/2026
    """
    a1 = np.array([[1], [2], [3]])
    a2 = np.array([[4], [5], [6]])
    r1, _ = _validacion_inputs(a1, a2)
    assert r1.ndim == 1


def test_error_dimensiones_invalidas_array_1():
    """
    Test dimensiones inválidas en array 1
    :authors: Tomás Macrade
    :date: 23/03/2026
    """
    with pytest.raises(ValueError, match="Los inputs deben ser vectores 1D"):
        _validacion_inputs(np.ones((2, 2)), [1, 2])

def test_error_dimensiones_invalidas_array_2():
    """
    Test dimensiones inválidas en array 2
    :authors: Tomás Macrade
    :date: 23/03/2026
    """
    with pytest.raises(ValueError, match="Los inputs deben ser vectores 1D"):
        _validacion_inputs([1, 2],np.ones((2, 2)) )


def test_error_distinto_largo():
    """
    Test error de conversión por elementos de distinto largo
    :authors: Tomás Macrade
    :date: 23/03/2026
    """
    with pytest.raises(ValueError, match="misma cantidad de elementos"):
        _validacion_inputs([1, 2], [1, 2, 3])


@pytest.mark.parametrize(
    "input_val, expected",
    [
        ("Binary", "binary"),
        (" MACRO ", "macro"),
        ("Weighted,", "weighted"),
        (None, None),
    ],
)
def test_normalizacion_exitosa(input_val, expected):
    """
    Test de limpieza y normalización del parámetro average.
    :authors: Tomás Macrade
    :date: 23/03/2026
    """
    assert _validacion_average(input_val) == expected


def test_error_tipo_invalido():
    """
    Test de error ante tipos distintos a string o None.
    :authors: Tomás Macrade
    :date: 23/03/2026
    """
    with pytest.raises(TypeError, match="average debe ser str o None"):
        _validacion_average(123)


def test_error_valor_no_permitido():
    """
    Test de error ante strings no reconocidos.
    :authors: Tomás Macrade
    :date: 23/03/2026
    """
    with pytest.raises(ValueError, match="El parámetro average debe ser"):
        _validacion_average("invalid_mode")


def test_calculo_basico_tp_fp_fn():
    """
    Test de cálculo de métricas por clase en escenario controlado.
    :authors: Tomás Macrade
    :date: 23/03/2026
    """
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    classes = np.array([0, 1])

    res = _compute_metric_components(y_true, y_pred, classes, ["TP", "FP", "FN"])

    assert np.array_equal(res[0], [1, 1, 1])
    assert np.array_equal(res[1], [1, 1, 1])
