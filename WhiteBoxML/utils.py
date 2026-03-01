"""
Funciones auxiliares para métricas.

Este módulo implementa utilidades internas que centralizan la
validación y transformación de inputs (por ejemplo, conversión
a arrays de NumPy y verificación de consistencia dimensional).

Estas funciones son utilizadas por las métricas públicas del
paquete y no forman parte de la API expuesta al usuario.
"""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike


def _validacion_inputs(
    array1: ArrayLike,
    array2: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Valida y prepara los inputs para métricas de regresión.

    :param array1: primer arreglo de elementos
    :param array2: segundo arreglo de elementos
    :return: tupla de arrays de numpy unidimensionales
    :authors: Tomás Macrade
    :date: 27/02/2026
    """
    try:
        array1 = np.asarray(array1)
        array2 = np.asarray(array2)
    except Exception as e:
        raise ValueError("Los inputs deben ser convertibles a arrays de NumPy.") from e

    for array in (array1, array2):
        if array.ndim == 2 and 1 in array.shape:
            array = array.reshape(-1)
        elif array.ndim != 1:
            raise ValueError(
                "Los inputs deben ser vectores 1D o matrices columna/fila."
            )

    if array1.shape != array2.shape:
        raise ValueError(
            "Los inputs deben tener la misma cantidad de elementos. "
            f"El primer array tiene {len(array1)} elementos "
            f"mientras que el segundo tiene {len(array2)}."
        )

    return array1, array2


def _validacion_average(average: str | None) -> str | None:
    """
    Check sobre los valores del parámetro average y
    normalización en caso de ser necesario.

    :param average: parámetro average
    :authors: Tomás Macrade
    :date: 29/02/2026
    """
    if average is None:
        return None

    if not isinstance(average, str):
        raise TypeError("average debe ser str o None.")

    average = average.strip().lower().replace(",", "")

    if average not in ("binary", "micro", "macro", "weighted", None):
        raise ValueError(
            'El parámetro average debe ser "binary", '
            '"micro", "macro", "weighted" o None.'
        )

    return average


def _compute_metric_components(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
    metrics: list[Literal["FP", "TP", "FN"]],
) -> np.ndarray:
    """
    Cálculo de las tasas de falsos positivos,
    falsos negativos y verdaderos positivos por clase.

    :param y_true: targets reales
    :param y_pred: targets predichos
    :param classes: array de clases a considerar
    :param metrics: lista de métricas a calcular
    :return: métricas calculadas en el órden que fueron requeridas en metrics por clase
    """

    classes = np.unique(np.concatenate((y_true, y_pred)))

    counts = []
    for cls in classes:
        metric_list = []
        for m in metrics:
            if m == "TP":
                counter = np.sum((y_pred == cls) & (y_true == cls))
            elif m == "FP":
                counter = np.sum((y_pred == cls) & (y_true != cls))
            elif m == "FN":
                counter = np.sum((y_pred != cls) & (y_true == cls))
            metric_list.append(counter)
        counts.append(metric_list)
    counts = np.array(counts)
    return counts
