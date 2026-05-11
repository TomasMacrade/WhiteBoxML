"""
Reporte de métricas.

Este módulo implementa la generación de reportes de métricas
tanto de clasificación como de regresión a través de una función
unificada ``summary``.

Para clasificación incluye:
- Accuracy
- Precision
- Recall
- F1 Score

Para regresión incluye:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Coeficiente de determinación (R^2)

:authors: Joaquín Palacio Feijóo
:date: 21/04/2026
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from whiteboxml import metricas
from whiteboxml.utils import _validacion_inputs


def _detect_task_type(y_true: np.ndarray) -> str:
    """
    Detecta automáticamente si el problema es de clasificación o regresión
    basándose en el tipo de datos de y_true.

    - Strings u objetos → clasificación
    - Booleanos → clasificación
    - Floats con decimales → regresión
    - Floats sin decimales / enteros → clasificación

    :param y_true: targets reales (ya convertidos a np.ndarray)
    :return: "classification" o "regression"
    :authors: Joaquín Palacio Feijóo
    :date: 21/04/2026
    """
    # Strings u objetos
    if y_true.dtype.kind in ("U", "O"):
        return "classification"

    # Booleanos
    if y_true.dtype.kind == "b":
        return "classification"

    # Floats
    if y_true.dtype.kind == "f":
        if np.all(np.isfinite(y_true)) and np.all(y_true == np.floor(y_true)):
            return "classification"
        return "regression"

    # Enteros u otros
    return "classification"


def _detect_average(y_true: np.ndarray) -> str:
    """
    Determina el tipo de average para métricas de clasificación
    según la cantidad de clases únicas en y_true.

    - 2 clases o menos → "binary"
    - Más de 2 clases → "macro"

    :param y_true: targets reales
    :return: "binary" o "macro"
    :authors: Joaquín Palacio Feijóo
    :date: 21/04/2026
    """
    n_classes = len(np.unique(y_true))
    if n_classes <= 2:
        return "binary"
    return "macro"


def _classification_summary(
    y_true: np.ndarray, y_pred: np.ndarray, average: str
) -> pd.DataFrame:
    """
    Generación de reporte de métricas de clasificación.

    Incluye:
    - Accuracy
    - Precision
    - Recall
    - F1 Score

    :param y_true: targets reales
    :param y_pred: targets predichos
    :param average: tipo de promedio para precision y recall
    :return: DataFrame con Accuracy, Precision, Recall y F1
    :authors: Joaquín Palacio Feijóo
    :date: 21/04/2026
    """
    accuracy = metricas.accuracy(y_true, y_pred)
    precision = metricas.precision(y_true, y_pred, average=average)
    recall = metricas.recall(y_true, y_pred, average=average)
    f1 = metricas.f1_score(y_true, y_pred, average=average)

    return pd.DataFrame(
        {
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall],
            "F1": [f1],
        }
    )


def _regression_summary(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Genera el reporte de métricas de regresión.

    :param y_true: targets reales
    :param y_pred: targets predichos
    :return: DataFrame con MSE, MAE y R²
    :authors: Joaquín Palacio Feijóo
    :date: 21/04/2026
    """
    mse = metricas.mean_squared_error(y_true, y_pred)
    mae = metricas.mean_absolute_error(y_true, y_pred)
    r2 = metricas.r2(y_true, y_pred)

    return pd.DataFrame(
        {
            "MSE": [mse],
            "MAE": [mae],
            "R^2": [r2],
        }
    )


def summary(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    mode: str = "auto",
) -> pd.DataFrame:
    """
    Generación unificada de reporte de métricas.

    Detecta automáticamente si el problema es de clasificación o regresión
    y genera el reporte correspondiente. En clasificación, determina si es
    binaria o multiclase para seleccionar el average adecuado.

    :param y_true: targets reales
    :param y_pred: targets predichos
    :param mode: tipo de problema ("auto", "classification", "regression").
        Si es "auto", se detecta automáticamente.
    :return: DataFrame con las métricas correspondientes
    :raises ValueError: si mode no es un valor válido
    :authors: Joaquín Palacio Feijóo
    :date: 21/04/2026
    """
    valid_modes = ("auto", "classification", "regression")
    if mode not in valid_modes:
        raise ValueError(
            f"El parámetro mode debe ser uno de {valid_modes}. " f'Se recibió: "{mode}"'
        )

    vector_true, vector_pred = _validacion_inputs(y_true, y_pred)

    if mode == "auto":
        mode = _detect_task_type(vector_true)

    if mode == "classification":
        average = _detect_average(vector_true)
        return _classification_summary(vector_true, vector_pred, average)

    return _regression_summary(vector_true, vector_pred)
