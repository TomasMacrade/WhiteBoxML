"""
Métricas de regresión.

Este módulo implementa métricas fundamentales para evaluar
modelos de regresión.

Las funciones aquí definidas operan sobre ArrayLike
y no dependen de librerías externas de machine learning.

Incluye:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Coeficiente de determinación (R^2)
"""

import numpy as np
from numpy.typing import ArrayLike

from WhiteBoxML.utils import _validacion_inputs


def mean_squared_error(
    y_true: ArrayLike,
    y_pred: ArrayLike
    ) -> float:
    """
    Cálculo del error cuadrático medio.

    :param y_true: targets reales
    :param y_pred: targets predichos
    :return: error cuadrático medio
    :authors: Tomás Macrade
    :date: 27/02/2026
    """

    vector_true,vector_pred = _validacion_inputs(y_true,y_pred)
    
    mse = np.mean((vector_true - vector_pred)**2)
    return float(mse)

def mean_absolute_error(
    y_true: ArrayLike,
    y_pred: ArrayLike
    ) -> float:
    """
    Cálculo del error absoluto medio.

    :param y_true: targets reales
    :param y_pred: targets predichos
    :return: error absoluto medio
    :authors: Tomás Macrade
    :date: 27/02/2026
    """

    vector_true,vector_pred = _validacion_inputs(y_true,y_pred)
    
    mae = np.mean(np.abs(vector_true - vector_pred))
    return float(mae)

def r2(
    y_true: ArrayLike,
    y_pred: ArrayLike
    ) -> float:
    """
    Cálculo del coeficiente de determinación.

    :param y_true: targets reales
    :param y_pred: targets predichos
    :return: coeficiente de determinación.
    :authors: Tomás Macrade
    :date: 27/02/2026
    """

    vector_true,vector_pred = _validacion_inputs(y_true,y_pred)
    residuals = vector_true - vector_pred
    mse = np.mean(residuals ** 2)
    var = np.var(vector_true)
    if var == 0:
        raise ValueError(
            "El coeficiente de determinación no está definido cuando la varianza de y_true es cero."
        )
    r2 = 1 - mse/var
    return float(r2)

    



