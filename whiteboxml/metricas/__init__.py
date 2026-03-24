"""
Metricas: Conjunto de métricas útiles para clasificación y regresión 

:authors: Tomás Macrade
:date: 27/02/2026
"""

from .clasificacion import accuracy, precision, recall
from .regresion import mean_absolute_error, mean_squared_error, r2

__all__ = [
    "accuracy",
    "precision",
    "recall",
    "mean_absolute_error",
    "mean_squared_error",
    "r2",
]
