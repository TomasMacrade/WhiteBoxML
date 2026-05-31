"""
Metricas: Conjunto de métricas útiles para clasificación y regresión 

:authors: Tomás Macrade
:date: 27/02/2026
"""

# pylint: disable=duplicate-code
from .clasificacion import accuracy, auc_pr, auc_roc, f1_score, precision, recall
from .regresion import mean_absolute_error, mean_squared_error, r2

__all__ = [
    "accuracy",
    "f1_score",
    "precision",
    "recall",
    "auc_roc",
    "auc_pr",
    "mean_absolute_error",
    "mean_squared_error",
    "r2",
]
