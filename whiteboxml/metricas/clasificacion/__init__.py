"""
Clasificación: Conjunto de métricas para clasificación

:authors: Tomás Macrade
:date: 27/02/2026
"""

from .clasificacion import accuracy, auc_pr, auc_roc, f1_score, precision, recall

__all__ = [
    "accuracy",
    "f1_score",
    "precision",
    "recall",
    "auc_roc",
    "auc_pr",
]
