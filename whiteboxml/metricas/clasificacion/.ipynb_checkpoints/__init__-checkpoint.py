"""
Clasificación: Conjunto de métricas para clasificación

:authors: Tomás Macrade
:date: 27/02/2026
"""

from .clasificacion import accuracy, precision, recall, plot_roc_curve, auc_roc

__all__ = ["accuracy", "precision", "recall", "plot_roc_curve", "auc_roc"]
