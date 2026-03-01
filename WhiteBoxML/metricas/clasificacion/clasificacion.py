"""
Métricas de clasificación.

Este módulo implementa métricas fundamentales para evaluar
modelos de clasificación.

Las funciones aquí definidas operan sobre ArrayLike
y no dependen de librerías externas de machine learning.

Incluye:
- Accuracy
- Precision
- Recall
"""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from whiteboxml.utils import (
    _compute_metric_components,
    _validacion_average,
    _validacion_inputs,
)


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Cálculo del accuracy.

    :param y_true: targets reales
    :param y_pred: targets predichos
    :return: accuracy
    :authors: Tomás Macrade
    :date: 28/02/2026
    """

    vector_true, vector_pred = _validacion_inputs(y_true, y_pred)
    accuracy = np.mean(vector_true == vector_pred)
    return float(accuracy)


def precision(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str | None = "binary",
    pos_label: Any = 1,
) -> float | np.ndarray:
    """
    Cálculo de la precision.

    :param y_true: targets reales
    :param y_pred: targets predichos
    :param average: define el tipo de average en
    clasificación multiclase ("binary","micro", "macro", "weighted", None)
    :param pos_label: valor a considerar como positivo
    en el caso de targets binarios. Ignorado si average != "binary".
    :return: score de precision o array con la precision por clase
    en caso de average = None
    :authors: Tomás Macrade
    :date: 28/02/2026
    """

    average = _validacion_average(average)
    vector_true, vector_pred = _validacion_inputs(y_true, y_pred)

    classes = np.unique(np.concatenate((vector_true, vector_pred)))

    if average == "binary":
        tp = np.sum((vector_pred == pos_label) & (vector_true == pos_label))
        fp = np.sum((vector_pred == pos_label) & (vector_true != pos_label))
        return float(tp / (tp + fp)) if tp + fp > 0 else 0.0

    if average == "micro":
        tp = np.sum(vector_true == vector_pred)
        total_muestras = vector_true.size
        return float(tp / total_muestras) if total_muestras > 0 else 0.0

    if average == "macro":
        components = _compute_metric_components(
            vector_true, vector_pred, classes, ["TP", "FP"]
        )
        tp, fp = components[:, 0], components[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            per_class_precision = np.nan_to_num(tp / (tp + fp))
        return float(np.mean(per_class_precision))

    if average == "weighted":
        components = _compute_metric_components(
            vector_true, vector_pred, classes, ["TP", "FP"]
        )
        tp, fp = components[:, 0], components[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            per_class_precision = np.nan_to_num(tp / (tp + fp))

        supports = np.array([np.sum(vector_true == c) for c in classes])
        total_support = np.sum(supports)
        if total_support == 0:
            return 0.0
        weights = supports / total_support

        return float(np.sum(per_class_precision * weights))

    if average is None:
        components = _compute_metric_components(
            vector_true, vector_pred, classes, ["TP", "FP"]
        )
        tp, fp = components[:, 0], components[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            per_class_precision = np.nan_to_num(tp / (tp + fp))
        return per_class_precision

    else:
        raise ValueError("Invalid average")


def recall(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str | None = "micro",
    pos_label: Any = 1,
) -> float | np.ndarray:
    """
    Cálculo del recall.

    :param y_true: targets reales
    :param y_pred: targets predichos
    :param average: define el tipo de average en
    clasificación multiclase ("binary","micro", "macro", "weighted", None)
    :param pos_label: valor a considerar como positivo en el caso de targets binarios
    :return: score de recall o array con la recall por clase en caso de average = None
    :authors: Tomás Macrade
    :date: 28/02/2026
    """

    average = _validacion_average(average)
    vector_true, vector_pred = _validacion_inputs(y_true, y_pred)

    classes = np.unique(np.concatenate((vector_true, vector_pred)))

    if average == "binary":
        tp = np.sum((vector_pred == pos_label) & (vector_true == pos_label))
        fn = np.sum((vector_pred != pos_label) & (vector_true == pos_label))
        return float(tp / (tp + fn)) if tp + fn > 0 else 0.0

    if average == "micro":
        tp = np.sum(vector_true == vector_pred)
        total_muestras = vector_true.size
        return float(tp / total_muestras) if total_muestras > 0 else 0.0

    if average == "macro":
        components = _compute_metric_components(
            vector_true, vector_pred, classes, ["TP", "FN"]
        )
        tp, fn = components[:, 0], components[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            per_class_recall = np.nan_to_num(tp / (tp + fn))
        return float(np.mean(per_class_recall))

    if average == "weighted":
        components = _compute_metric_components(
            vector_true, vector_pred, classes, ["TP", "FN"]
        )
        tp, fn = components[:, 0], components[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            per_class_recall = np.nan_to_num(tp / (tp + fn))

        supports = np.array([np.sum(vector_true == c) for c in classes])
        total_support = np.sum(supports)
        if total_support == 0:
            return 0.0
        weights = supports / total_support

        return float(np.sum(per_class_recall * weights))

    if average is None:
        components = _compute_metric_components(
            vector_true, vector_pred, classes, ["TP", "FN"]
        )
        tp, fn = components[:, 0], components[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            per_class_recall = np.nan_to_num(tp / (tp + fn))
        return per_class_recall

    else:
        raise ValueError("Invalid average")
