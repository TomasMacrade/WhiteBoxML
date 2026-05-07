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

:authors: Tomás Macrade
:date: 28/02/2026
"""

from typing import Any, Dict, Union

import numpy as np
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt


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
    return float(np.mean(vector_true == vector_pred))


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

    classes = np.unique(vector_true)

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
        weights = supports / total_support

        return float(np.sum(per_class_precision * weights))

    components = _compute_metric_components(
        vector_true, vector_pred, classes, ["TP", "FP"]
    )
    tp, fp = components[:, 0], components[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_precision = np.nan_to_num(tp / (tp + fp))
    return per_class_precision


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

    classes = np.unique(vector_true)

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
        weights = supports / total_support

        return float(np.sum(per_class_recall * weights))

    components = _compute_metric_components(
        vector_true, vector_pred, classes, ["TP", "FN"]
    )
    tp, fn = components[:, 0], components[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_recall = np.nan_to_num(tp / (tp + fn))
    return per_class_recall


def auc_roc(
    y_true: ArrayLike, scores: ArrayLike, thresholds: int = 50, show_plot: bool = True
) -> Dict[str, Union[np.ndarray, float]]:
    """

    Cálculo del AUC ROC y optimización del umbral

    Calcula la curva ROC barriendo una serie de umbrales y determina el mejor punto de corte.
    También genera una visualización de la curva.


    :param y_true: targets de clase (0 o 1).
    :type y_true: ArrayLike
    :param scores: probabilidad de scores o probabilidades asignados por el modelo.
    :type scores: ArrayLike
    :param thresholds: número de umbrales a evaluar, por defecto 50.
    :type thresholds: int
    :param show_plot: Generar visualización de la AUC ROC.
    :type: bool
    :return: Diccionario con TVP, TFP (arrays), valor AUC y mejor_umbral (floats).
    :rtype: Dict[str, Union[np.ndarray, float]]
    :authors: Emiliano David Santis
    :date: 4/05/2026

    """
    _validacion_inputs(y_true, scores)

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    positivos: np.ndarray = y_true == 1
    negativos: np.ndarray = y_true == 0

    n_positivos: int = int(np.sum(positivos))
    n_negativos: int = int(np.sum(negativos))

    umbrales = np.linspace(1, 0, thresholds).reshape(-1, 1)

    predicciones = scores >= umbrales

    vps = np.sum(predicciones[:, positivos], axis=1)
    fps = np.sum(predicciones[:, negativos], axis=1)

    tvp = vps / n_positivos if n_positivos > 0 else np.zeros(thresholds)
    tfp = fps / n_negativos if n_negativos > 0 else np.zeros(thresholds)

    distancia = np.sqrt((1.0 - tvp) ** 2 + (tfp) ** 2)

    idx = np.argmin(distancia)

    mejor_umbral = float(umbrales[idx].item())

    try:
        auc = float(np.abs(np.trapezoid(tvp, tfp)))
    except ImportError:
        auc = float(np.abs(np.trapz(tvp, tfp)))

    if show_plot:
        plot_roc_curve(tfp, tvp, auc, mejor_umbral, idx)

    return {"TVP": tvp, "TFP": tfp, "AUC": auc, "Mejor Umbral": mejor_umbral}


def plot_roc_curve(
    tfp: np.ndarray, tvp: np.ndarray, auc: float, mejor_umbral: float, idx_optimo: int
) -> None:
    """

    Genera la visualización de la curva AUC ROC.

    :param tfp: Array con la tasa de falsos positivos.
    :type tfp: np.ndarray
    :param tvp: Array con la tasa de verdaderos positivos.
    :type tvp: np.ndarray
    :param auc: Valor del área bajo la curva calculado.
    :type auc: float
    :param mejor_umbral: Valor del umbral óptimo identificado.
    :type mejor_umbral: float
    :param idx_optimo: Índice del punto óptimo en los arrays.
    :type idx_optimo: int
    :return: None
    :rtype: None
    :authors: Emiliano David Santis
    :date: 4/05/2026


    """
    _, ax = plt.subplots(figsize=(6, 6))

    ax.plot(tfp, tvp, label=f"AUC = {auc:.4f}", color="tab:blue", lw=2)
    ax.plot([0, 1], [0, 1], "r--", label="Azar (AUC = 0.5)")

    ax.set_xlabel("Tasa de falsos positivos (TFP)")
    ax.set_ylabel("Tasa de verdaderos positivos (TVP)")
    ax.set_title("Curva AUC ROC")

    ax.fill_between(tfp, tvp, alpha=0.3, color="tab:blue")
    ax.scatter(
        tfp[idx_optimo],
        tvp[idx_optimo],
        color="green",
        s=66,
        zorder=5,
        label=f"Umbral Óptimo: {mejor_umbral:.2f}",
    )

    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.show()
