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
- F1-Score

:authors: Tomás Macrade
:date: 28/02/2026
"""

from typing import Any, Dict, Union

import matplotlib.pyplot as plt
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

    classes = np.union1d(vector_true, vector_pred)

    if average == "binary":
        tp = np.sum((vector_pred == pos_label) & (vector_true == pos_label))
        fp = np.sum((vector_pred == pos_label) & (vector_true != pos_label))
        return float(tp / (tp + fp)) if tp + fp > 0 else 0.0

    if average == "micro":
        tp = np.sum(vector_true == vector_pred)
        total_muestras = vector_true.size
        return float(tp / total_muestras) if total_muestras > 0 else 0.0

    components = _compute_metric_components(
        vector_true, vector_pred, classes, ["TP", "FP"]
    )

    tp, fp = components[:, 0], components[:, 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_precision = np.nan_to_num(tp / (tp + fp))

    if average == "macro":
        return float(np.mean(per_class_precision))

    if average == "weighted":
        supports = np.array([np.sum(vector_true == c) for c in classes])
        total_support = np.sum(supports)
        weights = supports / total_support

        return float(np.sum(per_class_precision * weights))

    return per_class_precision


def recall(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str | None = "binary",
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

    classes = np.union1d(vector_true, vector_pred)

    if average == "binary":
        tp = np.sum((vector_pred == pos_label) & (vector_true == pos_label))
        fn = np.sum((vector_pred != pos_label) & (vector_true == pos_label))
        return float(tp / (tp + fn)) if tp + fn > 0 else 0.0

    if average == "micro":
        tp = np.sum(vector_true == vector_pred)
        total_muestras = vector_true.size
        return float(tp / total_muestras) if total_muestras > 0 else 0.0

    components = _compute_metric_components(
        vector_true, vector_pred, classes, ["TP", "FN"]
    )

    tp, fn = components[:, 0], components[:, 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_recall = np.nan_to_num(tp / (tp + fn))

    if average == "macro":
        return float(np.mean(per_class_recall))

    if average == "weighted":
        supports = np.array([np.sum(vector_true == c) for c in classes])
        total_support = np.sum(supports)
        weights = supports / total_support

        return float(np.sum(per_class_recall * weights))

    return per_class_recall


# pylint: disable=too-many-locals
def f1_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str | None = "binary",
    pos_label: Any = 1,
) -> float | np.ndarray:
    """
    Cálculo del F1 Score.
    El F1 Score es la media armónica entre precision y recall.

    F1 = 2 * (precision * recall) / (precision + recall)

    :param y_true: valores reales (ground truth)
    :param y_pred: valores predichos por el modelo
    :param average: tipo de promedio en multiclase
    :param pos_label: clase positiva (solo para binary)
    :return: valor de F1 o vector por clase
    :authors: Cecilia Gómez
    :date: 24/04/2026
    """
    average = _validacion_average(average)

    vector_true, vector_pred = _validacion_inputs(y_true, y_pred)

    classes = np.union1d(vector_true, vector_pred)

    if average == "binary":

        tp = np.sum((vector_pred == pos_label) & (vector_true == pos_label))
        fp = np.sum((vector_pred == pos_label) & (vector_true != pos_label))
        fn = np.sum((vector_pred != pos_label) & (vector_true == pos_label))
        precision_score = float(tp / (tp + fp)) if tp + fp > 0 else 0.0
        recall_score = float(tp / (tp + fn)) if tp + fn > 0 else 0.0
        denominator = precision_score + recall_score

        return (
            2 * precision_score * recall_score / denominator if denominator > 0 else 0.0
        )

    if average == "micro":
        return accuracy(vector_true, vector_pred)

    components = _compute_metric_components(
        vector_true, vector_pred, classes, ["TP", "FP", "FN"]
    )

    tp, fp, fn = components[:, 0], components[:, 1], components[:, 2]

    # Calculamos precision y recall por clase
    with np.errstate(divide="ignore", invalid="ignore"):

        per_class_precision = np.nan_to_num(tp / (tp + fp))
        per_class_recall = np.nan_to_num(tp / (tp + fn))

        # F1 por clase
        per_class_f1 = np.nan_to_num(
            2
            * per_class_precision
            * per_class_recall
            / (per_class_precision + per_class_recall)
        )

    if average == "macro":
        # Promedio simple entre clases
        return float(np.mean(per_class_f1))

    if average == "weighted":

        # Cantidad de ejemplos por clase (soporte)
        supports = np.array([np.sum(vector_true == c) for c in classes])

        # Total de ejemplos
        total_support = np.sum(supports)

        # Pesos relativos de cada clase
        weights = supports / total_support

        # Promedio ponderado
        return float(np.sum(per_class_f1 * weights))

    return per_class_f1


def auc_roc(
    y_true: ArrayLike, scores: ArrayLike, thresholds: int = 50, show_plot: bool = True
) -> Dict[str, Union[np.ndarray, float]]:
    """

    Cálculo del AUC ROC y optimización del umbral

    Calcula la curva ROC barriendo una serie de umbrales
    y determina el mejor punto de corte.
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

    n_positivos = int(np.sum(y_true == 1))
    n_negativos = int(np.sum(y_true == 0))

    umbrales_raw = np.linspace(np.max(scores) + 1e-9, np.min(scores) - 1e-9, thresholds)
    predicciones = scores >= umbrales_raw.reshape(-1, 1)

    tvp_raw = np.sum(predicciones[:, y_true == 1], axis=1) / n_positivos
    tfp_raw = np.sum(predicciones[:, y_true == 0], axis=1) / n_negativos

    indices_orden = np.lexsort((tvp_raw, tfp_raw))
    tfp = tfp_raw[indices_orden]
    tvp = tvp_raw[indices_orden]
    umbrales = umbrales_raw[indices_orden]

    tvp = np.maximum.accumulate(tvp)

    try:
        auc = float(np.abs(np.trapezoid(tvp, tfp)))
    except AttributeError:
        auc = float(np.abs(np.trapz(tvp, tfp)))

    distancia = np.sqrt((1.0 - tvp) ** 2 + (tfp) ** 2)

    idx = np.argmin(distancia)

    mejor_umbral = float(umbrales[idx].item())

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
    if auc >= 0.9:
        color_curva = "#2ca02c"
    elif auc >= 0.7:
        color_curva = "#0e82ff"
    else:
        color_curva = "#d62728"

    _, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    ax.plot(tfp, tvp, label=f"AUC = {auc:.4f}", color=color_curva, lw=2.5)
    ax.plot(
        [0, 1], [0, 1], color="black", linestyle="--", alpha=0.3, label="Azar (0.5)"
    )

    es_relevante = 0.55 <= auc <= 0.995

    if es_relevante:
        ax.scatter(
            tfp[idx_optimo],
            tvp[idx_optimo],
            color="gold",
            s=120,
            edgecolor="black",
            zorder=5,
            label=f"Umbral Óptimo: {mejor_umbral:.2f}",
        )
    else:
        print(f"Punto óptimo omitido (AUC {auc:.2f} fuera de rango de relevancia)")

    ax.set_xlabel("Tasa de falsos positivos (TFP)")
    ax.set_ylabel("Tasa de verdaderos positivos (TVP)")
    ax.set_title("Curva AUC ROC", fontweight="bold")

    ax.fill_between(tfp, tvp, alpha=0.15, color=color_curva)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def auc_pr(
    y_true: ArrayLike,
    scores: ArrayLike,
    thresholds: int = 50,
    show_plot: bool = True,
) -> Dict[str, Union[np.ndarray, float]]:
    """Cálculo del AUC PR y optimización del umbral por F1-Score

    Calcula la curva Precision-Recall barriendo una serie de umbrales y determina
    el mejor punto de corte basado en la maximización del F1-Score.
    También genera una visualización adaptativa de la curva según su rendimiento.


    :param y_true: targets de clase (0 o 1).
    :type y_true: ArrayLike
    :param scores: probabilidad o scores asignados por el modelo.
    :type scores: ArrayLike
    :param thresholds: número de umbrales a evaluar, por defecto 50.
    :type thresholds: int
    :param show_plot: Generar visualización de la AUC PR.
    :type show_plot: bool
    :return: Diccionario con Precision, Recall (arrays),
    valor AUC_PR y Mejor Umbral (F1) (floats).
    :rtype: Dict[str, Union[np.ndarray, float]]
    :authors: Emiliano David Santis
    :date: 14/05/2026
    """
    _validacion_inputs(y_true, scores)

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    n_positivos = int(np.sum(y_true == 1))
    prevalencia = n_positivos / len(y_true) if len(y_true) > 0 else 0

    umbrales_raw = np.linspace(np.max(scores) + 1e-9, np.min(scores) - 1e-9, thresholds)
    predicciones = scores >= umbrales_raw.reshape(-1, 1)

    vps = np.sum(predicciones[:, y_true == 1], axis=1)
    fps = np.sum(predicciones[:, y_true == 0], axis=1)

    rec_vals = vps / n_positivos if n_positivos > 0 else np.zeros(thresholds)

    denominador_prec = vps + fps
    prec_vals = np.where(denominador_prec > 0, vps / denominador_prec, 0.0)

    indices_orden = np.argsort(rec_vals)
    rec_sorted = rec_vals[indices_orden]
    prec_sorted = prec_vals[indices_orden]

    if n_positivos > 0:
        auc_score = float(np.sum(prec_sorted[1:] * np.diff(rec_sorted)))
    else:
        auc_score = 0.0

    f1_scores = np.where(
        (prec_vals + rec_vals) > 0,
        (2 * prec_vals * rec_vals) / (prec_vals + rec_vals),
        0,
    )
    idx_f1 = np.argmax(f1_scores)
    mejor_umbral = float(umbrales_raw[idx_f1].item())

    if show_plot:
        plot_pr_curve(
            recall=rec_sorted,
            precision=prec_sorted,
            auc_pr=auc_score,
            mejor_umbral=mejor_umbral,
            idx_f1=idx_f1,
            prevalencia=prevalencia,
            rec_original=rec_vals,
            prec_original=prec_vals,
        )

    return {
        "Precision": prec_sorted,
        "Recall": rec_sorted,
        "AUC_PR": auc_score,
        "Mejor Umbral (F1)": mejor_umbral,
    }


def plot_pr_curve(
    recall: np.ndarray,
    precision: np.ndarray,
    auc_pr: float,
    mejor_umbral: float,
    idx_f1: int,
    prevalencia: float,
    rec_original: np.ndarray,
    prec_original: np.ndarray,
) -> None:
    """Genera la visualización de la curva Precision-Recall con escala fija (0-1).
    Incorpora un sistema de color adaptativo basado en la mejora sobre la prevalencia.


    :param recall: Array ordenado con los valores de recall.
    :type recall: np.ndarray
    :param precision: Array ordenado con los valores de precisión.
    :type precision: np.ndarray
    :param auc_pr: Valor del área bajo la curva PR calculada.
    :type auc_pr: float
    :param mejor_umbral: Valor del umbral óptimo identificado mediante F1.
    :type mejor_umbral: float
    :param idx_f1: Índice del punto óptimo en los arrays originales.
    :type idx_f1: int
    :param prevalencia: Proporción de casos positivos reales en el dataset.
    :type prevalencia: float
    :param rec_original: Array de recall sin ordenar mapeado a los umbrales crudos.
    :type rec_original: np.ndarray
    :param prec_original: Array de precisión sin ordenar mapeado a los umbrales crudos.
    :type prec_original: np.ndarray
    :return: None
    :rtype: None
    :authors: Emiliano David Santis
    :date: 14/05/2026
    """

    rango_mejora = 1.0 - prevalencia
    if auc_pr >= prevalencia + (rango_mejora * 0.75):
        color_curva = "#2ca02c"  # Verde: Excelente
    elif auc_pr > prevalencia + (rango_mejora * 0.25):
        color_curva = "#0e82ff"  # Azul: Aceptable
    else:
        color_curva = "#d62728"  # Rojo: Pobre

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        recall,
        precision,
        label=f"AUC-PR = {auc_pr:.4f}",
        color=color_curva,
        lw=2.5,
        zorder=3,
    )
    ax.fill_between(recall, precision, alpha=0.15, color=color_curva, zorder=2)

    ax.axhline(
        y=prevalencia,
        color="#555555",
        linestyle="--",
        alpha=0.6,
        label=f"Azar/Prevalencia ({prevalencia:.2f})",
        zorder=1,
    )

    ax.scatter(
        rec_original[idx_f1],
        prec_original[idx_f1],
        color="gold",
        s=130,
        edgecolor="black",
        linewidth=1.5,
        zorder=5,
        label=f"Punto Óptimo F1\n(Umbral: {mejor_umbral:.2f})",
    )

    ax.set_xlabel("Recall (Exhaustividad)", fontsize=10)
    ax.set_ylabel("Precisión", fontsize=10)
    ax.set_title("Curva Precision-Recall", fontweight="bold", fontsize=12, pad=15)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])

    ax.legend(loc="lower left", frameon=True, shadow=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    plt.show()
