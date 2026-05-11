"""
Módulo para cálculo y visualización de curvas ROC.

:authors: Claudio Gabriel Alonso
:date: 21/04/2026
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def calculate_auc(fpr: ArrayLike, tpr: ArrayLike) -> float:
    """
    Calcula el área bajo la curva ROC (AUC) usando integración numérica.

    :param fpr: False Positive Rate.
    :param tpr: True Positive Rate.
    :return: Valor del AUC.

    :authors: Claudio Gabriel Alonso, Fernanda Alcaraz
    :date: 20/04/2026
    """

    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)

    return np.trapz(tpr, fpr)


def calculate_roc_curve(
    y_true: ArrayLike, y_pred_proba: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula la curva ROC a partir de etiquetas reales y probabilidades predichas.

    :param y_true: Array de etiquetas reales (0 o 1).
    :param y_pred_proba: Array de probabilidades estimadas para la clase positiva.
    :return: Tupla (fpr, tpr, thresholds).

    :authors: Claudio Gabriel Alonso
    :date: 20/04/2026
    """

    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = y_true[sorted_indices]
    thresholds = y_pred_proba[sorted_indices]

    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)

    n_pos = np.sum(y_true)
    n_neg = np.sum(1 - y_true)

    tpr = tp / n_pos if n_pos > 0 else np.zeros_like(tp)
    fpr = fp / n_neg if n_neg > 0 else np.zeros_like(fp)

    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))
    thresholds = np.concatenate(([1.0], thresholds))

    return fpr, tpr, thresholds


def plot_roc_curve(
    y_true: ArrayLike,
    y_pred_proba: ArrayLike,
    show_diagonal: bool = True,
    show: bool = True,
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Calcula y grafica la curva ROC a partir de los datos de entrada.

    :param y_true: Array de etiquetas reales (0 o 1).
    :param y_pred_proba: Array de probabilidades estimadas.
    :param show_diagonal: Si True, muestra la línea base.
    :param show: Si True, muestra el gráfico. Si False, devuelve la figura.
    :return: fig, ax si show=False, caso contrario None.

    :authors: Claudio Gabriel Alonso
    :date: 29/04/2026
    """

    fpr, tpr, _ = calculate_roc_curve(y_true, y_pred_proba)
    auc = calculate_auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")

    if show_diagonal:
        ax.plot([0, 1], [0, 1], linestyle="--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid()

    if show:
        plt.show()
        return None
    return fig, ax


def calculate_precision_recall_curve(y_true, y_pred_proba):
    """
    Calcular la curva de Precision-Recall.

    :param y_true: Array de etiquetas reales (0 o 1).
    :param y_pred_proba: Array de probabilidades estimadas.
    :return: Tupla (recall, precision, thresholds).

    :authors: Fernanda Alcaraz
    :date: 09/05/2026
    """

    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    thresholds = sorted(
        set(y_pred_proba),
        reverse=True,
    )

    precision = [1.0]
    recall = [0.0]

    positives = sum(y_true)

    for threshold in thresholds:
        y_pred = [1 if p >= threshold else 0 for p in y_pred_proba]

        tp = sum(
            yt == 1 and yp == 1
            for yt, yp in zip(
                y_true,
                y_pred,
            )
        )

        fp = sum(
            yt == 0 and yp == 1
            for yt, yp in zip(
                y_true,
                y_pred,
            )
        )

        if tp + fp > 0:
            prec = tp / (tp + fp)
        else:
            prec = 1.0

        rec = tp / positives

        precision.append(prec)
        recall.append(rec)

    return recall, precision, thresholds


def plot_pr_curve(
    y_true: ArrayLike,
    y_pred_proba: ArrayLike,
    show: bool = True,
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Calcula y grafica la curva PR a partir de los datos de entrada.

    :param y_true: Array de etiquetas reales (0 o 1).
    :param y_pred_proba: Array de probabilidades estimadas.
    :param show: Si True, muestra el gráfico. Si False, devuelve la figura.
    :return: fig, ax si show=False, caso contrario None.

    :authors: Fernanda Alcaraz
    :date: 09/05/2026
    """

    recall, precision, _ = calculate_precision_recall_curve(y_true, y_pred_proba)

    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.grid()

    if show:
        plt.show()
        return None
    return fig, ax
