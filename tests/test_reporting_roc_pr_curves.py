"""
Tests del módulo ROC y PR.

:authors: Claudio Gabriel Alonso, Fernanda Alcaraz
:date: 21/04/2026
"""

import matplotlib
import numpy as np

from whiteboxml.reporting.visualizacion.roc_pr_curves import (
    calculate_auc,
    calculate_precision_recall_curve,
    calculate_roc_curve,
    plot_pr_curve,
    plot_roc_curve,
)

matplotlib.use("Agg")


def test_auc_perfect():
    """
    Test AUC perfecto (modelo ideal)
    :authors: Claudio Gabriel Alonso
    :date: 21/04/2026
    """

    fpr = [0.0, 0.0, 1.0]
    tpr = [0.0, 1.0, 1.0]

    auc = calculate_auc(fpr, tpr)

    assert auc == 1.0


def test_auc_random():
    """
    Test AUC modelo aleatorio
    :authors: Claudio Gabriel Alonso
    :date: 21/04/2026
    """

    fpr = [0.0, 1.0]
    tpr = [0.0, 1.0]

    auc = calculate_auc(fpr, tpr)

    assert auc == 0.5


def test_roc_curve_basic():
    """
    Test básico de la curva ROC
    :authors: Claudio Gabriel Alonso
    :date: 21/04/2026
    """

    y_true = [0, 0, 1, 1]
    y_pred_proba = [0.1, 0.4, 0.35, 0.8]

    fpr, tpr, thresholds = calculate_roc_curve(y_true, y_pred_proba)

    # Validaciones básicas
    assert len(fpr) == len(tpr) == len(thresholds)
    assert fpr[0] == 0.0
    assert tpr[0] == 0.0

    # Monotonía
    assert np.all(np.diff(fpr) >= 0)
    assert np.all(np.diff(tpr) >= 0)


def test_roc_curve_perfect():
    """
    Test curva ROC perfecta
    :authors: Claudio Gabriel Alonso
    :date: 21/04/2026
    """

    y_true = [0, 0, 1, 1]
    y_pred_proba = [0.1, 0.2, 0.8, 0.9]

    fpr, tpr, _ = calculate_roc_curve(y_true, y_pred_proba)
    auc = calculate_auc(fpr, tpr)

    assert auc == 1.0


def test_plot_roc_curve_runs():
    """
    Test que el plot de ROC se ejecuta sin errores
    :authors: Claudio Gabriel Alonso
    :date: 29/04/2026
    """

    y_true = [0, 1, 0, 1]
    y_pred_proba = [0.2, 0.8, 0.4, 0.6]

    fig, ax = plot_roc_curve(y_true, y_pred_proba, show=False)

    assert fig is not None
    assert ax is not None


def test_pr_curve_basic():
    """
    Test básico de la curva PR
    """

    y_true = [0, 0, 1, 1]
    y_pred_proba = [0.1, 0.4, 0.35, 0.8]

    recall, precision, thresholds = calculate_precision_recall_curve(
        y_true,
        y_pred_proba,
    )

    assert len(recall) == len(precision)
    assert len(recall) >= len(thresholds)


def test_plot_pr_curve_runs():
    """
    Test del plot de PR
    """

    y_true = [0, 1, 0, 1]
    y_pred_proba = [0.2, 0.8, 0.4, 0.6]

    fig, ax = plot_pr_curve(
        y_true,
        y_pred_proba,
        show=False,
    )

    assert fig is not None
    assert ax is not None
