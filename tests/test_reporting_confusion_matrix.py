"""
Tests de matriz de confusión para el módulo de reporting.

:authors: Nahuel Nicolas Alvarez
:date: 19/04/2026
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from whiteboxml.reporting.visualizacion.confusion_matrix import (
    confusion_matrix,
    plot_confusion_matrix,
)

matplotlib.use("Agg")


def test_confusion_matrix_binaria_basica():
    """
    Test matriz de confusión binaria básica.
    :authors: Alvarez Nahuel Nicolas
    :date: 19/04/2026
    """
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]

    expected = np.array([[1, 1], [1, 1]])

    result = confusion_matrix(y_true, y_pred)

    assert np.array_equal(result, expected)


def test_plot_confusion_matrix_con_metricas_muestra_tabla():
    """
    Testea que el plot con métricas muestre panel lateral con tabla y accuracy.

    :authors: Alvarez Nahuel Nicolas
    :date: 29/04/2026
    """
    cm = np.array([[5, 1, 0], [1, 4, 1], [0, 1, 6]])
    labels = [0, 1, 2]

    fig, _ = plot_confusion_matrix(cm, labels=labels, show_class_metrics=True)

    assert len(fig.axes) == 3

    metrics_axes = [axis for axis in fig.axes if len(axis.tables) > 0]
    assert len(metrics_axes) == 1

    metrics_ax = metrics_axes[0]
    assert "Accuracy:" in metrics_ax.get_title()

    table = metrics_ax.tables[0]
    assert table[(0, 0)].get_text().get_text() == "Clase"
    assert table[(0, 0)].get_facecolor() == to_rgba("#1e3a8a")
    assert table[(0, 0)].get_text().get_color() == "white"

    assert table[(1, 0)].get_facecolor() == to_rgba("#dbeafe")
    assert table[(2, 0)].get_facecolor() == to_rgba("#eff6ff")

    plt.close(fig)


def test_plot_confusion_matrix_sin_metricas_sin_tabla():
    """
    Testea que el plot sin métricas no agregue panel lateral.

    :authors: Alvarez Nahuel Nicolas
    :date: 29/04/2026
    """
    cm = np.array([[2, 0], [1, 3]])
    labels = [0, 1]

    fig, _ = plot_confusion_matrix(cm, labels=labels, show_class_metrics=False)

    assert len(fig.axes) == 2
    assert all(len(axis.tables) == 0 for axis in fig.axes)

    plt.close(fig)
