"""
Visualización: Módulo de visualización de WhiteBoxML.

:authors: Nahuel Nicolas Alvarez
:date: 15/04/2026
"""

from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.table import Table
from numpy.typing import ArrayLike

from whiteboxml.utils import _validacion_inputs


class _ClassMetrics(TypedDict):
    """
    Estructura tipada de métricas por clase para el panel lateral.

    :authors: Nahuel Nicolas Alvarez
    :date: 29/04/2026
    """

    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    support: np.ndarray
    accuracy: float


def confusion_matrix(
    y_true: ArrayLike, y_pred: ArrayLike, labels: ArrayLike | None = None
) -> np.ndarray:
    """
    Genera una matriz de confusión a partir de los valores reales y predichos.

    Las filas representan las clases reales y las columnas las clases predichas.

    :param y_true: Array de valores reales.
    :param y_pred: Array de valores predichos.
    :param labels: Lista de etiquetas para las clases.
        Si es None, se inferirán de los datos.
    :return: Matriz de confusión como un array de NumPy.

    :authors: Nahuel Nicolas Alvarez
    :date: 19/04/2026
    """
    y_true, y_pred = _validacion_inputs(y_true, y_pred)

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    else:
        labels = np.asarray(labels)

    return np.array(
        [[np.sum((y_true == i) & (y_pred == j)) for j in labels] for i in labels],
        dtype=int,
    )


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: ArrayLike,
    figsize: tuple[int, int] = (6, 5),
    cmap: str = "Blues",
    show_class_metrics: bool = True,
) -> tuple[Figure, Axes]:
    """
    Genera una visualización de una matriz de confusión usando Matplotlib.

    :param cm: Matriz de confusión calculada previamente.
    :param labels: Etiquetas de las clases a mostrar en los ejes.
    :param figsize: Tamaño de la figura.
        Si show_class_metrics es True, se usa un tamaño fijo para evitar solapamientos.
    :param cmap: Mapa de colores utilizado para el gráfico.
    :param show_class_metrics: Si True, muestra métricas por clase
        (precision, recall, f1 y soporte) al costado del gráfico.
    :return: Figura y ejes del gráfico generado.

    :authors: Nahuel Nicolas Alvarez
    :date: 19/04/2026
    """
    cm = np.asarray(cm)
    labels = np.asarray(labels)

    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("cm debe ser una matriz cuadrada de dos dimensiones.")

    if cm.shape[0] != len(labels):
        raise ValueError(
            "La cantidad de labels debe coincidir con las dimensiones de la matriz."
        )

    if show_class_metrics:
        fig = plt.figure(figsize=(12, 6))
        grid = GridSpec(1, 2, figure=fig, width_ratios=[3, 2], wspace=0.25)
        ax = fig.add_subplot(grid[0, 0])
        ax_metrics = fig.add_subplot(grid[0, 1])
    else:
        fig, ax = plt.subplots(figsize=figsize)

    image = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cantidad de observaciones", rotation=90)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicción",
        ylabel="Valor real",
        title="Matriz de confusión",
    )
    ax.grid(False)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    _annotate_confusion_matrix(ax, cm)

    if show_class_metrics:
        metrics = _compute_class_metrics(cm)
        _draw_metrics_table(ax_metrics, labels, metrics)

    fig.tight_layout()
    return fig, ax


def _annotate_confusion_matrix(ax: Axes, cm: np.ndarray) -> None:
    """
    Agrega anotaciones de conteo sobre cada celda de la matriz.

    :param ax: Ejes donde se dibuja la matriz.
    :param cm: Matriz de confusión.
    :return: None.
    :authors: Nahuel Nicolas Alvarez
    :date: 29/04/2026
    """
    threshold = cm.max() / 2 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "#1f2937",
                fontsize=10,
                fontweight="bold" if i == j else "normal",
            )


def _compute_class_metrics(cm: np.ndarray) -> _ClassMetrics:
    """
    Calcula métricas por clase a partir de la matriz de confusión.

    :param cm: Matriz de confusión.
    :return: Diccionario con precision, recall, f1, support y accuracy.
    :authors: Nahuel Nicolas Alvarez
    :date: 29/04/2026
    """
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    actual = cm.sum(axis=1).astype(float)
    total = float(cm.sum())

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.nan_to_num(tp / predicted)
        recall = np.nan_to_num(tp / actual)
        f1 = np.nan_to_num((2 * precision * recall) / (precision + recall))
        accuracy = np.nan_to_num(tp.sum() / total) if total > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": actual,
        "accuracy": float(accuracy),
    }


def _build_metrics_rows(
    labels: np.ndarray,
    metrics: _ClassMetrics,
    decimals: int = 3,
) -> list[list[str]]:
    """
    Construye las filas de texto para la tabla de métricas por clase.

    :param labels: Etiquetas de clase.
    :param metrics: Diccionario tipado con métricas por clase.
    :param decimals: Cantidad de decimales a mostrar.
    :return: Filas de texto para la tabla.
    :authors: Nahuel Nicolas Alvarez
    :date: 29/04/2026
    """
    return [
        [
            str(cls),
            f"{p_i:.{decimals}f}",
            f"{r_i:.{decimals}f}",
            f"{f_i:.{decimals}f}",
            str(int(s_i)),
        ]
        for cls, p_i, r_i, f_i, s_i in zip(
            labels,
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["support"],
        )
    ]


def _style_metrics_table(table: Table) -> None:
    """
    Aplica estilo visual al encabezado y filas de la tabla de métricas.

    :param table: Tabla de Matplotlib a estilizar.
    :return: None.
    :authors: Nahuel Nicolas Alvarez
    :date: 29/04/2026
    """
    header_color = "#1e3a8a"
    row_light = "#dbeafe"
    row_white = "#eff6ff"

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(row_light if row % 2 else row_white)

    for col in range(5):
        table[(0, col)].set_edgecolor("white")


def _draw_metrics_table(
    ax: Axes,
    labels: np.ndarray,
    metrics: _ClassMetrics,
    decimals: int = 3,
) -> None:
    """
    Dibuja una tabla de métricas por clase en un panel lateral.

    :param ax: Ejes donde se dibuja la tabla.
    :param labels: Etiquetas de clase.
    :param metrics: Diccionario con métricas por clase y accuracy global.
    :param decimals: Cantidad de decimales a mostrar.
    :return: None.
    :authors: Nahuel Nicolas Alvarez
    :date: 29/04/2026
    """
    accuracy = metrics["accuracy"]

    ax.axis("off")
    ax.set_title(f"Metricas por clase (Accuracy: {accuracy:.{decimals}f})", fontsize=10)

    row_data = _build_metrics_rows(labels, metrics, decimals)

    table = ax.table(
        cellText=row_data,
        colLabels=["Clase", "Precision", "Recall", "F1", "Soporte"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    _style_metrics_table(table)
