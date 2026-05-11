"""
Tests del módulo de reportes
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from whiteboxml import reporting


def test_summary_classification_perfect():
    """
    Test reporte de clasificación binaria con predicción perfecta.
    :authors: Joaquín Palacio Feijóo
    :date: 21/04/2026
    """
    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 1, 1]

    expected_df = pd.DataFrame(
        {
            "Accuracy": [1.0],
            "Precision": [1.0],
            "Recall": [1.0],
            "F1": [1.0],
        }
    )

    df = reporting.summary(y_true, y_pred)

    assert_frame_equal(df, expected_df)


def test_summary_classification_imperfect():
    """
    Test reporte de clasificación binaria con errores.
    :authors: Joaquín Palacio Feijóo
    :date: 21/04/2026
    """
    y_true = [0, 0, 0, 1]
    y_pred = [1, 0, 1, 1]

    expected_df = pd.DataFrame(
        {
            "Accuracy": [0.5],
            "Precision": [1 / 3],
            "Recall": [1.0],
            "F1": [0.5],
        }
    )

    df = reporting.summary(y_true, y_pred)

    assert_frame_equal(df, expected_df)


def test_summary_classification_explicit_mode():
    """
    Test reporte forzando mode='classification'.
    :authors: Joaquín Palacio Feijóo
    :date: 21/04/2026
    """
    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 1, 1]

    expected_df = pd.DataFrame(
        {
            "Accuracy": [1.0],
            "Precision": [1.0],
            "Recall": [1.0],
            "F1": [1.0],
        }
    )

    df = reporting.summary(y_true, y_pred, mode="classification")

    assert_frame_equal(df, expected_df)


def test_summary_multiclass_auto():
    """
    Test auto-detección de clasificación multiclase (usa average='macro').
    :authors: Joaquín Palacio Feijóo
    :date: 21/04/2026
    """
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 2, 2]

    expected_df = pd.DataFrame(
        {
            "Accuracy": [1.0],
            "Precision": [1.0],
            "Recall": [1.0],
            "F1": [1.0],
        }
    )

    df = reporting.summary(y_true, y_pred)

    assert_frame_equal(df, expected_df)


def test_summary_multiclass_imperfect():
    """
    Test clasificación multiclase imperfecta con average='macro'.
    :authors: Joaquín Palacio Feijóo
    :date: 21/04/2026
    """
    y_true = [0, 0, 0, 1]
    y_pred = [1, 0, 1, 1]

    expected_df = pd.DataFrame(
        {
            "Accuracy": [0.5],
            "Precision": [1 / 3],
            "Recall": [1.0],
            "F1": [0.5],
        }
    )

    df = reporting.summary(y_true, y_pred, mode="classification")

    assert_frame_equal(df, expected_df)


def test_summary_regression_perfect():
    """
    Test reporte de regresión con predicción perfecta.
    :authors: Joaquín Palacio Feijóo
    :date: 21/04/2026
    """
    y_true = [1.5, 2.3, 3.7]
    y_pred = [1.5, 2.3, 3.7]

    expected_df = pd.DataFrame(
        {
            "MSE": [0.0],
            "MAE": [0.0],
            "R^2": [1.0],
        }
    )

    df = reporting.summary(y_true, y_pred)

    assert_frame_equal(df, expected_df)


def test_summary_regression_imperfect():
    """
    Test reporte de regresión con errores.
    :authors: Joaquín Palacio Feijóo
    :date: 21/04/2026
    """
    y_true = [1.5, 2.5, 3.5]
    y_pred = [1.0, 2.0, 3.0]

    df = reporting.summary(y_true, y_pred)

    assert df["MSE"].iloc[0] == pytest.approx(0.25)
    assert df["MAE"].iloc[0] == pytest.approx(0.5)
    assert "R^2" in df.columns


def test_summary_regression_explicit_mode():
    """
    Test reporte forzando mode='regression' con datos enteros.
    :authors: Joaquín Palacio Feijóo
    :date: 21/04/2026
    """
    y_true = [0, 0, 0, 1]
    y_pred = [1, 0, 1, 1]

    expected_df = pd.DataFrame(
        {
            "MSE": [0.5],
            "MAE": [0.5],
            "R^2": [-1.6666666666666665],
        }
    )

    df = reporting.summary(y_true, y_pred, mode="regression")

    assert_frame_equal(df, expected_df)


def test_summary_auto_detects_classification_from_int():
    """
    Test que datos enteros se detectan como clasificación.
    :authors: Joaquín Palacio Feijóo
    :date: 21/04/2026
    """
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]

    df = reporting.summary(y_true, y_pred)

    assert "Accuracy" in df.columns
    assert "Precision" in df.columns
    assert "Recall" in df.columns
    assert "F1" in df.columns


def test_summary_auto_detects_regression_from_float():
    """
    Test que datos float con decimales se detectan como regresión.
    :authors: Joaquín Palacio Feijóo
    :date: 21/04/2026
    """
    y_true = [1.1, 2.2, 3.3]
    y_pred = [1.0, 2.0, 3.0]

    df = reporting.summary(y_true, y_pred)

    assert "MSE" in df.columns
    assert "MAE" in df.columns
    assert "R^2" in df.columns


def test_summary_auto_detects_classification_from_strings():
    """
    Test que datos string se detectan como clasificación.
    :authors: Joaquín Palacio Feijóo
    :date: 21/04/2026
    """
    y_true = np.array(["cat", "dog", "cat", "cat"])
    y_pred = np.array(["cat", "dog", "dog", "cat"])

    df = reporting.summary(y_true, y_pred)

    assert "Accuracy" in df.columns


def test_summary_invalid_mode():
    """
    Test que un mode inválido lanza ValueError.
    :authors: Joaquín Palacio Feijóo
    :date: 21/04/2026
    """
    with pytest.raises(ValueError, match="mode"):
        reporting.summary([1, 2], [1, 2], mode="invalid")
