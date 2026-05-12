from __future__ import annotations
import numpy as np
from whiteboxml.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder


def test_standard_scaler() -> None:
    """
    Verifica que StandardScaler normalice los datos
    a media 0 y desviación estándar 1 por columna.

    :return: None
    :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
    :date: 07/05/2026
    """

    X: np.ndarray = np.array([[1, 2], [3, 4], [5, 6]])

    scaler: StandardScaler = StandardScaler()
    X_scaled: np.ndarray = scaler.fit_transform(X)

    assert np.allclose(np.mean(X_scaled, axis=0), [0, 0])
    assert np.allclose(np.std(X_scaled, axis=0), [1, 1])


def test_minmax_scaler() -> None:
    """
    Verifica que MinMaxScaler escale los datos
    al rango [0, 1].

    :return: None
    :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
    :date: 07/05/2026
    """

    X: np.ndarray = np.array([[1, 2], [3, 4], [5, 6]])

    scaler: MinMaxScaler = MinMaxScaler()
    X_scaled: np.ndarray = scaler.fit_transform(X)

    assert np.allclose(np.min(X_scaled, axis=0), [0, 0])
    assert np.allclose(np.max(X_scaled, axis=0), [1, 1])


def test_label_encoder() -> None:
    """
    Verifica que LabelEncoder:
    - asigne enteros a categorías
    - mantenga consistencia en valores repetidos
    - diferencie categorías distintas

    :return: None
    :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
    :date: 07/05/2026
    """

    y: np.ndarray = np.array(["rojo", "azul", "rojo"])

    encoder: LabelEncoder = LabelEncoder()
    y_encoded: np.ndarray = encoder.fit_transform(y)

    assert len(set(y_encoded)) == 2
    assert y_encoded[0] == y_encoded[2]
    assert y_encoded[0] != y_encoded[1]


def test_onehot_encoder() -> None:
    """
    Verifica que OneHotEncoder:
    - genere una matriz con dimensiones correctas
    - tenga un único valor 1 por fila
    - sea consistente para categorías repetidas

    :return: None
    :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
    :date: 07/05/2026
    """

    y: np.ndarray = np.array(["rojo", "azul", "rojo"])

    encoder: OneHotEncoder = OneHotEncoder()
    y_encoded: np.ndarray = encoder.fit_transform(y)

    assert y_encoded.shape == (3, 2)
    assert np.all(np.sum(y_encoded, axis=1) == 1)
    assert np.array_equal(y_encoded[0], y_encoded[2])