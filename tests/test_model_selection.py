from __future__ import annotations
import numpy as np
from whiteboxml.model_selection import train_test_split


def test_split_shapes() -> None:
    """
    Verifica que train_test_split genere tamaños correctos
    para los conjuntos de entrenamiento y test según test_size.

    :return: None
    :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
    :date: 07/05/2026
    """

    X: np.ndarray = np.arange(100).reshape(50, 2)
    y: np.ndarray = np.arange(50)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    assert len(X_train) == 40
    assert len(X_test) == 10
    assert len(y_train) == 40
    assert len(y_test) == 10


def test_reproducibility() -> None:
    """
    Verifica que usando el mismo random_state,
    el resultado del split sea determinista.

    :return: None
    :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
    :date: 07/05/2026
    """

    X: np.ndarray = np.arange(20).reshape(10, 2)
    y: np.ndarray = np.arange(10)

    split1 = train_test_split(X, y, random_state=42)
    split2 = train_test_split(X, y, random_state=42)

    for a, b in zip(split1, split2):
        assert np.array_equal(a, b)


def test_no_overlap() -> None:
    """
    Verifica que los conjuntos de entrenamiento y test
    no compartan muestras (no haya solapamiento).

    :return: None
    :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
    :date: 07/05/2026
    """

    X: np.ndarray = np.arange(20).reshape(10, 2)
    y: np.ndarray = np.arange(10)

    X_train, X_test, _, _ = train_test_split(X, y, random_state=42)

    train_set = set(map(tuple, X_train))
    test_set = set(map(tuple, X_test))

    assert train_set.isdisjoint(test_set)