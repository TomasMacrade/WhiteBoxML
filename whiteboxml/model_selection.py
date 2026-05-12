from __future__ import annotations
import numpy as np


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide un dataset en conjuntos de entrenamiento y test.

    :param X: Variables de entrada (features).
    :type X: np.ndarray
    :param y: Variable objetivo (labels).
    :type y: np.ndarray
    :param test_size: Proporción del dataset destinada a test (ej: 0.2 = 20%).
    :type test_size: float
    :param random_state: Semilla para reproducibilidad.
    :type random_state: int | None
    :return: Tupla con X_train, X_test, y_train, y_test.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    :authors: Lucas Capocasa, Matias Moreyra,  Santiago Pastori, Franco Aranda
    :date: 07/05/2026
    """

    if random_state is not None:
        np.random.seed(random_state)

    X = np.array(X)
    y = np.array(y)

    n_samples = X.shape[0]

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    test_count = int(n_samples * test_size)

    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
    )