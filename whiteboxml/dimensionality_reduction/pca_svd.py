"""
Principal Component Analysis using Custom Singular Value Decomposition (SVD).

This module implements PCA by calculating SVD from scratch using power iteration and deflation, strictly following the mathematical derivation of the covariance eigenvalue decomposition.

Authors: Toledo, Calina, Canteros
Date: June 1, 2026
References:

- Jolliffe, I. T. (2002). Principal Component Analysis. Springer.

- Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Union, Tuple
import matplotlib.pyplot as plt


def _svd_from_scratch(
    X_c: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 500
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula la SVD de una matriz centrada usando el método de las potencias y deflación.

    Encuentra U, S, V tales que X_c ≈ U @ diag(S) @ V^T mediante la extracción iterativa
    del autovector dominante de X_c^T X_c y la posterior deflación de la matriz.

    :param X_c: Matriz de datos centrada de forma (n_samples, n_features).
    :type X_c: np.ndarray
    :param tol: Tolerancia de convergencia para la iteración del autovector.
    :type tol: float
    :param max_iter: Número máximo de iteraciones por componente.
    :type max_iter: int
    :return: Tupla (U, S, V) donde U ∈ ℝ^(n×k), S ∈ ℝ^k, V ∈ ℝ^(m×k).
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]

    :authors: Toledo, Calina, Canteros
    :date: 01/06/2026

    Notas
    -----
    Esta implementación sigue la equivalencia matemática:
    eig(X_c^T X_c) → (λ_i, v_i) ⇒ σ_i = √λ_i, u_i = X_c v_i / σ_i.
    La deflación garantiza la ortogonalidad entre componentes sucesivos.
    """
    n, m = X_c.shape
    k_max = min(n, m)

    # Matriz simétrica semidefinida positiva: C = X_c^T X_c
    C = X_c.T @ X_c

    U = np.zeros((n, k_max))
    V = np.zeros((m, k_max))
    S = np.zeros(k_max)

    for i in range(k_max):
        # Método de las Potencias: encontrar el autovector dominante de C
        v = np.random.RandomState(i).randn(m)
        v = v / np.linalg.norm(v)

        for _ in range(max_iter):
            v_new = C @ v
            norm_v = np.linalg.norm(v_new)

            if norm_v < tol:  # Autovalor cero alcanzado
                v_new = v
                break

            v_new = v_new / norm_v
            if np.linalg.norm(v_new - v) < tol:
                v = v_new
                break
            v = v_new

        # Calcular el autovalor y el valor singular
        lam = float(v @ (C @ v))
        sigma = np.sqrt(max(lam, 0.0))  # Forzar a cero si hay negativos numéricos

        if sigma < tol:
            # Los componentes restantes son numéricamente cero
            k_max = i
            break

        S[i] = sigma
        V[:, i] = v

        # Vector singular izquierdo: u_i = X_c v_i / σ_i
        U[:, i] = (X_c @ v) / sigma

        # Deflación: eliminar el componente encontrado de C
        # C ← C - λ v v^T  (preserva la simetría)
        C = C - lam * np.outer(v, v)

    # Recortar al rango numérico real y devolver V^T en el formato estándar de SVD.
    return U[:, :k_max], S[:k_max], V[:, :k_max].T


class PCAFromSVD:
    """
    Análisis de Componentes Principales mediante una implementación propia de SVD.

    Calcula PCA aplicando una SVD desde cero a los datos centrados (y opcionalmente
    escalados), adhiriéndose estrictamente a la descomposición espectral de la
    matriz de covarianza sin depender de librerías optimizadas de álgebra lineal.

    :param n_components: Número de componentes a conservar. Si es None, se conservan todos.
        Si es int, especifica el número exacto. Si es float en (0,1), selecciona el número
        mínimo de componentes para retener al menos esa fracción de la varianza total.
    :type n_components: Optional[Union[int, float]]
    :param scale: Indica si se deben estandarizar las variables a media cero y varianza unitaria
        antes de aplicar SVD. Recomendado cuando las variables tienen escalas diferentes.
    :type scale: bool
    :param svd_tol: Tolerancia de convergencia para el algoritmo SVD desde cero.
    :type svd_tol: float
    :param svd_max_iter: Máximo de iteraciones por componente en el método de las potencias de SVD.
    :type svd_max_iter: int

    :ivar mean_: Media de cada variable calculada durante el fit.
    :vartype mean_: np.ndarray
    :ivar std_: Desviación estándar de cada variable (solo si scale=True).
    :vartype std_: Optional[np.ndarray]
    :ivar components_: Ejes principales en el espacio de características (vectores singulares derechos).
    :vartype components_: np.ndarray
    :ivar explained_variance_: Varianza explicada por cada componente seleccionado.
    :vartype explained_variance_: np.ndarray
    :ivar explained_variance_ratio_: Porcentaje de varianza total explicada.
    :vartype explained_variance_ratio_: np.ndarray
    :ivar singular_values_: Valores singulares obtenidos de la descomposición SVD.
    :vartype singular_values_: np.ndarray
    :ivar n_components_: Número real de componentes seleccionados.
    :vartype n_components_: int

    :authors: Toledo, Calina, Canteros
    :date: 01/06/2026

    Ejemplo
    -------
    >>> import numpy as np
    >>> X = np.random.randn(100, 5)
    >>> pca = PCAFromSVD(n_components=0.95, scale=True)
    >>> X_reduced = pca.fit_transform(X)
    """

    def __init__(
        self,
        n_components: Optional[Union[int, float]] = None,
        scale: bool = False,
        svd_tol: float = 1e-10,
        svd_max_iter: int = 500
    ) -> None:
        """
        Inicializa el modelo PCA con los parámetros especificados.

        :param n_components: Número de componentes a conservar.
        :type n_components: Optional[Union[int, float]]
        :param scale: Indica si se deben estandarizar las variables.
        :type scale: bool
        :param svd_tol: Tolerancia para la convergencia de SVD.
        :type svd_tol: float
        :param svd_max_iter: Iteraciones máximas para el método de las potencias de SVD.
        :type svd_max_iter: int
        """
        self.n_components = n_components
        self.scale = scale
        self.svd_tol = svd_tol
        self.svd_max_iter = svd_max_iter

        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None
        self.n_components_: Optional[int] = None

    def fit(self, X: np.ndarray) -> PCAFromSVD:
        """
        Entrena el modelo PCA calculando la SVD propia de los datos centrados (y escalados).

        :param X: Matriz de datos de entrenamiento de forma (n_samples, n_features).
        :type X: np.ndarray
        :return: La instancia entrenada del modelo PCA.
        :rtype: PCAFromSVD
        :raises ValueError: Si n_components es inválido o si los datos tienen una forma incorrecta.

        :authors: Toledo, Calina, Canteros
        :date: 01/06/2026

        Notas
        -----
        Utiliza el método de las potencias + deflación para calcular SVD desde cero.
        Evita usar numpy.linalg.svd para cumplir con los requerimientos académicos de desarrollo propio.
        """
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        if n_samples < 2:
            raise ValueError("At least 2 samples are required for PCA")

        # Paso 1: Centrar
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Opcional: Escalar
        if self.scale:
            self.std_ = np.std(X, axis=0, ddof=0)
            self.std_[self.std_ == 0] = 1.0
            X_centered = X_centered / self.std_

        # Paso 2: SVD Propia
        U, S, Vt = _svd_from_scratch(
            X_centered, tol=self.svd_tol, max_iter=self.svd_max_iter
        )

        # Guardar resultados (forma de Vt: (k, m))
        self.singular_values_ = S
        self.components_ = Vt

        # Paso 3: Cálculo de varianza
        self.explained_variance_ = (S ** 2) / (n_samples - 1)
        total_variance = np.sum(self.explained_variance_)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        # Paso 4: Determinar k
        self.n_components_ = self._determine_n_components(len(S))

        # Paso 5: Recortar (Truncar)
        k = self.n_components_
        self.components_ = self.components_[:k]
        self.explained_variance_ = self.explained_variance_[:k]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:k]
        self.singular_values_ = self.singular_values_[:k]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Proyecta los datos sobre los ejes de los componentes principales.

        :param X: Matriz de datos a transformar de forma (n_samples, n_features).
        :type X: np.ndarray
        :return: Datos transformados en el espacio de componentes principales.
        :rtype: np.ndarray
        :raises RuntimeError: Si aún no se ha llamado a fit().

        :authors: Toledo, Calina, Canteros
        :date: 01/06/2026
        """
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("You must call fit() before transform()")

        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean_
        if self.scale and self.std_ is not None:
            X_centered = X_centered / self.std_

        return X_centered @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Entrena el modelo y transforma los datos en un solo paso."""
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Reconstruye los datos desde el espacio de componentes principales.

        :param Z: Datos en el espacio de CP de forma (n_samples, n_components).
        :type Z: np.ndarray
        :return: Datos reconstruidos en el espacio original.
        :rtype: np.ndarray
        :raises RuntimeError: Si aún no se ha llamado a fit().

        :authors: Toledo, Calina, Canteros
        :date: 01/06/2026
        """
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("You must call fit() before inverse_transform()")

        Z = np.asarray(Z, dtype=float)
        X_centered_rec = Z @ self.components_
        if self.scale and self.std_ is not None:
            X_centered_rec = X_centered_rec * self.std_
        return X_centered_rec + self.mean_

    def _determine_n_components(self, n_available: int) -> int:
        """Determina el número efectivo de componentes."""
        if self.n_components is None:
            return n_available
        if isinstance(self.n_components, int):
            if self.n_components < 1 or self.n_components > n_available:
                raise ValueError(f"n_components must be between 1 and {n_available}")
            return self.n_components
        if isinstance(self.n_components, float):
            if not 0 < self.n_components < 1:
                raise ValueError("n_components float must be in (0, 1)")
            cum_var = np.cumsum(self.explained_variance_ratio_)
            idx = np.where(cum_var >= self.n_components)[0]
            return int(idx[0] + 1) if len(idx) > 0 else n_available
        raise ValueError(f"Invalid n_components type: {type(self.n_components)}")

    def plot_variance(self, cumulative: bool = False, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Grafica la varianza explicada. Ver la versión anterior para el docstring completo."""
        if self.explained_variance_ratio_ is None:
            raise RuntimeError("You must call fit() before plotting variance")
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        n_comp = len(self.explained_variance_ratio_)
        x = np.arange(1, n_comp + 1)
        if cumulative:
            y = np.cumsum(self.explained_variance_ratio_)
            ax.plot(x, y, marker='o', linestyle='--', color='tab:blue', linewidth=2)
            ax.set_ylabel('Cumulative Explained Variance')
        else:
            y = self.explained_variance_ratio_
            ax.bar(x, y, alpha=0.7, color='tab:blue', edgecolor='black')
            ax.set_ylabel('Explained Variance by Component')
        ax.set_xlabel('Principal Component')
        ax.set_title('Variance Analysis (PCA via Custom SVD)', fontweight='bold')
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3, axis='y')
        return ax