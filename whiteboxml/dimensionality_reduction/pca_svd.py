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
    Compute SVD of a centered matrix using power iteration and deflation.
    
    Finds U, S, V such that X_c ≈ U @ diag(S) @ V^T by iteratively extracting
    the dominant eigenvector of X_c^T X_c and deflating the matrix.
    
    :param X_c: Centered data matrix of shape (n_samples, n_features).
    :type X_c: np.ndarray
    :param tol: Convergence tolerance for eigenvector iteration.
    :type tol: float
    :param max_iter: Maximum iterations per component.
    :type max_iter: int
    :return: Tuple (U, S, V) where U ∈ ℝ^(n×k), S ∈ ℝ^k, V ∈ ℝ^(m×k).
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    
    :authors: Toledo, Calina, Canteros
    :date: 01/06/2026
    
    Notes
    -----
    This implementation follows the mathematical equivalence:
    eig(X_c^T X_c) → (λ_i, v_i) ⇒ σ_i = λ_i, u_i = X_c v_i / σ_i.
    Deflation ensures orthogonality between successive components.
    """
    n, m = X_c.shape
    k_max = min(n, m)
    
    # Symmetric positive semi-definite matrix: C = X_c^T X_c
    C = X_c.T @ X_c
    
    U = np.zeros((n, k_max))
    V = np.zeros((m, k_max))
    S = np.zeros(k_max)
    
    for i in range(k_max):
        # 🔹 Power Iteration: find dominant eigenvector of C
        v = np.random.RandomState(i).randn(m)
        v = v / np.linalg.norm(v)
        
        for _ in range(max_iter):
            v_new = C @ v
            norm_v = np.linalg.norm(v_new)
            
            if norm_v < tol:  # Zero eigenvalue reached
                v_new = v
                break
                
            v_new = v_new / norm_v
            if np.linalg.norm(v_new - v) < tol:
                v = v_new
                break
            v = v_new
        
        # 🔹 Compute eigenvalue and singular value
        lam = float(v @ (C @ v))
        sigma = np.sqrt(max(lam, 0.0))  # Clamp numerical negatives
        
        if sigma < tol:
            # Remaining components are numerically zero
            k_max = i
            break
            
        S[i] = sigma
        V[:, i] = v
        
        # 🔹 Left singular vector: u_i = X_c v_i / σ_i
        U[:, i] = (X_c @ v) / sigma
        
        # 🔹 Deflation: remove found component from C
        # C ← C - λ v v^T  (preserves symmetry)
        C = C - lam * np.outer(v, v)
    
    # Trim to actual numerical rank and return V^T in the standard SVD format.
    return U[:, :k_max], S[:k_max], V[:, :k_max].T


class PCAFromSVD:
    """
    Principal Component Analysis via custom SVD implementation.
    
    Computes PCA by applying a from-scratch SVD to centered (and optionally
    scaled) data, adhering strictly to the spectral decomposition of the
    covariance matrix without relying on optimized linear algebra backends.
    
    :param n_components: Number of components to keep. If None, all components
        are kept. If int, specifies the exact number. If float in (0,1), selects
        the minimum number of components to retain at least that fraction of
        total variance.
    :type n_components: Optional[Union[int, float]]
    :param scale: Whether to standardize features to zero mean and unit variance
        before applying SVD. Recommended when features have different scales.
    :type scale: bool
    :param svd_tol: Convergence tolerance for the from-scratch SVD algorithm.
    :type svd_tol: float
    :param svd_max_iter: Maximum iterations per component in SVD power method.
    :type svd_max_iter: int
    
    :ivar mean_: Mean of each feature computed during fit.
    :vartype mean_: np.ndarray
    :ivar std_: Standard deviation of each feature (only if scale=True).
    :vartype std_: Optional[np.ndarray]
    :ivar components_: Principal axes in feature space (right singular vectors).
    :vartype components_: np.ndarray
    :ivar explained_variance_: Variance explained by each selected component.
    :vartype explained_variance_: np.ndarray
    :ivar explained_variance_ratio_: Percentage of total variance explained.
    :vartype explained_variance_ratio_: np.ndarray
    :ivar singular_values_: Singular values from SVD decomposition.
    :vartype singular_values_: np.ndarray
    :ivar n_components_: Actual number of components selected.
    :vartype n_components_: int
    
    :authors: Toledo, Calina, Canteros
    :date: 01/06/2026
    
    Example
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
        Initialize PCA model with specified parameters.
        
        :param n_components: Number of components to retain.
        :type n_components: Optional[Union[int, float]]
        :param scale: Whether to standardize features.
        :type scale: bool
        :param svd_tol: Tolerance for SVD convergence.
        :type svd_tol: float
        :param svd_max_iter: Max iterations for SVD power method.
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
        Fit the PCA model by computing custom SVD of centered (and scaled) data.
        
        :param X: Training data matrix of shape (n_samples, n_features).
        :type X: np.ndarray
        :return: The fitted PCA model instance.
        :rtype: PCAFromSVD
        :raises ValueError: If n_components is invalid or data has incorrect shape.
        
        :authors: Tu Nombre
        :date: 01/06/2026
        
        Notes
        -----
        Uses power iteration + deflation to compute SVD from scratch.
        Avoids numpy.linalg.svd to satisfy academic "from-scratch" requirements.
        """
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        
        if n_samples < 2:
            raise ValueError("At least 2 samples are required for PCA")
        
        # Step 1: Center
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Optional: Scale
        if self.scale:
            self.std_ = np.std(X, axis=0, ddof=0)
            self.std_[self.std_ == 0] = 1.0
            X_centered = X_centered / self.std_
        
        # Step 2: Custom SVD
        U, S, Vt = _svd_from_scratch(
            X_centered, tol=self.svd_tol, max_iter=self.svd_max_iter
        )
        
        # Store results (Vt shape: (k, m))
        self.singular_values_ = S
        self.components_ = Vt
        
        # Step 3: Variance calculation
        self.explained_variance_ = (S ** 2) / (n_samples - 1)
        total_variance = np.sum(self.explained_variance_)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        # Step 4: Determine k
        self.n_components_ = self._determine_n_components(len(S))
        
        # Step 5: Truncate
        k = self.n_components_
        self.components_ = self.components_[:k]
        self.explained_variance_ = self.explained_variance_[:k]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:k]
        self.singular_values_ = self.singular_values_[:k]
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto the principal component axes.
        
        :param X: Data matrix to transform of shape (n_samples, n_features).
        :type X: np.ndarray
        :return: Transformed data in principal component space.
        :rtype: np.ndarray
        :raises RuntimeError: If fit has not been called.
        
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
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from principal component space.
        
        :param Z: Data in PC space of shape (n_samples, n_components).
        :type Z: np.ndarray
        :return: Reconstructed data in original space.
        :rtype: np.ndarray
        :raises RuntimeError: If fit has not been called.
        
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
        """Determine effective number of components."""
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
        """Plot explained variance. See previous version for full docstring."""
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