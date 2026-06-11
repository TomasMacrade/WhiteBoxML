"""
Test suite for PCAFromSVD implementation.

Tests validate mathematical correctness, numerical stability, and
API compatibility with expected behavior.

:authors: Toledo, Calina, Canteros
:date: 01/06/2026
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

# Importamos la clase y la función interna del SVD
from whiteboxml.dimensionality_reduction.pca_svd import PCAFromSVD, _svd_from_scratch


# ==========================================
# TESTS PARA LA FUNCIÓN SVD FROM SCRATCH
# ==========================================

def test_svd_from_scratch_reconstruccion_exacta():
    """
    Verify that custom SVD reconstructs the centered matrix within tolerance.
    """
    rng = np.random.RandomState(42)
    X = rng.randn(30, 8)
    X_c = X - np.mean(X, axis=0)
    
    U, S, Vt = _svd_from_scratch(X_c, tol=1e-12, max_iter=1000)
    X_reconstructed = U @ np.diag(S) @ Vt
    
    assert_allclose(X_reconstructed, X_c, rtol=1e-8, atol=1e-8)


# ==========================================
# TESTS PARA LA CLASE PCAFromSVD
# ==========================================

def test_pca_svd_retiene_varianza_umbral():
    """Verify that n_components as float retains at least specified variance."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    X[:, 0] = 2 * X[:, 1] + rng.randn(100) * 0.1
    
    pca = PCAFromSVD(n_components=0.90, scale=True)
    pca.fit(X)
    
    cumulative_variance = np.sum(pca.explained_variance_ratio_)
    assert cumulative_variance >= 0.90


def test_pca_svd_coherencia_reconstruccion():
    """Verify that inverse_transform(transform(X)) reconstructs ORIGINAL X."""
    rng = np.random.RandomState(0)
    X = rng.randn(50, 4)
    
    pca = PCAFromSVD(n_components=4, scale=False)
    X_transformed = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)
    
    # Tolerancia relajada para método iterativo from-scratch (precisión ~1e-9)
    assert_allclose(X_reconstructed, X, rtol=1e-7, atol=1e-7)


def test_pca_svd_componentes_no_correlacionadas():
    """Verify that transformed components are uncorrelated."""
    rng = np.random.RandomState(123)
    X = rng.randn(80, 6)
    
    pca = PCAFromSVD(n_components=6, scale=True)
    X_transformed = pca.fit_transform(X)
    
    cov_transformed = np.cov(X_transformed, rowvar=False)
    n_comp = X_transformed.shape[1]
    mask = ~np.eye(n_comp, dtype=bool)
    off_diagonal = cov_transformed[mask]
    
    # Tolerancia relajada para método iterativo from-scratch
    assert_allclose(off_diagonal, 0, atol=1e-6)


def test_pca_svd_orden_autovalores_decreciente():
    """Verify that eigenvalues are in decreasing order."""
    rng = np.random.RandomState(456)
    X = rng.randn(100, 8)
    
    pca = PCAFromSVD(n_components=None, scale=True)
    pca.fit(X)
    
    variances = pca.explained_variance_
    assert np.all(np.diff(variances) <= 1e-10)


def test_pca_svd_fit_transform_igual_a_fit_luego_transform():
    """Verify that fit_transform(X) equals fit(X).transform(X)."""
    rng = np.random.RandomState(789)
    X = rng.randn(60, 5)
    
    pca1 = PCAFromSVD(n_components=3, scale=True)
    X_transformed_1 = pca1.fit_transform(X)
    
    pca2 = PCAFromSVD(n_components=3, scale=True)
    pca2.fit(X)
    X_transformed_2 = pca2.transform(X)
    
    assert_allclose(X_transformed_1, X_transformed_2, rtol=1e-14)


def test_pca_svd_requiere_fit_antes_transform():
    """Verify that transform raises error if fit not called first."""
    pca = PCAFromSVD(n_components=2)
    X = np.random.randn(30, 4)
    
    with pytest.raises(RuntimeError, match="must call fit"):
        pca.transform(X)


def test_pca_svd_requiere_fit_antes_inverse_transform():
    """Verify that inverse_transform raises error if fit not called first."""
    pca = PCAFromSVD(n_components=2)
    Z = np.random.randn(30, 2)
    
    with pytest.raises(RuntimeError, match="must call fit"):
        pca.inverse_transform(Z)


def test_pca_svd_numero_componentes_invalido():
    """Verify that invalid n_components raises ValueError."""
    pca = PCAFromSVD(n_components=100)
    X = np.random.randn(50, 10)
    
    with pytest.raises(ValueError):
        pca.fit(X)


def test_pca_svd_escala_vs_sin_escala_produce_resultados_diferentes():
    """Verify that scaling affects the resulting components."""
    rng = np.random.RandomState(321)
    X = rng.randn(100, 3)
    X[:, 0] *= 1000
    
    pca_scaled = PCAFromSVD(n_components=3, scale=True)
    pca_scaled.fit(X)
    
    pca_unscaled = PCAFromSVD(n_components=3, scale=False)
    pca_unscaled.fit(X)
    
    corr_diff = np.abs(np.corrcoef(
        pca_scaled.components_[0].flatten(),
        pca_unscaled.components_[0].flatten()
    )[0, 1])
    
    assert corr_diff < 0.95


def test_pca_svd_varianza_explicada_suma_uno():
    """Verify that total explained variance ratio sums to 1.0."""
    rng = np.random.RandomState(654)
    X = rng.randn(70, 5)
    
    pca = PCAFromSVD(n_components=None, scale=True)
    pca.fit(X)
    
    total_variance_ratio = np.sum(pca.explained_variance_ratio_)
    assert_allclose(total_variance_ratio, 1.0, rtol=1e-10)