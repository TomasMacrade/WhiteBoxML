"""
Tests for the WhiteBoxML regularization module.

Covers: L1Regularizer, L2Regularizer, ElasticNetRegularizer, and the
Regularizer base class (validation and ABC contract).

:authors: Agustina Acosta, Rocio Barbetta, Dylan Guitard Jimenez
:date: 24/03/2026
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from whiteboxml.regularizations import (
    ElasticNetRegularizer,
    L1Regularizer,
    L2Regularizer,
    Regularizer,
)

# ---------------------------------------------------------------------------
# ABC contract
# ---------------------------------------------------------------------------


def test_regularizer_cannot_be_instantiated_directly():
    """Regularizer is abstract and must not be instantiatable directly."""
    with pytest.raises(TypeError):
        Regularizer()  # type: ignore[abstract]


def test_subclass_missing_gradient_cannot_be_instantiated():
    """A subclass that does not implement gradient() must fail on instantiation."""

    class Incomplete(Regularizer):
        def loss(self, weights: np.ndarray) -> float:
            return 0.0

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Shared validations
# ---------------------------------------------------------------------------


def test_negative_alpha_raises_error():
    """A negative alpha must raise ValueError in any regularizer."""
    with pytest.raises(ValueError):
        L2Regularizer(alpha=-0.1)


def test_3d_weights_raise_error():
    """Weights with more than 2 dimensions must raise ValueError."""
    reg = L2Regularizer(alpha=0.1)
    with pytest.raises(ValueError):
        reg.loss(np.ones((2, 2, 2)))


def test_scalar_weights_accepted():
    """A scalar must be accepted and treated as a 1D array."""
    reg = L2Regularizer(alpha=1.0)
    assert reg.loss(3.0) == pytest.approx(9.0)


def test_list_weights_accepted():
    """A Python list must be accepted and converted to ndarray."""
    reg = L1Regularizer(alpha=1.0)
    assert reg.loss([1.0, -2.0, 3.0]) == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# L2Regularizer
# ---------------------------------------------------------------------------


def test_l2_loss_formula():
    """L2 loss must equal alpha * sum(w²)."""
    reg = L2Regularizer(alpha=0.5)
    weights = np.array([1.0, 2.0, 3.0])
    expected = 0.5 * (1 + 4 + 9)
    assert reg.loss(weights) == pytest.approx(expected)


def test_l2_gradient_formula():
    """L2 gradient must equal 2 * alpha * w."""
    reg = L2Regularizer(alpha=0.5)
    weights = np.array([1.0, 2.0, 3.0])
    expected = 2 * 0.5 * weights
    assert_allclose(reg.gradient(weights), expected)


def test_l2_gradient_same_shape():
    """L2 gradient must have the same shape as the weights."""
    reg = L2Regularizer(alpha=0.1)
    weights = np.ones((4, 3))
    assert reg.gradient(weights).shape == weights.shape


def test_l2_zero_alpha_zero_loss():
    """With alpha=0, L2 loss must be exactly 0."""
    reg = L2Regularizer(alpha=0.0)
    assert reg.loss(np.array([10.0, -5.0])) == pytest.approx(0.0)


def test_l2_zero_alpha_zero_gradient():
    """With alpha=0, L2 gradient must be the zero vector."""
    reg = L2Regularizer(alpha=0.0)
    weights = np.array([3.0, -1.0])
    assert_allclose(reg.gradient(weights), np.zeros_like(weights))


def test_l2_repr():
    """L2Regularizer __repr__ must include the alpha value."""
    reg = L2Regularizer(alpha=0.01)
    assert "0.01" in repr(reg)


# ---------------------------------------------------------------------------
# L1Regularizer
# ---------------------------------------------------------------------------


def test_l1_loss_formula():
    """L1 loss must equal alpha * sum(|w|)."""
    reg = L1Regularizer(alpha=2.0)
    weights = np.array([1.0, -2.0, 3.0])
    expected = 2.0 * (1 + 2 + 3)
    assert reg.loss(weights) == pytest.approx(expected)


def test_l1_gradient_formula():
    """L1 gradient must equal alpha * sign(w)."""
    reg = L1Regularizer(alpha=2.0)
    weights = np.array([1.0, -2.0, 3.0])
    expected = 2.0 * np.sign(weights)
    assert_allclose(reg.gradient(weights), expected)


def test_l1_gradient_same_shape():
    """L1 gradient must have the same shape as the weights."""
    reg = L1Regularizer(alpha=0.1)
    weights = np.ones((4, 3))
    assert reg.gradient(weights).shape == weights.shape


def test_l1_zero_alpha_zero_loss():
    """With alpha=0, L1 loss must be exactly 0."""
    reg = L1Regularizer(alpha=0.0)
    assert reg.loss(np.array([10.0, -5.0])) == pytest.approx(0.0)


def test_l1_zero_weights_zero_gradient():
    """With weights=0, L1 gradient must be the zero vector."""
    reg = L1Regularizer(alpha=1.0)
    assert_allclose(reg.gradient(np.zeros(3)), np.zeros(3))


def test_l1_repr():
    """L1Regularizer __repr__ must include the alpha value."""
    reg = L1Regularizer(alpha=0.05)
    assert "0.05" in repr(reg)


# ---------------------------------------------------------------------------
# ElasticNetRegularizer
# ---------------------------------------------------------------------------


def test_elasticnet_invalid_l1_ratio_above_one():
    """l1_ratio above 1 must raise ValueError."""
    with pytest.raises(ValueError):
        ElasticNetRegularizer(alpha=0.1, l1_ratio=1.5)


def test_elasticnet_negative_l1_ratio():
    """Negative l1_ratio must raise ValueError."""
    with pytest.raises(ValueError):
        ElasticNetRegularizer(alpha=0.1, l1_ratio=-0.1)


def test_elasticnet_pure_l2_equals_l2():
    """ElasticNet with l1_ratio=0 must be equivalent to L2."""
    weights = np.array([1.0, -2.0, 3.0])
    alpha = 0.3
    en = ElasticNetRegularizer(alpha=alpha, l1_ratio=0.0)
    l2 = L2Regularizer(alpha=alpha)
    assert en.loss(weights) == pytest.approx(l2.loss(weights))
    assert_allclose(en.gradient(weights), l2.gradient(weights))


def test_elasticnet_pure_l1_equals_l1():
    """ElasticNet with l1_ratio=1 must be equivalent to L1."""
    weights = np.array([1.0, -2.0, 3.0])
    alpha = 0.3
    en = ElasticNetRegularizer(alpha=alpha, l1_ratio=1.0)
    l1 = L1Regularizer(alpha=alpha)
    assert en.loss(weights) == pytest.approx(l1.loss(weights))
    assert_allclose(en.gradient(weights), l1.gradient(weights))


def test_elasticnet_loss_formula():
    """ElasticNet loss must combine L1 and L2 terms according to l1_ratio."""
    alpha, rho = 1.0, 0.5
    weights = np.array([1.0, -2.0, 3.0])
    reg = ElasticNetRegularizer(alpha=alpha, l1_ratio=rho)
    expected = alpha * (rho * np.sum(np.abs(weights)) + (1 - rho) * np.sum(weights**2))
    assert reg.loss(weights) == pytest.approx(expected)


def test_elasticnet_gradient_formula():
    """ElasticNet gradient must combine sign(w) and w terms according to l1_ratio."""
    alpha, rho = 1.0, 0.5
    weights = np.array([1.0, -2.0, 3.0])
    reg = ElasticNetRegularizer(alpha=alpha, l1_ratio=rho)
    expected = alpha * (rho * np.sign(weights) + 2 * (1 - rho) * weights)
    assert_allclose(reg.gradient(weights), expected)


def test_elasticnet_gradient_same_shape():
    """ElasticNet gradient must have the same shape as the weights."""
    reg = ElasticNetRegularizer(alpha=0.1, l1_ratio=0.5)
    weights = np.ones((4, 3))
    assert reg.gradient(weights).shape == weights.shape


def test_elasticnet_repr():
    """ElasticNetRegularizer __repr__ must include alpha and l1_ratio."""
    reg = ElasticNetRegularizer(alpha=0.1, l1_ratio=0.3)
    assert "0.1" in repr(reg)
    assert "0.3" in repr(reg)
