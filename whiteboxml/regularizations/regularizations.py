from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Regularizer(ABC):
    """
    Abstract base class for model regularizers.

    Defines the contract that every regularizer must fullfill: compute
    the penalty term added to the model's loss function (``loss``)
    and its gradient with respect to the weights (``gradient``).
    Models consume this interface without knowing the concrete type.

    :param alpha: Regularization strength. Higher values penalize large
        weights more heavily. Must be >= 0.
    :type alpha: float
    :authors: Agustina Acosta, Rocio Barbetta, Dylan Guitard Jimenez
    :date: 2024-06
    """

    def __init__(self, alpha: float = 0.01) -> None:
        """Initialize the regularizer with the penalty strength.

        :param alpha: Penalty scaling factor. Must be >= 0.
        :type alpha: float
        :raises ValueError: If alpha is negative.
        :authors: Agustina Acosta, Rocio Barbetta, Dylan Guitard Jimenez
        :date: 2024-06
        """
        if alpha < 0:
            raise ValueError("alpha must be >= 0.")
        self.alpha = alpha

    @abstractmethod
    def loss(self, weights: np.ndarray) -> float:
        """
        Compute the penalty term to be added to the model's loss.

        :param weights: Weight vector or matrix of the model.
        :type weights: np.ndarray
        :return: Scalar penalty value.
        :rtype: float
        :authors: Agustina Acosta, Rocio Barbetta, Dylan Guitard Jimenez
        :date: 24/03/2026
        """

    @abstractmethod
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the penalty with respect to the weights.

        :param weights: Weight vector or matrix of the model.
        :type weights: np.ndarray
        :return: Gradient with the same shape as ``weights``.
        :rtype: np.ndarray
        :authors: Agustina Acosta, Rocio Barbetta, Dylan Guitard Jimenez
        :date: 24/03/2026
        """

    def _validate_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Convert and validate the weight array before operating on it.

        Accepts scalars, lists, and arrays; converts them to a float
        ''np.ndarray'', Rejects tensors with more than two dimensions.


        :param weights: Weights to validate.
        :type weights: np.ndarray
        :return: Validated array converted to float.
        :rtype: np.ndarray
        :raises ValueError: If ``weights`` has more than 2 dimensions.
        :authors: Agustina Acosta, Rocio Barbetta, Dylan Guitard Jimenez
        :date: 24/03/2026
        """
        weights = np.atleast_1d(np.asarray(weights, dtype=float))
        if weights.ndim > 2:
            raise ValueError(
                f"weights must be a vector or matrix (1D or 2D), "
                f"got shape {weights.shape}."
            )
        return weights


class L2Regularizer(Regularizer):
    """
    L2 regularization (Ridge).

    Penalizes the magnitude of all weights equally, shrinking them
    towards zero but without eliminating them. Well suited when many variables
    are expected to contribute to the model.

    .. math::

        \\mathcal{L}_{L2} = \\alpha \\sum w_i^2

        \\nabla_{L2} = 2\\alpha \\, w

    :param alpha: Penalty strength. Must be >= 0.
    :type alpha: float
    :authors: Agustina Acosta, Rocio Barbetta, Dylan Guitard Jimenez
    :date: 24/03/2026
    """

    def loss(self, weights: np.ndarray) -> float:
        """
        Compute the L2 penalty: ``alpha * sum(w²)``.

        :param weights: Weight vector or matrix of the model.
        :type weights: np.ndarray
        :return: Scalar L2 penalty value.
        :rtype: float
        :authors: Agustina Acosta, Rocio Barbetta, Dylan Guitard Jimenez
        :date: 24/03/2026
        """
        weights = self._validate_weights(weights)
        return float(self.alpha * np.sum(weights**2))

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Compute the L2 gradient: ``2 * alpha * w``.

        :param weights: Weight vector or matrix of the model.
        :type weights: np.ndarray
        :return: Gradient with the same shape as ``weights``.
        :rtype: np.ndarray
        :authors: Agustina Acosta, Rocio Barbetta, Dylan Guitard Jimenez
        :date: 24/03/2026
        """
        weights = self._validate_weights(weights)
        return 2 * self.alpha * weights

    def __repr__(self) -> str:
        return f"L2Regularizer(alpha={self.alpha})"


class L1Regularizer(Regularizer):
    """
    L1 regularization (Lasso).

    Pushes small weights towards exactly zero, producing sparse models.
    Performs implicit feature selection, making it especially useful in
    high-dimensional settings.

    .. math::

        \\mathcal{L}_{L1} = \\alpha \\sum |w_i|

        \\nabla_{L1} = \\alpha \\, \\text{sign}(w)

    :param alpha: Penalty strength. Must be >= 0.
    :type alpha: float
    :authors: Agustina Acosta, Rocio Barbetta, Dylan Guitard Jimenez
    :date: 24/03/2026
    """

    def loss(self, weights: np.ndarray) -> float:
        """
        Compute the L1 penalty: ``alpha * sum(|w|)``.

        :param weights: Weight vector or matrix of the model.
        :type weights: np.ndarray
        :return: Scalar L1 penalty value.
        :rtype: float
        :authors: Agustina Acosta, Rocio Barbetta, Dylan Guitard Jimenez
        :date: 24/03/2026
        """
        weights = self._validate_weights(weights)
        return float(self.alpha * np.sum(np.abs(weights)))

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Compute the L1 gradient: ``alpha * sign(w)``.

        :param weights: Weight vector or matrix of the model.
        :type weights: np.ndarray
        :return: Gradient with the same shape as ``weights``.
        :rtype: np.ndarray
        :authors: Agustina Acosta, Rocio Barbetta, Dylan Guitard Jimenez
        :date: 24/03/2026
        """
        weights = self._validate_weights(weights)
        return self.alpha * np.sign(weights)

    def __repr__(self) -> str:
        return f"L1Regularizer(alpha={self.alpha})"


class ElasticNetRegularizer(Regularizer):
    """
    ElasticNet regularization (combination of L1 and L2).

    Blends both penalties via ``l1_ratio``, enabling simultaneous
    feature selection (L1) and stability under multicollinearity (L2).

    .. math::

        \\mathcal{L} = \\alpha \\left(
            \\rho \\sum |w_i| + (1-\\rho) \\sum w_i^2
        \\right)

        \\nabla = \\alpha \\left(
            \\rho \\, \\text{sign}(w) + 2(1-\\rho) \\, w
        \\right)

    where :math:`\\rho` = ``l1_ratio``.

    :param alpha: Overall penalty strength. Must be >= 0.
    :type alpha: float
    :param l1_ratio: Balance between L1 and L2. ``0.0`` = pure L2,
        ``1.0`` = pure L1. Must be in [0.0, 1.0].
    :type l1_ratio: float
    :authors: Agustina Acosta, Rocio Barbetta, Dylan Guitard Jimenez
    :date: 24/03/2026

    """

    def __init__(self, alpha: float = 0.01, l1_ratio: float = 0.5) -> None:
        """
        Initialize ElasticNet with penalty strength and L1/L2 balance.

        :param alpha: Penalty scaling factor. Must be >= 0.
        :type alpha: float
        :param l1_ratio: L1 proportion in the mix. Must be in [0.0, 1.0].
        :type l1_ratio: float
        :raises ValueError: If ``alpha`` < 0 or ``l1_ratio`` outside [0, 1].
        :authors: Agustina Acosta, Rocio Barbetta, Dylan Guitard Jimenez
        :date: 24/03/2026
        """
        super().__init__(alpha)
        if not (0.0 <= l1_ratio <= 1.0):
            raise ValueError("l1_ratio must be between 0.0 and 1.0.")
        self.l1_ratio = l1_ratio

    def loss(self, weights: np.ndarray) -> float:
        """
        Compute the ElasticNet penalty combining L1 and L2 terms.

        :param weights: Weight vector or matrix of the model.
        :type weights: np.ndarray
        :return: Scalar ElasticNet penalty value.
        :rtype: float
        :authors: Agustina Acosta, Rocio Barbetta, Dylan Guitard Jimenez
        :date: 24/03/2026
        """
        weights = self._validate_weights(weights)
        l1 = self.l1_ratio * np.sum(np.abs(weights))
        l2 = (1 - self.l1_ratio) * np.sum(weights**2)
        return float(self.alpha * (l1 + l2))

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Compute the ElasticNet gradient combining L1 and L2 gradients.

        :param weights: Weight vector or matrix of the model.
        :type weights: np.ndarray
        :return: Gradient with the same shape as ``weights``.
        :rtype: np.ndarray
        :authors: Agustina Acosta, Rocio Barbetta, Dylan Guitard Jimenez
        :date: 24/03/2026
        """
        weights = self._validate_weights(weights)
        l1 = self.l1_ratio * np.sign(weights)
        l2 = 2 * (1 - self.l1_ratio) * weights
        return self.alpha * (l1 + l2)

    def __repr__(self) -> str:
        return f"ElasticNetRegularizer(alpha={self.alpha}, l1_ratio={self.l1_ratio})"
