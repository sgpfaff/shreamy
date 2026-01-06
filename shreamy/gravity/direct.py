"""
Direct N^2 summation gravity solver.
"""

import numpy as np
from typing import Optional

from .base import GravitySolver


class DirectSummation(GravitySolver):
    """
    Direct N^2 summation gravity solver.

    Computes gravitational forces by directly summing over all particle
    pairs. This is O(N^2) in complexity but exact (up to softening).

    Best for small N (< 1000) or when high accuracy is needed.

    Parameters
    ----------
    softening : float, optional
        Gravitational softening length. Plummer softening is used:
        a -> a / (r^2 + softening^2)^(3/2)
    G : float, default 1.0
        Gravitational constant.
    use_vectorized : bool, default True
        Use vectorized numpy operations. Faster for N > ~100.
    """

    def __init__(
        self,
        softening: Optional[float] = None,
        G: float = 1.0,
        use_vectorized: bool = True,
    ):
        """Initialize direct summation solver."""
        super().__init__(softening=softening, G=G)
        self._use_vectorized = use_vectorized

    def accelerations(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
    ) -> np.ndarray:
        """
        Compute accelerations via direct summation.

        Uses Plummer softening:
        a_i = -G * sum_j m_j * (r_i - r_j) / (|r_i - r_j|^2 + eps^2)^(3/2)
        """
        raise NotImplementedError

    def potential(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
    ) -> np.ndarray:
        """Compute potential via direct summation."""
        raise NotImplementedError

    def potential_energy(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
    ) -> float:
        """
        Compute total gravitational potential energy.

        W = -G * sum_{i<j} m_i * m_j / |r_i - r_j|
        """
        raise NotImplementedError

    @property
    def use_vectorized(self) -> bool:
        """Whether vectorized computation is enabled."""
        return self._use_vectorized
