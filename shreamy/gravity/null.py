"""
Null gravity solver for test particle limit.
"""

import numpy as np

from .base import GravitySolver


class NoGravity(GravitySolver):
    """
    Null gravity solver that returns zero accelerations.

    Useful for test particle simulations where particles only feel
    an external potential.
    """

    def __init__(self, **kwargs):
        """Initialize null gravity solver."""
        super().__init__(softening=0.0, G=1.0)

    def accelerations(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
    ) -> np.ndarray:
        """Return zero accelerations."""
        return np.zeros_like(positions)

    def potential(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
    ) -> np.ndarray:
        """Return zero potential."""
        return np.zeros(len(positions))

    def potential_energy(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
    ) -> float:
        """Return zero potential energy."""
        return 0.0
