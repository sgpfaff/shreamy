"""
Plummer sphere potential.
"""

import numpy as np

from .base import AnalyticPotential


class PlummerPotential(AnalyticPotential):
    """
    Plummer sphere potential.

    Phi(r) = -G * M / sqrt(r^2 + b^2)

    Parameters
    ----------
    M : float
        Total mass.
    b : float
        Scale radius.
    G : float, default 1.0
        Gravitational constant.
    """

    def __init__(self, M: float, b: float, G: float = 1.0):
        """Initialize Plummer potential."""
        self._M = M
        self._b = b
        self._G = G

    def acceleration(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """Compute Plummer acceleration."""
        raise NotImplementedError

    def potential_value(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """Compute Plummer potential."""
        raise NotImplementedError

    @property
    def M(self) -> float:
        """Total mass."""
        return self._M

    @property
    def b(self) -> float:
        """Scale radius."""
        return self._b

    @property
    def G(self) -> float:
        """Gravitational constant."""
        return self._G
