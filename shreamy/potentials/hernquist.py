"""
Hernquist profile potential.
"""

import numpy as np

from .base import AnalyticPotential


class HernquistPotential(AnalyticPotential):
    """
    Hernquist profile potential.

    Phi(r) = -G * M / (r + a)

    Parameters
    ----------
    M : float
        Total mass.
    a : float
        Scale radius.
    G : float, default 1.0
        Gravitational constant.
    """

    def __init__(self, M: float, a: float, G: float = 1.0):
        """Initialize Hernquist potential."""
        self._M = M
        self._a = a
        self._G = G

    def acceleration(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """Compute Hernquist acceleration."""
        raise NotImplementedError

    def potential_value(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """Compute Hernquist potential."""
        raise NotImplementedError

    @property
    def M(self) -> float:
        """Total mass."""
        return self._M

    @property
    def a(self) -> float:
        """Scale radius."""
        return self._a

    @property
    def G(self) -> float:
        """Gravitational constant."""
        return self._G
