"""
Base class for gravity solvers.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class GravitySolver(ABC):
    """
    Abstract base class for gravity solvers.

    A gravity solver computes the gravitational accelerations on particles
    due to other particles. Different implementations provide different
    trade-offs between speed and accuracy.

    Parameters
    ----------
    softening : float, optional
        Gravitational softening length to prevent singularities.
        If None, no softening is applied.
    G : float, default 1.0
        Gravitational constant in the chosen unit system.
    """

    def __init__(
        self,
        softening: Optional[float] = None,
        G: float = 1.0,
    ):
        """Initialize the gravity solver."""
        self._softening = softening if softening is not None else 0.0
        self._G = G

    @abstractmethod
    def accelerations(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
    ) -> np.ndarray:
        """
        Compute gravitational accelerations on all particles.

        Parameters
        ----------
        positions : ndarray
            Particle positions of shape (N, 3).
        masses : ndarray
            Particle masses of shape (N,).

        Returns
        -------
        ndarray
            Accelerations of shape (N, 3).
        """
        pass

    @abstractmethod
    def potential(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
    ) -> np.ndarray:
        """
        Compute gravitational potential at each particle position.

        Parameters
        ----------
        positions : ndarray
            Particle positions of shape (N, 3).
        masses : ndarray
            Particle masses of shape (N,).

        Returns
        -------
        ndarray
            Potential values of shape (N,).
        """
        pass

    def potential_energy(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
    ) -> float:
        """
        Compute total gravitational potential energy of the system.

        Returns
        -------
        float
            Total potential energy (negative).
        """
        raise NotImplementedError

    @property
    def softening(self) -> float:
        """Gravitational softening length."""
        return self._softening

    @softening.setter
    def softening(self, value: float) -> None:
        """Set the softening length."""
        self._softening = value

    @property
    def G(self) -> float:
        """Gravitational constant."""
        return self._G
