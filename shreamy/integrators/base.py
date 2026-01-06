"""
Base class for numerical integrators.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Tuple


class Integrator(ABC):
    """
    Abstract base class for N-body integrators.

    An integrator advances the particle system from time t to t + dt,
    given an acceleration function that computes a = a(x, v, t).

    Parameters
    ----------
    acceleration_func : callable
        Function that computes accelerations given (positions, velocities, t).
        Signature: acceleration_func(pos, vel, t) -> ndarray of shape (N, 3).
    """

    def __init__(self, acceleration_func: Callable):
        """Initialize the integrator with an acceleration function."""
        self._acceleration_func = acceleration_func

    @abstractmethod
    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        t: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance the system by one time step.

        Parameters
        ----------
        positions : ndarray
            Current positions of shape (N, 3).
        velocities : ndarray
            Current velocities of shape (N, 3).
        t : float
            Current time.
        dt : float
            Time step.

        Returns
        -------
        new_positions : ndarray
            Updated positions of shape (N, 3).
        new_velocities : ndarray
            Updated velocities of shape (N, 3).
        """
        pass

    def integrate(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        times: np.ndarray,
        save_every: int = 1,
        progressbar: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Integrate the system over multiple time steps.

        Parameters
        ----------
        positions : ndarray
            Initial positions of shape (N, 3).
        velocities : ndarray
            Initial velocities of shape (N, 3).
        times : ndarray
            Array of times at which to evaluate. Must be monotonic.
        save_every : int, default 1
            Save every N-th step.
        progressbar : bool, default True
            Show progress bar.

        Returns
        -------
        saved_times : ndarray
            Times at which state was saved.
        saved_positions : ndarray
            Positions at each saved time, shape (n_saves, N, 3).
        saved_velocities : ndarray
            Velocities at each saved time, shape (n_saves, N, 3).
        """
        raise NotImplementedError

    @property
    def acceleration_func(self) -> Callable:
        """The acceleration function."""
        return self._acceleration_func

    @acceleration_func.setter
    def acceleration_func(self, func: Callable) -> None:
        """Set a new acceleration function."""
        self._acceleration_func = func

    @property
    @abstractmethod
    def order(self) -> int:
        """Order of accuracy of the integrator."""
        pass

    @property
    @abstractmethod
    def is_symplectic(self) -> bool:
        """Whether the integrator is symplectic (energy-conserving)."""
        pass
