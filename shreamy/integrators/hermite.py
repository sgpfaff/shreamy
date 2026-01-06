"""
Fourth-order Hermite integrator.
"""

import numpy as np
from typing import Tuple, Callable, Optional

from .base import Integrator


class Hermite(Integrator):
    """
    Fourth-order Hermite integrator with predictor-corrector scheme.

    A high-order integrator commonly used in direct N-body codes.
    Requires computation of both acceleration and its time derivative (jerk).

    Best for: high-accuracy simulations, especially with close encounters.

    Parameters
    ----------
    acceleration_func : callable
        Function that computes accelerations.
    jerk_func : callable, optional
        Function that computes time derivative of acceleration.
        If None, jerk is estimated numerically.
    n_iterations : int, default 1
        Number of corrector iterations.
    """

    def __init__(
        self,
        acceleration_func: Callable,
        jerk_func: Optional[Callable] = None,
        n_iterations: int = 1,
    ):
        """Initialize Hermite integrator."""
        super().__init__(acceleration_func)
        self._jerk_func = jerk_func
        self._n_iterations = n_iterations

    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        t: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance using Hermite predictor-corrector scheme.

        Predictor step:
            x_p = x + v*dt + (1/2)*a*dt^2 + (1/6)*j*dt^3
            v_p = v + a*dt + (1/2)*j*dt^2

        Corrector step:
            x_c = x + (1/2)*(v + v_p)*dt + (1/12)*(a - a_p)*dt^2
            v_c = v + (1/2)*(a + a_p)*dt + (1/12)*(j - j_p)*dt^2
        """
        raise NotImplementedError

    def _compute_jerk(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Compute jerk (time derivative of acceleration).

        If jerk_func is provided, use it. Otherwise, estimate numerically.
        """
        raise NotImplementedError

    @property
    def jerk_func(self) -> Optional[Callable]:
        """The jerk function."""
        return self._jerk_func

    @property
    def n_iterations(self) -> int:
        """Number of corrector iterations."""
        return self._n_iterations

    @property
    def order(self) -> int:
        return 4

    @property
    def is_symplectic(self) -> bool:
        return False
