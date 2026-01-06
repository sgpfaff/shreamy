"""
Time-dependent potential wrapper.
"""

import numpy as np
from typing import Callable

from .base import HostPotential


class TimeDependentPotential(HostPotential):
    """
    A time-dependent potential wrapper.

    Allows for potentials that evolve over time, such as a growing
    or adiabatically contracting halo.

    Parameters
    ----------
    potential : HostPotential
        The base potential.
    time_evolution : callable
        Function that returns a scaling factor as a function of time.
        Signature: time_evolution(t) -> float
    """

    def __init__(
        self,
        potential: HostPotential,
        time_evolution: Callable[[float], float],
    ):
        """Initialize time-dependent potential."""
        self._potential = potential
        self._time_evolution = time_evolution

    def acceleration(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """Compute time-scaled acceleration."""
        raise NotImplementedError

    def potential_value(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """Compute time-scaled potential."""
        raise NotImplementedError

    @property
    def base_potential(self) -> HostPotential:
        """The underlying static potential."""
        return self._potential

    def scale_factor(self, t: float) -> float:
        """Get the scaling factor at time t."""
        return self._time_evolution(t)
