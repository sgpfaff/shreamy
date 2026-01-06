"""
Leapfrog integrators.
"""

import numpy as np
from typing import Tuple, Callable

from .base import Integrator


class Leapfrog(Integrator):
    """
    Leapfrog (kick-drift-kick) integrator.

    A second-order symplectic integrator that is well-suited for
    gravitational N-body simulations. Conserves energy to good
    precision over long timescales.

    This uses the velocity Verlet formulation:
        v_{n+1/2} = v_n + (dt/2) * a_n
        x_{n+1} = x_n + dt * v_{n+1/2}
        v_{n+1} = v_{n+1/2} + (dt/2) * a_{n+1}
    """

    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        t: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advance using leapfrog KDK scheme."""
        raise NotImplementedError

    @property
    def order(self) -> int:
        return 2

    @property
    def is_symplectic(self) -> bool:
        return True


class LeapfrogDKD(Integrator):
    """
    Leapfrog (drift-kick-drift) integrator.

    Alternative formulation of leapfrog that drifts positions first.
    Equivalent to KDK for constant time steps but can differ for
    adaptive stepping.

        x_{n+1/2} = x_n + (dt/2) * v_n
        v_{n+1} = v_n + dt * a_{n+1/2}
        x_{n+1} = x_{n+1/2} + (dt/2) * v_{n+1}
    """

    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        t: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advance using DKD leapfrog scheme."""
        raise NotImplementedError

    @property
    def order(self) -> int:
        return 2

    @property
    def is_symplectic(self) -> bool:
        return True
