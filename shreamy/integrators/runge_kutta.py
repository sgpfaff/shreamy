"""
Fourth-order Runge-Kutta integrator.
"""

import numpy as np
from typing import Tuple

from .base import Integrator


class RungeKutta4(Integrator):
    """
    Fourth-order Runge-Kutta integrator.

    A classic fourth-order integrator with good accuracy but not
    symplectic. Energy will drift over long integrations.

    Best for: short integrations where high accuracy is needed,
    or when potential is explicitly time-dependent.

    The RK4 scheme:
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt*k1/2)
        k3 = f(t + dt/2, y + dt*k2/2)
        k4 = f(t + dt, y + dt*k3)
        y_{n+1} = y_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    """

    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        t: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advance using RK4 scheme."""
        raise NotImplementedError

    @property
    def order(self) -> int:
        return 4

    @property
    def is_symplectic(self) -> bool:
        return False
