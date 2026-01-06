"""
Yoshida symplectic integrators.
"""

import numpy as np
from typing import Tuple

from .base import Integrator


class Yoshida4(Integrator):
    """
    Fourth-order Yoshida symplectic integrator.

    A higher-order symplectic integrator constructed from leapfrog
    steps. Better energy conservation than leapfrog for the same
    total number of force evaluations.

    Best for: long integrations where energy conservation is critical.

    The 4th order Yoshida scheme uses coefficients:
        w0 = -2^(1/3) / (2 - 2^(1/3))
        w1 = 1 / (2 - 2^(1/3))

    And performs a sequence of leapfrog steps with these weights.

    References
    ----------
    Yoshida, H. (1990). "Construction of higher order symplectic integrators"
    Physics Letters A, 150(5-7), 262-268.
    """

    # Yoshida coefficients for 4th order
    _c1 = 1.0 / (2.0 * (2.0 - 2.0 ** (1.0 / 3.0)))
    _c4 = _c1
    _c2 = (1.0 - 2.0 ** (1.0 / 3.0)) / (2.0 * (2.0 - 2.0 ** (1.0 / 3.0)))
    _c3 = _c2

    _d1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    _d3 = _d1
    _d2 = -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0))

    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        t: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance using 4th order Yoshida scheme.

        Sequence: c1, d1, c2, d2, c3, d3, c4
        where c = drift coefficient, d = kick coefficient
        """
        raise NotImplementedError

    @property
    def order(self) -> int:
        return 4

    @property
    def is_symplectic(self) -> bool:
        return True


class Yoshida6(Integrator):
    """
    Sixth-order Yoshida symplectic integrator.

    Even higher order than Yoshida4, with 7 force evaluations per step.
    Use when very high energy conservation is needed.
    """

    # 6th order coefficients (solution A from Yoshida 1990)
    _w1 = 0.78451361047755726382
    _w2 = 0.23557321335935813368
    _w3 = -1.17767998417887100695
    _w4 = 1.31518632068391121889
    _w5 = _w3
    _w6 = _w2
    _w7 = _w1

    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        t: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advance using 6th order Yoshida scheme."""
        raise NotImplementedError

    @property
    def order(self) -> int:
        return 6

    @property
    def is_symplectic(self) -> bool:
        return True
