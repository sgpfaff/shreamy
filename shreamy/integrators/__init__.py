"""
Integrators subpackage for shreamy.

This subpackage provides numerical integration schemes for the N-body problem.
Different integrators offer various trade-offs between speed, accuracy,
and energy conservation.

Classes
-------
Integrator
    Abstract base class for integrators.
Leapfrog
    Second-order symplectic integrator (KDK).
LeapfrogDKD
    Leapfrog with drift-kick-drift ordering.
RungeKutta4
    Fourth-order Runge-Kutta integrator.
Hermite
    Fourth-order Hermite predictor-corrector.
Yoshida4
    Fourth-order symplectic Yoshida integrator.
"""

from .base import Integrator
from .leapfrog import Leapfrog, LeapfrogDKD
from .runge_kutta import RungeKutta4
from .hermite import Hermite
from .yoshida import Yoshida4
from .utils import (
    get_integrator,
    estimate_timestep,
    adaptive_timestep,
)
from .block_timesteps import BlockTimeSteps

__all__ = [
    "Integrator",
    "Leapfrog",
    "LeapfrogDKD",
    "RungeKutta4",
    "Hermite",
    "Yoshida4",
    "get_integrator",
    "estimate_timestep",
    "adaptive_timestep",
    "BlockTimeSteps",
]
