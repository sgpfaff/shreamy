"""
Utility functions for integrators.
"""

import numpy as np
from typing import Optional, Callable

from .base import Integrator
from .leapfrog import Leapfrog, LeapfrogDKD
from .runge_kutta import RungeKutta4
from .hermite import Hermite
from .yoshida import Yoshida4


def get_integrator(
    method: str,
    acceleration_func: Callable,
    **kwargs,
) -> Integrator:
    """
    Factory function to create an integrator.

    Parameters
    ----------
    method : str
        Integrator method:
        - 'leapfrog' or 'kdk' : Leapfrog kick-drift-kick
        - 'dkd' : Leapfrog drift-kick-drift
        - 'rk4' : 4th order Runge-Kutta
        - 'hermite' : 4th order Hermite
        - 'yoshida4' : 4th order Yoshida
    acceleration_func : callable
        Function to compute accelerations.
    **kwargs
        Additional arguments passed to the integrator constructor.

    Returns
    -------
    Integrator
    """
    method = method.lower()

    if method in ("leapfrog", "kdk"):
        return Leapfrog(acceleration_func, **kwargs)
    elif method == "dkd":
        return LeapfrogDKD(acceleration_func, **kwargs)
    elif method == "rk4":
        return RungeKutta4(acceleration_func, **kwargs)
    elif method == "hermite":
        return Hermite(acceleration_func, **kwargs)
    elif method == "yoshida4":
        return Yoshida4(acceleration_func, **kwargs)
    else:
        raise ValueError(f"Unknown integrator method: {method}")


def estimate_timestep(
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    eta: float = 0.01,
    method: str = "aarseth",
) -> float:
    """
    Estimate an appropriate time step from the current state.

    Parameters
    ----------
    positions : ndarray
        Particle positions.
    velocities : ndarray
        Particle velocities.
    accelerations : ndarray
        Particle accelerations.
    eta : float, default 0.01
        Accuracy parameter (smaller = more accurate).
    method : str, default 'aarseth'
        Method for estimating time step:
        - 'aarseth' : Based on Aarseth criterion (a/|a_dot|)
        - 'dynamical' : Based on dynamical time
        - 'courant' : Courant-like condition

    Returns
    -------
    float
        Recommended time step.
    """
    raise NotImplementedError


def adaptive_timestep(
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    dt_current: float,
    eta: float = 0.01,
    dt_min: Optional[float] = None,
    dt_max: Optional[float] = None,
) -> float:
    """
    Compute an adaptive time step based on current state.

    Parameters
    ----------
    positions : ndarray
        Particle positions.
    velocities : ndarray
        Particle velocities.
    accelerations : ndarray
        Particle accelerations.
    dt_current : float
        Current time step.
    eta : float, default 0.01
        Accuracy parameter.
    dt_min : float, optional
        Minimum allowed time step.
    dt_max : float, optional
        Maximum allowed time step.

    Returns
    -------
    float
        New time step.
    """
    raise NotImplementedError
