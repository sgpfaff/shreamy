"""
Utility functions for gravity calculations.
"""

import numpy as np
from typing import Optional

from .base import GravitySolver
from .direct import DirectSummation
from .tree import BarnesHut
from .null import NoGravity


def get_gravity_solver(
    method: str = "auto",
    n_particles: Optional[int] = None,
    **kwargs,
) -> GravitySolver:
    """
    Factory function to create a gravity solver.

    Parameters
    ----------
    method : str, default 'auto'
        Solver method:
        - 'auto' : Automatically choose based on n_particles
        - 'direct' : Direct N^2 summation
        - 'tree' or 'barnes-hut' : Barnes-Hut tree code
        - 'none' : No self-gravity
    n_particles : int, optional
        Number of particles (used for 'auto' selection).
    **kwargs
        Additional arguments passed to the solver constructor.

    Returns
    -------
    GravitySolver
    """
    method = method.lower()

    if method == "none":
        return NoGravity(**kwargs)
    elif method == "direct":
        return DirectSummation(**kwargs)
    elif method in ("tree", "barnes-hut", "barneshut"):
        return BarnesHut(**kwargs)
    elif method == "auto":
        if n_particles is None:
            # Default to direct if we don't know N
            return DirectSummation(**kwargs)
        elif n_particles < 1000:
            return DirectSummation(**kwargs)
        else:
            return BarnesHut(**kwargs)
    else:
        raise ValueError(f"Unknown gravity solver method: {method}")


def estimate_softening(
    positions: np.ndarray,
    masses: np.ndarray,
    method: str = "mean_interparticle",
) -> float:
    """
    Estimate an appropriate softening length from particle data.

    Parameters
    ----------
    positions : ndarray
        Particle positions of shape (N, 3).
    masses : ndarray
        Particle masses of shape (N,).
    method : str, default 'mean_interparticle'
        Method for estimating softening:
        - 'mean_interparticle' : Based on mean inter-particle separation
        - 'half_mass' : Based on half-mass radius

    Returns
    -------
    float
        Recommended softening length.
    """
    raise NotImplementedError


def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distance matrix between particles.

    Parameters
    ----------
    positions : ndarray
        Particle positions of shape (N, 3).

    Returns
    -------
    ndarray
        Distance matrix of shape (N, N).
    """
    raise NotImplementedError


def pairwise_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    softening: float = 0.0,
    G: float = 1.0,
) -> np.ndarray:
    """
    Compute pairwise gravitational accelerations (vectorized).

    Parameters
    ----------
    positions : ndarray
        Particle positions of shape (N, 3).
    masses : ndarray
        Particle masses of shape (N,).
    softening : float
        Softening length.
    G : float
        Gravitational constant.

    Returns
    -------
    ndarray
        Accelerations of shape (N, 3).
    """
    raise NotImplementedError
