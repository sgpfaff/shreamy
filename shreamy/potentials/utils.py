"""
Utility functions for potentials.
"""

import numpy as np

from .base import HostPotential


def dynamical_time(
    potential: HostPotential,
    r: float,
) -> float:
    """
    Compute the dynamical time at a given radius.

    t_dyn = sqrt(r^3 / (G * M(r)))

    Parameters
    ----------
    potential : HostPotential
        The gravitational potential.
    r : float
        Radius at which to compute dynamical time.

    Returns
    -------
    float
        Dynamical time.
    """
    raise NotImplementedError


def tidal_radius(
    satellite_mass: float,
    satellite_position: np.ndarray,
    host_potential: HostPotential,
) -> float:
    """
    Compute the tidal radius of a satellite in the host potential.

    Uses the Jacobi approximation:
    r_t = (M_sat / (3 * M_host(<r)))^(1/3) * r

    Parameters
    ----------
    satellite_mass : float
        Mass of the satellite.
    satellite_position : ndarray
        Position of satellite center [x, y, z].
    host_potential : HostPotential
        The host galaxy potential.

    Returns
    -------
    float
        Tidal radius.
    """
    raise NotImplementedError


def enclosed_mass(
    potential: HostPotential,
    r: float,
) -> float:
    """
    Estimate enclosed mass at radius r from the potential.

    Uses M(<r) = r * v_c^2 / G

    Parameters
    ----------
    potential : HostPotential
    r : float
        Radius.

    Returns
    -------
    float
        Enclosed mass.
    """
    raise NotImplementedError
