"""
Utility functions for satellite models.
"""

import numpy as np
from typing import Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..particle import ParticleSet, ParticleHistory


def rejection_sample_spherical(
    density_func,
    n_particles: int,
    r_max: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample positions from a spherical density profile using rejection sampling.

    Parameters
    ----------
    density_func : callable
        Function density(r) that returns the density at radius r.
    n_particles : int
        Number of particles to sample.
    r_max : float
        Maximum radius for sampling.
    rng : np.random.Generator, optional
        Random number generator. If None, uses default.

    Returns
    -------
    positions : ndarray, shape (n_particles, 3)
        Sampled 3D positions.

    Notes
    -----
    This uses rejection sampling in spherical coordinates with a
    proposal distribution uniform in r^3 (to account for volume element).
    """
    raise NotImplementedError


def sample_isotropic_velocities(
    positions: np.ndarray,
    velocity_dispersion_func,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample isotropic velocities given positions and velocity dispersion profile.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Particle positions.
    velocity_dispersion_func : callable
        Function sigma(r) that returns velocity dispersion at radius r.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    velocities : ndarray, shape (N, 3)
        Sampled velocities.
    """
    raise NotImplementedError


def hernquist_density(r: np.ndarray, M: float, a: float) -> np.ndarray:
    """
    Hernquist density profile.

    Parameters
    ----------
    r : array-like
        Radii.
    M : float
        Total mass.
    a : float
        Scale radius.

    Returns
    -------
    rho : array-like
        Density at each radius.
    """
    return M * a / (2 * np.pi * r * (r + a) ** 3)


def plummer_density(r: np.ndarray, M: float, b: float) -> np.ndarray:
    """
    Plummer density profile.

    Parameters
    ----------
    r : array-like
        Radii.
    M : float
        Total mass.
    b : float
        Plummer scale length.

    Returns
    -------
    rho : array-like
        Density at each radius.
    """
    return 3 * M / (4 * np.pi * b ** 3) * (1 + (r / b) ** 2) ** (-2.5)


def nfw_density(r: np.ndarray, M_s: float, r_s: float) -> np.ndarray:
    """
    NFW density profile.

    Parameters
    ----------
    r : array-like
        Radii.
    M_s : float
        Scale mass.
    r_s : float
        Scale radius.

    Returns
    -------
    rho : array-like
        Density at each radius.
    """
    x = r / r_s
    return M_s / (4 * np.pi * r_s ** 3) / (x * (1 + x) ** 2)


def king_w0_to_concentration(W0: float) -> float:
    """
    Convert King model central potential parameter to concentration.

    Parameters
    ----------
    W0 : float
        Dimensionless central potential.

    Returns
    -------
    c : float
        Concentration r_t / r_c.
    """
    raise NotImplementedError


def compute_escape_velocity(
    r: np.ndarray,
    potential_func,
) -> np.ndarray:
    """
    Compute escape velocity from a potential.

    Parameters
    ----------
    r : array-like
        Radii.
    potential_func : callable
        Function potential(r) returning potential at r.

    Returns
    -------
    v_esc : array-like
        Escape velocity at each radius.
    """
    # v_esc = sqrt(-2 * phi) for phi < 0
    return np.sqrt(-2 * potential_func(r))


def spherical_to_cartesian(
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """
    Convert spherical to Cartesian coordinates.

    Parameters
    ----------
    r : array-like
        Radial distances.
    theta : array-like
        Polar angle (0 to pi).
    phi : array-like
        Azimuthal angle (0 to 2*pi).

    Returns
    -------
    xyz : ndarray, shape (..., 3)
        Cartesian coordinates.
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=-1)


def add_bulk_motion(
    positions: np.ndarray,
    velocities: np.ndarray,
    center_position: np.ndarray,
    center_velocity: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add bulk position and velocity to particle distribution.

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Particle positions relative to center.
    velocities : ndarray, shape (N, 3)
        Particle velocities relative to center.
    center_position : ndarray, shape (3,)
        Center position to add.
    center_velocity : ndarray, shape (3,)
        Center velocity to add.

    Returns
    -------
    new_positions : ndarray, shape (N, 3)
        Positions in external frame.
    new_velocities : ndarray, shape (N, 3)
        Velocities in external frame.
    """
    return positions + center_position, velocities + center_velocity


# =============================================================================
# Virialization Utilities
# =============================================================================


def virialize_particles(
    particles: "ParticleSet",
    potential_func=None,
    t_max: Optional[float] = None,
    dt: Optional[float] = None,
    n_dynamical_times: float = 3.0,
    return_history: bool = False,
) -> Union["ParticleSet", Tuple["ParticleSet", "ParticleHistory"]]:
    """
    Virialize a particle system by running a short N-body integration in isolation.

    This function runs the particles with self-gravity only (no external
    potential) to allow them to settle into dynamical equilibrium.

    Parameters
    ----------
    particles : ParticleSet
        Initial particle distribution to virialize.
    potential_func : callable, optional
        Function potential(r) for the satellite's own potential. Used to
        estimate the dynamical time if t_max is not provided.
    t_max : float, optional
        Total integration time. If None, estimated from n_dynamical_times
        and the dynamical time of the system.
    dt : float, optional
        Integration time step. If None, automatically determined.
    n_dynamical_times : float, default 3.0
        Number of dynamical times to integrate for (used if t_max is None).
    return_history : bool, default False
        If True, return full time evolution as ParticleHistory.

    Returns
    -------
    ParticleSet or tuple
        If return_history=False: Virialized particles.
        If return_history=True: (ParticleSet, ParticleHistory).

    Notes
    -----
    The integration is performed with self-gravity only, using direct
    summation for small N or tree code for larger N. The satellite's
    center of mass is held fixed at the origin.

    Examples
    --------
    >>> particles = custom_satellite.sample(10000)
    >>> virialized = virialize_particles(particles, t_max=5.0)
    >>>
    >>> # With history for diagnostics
    >>> virialized, history = virialize_particles(
    ...     particles, t_max=5.0, return_history=True
    ... )
    >>> # Check virial ratio evolution
    >>> for t in history.times:
    ...     ratio = compute_virial_ratio(history.at_time(t))
    ...     print(f"t={t:.2f}: 2K/|W| = {ratio:.3f}")
    """
    raise NotImplementedError


def compute_virial_ratio(particles: "ParticleSet", softening: float = 0.0) -> float:
    """
    Compute the virial ratio 2K/|W| for a particle system.

    For a virialized system in equilibrium, this should be approximately 1.0.

    Parameters
    ----------
    particles : ParticleSet
        Particle system to analyze.
    softening : float, default 0.0
        Gravitational softening length for potential energy calculation.

    Returns
    -------
    float
        Virial ratio 2K/|W|.

    Notes
    -----
    K = total kinetic energy = 0.5 * sum(m_i * |v_i|^2)
    W = total potential energy = -0.5 * sum_i sum_{j>i} G*m_i*m_j / r_ij

    For virial equilibrium: 2K + W = 0, so 2K/|W| = 1.

    Values < 1 indicate the system is too cold (will collapse).
    Values > 1 indicate the system is too hot (will expand).
    """
    raise NotImplementedError


def compute_kinetic_energy(particles: "ParticleSet") -> float:
    """
    Compute the total kinetic energy of a particle system.

    Parameters
    ----------
    particles : ParticleSet
        Particle system.

    Returns
    -------
    float
        Total kinetic energy K = 0.5 * sum(m * v^2).
    """
    raise NotImplementedError


def compute_potential_energy(
    particles: "ParticleSet",
    softening: float = 0.0,
) -> float:
    """
    Compute the total gravitational potential energy of a particle system.

    Parameters
    ----------
    particles : ParticleSet
        Particle system.
    softening : float, default 0.0
        Gravitational softening length.

    Returns
    -------
    float
        Total potential energy W = -0.5 * sum_i sum_{j!=i} G*m_i*m_j / r_ij.
    """
    raise NotImplementedError


def estimate_dynamical_time(particles: "ParticleSet") -> float:
    """
    Estimate the dynamical time of a particle system.

    The dynamical time is estimated as t_dyn ~ sqrt(r_h^3 / (G * M))
    where r_h is the half-mass radius and M is the total mass.

    Parameters
    ----------
    particles : ParticleSet
        Particle system.

    Returns
    -------
    float
        Estimated dynamical time in natural units.
    """
    raise NotImplementedError


def compute_half_mass_radius(particles: "ParticleSet") -> float:
    """
    Compute the half-mass radius of a particle system.

    Parameters
    ----------
    particles : ParticleSet
        Particle system.

    Returns
    -------
    float
        Half-mass radius (radius containing 50% of the mass).
    """
    raise NotImplementedError
