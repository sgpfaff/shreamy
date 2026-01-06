"""
Base class for satellite galaxy models.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..particle import ParticleSet, ParticleHistory


class SatelliteModel(ABC):
    """
    Abstract base class for satellite galaxy models.

    A SatelliteModel defines the mass distribution and velocity structure
    of a satellite galaxy. Particles can be sampled from this model using
    the sample() method.

    Parameters
    ----------
    mass : float
        Total mass of the satellite.
    position : array-like, optional
        Initial position of satellite center [x, y, z].
    velocity : array-like, optional
        Initial velocity of satellite center [vx, vy, vz].
    """

    def __init__(
        self,
        mass: float,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
    ):
        """Initialize the satellite model."""
        self._mass = mass
        self._position = np.asarray(position) if position is not None else np.zeros(3)
        self._velocity = np.asarray(velocity) if velocity is not None else np.zeros(3)

    @abstractmethod
    def sample(
        self,
        n_particles: int,
        seed: Optional[int] = None,
        virialize: bool = False,
        virialize_time: Optional[float] = None,
        virialize_dt: Optional[float] = None,
        return_virialize_info: bool = False,
    ) -> Union["ParticleSet", Tuple["ParticleSet", "ParticleHistory"]]:
        """
        Sample particles from the distribution function.

        Parameters
        ----------
        n_particles : int
            Number of particles to sample.
        seed : int, optional
            Random seed for reproducibility.
        virialize : bool, default False
            If True, run a short N-body integration in isolation (no external
            potential) to allow the satellite to settle into dynamical
            equilibrium before returning particles. This is useful for
            custom satellites or when the velocity sampling may not be
            perfectly self-consistent.
        virialize_time : float, optional
            Duration of the virialization integration in dynamical times.
            Default is ~2-3 dynamical times if not specified.
        virialize_dt : float, optional
            Time step for virialization integration. If None, automatically
            determined from the dynamical time.
        return_virialize_info : bool, default False
            If True and virialize=True, return a tuple of (ParticleSet,
            ParticleHistory) where ParticleHistory contains the full time
            evolution during virialization. Useful for verifying that
            virialization was successful.

        Returns
        -------
        ParticleSet or tuple
            If return_virialize_info=False (default): Sampled particles.
            If return_virialize_info=True: Tuple of (ParticleSet, ParticleHistory).

        Examples
        --------
        >>> satellite = PlummerSatellite(mass=1e9, b=1.0)
        >>> # Simple sampling (velocities from analytic DF)
        >>> particles = satellite.sample(10000)
        >>>
        >>> # With virialization
        >>> particles = satellite.sample(10000, virialize=True)
        >>>
        >>> # With virialization and diagnostics
        >>> particles, history = satellite.sample(
        ...     10000, virialize=True, return_virialize_info=True
        ... )
        >>> # Check virial ratio over time
        >>> for t in history.times:
        ...     snap = history.at_time(t)
        ...     print(f"t={t:.2f}: 2K/|W| = {compute_virial_ratio(snap):.3f}")
        """
        pass

    @abstractmethod
    def density(self, r: np.ndarray) -> np.ndarray:
        """
        Compute the density at given radii.

        Parameters
        ----------
        r : ndarray
            Radii at which to compute density.

        Returns
        -------
        ndarray
            Density values.
        """
        pass

    @abstractmethod
    def potential(self, r: np.ndarray) -> np.ndarray:
        """
        Compute the gravitational potential at given radii.

        Parameters
        ----------
        r : ndarray
            Radii at which to compute potential.

        Returns
        -------
        ndarray
            Potential values.
        """
        pass

    @property
    def mass(self) -> float:
        """Total mass of the satellite."""
        return self._mass

    @property
    def position(self) -> np.ndarray:
        """Position of the satellite center."""
        return self._position

    @position.setter
    def position(self, value: np.ndarray) -> None:
        """Set the position of the satellite center."""
        self._position = np.asarray(value)

    @property
    def velocity(self) -> np.ndarray:
        """Velocity of the satellite center."""
        return self._velocity

    @velocity.setter
    def velocity(self, value: np.ndarray) -> None:
        """Set the velocity of the satellite center."""
        self._velocity = np.asarray(value)

    def half_mass_radius(self) -> float:
        """Compute the half-mass radius."""
        raise NotImplementedError

    def velocity_dispersion(self) -> float:
        """Compute the central velocity dispersion."""
        raise NotImplementedError

    def dynamical_time(self) -> float:
        """
        Compute the dynamical time of the satellite.

        Returns
        -------
        float
            Dynamical time in natural units.

        Notes
        -----
        The dynamical time is estimated as t_dyn ~ sqrt(r_h^3 / (G * M))
        where r_h is the half-mass radius.
        """
        raise NotImplementedError

    def _run_virialization(
        self,
        particles: "ParticleSet",
        virialize_time: Optional[float] = None,
        virialize_dt: Optional[float] = None,
        return_history: bool = False,
    ) -> Union["ParticleSet", Tuple["ParticleSet", "ParticleHistory"]]:
        """
        Run a short N-body integration to virialize the particle system.

        This is a helper method that runs the particles in isolation
        (self-gravity only, no external potential) to allow them to
        settle into dynamical equilibrium.

        Parameters
        ----------
        particles : ParticleSet
            Initial particle distribution.
        virialize_time : float, optional
            Duration in dynamical times. Default is 3 dynamical times.
        virialize_dt : float, optional
            Time step. Default is ~0.01 dynamical times.
        return_history : bool, default False
            If True, return full time evolution.

        Returns
        -------
        ParticleSet or tuple
            Final virialized particles, optionally with history.
        """
        raise NotImplementedError

    @staticmethod
    def compute_virial_ratio(particles: "ParticleSet") -> float:
        """
        Compute the virial ratio 2K/|W| for a particle system.

        For a virialized system in equilibrium, this should be ~1.0.

        Parameters
        ----------
        particles : ParticleSet
            Particle system to analyze.

        Returns
        -------
        float
            Virial ratio 2K/|W|.

        Notes
        -----
        K = total kinetic energy = 0.5 * sum(m * v^2)
        W = total potential energy = -0.5 * sum_i sum_j G*m_i*m_j/r_ij
        For virial equilibrium: 2K + W = 0, so 2K/|W| = 1
        """
        raise NotImplementedError
