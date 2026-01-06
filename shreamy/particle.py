"""
Particle module for shreamy.

This module provides data structures for representing particles and their
phase space coordinates. The main class is ParticleSet, which efficiently
stores and manipulates particle data.
"""

import numpy as np
from typing import Optional, Union, Tuple, Sequence
from dataclasses import dataclass


class ParticleSet:
    """
    A collection of particles with positions, velocities, and masses.

    This is the fundamental data structure for particle data in shreamy.
    It provides efficient storage and access to phase space coordinates.

    Parameters
    ----------
    positions : array-like
        Particle positions of shape (N, 3) with columns [x, y, z].
    velocities : array-like
        Particle velocities of shape (N, 3) with columns [vx, vy, vz].
    masses : array-like, optional
        Particle masses of shape (N,). If None, all particles have unit mass.
    ids : array-like, optional
        Unique particle IDs. If None, assigned sequentially from 0.
    components : array-like, optional
        Component labels for each particle (e.g., 'dm', 'stellar').
        If None, all particles are labeled 'default'.

    Attributes
    ----------
    n_particles : int
        Number of particles.
    total_mass : float
        Sum of all particle masses.
    component_names : list of str
        Unique component names present in this ParticleSet.
    """

    def __init__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: Optional[np.ndarray] = None,
        ids: Optional[np.ndarray] = None,
        components: Optional[np.ndarray] = None,
    ):
        """Initialize ParticleSet with phase space data."""
        raise NotImplementedError("ParticleSet.__init__ not yet implemented")

    @classmethod
    def from_phase_space(
        cls,
        phase_space: np.ndarray,
        masses: Optional[np.ndarray] = None,
    ) -> "ParticleSet":
        """
        Create a ParticleSet from a 6D phase space array.

        Parameters
        ----------
        phase_space : array-like
            Array of shape (N, 6) with columns [x, y, z, vx, vy, vz].
        masses : array-like, optional
            Particle masses.

        Returns
        -------
        ParticleSet
        """
        raise NotImplementedError

    @classmethod
    def from_cylindrical(
        cls,
        R: np.ndarray,
        phi: np.ndarray,
        z: np.ndarray,
        vR: np.ndarray,
        vT: np.ndarray,
        vz: np.ndarray,
        masses: Optional[np.ndarray] = None,
    ) -> "ParticleSet":
        """
        Create a ParticleSet from cylindrical coordinates.

        Parameters
        ----------
        R : array-like
            Cylindrical radius.
        phi : array-like
            Azimuthal angle (radians).
        z : array-like
            Height above plane.
        vR : array-like
            Radial velocity.
        vT : array-like
            Tangential velocity.
        vz : array-like
            Vertical velocity.
        masses : array-like, optional
            Particle masses.

        Returns
        -------
        ParticleSet
        """
        raise NotImplementedError

    @classmethod
    def from_spherical(
        cls,
        r: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        vr: np.ndarray,
        vtheta: np.ndarray,
        vphi: np.ndarray,
        masses: Optional[np.ndarray] = None,
    ) -> "ParticleSet":
        """
        Create a ParticleSet from spherical coordinates.

        Parameters
        ----------
        r : array-like
            Spherical radius.
        theta : array-like
            Polar angle from z-axis (radians).
        phi : array-like
            Azimuthal angle (radians).
        vr : array-like
            Radial velocity.
        vtheta : array-like
            Polar velocity.
        vphi : array-like
            Azimuthal velocity.
        masses : array-like, optional
            Particle masses.

        Returns
        -------
        ParticleSet
        """
        raise NotImplementedError

    # =========================================================================
    # Properties for accessing phase space
    # =========================================================================

    @property
    def positions(self) -> np.ndarray:
        """Particle positions of shape (N, 3)."""
        raise NotImplementedError

    @property
    def velocities(self) -> np.ndarray:
        """Particle velocities of shape (N, 3)."""
        raise NotImplementedError

    @property
    def masses(self) -> np.ndarray:
        """Particle masses of shape (N,)."""
        raise NotImplementedError

    @property
    def x(self) -> np.ndarray:
        """x coordinates."""
        raise NotImplementedError

    @property
    def y(self) -> np.ndarray:
        """y coordinates."""
        raise NotImplementedError

    @property
    def z(self) -> np.ndarray:
        """z coordinates."""
        raise NotImplementedError

    @property
    def vx(self) -> np.ndarray:
        """x velocities."""
        raise NotImplementedError

    @property
    def vy(self) -> np.ndarray:
        """y velocities."""
        raise NotImplementedError

    @property
    def vz(self) -> np.ndarray:
        """z velocities."""
        raise NotImplementedError

    @property
    def n_particles(self) -> int:
        """Number of particles."""
        raise NotImplementedError

    @property
    def total_mass(self) -> float:
        """Total mass of all particles."""
        raise NotImplementedError

    @property
    def phase_space(self) -> np.ndarray:
        """Full phase space array of shape (N, 6)."""
        raise NotImplementedError

    # =========================================================================
    # Component Access
    # =========================================================================

    @property
    def components(self) -> np.ndarray:
        """Component labels for each particle."""
        raise NotImplementedError

    @property
    def component_names(self) -> list:
        """List of unique component names in this ParticleSet."""
        raise NotImplementedError

    def get_component(self, component: str) -> "ParticleSet":
        """
        Return a new ParticleSet containing only particles of the specified component.

        Parameters
        ----------
        component : str
            Component name (e.g., 'dm', 'stellar').

        Returns
        -------
        ParticleSet
            Subset containing only particles with matching component.

        Raises
        ------
        ValueError
            If component name is not found.
        """
        raise NotImplementedError

    def get_component_mask(self, component: str) -> np.ndarray:
        """
        Return a boolean mask for particles of the specified component.

        Parameters
        ----------
        component : str
            Component name.

        Returns
        -------
        ndarray
            Boolean mask of shape (N,).
        """
        raise NotImplementedError

    def has_component(self, component: str) -> bool:
        """Check if a component exists in this ParticleSet."""
        raise NotImplementedError

    @staticmethod
    def concatenate(particle_sets: list, component_labels: Optional[list] = None) -> "ParticleSet":
        """
        Concatenate multiple ParticleSets into one.

        Parameters
        ----------
        particle_sets : list of ParticleSet
            ParticleSets to concatenate.
        component_labels : list of str, optional
            Component labels to assign to each input ParticleSet.
            If None, preserves existing component labels.

        Returns
        -------
        ParticleSet
            Combined ParticleSet with all particles.

        Examples
        --------
        >>> dm_particles = satellite_dm.sample(50000)
        >>> star_particles = satellite_stars.sample(10000)
        >>> all_particles = ParticleSet.concatenate(
        ...     [dm_particles, star_particles],
        ...     component_labels=['dm', 'stellar']
        ... )
        """
        raise NotImplementedError

    # =========================================================================
    # Cylindrical and Spherical Coordinates
    # =========================================================================

    @property
    def R(self) -> np.ndarray:
        """Cylindrical radius R = sqrt(x^2 + y^2)."""
        raise NotImplementedError

    @property
    def r(self) -> np.ndarray:
        """Spherical radius r = sqrt(x^2 + y^2 + z^2)."""
        raise NotImplementedError

    @property
    def phi(self) -> np.ndarray:
        """Azimuthal angle phi = arctan2(y, x)."""
        raise NotImplementedError

    @property
    def theta(self) -> np.ndarray:
        """Polar angle theta = arccos(z/r)."""
        raise NotImplementedError

    # =========================================================================
    # Operations
    # =========================================================================

    def copy(self) -> "ParticleSet":
        """Return a deep copy of the ParticleSet."""
        raise NotImplementedError

    def shift(self, dr: np.ndarray = None, dv: np.ndarray = None) -> "ParticleSet":
        """
        Return a new ParticleSet shifted by constant offsets.

        Parameters
        ----------
        dr : array-like, optional
            Position offset [dx, dy, dz].
        dv : array-like, optional
            Velocity offset [dvx, dvy, dvz].

        Returns
        -------
        ParticleSet
        """
        raise NotImplementedError

    def rotate(self, rotation_matrix: np.ndarray) -> "ParticleSet":
        """
        Return a new ParticleSet rotated by a rotation matrix.

        Parameters
        ----------
        rotation_matrix : array-like
            3x3 rotation matrix.

        Returns
        -------
        ParticleSet
        """
        raise NotImplementedError

    def center_of_mass(self) -> np.ndarray:
        """Return the mass-weighted center of mass [x, y, z]."""
        raise NotImplementedError

    def center_of_mass_velocity(self) -> np.ndarray:
        """Return the mass-weighted center of mass velocity [vx, vy, vz]."""
        raise NotImplementedError

    def recenter(self) -> "ParticleSet":
        """
        Return a new ParticleSet centered on the center of mass,
        with zero net momentum.
        """
        raise NotImplementedError

    # =========================================================================
    # Selection and Indexing
    # =========================================================================

    def __getitem__(self, key) -> "ParticleSet":
        """Select a subset of particles by index or boolean mask."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of particles."""
        raise NotImplementedError

    # =========================================================================
    # I/O
    # =========================================================================

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return particle data as numpy arrays.

        Returns
        -------
        positions : ndarray
            Shape (N, 3)
        velocities : ndarray
            Shape (N, 3)
        masses : ndarray
            Shape (N,)
        """
        raise NotImplementedError


@dataclass
class Snapshot:
    """
    A snapshot of the particle system at a single time.

    This is used internally to store the state of the system at each
    saved time step during integration.

    Attributes
    ----------
    t : float
        Time of the snapshot.
    particles : ParticleSet
        Particle data at this time.
    """

    t: float
    particles: ParticleSet


class ParticleHistory:
    """
    Stores the time evolution of a particle system.

    This class efficiently stores particle data at multiple times,
    allowing for time-slicing and interpolation.

    Parameters
    ----------
    snapshots : list of Snapshot
        Initial list of snapshots.
    """

    def __init__(self, snapshots: Optional[Sequence[Snapshot]] = None):
        """Initialize with optional list of snapshots."""
        raise NotImplementedError

    def add_snapshot(self, t: float, particles: ParticleSet) -> None:
        """Add a snapshot at time t."""
        raise NotImplementedError

    def at_time(self, t: float, interpolate: bool = False) -> ParticleSet:
        """
        Return particle state at time t.

        Parameters
        ----------
        t : float
            Requested time.
        interpolate : bool, default False
            If True and t is between saved times, interpolate.
            If False, return nearest saved time.

        Returns
        -------
        ParticleSet
        """
        raise NotImplementedError

    @property
    def times(self) -> np.ndarray:
        """Array of all saved times."""
        raise NotImplementedError

    @property
    def n_snapshots(self) -> int:
        """Number of saved snapshots."""
        raise NotImplementedError

    def __getitem__(self, key) -> "ParticleHistory":
        """
        Slice particle history by particle index.

        Returns a new ParticleHistory with only selected particles.
        """
        raise NotImplementedError
