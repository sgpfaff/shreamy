"""
Particle module for shreamy.

This module provides data structures for representing particles and their
phase space coordinates. The main class is ParticleSet, which efficiently
stores and manipulates particle data.
"""

import numpy as np
from typing import Dict, Optional, Union, Tuple, Sequence
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
        # Convert to numpy arrays and ensure correct shape
        self._positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
        self._velocities = np.atleast_2d(np.asarray(velocities, dtype=np.float64))
        
        # Validate shapes
        if self._positions.shape[1] != 3:
            raise ValueError(f"positions must have shape (N, 3), got {self._positions.shape}")
        if self._velocities.shape[1] != 3:
            raise ValueError(f"velocities must have shape (N, 3), got {self._velocities.shape}")
        if self._positions.shape[0] != self._velocities.shape[0]:
            raise ValueError(
                f"positions and velocities must have same number of particles, "
                f"got {self._positions.shape[0]} and {self._velocities.shape[0]}"
            )
        
        n = self._positions.shape[0]
        
        # Handle masses
        if masses is None:
            self._masses = np.ones(n, dtype=np.float64)
        else:
            self._masses = np.asarray(masses, dtype=np.float64).ravel()
            if len(self._masses) != n:
                raise ValueError(f"masses must have length {n}, got {len(self._masses)}")
        
        # Handle IDs
        if ids is None:
            self._ids = np.arange(n, dtype=np.int64)
        else:
            self._ids = np.asarray(ids, dtype=np.int64).ravel()
            if len(self._ids) != n:
                raise ValueError(f"ids must have length {n}, got {len(self._ids)}")
        
        # Handle component labels
        if components is None:
            self._components = np.array(['default'] * n, dtype=object)
        else:
            self._components = np.asarray(components, dtype=object).ravel()
            if len(self._components) != n:
                raise ValueError(f"components must have length {n}, got {len(self._components)}")

    @classmethod
    def from_phase_space(
        cls,
        phase_space: np.ndarray,
        masses: Optional[np.ndarray] = None,
        ids: Optional[np.ndarray] = None,
        components: Optional[np.ndarray] = None,
    ) -> "ParticleSet":
        """
        Create a ParticleSet from a 6D phase space array.

        Parameters
        ----------
        phase_space : array-like
            Array of shape (N, 6) with columns [x, y, z, vx, vy, vz].
        masses : array-like, optional
            Particle masses.
        ids : array-like, optional
            Particle IDs.
        components : array-like, optional
            Component labels.

        Returns
        -------
        ParticleSet
        """
        phase_space = np.atleast_2d(np.asarray(phase_space))
        if phase_space.shape[1] != 6:
            raise ValueError(f"phase_space must have shape (N, 6), got {phase_space.shape}")
        positions = phase_space[:, :3]
        velocities = phase_space[:, 3:]
        return cls(positions, velocities, masses, ids, components)

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
        ids: Optional[np.ndarray] = None,
        components: Optional[np.ndarray] = None,
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
        ids : array-like, optional
            Particle IDs.
        components : array-like, optional
            Component labels.

        Returns
        -------
        ParticleSet
        """
        R = np.asarray(R)
        phi = np.asarray(phi)
        z = np.asarray(z)
        vR = np.asarray(vR)
        vT = np.asarray(vT)
        vz = np.asarray(vz)
        
        # Convert positions: (R, phi, z) -> (x, y, z)
        x = R * np.cos(phi)
        y = R * np.sin(phi)
        positions = np.column_stack([x, y, z])
        
        # Convert velocities: (vR, vT, vz) -> (vx, vy, vz)
        # vx = vR * cos(phi) - vT * sin(phi)
        # vy = vR * sin(phi) + vT * cos(phi)
        vx = vR * np.cos(phi) - vT * np.sin(phi)
        vy = vR * np.sin(phi) + vT * np.cos(phi)
        velocities = np.column_stack([vx, vy, vz])
        
        return cls(positions, velocities, masses, ids, components)

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
        ids: Optional[np.ndarray] = None,
        components: Optional[np.ndarray] = None,
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
        ids : array-like, optional
            Particle IDs.
        components : array-like, optional
            Component labels.

        Returns
        -------
        ParticleSet
        """
        r = np.asarray(r)
        theta = np.asarray(theta)
        phi = np.asarray(phi)
        vr = np.asarray(vr)
        vtheta = np.asarray(vtheta)
        vphi = np.asarray(vphi)
        
        # Convert positions: (r, theta, phi) -> (x, y, z)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        
        x = r * sin_theta * cos_phi
        y = r * sin_theta * sin_phi
        z = r * cos_theta
        positions = np.column_stack([x, y, z])
        
        # Convert velocities: (vr, vtheta, vphi) -> (vx, vy, vz)
        # Using spherical unit vectors:
        # e_r = (sin_theta*cos_phi, sin_theta*sin_phi, cos_theta)
        # e_theta = (cos_theta*cos_phi, cos_theta*sin_phi, -sin_theta)
        # e_phi = (-sin_phi, cos_phi, 0)
        vx = (vr * sin_theta * cos_phi + 
              vtheta * cos_theta * cos_phi - 
              vphi * sin_phi)
        vy = (vr * sin_theta * sin_phi + 
              vtheta * cos_theta * sin_phi + 
              vphi * cos_phi)
        vz = vr * cos_theta - vtheta * sin_theta
        velocities = np.column_stack([vx, vy, vz])
        
        return cls(positions, velocities, masses, ids, components)

    # =========================================================================
    # Properties for accessing phase space
    # =========================================================================

    @property
    def positions(self) -> np.ndarray:
        """Particle positions of shape (N, 3)."""
        return self._positions

    @property
    def velocities(self) -> np.ndarray:
        """Particle velocities of shape (N, 3)."""
        return self._velocities

    @property
    def masses(self) -> np.ndarray:
        """Particle masses of shape (N,)."""
        return self._masses

    @property
    def ids(self) -> np.ndarray:
        """Particle IDs of shape (N,)."""
        return self._ids

    @property
    def x(self) -> np.ndarray:
        """x coordinates."""
        return self._positions[:, 0]

    @property
    def y(self) -> np.ndarray:
        """y coordinates."""
        return self._positions[:, 1]

    @property
    def z(self) -> np.ndarray:
        """z coordinates."""
        return self._positions[:, 2]

    @property
    def vx(self) -> np.ndarray:
        """x velocities."""
        return self._velocities[:, 0]

    @property
    def vy(self) -> np.ndarray:
        """y velocities."""
        return self._velocities[:, 1]

    @property
    def vz(self) -> np.ndarray:
        """z velocities."""
        return self._velocities[:, 2]

    @property
    def n_particles(self) -> int:
        """Number of particles."""
        return self._positions.shape[0]

    @property
    def total_mass(self) -> float:
        """Total mass of all particles."""
        return float(np.sum(self._masses))

    @property
    def phase_space(self) -> np.ndarray:
        """Full phase space array of shape (N, 6)."""
        return np.hstack([self._positions, self._velocities])

    # =========================================================================
    # Component Access
    # =========================================================================

    @property
    def components(self) -> np.ndarray:
        """Component labels for each particle."""
        return self._components

    @property
    def component_names(self) -> list:
        """List of unique component names in this ParticleSet."""
        return list(np.unique(self._components))

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
        if not self.has_component(component):
            raise ValueError(
                f"Component '{component}' not found. "
                f"Available components: {self.component_names}"
            )
        mask = self.get_component_mask(component)
        return self[mask]

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
        return self._components == component

    def has_component(self, component: str) -> bool:
        """Check if a component exists in this ParticleSet."""
        return component in self._components

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
        if len(particle_sets) == 0:
            raise ValueError("Cannot concatenate empty list of ParticleSets")
        
        if component_labels is not None and len(component_labels) != len(particle_sets):
            raise ValueError(
                f"component_labels must have same length as particle_sets, "
                f"got {len(component_labels)} and {len(particle_sets)}"
            )
        
        positions_list = []
        velocities_list = []
        masses_list = []
        ids_list = []
        components_list = []
        
        id_offset = 0
        for i, ps in enumerate(particle_sets):
            positions_list.append(ps.positions)
            velocities_list.append(ps.velocities)
            masses_list.append(ps.masses)
            # Offset IDs to ensure uniqueness
            ids_list.append(ps.ids + id_offset)
            id_offset += ps.n_particles
            
            if component_labels is not None:
                # Override with provided labels
                components_list.append(np.array([component_labels[i]] * ps.n_particles, dtype=object))
            else:
                # Preserve existing labels
                components_list.append(ps.components)
        
        return ParticleSet(
            positions=np.vstack(positions_list),
            velocities=np.vstack(velocities_list),
            masses=np.concatenate(masses_list),
            ids=np.concatenate(ids_list),
            components=np.concatenate(components_list),
        )

    # =========================================================================
    # Cylindrical and Spherical Coordinates
    # =========================================================================

    @property
    def R(self) -> np.ndarray:
        """Cylindrical radius R = sqrt(x^2 + y^2)."""
        return np.sqrt(self.x**2 + self.y**2)

    @property
    def r(self) -> np.ndarray:
        """Spherical radius r = sqrt(x^2 + y^2 + z^2)."""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    @property
    def phi(self) -> np.ndarray:
        """Azimuthal angle phi = arctan2(y, x)."""
        return np.arctan2(self.y, self.x)

    @property
    def theta(self) -> np.ndarray:
        """Polar angle theta = arccos(z/r)."""
        r = self.r
        # Handle r=0 case to avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.arccos(np.clip(self.z / r, -1.0, 1.0))
            result = np.where(r == 0, 0.0, result)
        return result

    @property
    def vR(self) -> np.ndarray:
        """Cylindrical radial velocity vR = (x*vx + y*vy) / R."""
        R = self.R
        with np.errstate(divide='ignore', invalid='ignore'):
            result = (self.x * self.vx + self.y * self.vy) / R
            result = np.where(R == 0, 0.0, result)
        return result

    @property
    def vT(self) -> np.ndarray:
        """Cylindrical tangential velocity vT = (x*vy - y*vx) / R."""
        R = self.R
        with np.errstate(divide='ignore', invalid='ignore'):
            result = (self.x * self.vy - self.y * self.vx) / R
            result = np.where(R == 0, 0.0, result)
        return result

    @property
    def vr_spherical(self) -> np.ndarray:
        """Spherical radial velocity vr = (x*vx + y*vy + z*vz) / r."""
        r = self.r
        with np.errstate(divide='ignore', invalid='ignore'):
            result = (self.x * self.vx + self.y * self.vy + self.z * self.vz) / r
            result = np.where(r == 0, 0.0, result)
        return result

    # =========================================================================
    # Operations
    # =========================================================================

    def copy(self) -> "ParticleSet":
        """Return a deep copy of the ParticleSet."""
        return ParticleSet(
            positions=self._positions.copy(),
            velocities=self._velocities.copy(),
            masses=self._masses.copy(),
            ids=self._ids.copy(),
            components=self._components.copy(),
        )

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
        new_positions = self._positions.copy()
        new_velocities = self._velocities.copy()
        
        if dr is not None:
            dr = np.asarray(dr).ravel()
            if len(dr) != 3:
                raise ValueError(f"dr must have length 3, got {len(dr)}")
            new_positions = new_positions + dr
        
        if dv is not None:
            dv = np.asarray(dv).ravel()
            if len(dv) != 3:
                raise ValueError(f"dv must have length 3, got {len(dv)}")
            new_velocities = new_velocities + dv
        
        return ParticleSet(
            positions=new_positions,
            velocities=new_velocities,
            masses=self._masses.copy(),
            ids=self._ids.copy(),
            components=self._components.copy(),
        )

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
        R = np.asarray(rotation_matrix)
        if R.shape != (3, 3):
            raise ValueError(f"rotation_matrix must have shape (3, 3), got {R.shape}")
        
        # Rotate positions and velocities: x' = R @ x
        new_positions = (R @ self._positions.T).T
        new_velocities = (R @ self._velocities.T).T
        
        return ParticleSet(
            positions=new_positions,
            velocities=new_velocities,
            masses=self._masses.copy(),
            ids=self._ids.copy(),
            components=self._components.copy(),
        )

    def center_of_mass(self) -> np.ndarray:
        """Return the mass-weighted center of mass [x, y, z]."""
        total_mass = self.total_mass
        if total_mass == 0:
            return np.zeros(3)
        return np.sum(self._positions * self._masses[:, np.newaxis], axis=0) / total_mass

    def center_of_mass_velocity(self) -> np.ndarray:
        """Return the mass-weighted center of mass velocity [vx, vy, vz]."""
        total_mass = self.total_mass
        if total_mass == 0:
            return np.zeros(3)
        return np.sum(self._velocities * self._masses[:, np.newaxis], axis=0) / total_mass

    def recenter(self) -> "ParticleSet":
        """
        Return a new ParticleSet centered on the center of mass,
        with zero net momentum.
        """
        com = self.center_of_mass()
        com_vel = self.center_of_mass_velocity()
        return self.shift(dr=-com, dv=-com_vel)

    # =========================================================================
    # Selection and Indexing
    # =========================================================================

    def __getitem__(self, key) -> "ParticleSet":
        """Select a subset of particles by index or boolean mask."""
        return ParticleSet(
            positions=self._positions[key],
            velocities=self._velocities[key],
            masses=self._masses[key],
            ids=self._ids[key],
            components=self._components[key],
        )

    def __len__(self) -> int:
        """Return the number of particles."""
        return self.n_particles

    def __repr__(self) -> str:
        """String representation."""
        comp_str = ", ".join(f"{name}: {np.sum(self._components == name)}" 
                             for name in self.component_names)
        return (f"ParticleSet(n_particles={self.n_particles}, "
                f"total_mass={self.total_mass:.3e}, components={{{comp_str}}})")

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
        return self._positions.copy(), self._velocities.copy(), self._masses.copy()

    def to_dict(self) -> Dict[str, np.ndarray]:
        """
        Return particle data as a dictionary.

        Returns
        -------
        data : dict
            Dictionary with keys 'positions', 'velocities', 'masses', 'components'.
        """
        result = {
            "positions": self._positions.copy(),
            "velocities": self._velocities.copy(),
            "masses": self._masses.copy(),
        }
        if self._components is not None:
            result["components"] = self._components.copy()
        return result

    def save(self, filename: str, format: str = "npy") -> None:
        """
        Save particle data to file.

        Parameters
        ----------
        filename : str
            Output filename.
        format : str, default "npy"
            File format: "npy" for numpy binary, "npz" for compressed.
        """
        data = self.to_dict()
        if format == "npy":
            np.save(filename, data)
        elif format == "npz":
            np.savez_compressed(filename, **data)
        else:
            raise ValueError(f"Unknown format: {format}")

    @classmethod
    def load(cls, filename: str) -> "ParticleSet":
        """
        Load particle data from file.

        Parameters
        ----------
        filename : str
            Input filename.

        Returns
        -------
        ParticleSet
        """
        # Handle both .npy and .npz files
        if filename.endswith(".npz"):
            data = np.load(filename, allow_pickle=True)
            positions = data["positions"]
            velocities = data["velocities"]
            masses = data["masses"]
            components = data.get("components", None)
        else:
            data = np.load(filename, allow_pickle=True).item()
            positions = data["positions"]
            velocities = data["velocities"]
            masses = data["masses"]
            components = data.get("components", None)

        return cls(
            positions=positions,
            velocities=velocities,
            masses=masses,
            components=components,
        )


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
        if snapshots is None:
            self._snapshots: list[Snapshot] = []
        else:
            # Sort by time
            self._snapshots = sorted(list(snapshots), key=lambda s: s.t)

    def add_snapshot(self, t: float, particles: ParticleSet) -> None:
        """
        Add a snapshot at time t.

        Parameters
        ----------
        t : float
            Time of the snapshot.
        particles : ParticleSet
            Particle data at this time.
        """
        snapshot = Snapshot(t=t, particles=particles.copy())
        # Insert in sorted order
        if len(self._snapshots) == 0 or t >= self._snapshots[-1].t:
            self._snapshots.append(snapshot)
        else:
            # Find insertion point
            for i, s in enumerate(self._snapshots):
                if t < s.t:
                    self._snapshots.insert(i, snapshot)
                    break

    def at_time(self, t: float, interpolate: bool = False) -> ParticleSet:
        """
        Return particle state at time t.

        Parameters
        ----------
        t : float
            Requested time.
        interpolate : bool, default False
            If True and t is between saved times, interpolate linearly.
            If False, return nearest saved time.

        Returns
        -------
        ParticleSet

        Raises
        ------
        ValueError
            If no snapshots are available.
        """
        if len(self._snapshots) == 0:
            raise ValueError("No snapshots available")

        times = self.times

        # Check if exactly at a saved time
        exact_idx = np.where(np.abs(times - t) < 1e-12)[0]
        if len(exact_idx) > 0:
            return self._snapshots[exact_idx[0]].particles.copy()

        # Find bracketing snapshots
        if t <= times[0]:
            return self._snapshots[0].particles.copy()
        if t >= times[-1]:
            return self._snapshots[-1].particles.copy()

        # Find indices that bracket t
        idx_after = np.searchsorted(times, t)
        idx_before = idx_after - 1

        if not interpolate:
            # Return nearest
            if abs(times[idx_before] - t) <= abs(times[idx_after] - t):
                return self._snapshots[idx_before].particles.copy()
            else:
                return self._snapshots[idx_after].particles.copy()

        # Linear interpolation
        t0 = times[idx_before]
        t1 = times[idx_after]
        alpha = (t - t0) / (t1 - t0)

        p0 = self._snapshots[idx_before].particles
        p1 = self._snapshots[idx_after].particles

        # Interpolate positions and velocities
        positions = (1 - alpha) * p0.positions + alpha * p1.positions
        velocities = (1 - alpha) * p0.velocities + alpha * p1.velocities

        return ParticleSet(
            positions=positions,
            velocities=velocities,
            masses=p0.masses,  # Masses don't change
            components=p0._components,  # Components don't change
        )

    @property
    def times(self) -> np.ndarray:
        """Array of all saved times."""
        return np.array([s.t for s in self._snapshots])

    @property
    def n_snapshots(self) -> int:
        """Number of saved snapshots."""
        return len(self._snapshots)

    def __len__(self) -> int:
        """Number of saved snapshots."""
        return self.n_snapshots

    def __getitem__(self, key) -> "ParticleHistory":
        """
        Slice particle history by particle index.

        Returns a new ParticleHistory with only selected particles.

        Parameters
        ----------
        key : int, slice, or array-like
            Particle indices to select.

        Returns
        -------
        ParticleHistory
            New history with selected particles.
        """
        new_snapshots = [
            Snapshot(t=s.t, particles=s.particles[key]) for s in self._snapshots
        ]
        return ParticleHistory(new_snapshots)

    def get_particle_trajectory(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the trajectory of a single particle.

        Parameters
        ----------
        idx : int
            Particle index.

        Returns
        -------
        times : ndarray
            Shape (n_snapshots,)
        positions : ndarray
            Shape (n_snapshots, 3)
        velocities : ndarray
            Shape (n_snapshots, 3)
        """
        times = self.times
        positions = np.array([s.particles.positions[idx] for s in self._snapshots])
        velocities = np.array([s.particles.velocities[idx] for s in self._snapshots])
        return times, positions, velocities

    def get_center_of_mass_trajectory(
        self, component: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the center of mass trajectory.

        Parameters
        ----------
        component : str, optional
            If given, compute center of mass for only this component.

        Returns
        -------
        times : ndarray
            Shape (n_snapshots,)
        positions : ndarray
            Shape (n_snapshots, 3)
        velocities : ndarray
            Shape (n_snapshots, 3)
        """
        times = self.times
        positions = []
        velocities = []

        for s in self._snapshots:
            if component is not None:
                p = s.particles.get_component(component)
            else:
                p = s.particles
            pos_com = p.center_of_mass()
            vel_com = p.center_of_mass_velocity()
            positions.append(pos_com)
            velocities.append(vel_com)

        return times, np.array(positions), np.array(velocities)

    def save(self, filename: str) -> None:
        """
        Save particle history to file.

        Parameters
        ----------
        filename : str
            Output filename (will be saved as .npz).
        """
        data = {}
        data["times"] = self.times

        for i, s in enumerate(self._snapshots):
            data[f"positions_{i}"] = s.particles.positions
            data[f"velocities_{i}"] = s.particles.velocities
            data[f"masses_{i}"] = s.particles.masses
            if s.particles._components is not None:
                data[f"components_{i}"] = s.particles._components

        data["n_snapshots"] = np.array([self.n_snapshots])

        np.savez_compressed(filename, **data)

    @classmethod
    def load(cls, filename: str) -> "ParticleHistory":
        """
        Load particle history from file.

        Parameters
        ----------
        filename : str
            Input filename.

        Returns
        -------
        ParticleHistory
        """
        data = np.load(filename, allow_pickle=True)
        n_snapshots = int(data["n_snapshots"][0])
        times = data["times"]

        snapshots = []
        for i in range(n_snapshots):
            positions = data[f"positions_{i}"]
            velocities = data[f"velocities_{i}"]
            masses = data[f"masses_{i}"]
            components = data.get(f"components_{i}", None)

            particles = ParticleSet(
                positions=positions,
                velocities=velocities,
                masses=masses,
                components=components,
            )
            snapshots.append(Snapshot(t=times[i], particles=particles))

        return cls(snapshots)

    def __repr__(self) -> str:
        """String representation."""
        if len(self._snapshots) == 0:
            return "ParticleHistory(empty)"

        n_particles = self._snapshots[0].particles.n_particles
        t_min = self.times[0]
        t_max = self.times[-1]
        return (
            f"ParticleHistory(n_particles={n_particles}, "
            f"n_snapshots={self.n_snapshots}, "
            f"t=[{t_min:.4f}, {t_max:.4f}])"
        )
