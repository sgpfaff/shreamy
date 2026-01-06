"""
Block (hierarchical) time step utilities.
"""

import numpy as np
from typing import Optional


class BlockTimeSteps:
    """
    Manages block (hierarchical) time steps for particles.

    In block time stepping, particles are grouped by their required
    time step, allowing frequent updates for particles needing small
    steps without updating the entire system.

    This is an advanced feature for optimizing simulations with
    widely varying time scales (e.g., dense cores and diffuse halos).

    Parameters
    ----------
    n_particles : int
        Number of particles.
    dt_min : float
        Minimum (smallest) allowed time step.
    dt_max : float
        Maximum (largest) allowed time step.
    n_levels : int, default 10
        Number of time step levels. Time steps are powers of 2 apart.
    """

    def __init__(
        self,
        n_particles: int,
        dt_min: float,
        dt_max: float,
        n_levels: int = 10,
    ):
        """Initialize block time step manager."""
        self._n_particles = n_particles
        self._dt_min = dt_min
        self._dt_max = dt_max
        self._n_levels = n_levels
        self._levels = np.zeros(n_particles, dtype=int)
        self._next_time = np.zeros(n_particles)

    def assign_levels(
        self,
        accelerations: np.ndarray,
        velocities: np.ndarray,
    ) -> np.ndarray:
        """
        Assign time step levels to particles based on their dynamics.

        Particles with high accelerations get smaller time steps (higher levels).

        Parameters
        ----------
        accelerations : ndarray
            Particle accelerations of shape (N, 3).
        velocities : ndarray
            Particle velocities of shape (N, 3).

        Returns
        -------
        ndarray
            Integer levels for each particle (0 = largest dt, n_levels-1 = smallest dt).
        """
        raise NotImplementedError

    def get_active_particles(self, t: float) -> np.ndarray:
        """
        Return indices of particles that should be updated at time t.

        Parameters
        ----------
        t : float
            Current simulation time.

        Returns
        -------
        ndarray
            Boolean mask or indices of active particles.
        """
        raise NotImplementedError

    def get_timestep(self, level: int) -> float:
        """
        Get the time step for a given level.

        Parameters
        ----------
        level : int
            Time step level.

        Returns
        -------
        float
            Time step for that level.
        """
        raise NotImplementedError

    def update_next_time(self, indices: np.ndarray, t: float) -> None:
        """
        Update the next update time for particles.

        Parameters
        ----------
        indices : ndarray
            Indices of particles that were just updated.
        t : float
            Current time.
        """
        raise NotImplementedError

    @property
    def levels(self) -> np.ndarray:
        """Current time step levels for all particles."""
        return self._levels

    @property
    def n_levels(self) -> int:
        """Number of time step levels."""
        return self._n_levels

    @property
    def dt_min(self) -> float:
        """Minimum time step."""
        return self._dt_min

    @property
    def dt_max(self) -> float:
        """Maximum time step."""
        return self._dt_max
