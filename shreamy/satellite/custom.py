"""
Custom user-defined satellite models.
"""

import numpy as np
from typing import Callable, Optional, Union, Tuple, TYPE_CHECKING

from .base import SatelliteModel

if TYPE_CHECKING:
    from ..particle import ParticleSet, ParticleHistory


class CustomSatellite(SatelliteModel):
    """
    A satellite with user-defined density and potential profiles.

    This allows users to define arbitrary satellite models by providing
    functions for the density and potential profiles.

    Parameters
    ----------
    mass : float
        Total mass of the satellite.
    density_func : callable
        Function density(r) -> rho that returns density at radius r.
    potential_func : callable, optional
        Function potential(r) -> phi that returns potential at radius r.
        If not provided, potential will raise NotImplementedError.
    sample_func : callable, optional
        Function sample(n, rng) -> (positions, velocities) that generates
        particle positions and velocities. If not provided, rejection
        sampling will be used with density_func.
    position : array-like, optional
        Initial center position [x, y, z].
    velocity : array-like, optional
        Initial center velocity [vx, vy, vz].
    r_max : float, default 10.0
        Maximum radius for sampling (in model units).

    Examples
    --------
    >>> def my_density(r):
    ...     return np.exp(-r) / (4 * np.pi * r**2)
    >>> sat = CustomSatellite(mass=1e9, density_func=my_density)
    >>> particles = sat.sample(10000)
    """

    def __init__(
        self,
        mass: float,
        density_func: Callable[[np.ndarray], np.ndarray],
        potential_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        sample_func: Optional[Callable] = None,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        r_max: float = 10.0,
    ):
        """Initialize custom satellite model."""
        super().__init__(mass, position, velocity)
        self._density_func = density_func
        self._potential_func = potential_func
        self._sample_func = sample_func
        self._r_max = r_max

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
        Sample particles from the custom distribution.

        If sample_func was provided, uses that. Otherwise uses rejection
        sampling with the density function.

        Parameters
        ----------
        n_particles : int
            Number of particles to sample.
        seed : int, optional
            Random seed for reproducibility.
        virialize : bool, default False
            If True, run a short N-body integration to virialize.
            This is especially recommended for custom satellites where
            the velocity distribution may not be self-consistent.
        virialize_time : float, optional
            Duration of virialization in dynamical times.
        virialize_dt : float, optional
            Time step for virialization.
        return_virialize_info : bool, default False
            If True and virialize=True, return (ParticleSet, ParticleHistory).

        Returns
        -------
        ParticleSet or tuple
            Sampled particles, optionally with virialization history.
        """
        raise NotImplementedError

    def density(self, r: np.ndarray) -> np.ndarray:
        """
        Compute density at radius r.

        Parameters
        ----------
        r : array-like
            Radii at which to compute density.

        Returns
        -------
        array-like
            Density values.
        """
        return self._density_func(r)

    def potential(self, r: np.ndarray) -> np.ndarray:
        """
        Compute potential at radius r.

        Parameters
        ----------
        r : array-like
            Radii at which to compute potential.

        Returns
        -------
        array-like
            Potential values.

        Raises
        ------
        NotImplementedError
            If potential_func was not provided.
        """
        if self._potential_func is None:
            raise NotImplementedError("No potential function provided")
        return self._potential_func(r)

    @property
    def r_max(self) -> float:
        """Maximum sampling radius."""
        return self._r_max
