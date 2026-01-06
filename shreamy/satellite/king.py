"""
King (lowered isothermal) satellite model.
"""

import numpy as np
from typing import Optional, Union, Tuple, TYPE_CHECKING

from .base import SatelliteModel

if TYPE_CHECKING:
    from ..particle import ParticleSet, ParticleHistory


class KingSatellite(SatelliteModel):
    """
    A satellite with a King (lowered isothermal) profile.

    King models have a finite extent (truncation radius) and are
    commonly used for globular clusters.

    Parameters
    ----------
    mass : float
        Total mass.
    core_radius : float
        King core radius (r_c).
    concentration : float
        King concentration (c = log10(r_t/r_c)).
    position : array-like, optional
        Initial position.
    velocity : array-like, optional
        Initial velocity.
    """

    def __init__(
        self,
        mass: float,
        core_radius: float,
        concentration: float,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
    ):
        """Initialize a King satellite."""
        super().__init__(mass, position, velocity)
        self._core_radius = core_radius
        self._concentration = concentration

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
        Sample particles from the King distribution.

        Parameters
        ----------
        n_particles : int
            Number of particles to sample.
        seed : int, optional
            Random seed for reproducibility.
        virialize : bool, default False
            If True, run a short N-body integration to virialize.
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
        """Compute King density."""
        raise NotImplementedError

    def potential(self, r: np.ndarray) -> np.ndarray:
        """Compute King potential."""
        raise NotImplementedError

    @property
    def core_radius(self) -> float:
        """The King core radius."""
        return self._core_radius

    @property
    def concentration(self) -> float:
        """The King concentration parameter."""
        return self._concentration

    @property
    def tidal_radius(self) -> float:
        """The truncation (tidal) radius."""
        return self._core_radius * 10 ** self._concentration
