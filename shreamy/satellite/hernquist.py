"""
Hernquist profile satellite model.
"""

import numpy as np
from typing import Optional, Union, Tuple, TYPE_CHECKING

from .base import SatelliteModel

if TYPE_CHECKING:
    from ..particle import ParticleSet, ParticleHistory


class HernquistSatellite(SatelliteModel):
    """
    A satellite with a Hernquist density profile.

    rho(r) = M*a / (2*pi*r) / (r + a)^3

    This cuspy profile is more realistic for elliptical galaxies and
    some dwarf galaxies.

    Parameters
    ----------
    mass : float
        Total mass.
    scale_radius : float
        Hernquist scale radius (a).
    position : array-like, optional
        Initial position.
    velocity : array-like, optional
        Initial velocity.
    """

    def __init__(
        self,
        mass: float,
        scale_radius: float,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
    ):
        """Initialize a Hernquist satellite."""
        super().__init__(mass, position, velocity)
        self._scale_radius = scale_radius

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
        Sample particles from the Hernquist distribution.

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
        """Compute Hernquist density."""
        raise NotImplementedError

    def potential(self, r: np.ndarray) -> np.ndarray:
        """Compute Hernquist potential."""
        raise NotImplementedError

    def half_mass_radius(self) -> float:
        """
        Compute the half-mass radius.

        For Hernquist: r_h = (1 + sqrt(2)) * a ≈ 2.414 * a
        """
        return (1.0 + np.sqrt(2.0)) * self._scale_radius

    @property
    def scale_radius(self) -> float:
        """The Hernquist scale radius."""
        return self._scale_radius
