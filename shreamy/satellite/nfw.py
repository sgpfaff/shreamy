"""
NFW satellite (dark matter subhalo) model.
"""

import numpy as np
from typing import Optional, Union, Tuple, TYPE_CHECKING

from .base import SatelliteModel

if TYPE_CHECKING:
    from ..particle import ParticleSet, ParticleHistory


class NFWSatellite(SatelliteModel):
    """
    A satellite with an NFW profile.

    Typically used for dark matter subhalos.

    rho(r) = rho_0 / (r/r_s) / (1 + r/r_s)^2

    Parameters
    ----------
    mass : float
        Virial mass (M_200).
    concentration : float
        NFW concentration (c = r_200 / r_s).
    position : array-like, optional
        Initial position.
    velocity : array-like, optional
        Initial velocity.
    """

    def __init__(
        self,
        mass: float,
        concentration: float,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
    ):
        """Initialize an NFW satellite."""
        super().__init__(mass, position, velocity)
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
        Sample particles from the NFW distribution.

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
        """Compute NFW density."""
        raise NotImplementedError

    def potential(self, r: np.ndarray) -> np.ndarray:
        """Compute NFW potential."""
        raise NotImplementedError

    @property
    def concentration(self) -> float:
        """NFW concentration parameter."""
        return self._concentration

    @property
    def scale_radius(self) -> float:
        """Scale radius r_s = r_200 / c."""
        raise NotImplementedError

    @property
    def virial_radius(self) -> float:
        """Virial radius r_200."""
        raise NotImplementedError
