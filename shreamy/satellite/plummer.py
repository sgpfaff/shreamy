"""
Plummer sphere satellite model.
"""

import numpy as np
from typing import Optional, Union, Tuple, TYPE_CHECKING

from .base import SatelliteModel

if TYPE_CHECKING:
    from ..particle import ParticleSet, ParticleHistory


class PlummerSatellite(SatelliteModel):
    """
    A satellite with a Plummer density profile.

    rho(r) = (3*M)/(4*pi*b^3) * (1 + r^2/b^2)^(-5/2)

    This is a simple, cored model commonly used for dwarf spheroidal
    satellites and globular clusters.

    Parameters
    ----------
    mass : float
        Total mass of the satellite.
    scale_radius : float
        Plummer scale radius (b).
    position : array-like, optional
        Initial position [x, y, z].
    velocity : array-like, optional
        Initial velocity [vx, vy, vz].

    Examples
    --------
    >>> from shreamy.satellite import PlummerSatellite
    >>> sat = PlummerSatellite(mass=1e9, scale_radius=1.0)
    >>> particles = sat.sample(n_particles=10000)
    """

    def __init__(
        self,
        mass: float,
        scale_radius: float,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
    ):
        """Initialize a Plummer satellite."""
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
        Sample particles from the Plummer distribution.

        Uses the von Neumann rejection method for velocities to ensure
        the sampled particles are in dynamical equilibrium.

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
        """Compute Plummer density."""
        raise NotImplementedError

    def potential(self, r: np.ndarray) -> np.ndarray:
        """Compute Plummer potential."""
        raise NotImplementedError

    def escape_velocity(self, r: np.ndarray) -> np.ndarray:
        """Compute escape velocity at radius r."""
        raise NotImplementedError

    def half_mass_radius(self) -> float:
        """
        Compute the half-mass radius.

        For a Plummer sphere: r_h = b / sqrt(2^(2/3) - 1) ≈ 1.305 * b
        """
        return self._scale_radius / np.sqrt(2.0 ** (2.0 / 3.0) - 1.0)

    def velocity_dispersion(self) -> float:
        """
        Compute the central velocity dispersion.

        For a Plummer sphere: sigma_0 = sqrt(G*M / (6*b))
        """
        raise NotImplementedError

    @property
    def scale_radius(self) -> float:
        """The Plummer scale radius."""
        return self._scale_radius
