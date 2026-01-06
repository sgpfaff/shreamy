"""
Satellite from galpy distribution function.
"""

import numpy as np
from typing import Optional, Union, Tuple, TYPE_CHECKING

from .base import SatelliteModel

if TYPE_CHECKING:
    from ..particle import ParticleSet, ParticleHistory


class GalpyDFSatellite(SatelliteModel):
    """
    A satellite initialized using a galpy distribution function.

    This allows using galpy's extensive library of distribution functions
    to create self-consistent satellite models.

    Parameters
    ----------
    df : galpy.df.DistributionFunction
        A galpy distribution function object.
    potential : galpy.potential.Potential
        The potential in which the DF is defined.
    mass : float
        Total mass to assign to sampled particles.
    position : array-like, optional
        Initial position.
    velocity : array-like, optional
        Initial velocity.
    ro : float, default 8.0
        galpy distance scale.
    vo : float, default 220.0
        galpy velocity scale.

    Examples
    --------
    >>> from galpy.df import isotropicHernquistdf
    >>> from galpy.potential import HernquistPotential
    >>> pot = HernquistPotential(amp=1e9, a=1.0)
    >>> df = isotropicHernquistdf(pot=pot)
    >>> sat = GalpyDFSatellite(df, pot, mass=1e9)
    >>> particles = sat.sample(10000)
    """

    def __init__(
        self,
        df,
        potential,
        mass: float,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        ro: float = 8.0,
        vo: float = 220.0,
    ):
        """Initialize satellite from galpy distribution function."""
        super().__init__(mass, position, velocity)
        self._df = df
        self._potential = potential
        self._ro = ro
        self._vo = vo

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
        Sample particles using galpy's DF sampling methods.

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
        """Compute density from galpy potential."""
        raise NotImplementedError

    def potential(self, r: np.ndarray) -> np.ndarray:
        """Compute potential from galpy potential."""
        raise NotImplementedError

    @property
    def df(self):
        """The galpy distribution function."""
        return self._df

    @property
    def galpy_potential(self):
        """The galpy potential."""
        return self._potential

    @property
    def ro(self) -> float:
        """Distance scale in kpc."""
        return self._ro

    @property
    def vo(self) -> float:
        """Velocity scale in km/s."""
        return self._vo
