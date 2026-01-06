"""
Galpy potential wrapper for shreamy.
"""

import numpy as np
from typing import Optional, Union, List

from .base import HostPotential


class GalpyPotentialWrapper(HostPotential):
    """
    Wrapper around galpy potential objects.

    This class provides an interface between shreamy and galpy potentials,
    translating between galpy's conventions and shreamy's internal format.

    Parameters
    ----------
    potential : galpy.potential.Potential or list
        A galpy Potential object or list of Potentials.
    ro : float, default 8.0
        Distance scale (kpc) for galpy natural units.
    vo : float, default 220.0
        Velocity scale (km/s) for galpy natural units.

    Examples
    --------
    >>> from galpy.potential import MWPotential2014
    >>> from shreamy.potentials import GalpyPotentialWrapper
    >>> pot = GalpyPotentialWrapper(MWPotential2014)
    >>> accel = pot.acceleration(x, y, z)
    """

    def __init__(
        self,
        potential,
        ro: float = 8.0,
        vo: float = 220.0,
    ):
        """Initialize wrapper around a galpy potential."""
        self._potential = potential
        self._ro = ro
        self._vo = vo

    def acceleration(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Compute acceleration using galpy's evaluateRforces and evaluatezforces.

        Converts between Cartesian and cylindrical coordinates as needed.
        """
        raise NotImplementedError

    def potential_value(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """Compute potential using galpy's evaluatePotentials."""
        raise NotImplementedError

    @property
    def galpy_potential(self):
        """Return the underlying galpy potential object."""
        return self._potential

    @property
    def ro(self) -> float:
        """Distance scale in kpc."""
        return self._ro

    @property
    def vo(self) -> float:
        """Velocity scale in km/s."""
        return self._vo


def from_galpy(potential, ro: float = 8.0, vo: float = 220.0) -> HostPotential:
    """
    Create a shreamy HostPotential from a galpy potential.

    This is a convenience function equivalent to GalpyPotentialWrapper.

    Parameters
    ----------
    potential : galpy.potential.Potential or list
        galpy potential or list of potentials.
    ro : float, default 8.0
        Distance scale (kpc).
    vo : float, default 220.0
        Velocity scale (km/s).

    Returns
    -------
    HostPotential
    """
    return GalpyPotentialWrapper(potential, ro=ro, vo=vo)
