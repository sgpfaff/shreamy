"""
Units module for shreamy.

This module provides unit handling compatible with galpy's natural unit
system and astropy units. It enables conversion between physical units
(kpc, km/s, Msun) and the internal dimensionless units used for numerical
stability.
"""

import numpy as np
from typing import Optional, Tuple, Union


# =============================================================================
# Physical constants (in cgs)
# =============================================================================

# Gravitational constant
G_CGS = 6.67430e-8  # cm^3 / (g * s^2)

# Conversions
KPC_TO_CM = 3.085677581e21
MSUN_TO_G = 1.98841e33
KM_TO_CM = 1e5
YR_TO_S = 3.15576e7
GYR_TO_S = YR_TO_S * 1e9


# =============================================================================
# Unit System Class
# =============================================================================


class UnitSystem:
    """
    Defines a unit system for the simulation.

    shreamy uses galpy's convention of natural units defined by:
    - ro: distance scale (default 8 kpc)
    - vo: velocity scale (default 220 km/s)

    These define derived scales:
    - time scale: ro / vo
    - mass scale: vo^2 * ro / G

    Parameters
    ----------
    ro : float, default 8.0
        Distance unit in kpc.
    vo : float, default 220.0
        Velocity unit in km/s.

    Examples
    --------
    >>> units = UnitSystem(ro=8.0, vo=220.0)
    >>> units.time_in_gyr
    0.0352...
    >>> units.mass_in_msun
    5.17...e10
    """

    def __init__(self, ro: float = 8.0, vo: float = 220.0):
        """Initialize the unit system."""
        self._ro = ro  # kpc
        self._vo = vo  # km/s

    # =========================================================================
    # Unit scale properties
    # =========================================================================

    @property
    def ro(self) -> float:
        """Distance scale in kpc."""
        return self._ro

    @property
    def vo(self) -> float:
        """Velocity scale in km/s."""
        return self._vo

    @property
    def time_in_gyr(self) -> float:
        """Time unit in Gyr (ro/vo converted to Gyr)."""
        # ro [kpc] / vo [km/s] -> [s]
        # 1 kpc = 3.086e16 km
        time_s = (self._ro * 3.085677581e16) / self._vo  # in seconds
        return time_s / GYR_TO_S

    @property
    def mass_in_msun(self) -> float:
        """
        Mass unit in solar masses.

        Defined such that G = 1 in natural units:
        M = vo^2 * ro / G
        """
        # vo in cm/s
        vo_cgs = self._vo * KM_TO_CM
        # ro in cm
        ro_cgs = self._ro * KPC_TO_CM
        # Mass in grams
        mass_cgs = vo_cgs**2 * ro_cgs / G_CGS
        return mass_cgs / MSUN_TO_G

    @property
    def G(self) -> float:
        """Gravitational constant in natural units (always 1)."""
        return 1.0

    # =========================================================================
    # Conversion to physical units
    # =========================================================================

    def position_to_physical(self, x: np.ndarray) -> np.ndarray:
        """Convert positions from natural units to kpc."""
        return x * self._ro

    def velocity_to_physical(self, v: np.ndarray) -> np.ndarray:
        """Convert velocities from natural units to km/s."""
        return v * self._vo

    def time_to_physical(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert time from natural units to Gyr."""
        return t * self.time_in_gyr

    def mass_to_physical(self, m: np.ndarray) -> np.ndarray:
        """Convert mass from natural units to solar masses."""
        return m * self.mass_in_msun

    def energy_to_physical(self, E: np.ndarray) -> np.ndarray:
        """Convert energy from natural units to (km/s)^2."""
        return E * self._vo**2

    def acceleration_to_physical(self, a: np.ndarray) -> np.ndarray:
        """Convert acceleration from natural units to km/s/Gyr."""
        # Natural units: vo^2 / ro
        # Physical: km/s / Gyr
        return a * (self._vo**2 / self._ro) * (self.time_in_gyr * 1e9)  # Convert per kpc to per Gyr

    # =========================================================================
    # Conversion from physical units
    # =========================================================================

    def position_from_physical(self, x_kpc: np.ndarray) -> np.ndarray:
        """Convert positions from kpc to natural units."""
        return x_kpc / self._ro

    def velocity_from_physical(self, v_kms: np.ndarray) -> np.ndarray:
        """Convert velocities from km/s to natural units."""
        return v_kms / self._vo

    def time_from_physical(self, t_gyr: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert time from Gyr to natural units."""
        return t_gyr / self.time_in_gyr

    def mass_from_physical(self, m_msun: np.ndarray) -> np.ndarray:
        """Convert mass from solar masses to natural units."""
        return m_msun / self.mass_in_msun

    # =========================================================================
    # String representation
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"UnitSystem(ro={self._ro} kpc, vo={self._vo} km/s)\n"
            f"  time unit: {self.time_in_gyr:.4f} Gyr\n"
            f"  mass unit: {self.mass_in_msun:.4e} Msun"
        )


# =============================================================================
# Default unit system (galpy standard)
# =============================================================================

default_units = UnitSystem(ro=8.0, vo=220.0)


def get_default_units() -> UnitSystem:
    """Return the default unit system."""
    return default_units


def set_default_units(ro: float = 8.0, vo: float = 220.0) -> None:
    """
    Set the default unit system.

    Parameters
    ----------
    ro : float
        Distance scale in kpc.
    vo : float
        Velocity scale in km/s.
    """
    global default_units
    default_units = UnitSystem(ro=ro, vo=vo)


# =============================================================================
# Astropy integration
# =============================================================================


def to_astropy_units():
    """
    Return astropy unit equivalencies for the current unit system.

    Requires astropy to be installed.

    Returns
    -------
    dict
        Dictionary mapping natural unit quantities to astropy units.
    """
    raise NotImplementedError("Astropy integration not yet implemented")


def from_astropy_quantity(quantity, target: str) -> np.ndarray:
    """
    Convert an astropy Quantity to shreamy natural units.

    Parameters
    ----------
    quantity : astropy.units.Quantity
        The quantity to convert.
    target : str
        Target unit type: 'position', 'velocity', 'time', 'mass'.

    Returns
    -------
    ndarray
        Values in natural units.
    """
    raise NotImplementedError("Astropy integration not yet implemented")


# =============================================================================
# Galpy integration
# =============================================================================


def from_galpy_orbit(orbit, ro: float = 8.0, vo: float = 220.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract positions and velocities from a galpy Orbit in natural units.

    Parameters
    ----------
    orbit : galpy.orbit.Orbit
        A galpy Orbit object.
    ro, vo : float
        Unit scales to use.

    Returns
    -------
    positions : ndarray
        Shape (N, 3) positions in natural units.
    velocities : ndarray
        Shape (N, 3) velocities in natural units.
    """
    raise NotImplementedError


def to_galpy_orbit(
    positions: np.ndarray,
    velocities: np.ndarray,
    ro: float = 8.0,
    vo: float = 220.0,
):
    """
    Create a galpy Orbit from positions and velocities.

    Parameters
    ----------
    positions : ndarray
        Shape (N, 3) positions in natural units.
    velocities : ndarray
        Shape (N, 3) velocities in natural units.
    ro, vo : float
        Unit scales.

    Returns
    -------
    galpy.orbit.Orbit
    """
    raise NotImplementedError


# =============================================================================
# Time conversion utilities
# =============================================================================


def gyr_to_natural(t_gyr: float, ro: float = 8.0, vo: float = 220.0) -> float:
    """Convert time in Gyr to natural units."""
    return t_gyr / UnitSystem(ro, vo).time_in_gyr


def natural_to_gyr(t: float, ro: float = 8.0, vo: float = 220.0) -> float:
    """Convert time in natural units to Gyr."""
    return t * UnitSystem(ro, vo).time_in_gyr


def dynamical_time(
    density: float,
    units: str = "natural",
) -> float:
    """
    Compute the dynamical time for a given density.

    t_dyn = sqrt(3 * pi / (16 * G * rho))

    Parameters
    ----------
    density : float
        Mass density.
    units : str, default 'natural'
        'natural' or 'physical'.

    Returns
    -------
    float
        Dynamical time.
    """
    raise NotImplementedError


def crossing_time(
    velocity_dispersion: float,
    radius: float,
) -> float:
    """
    Compute the crossing time for a system.

    t_cross = R / sigma

    Parameters
    ----------
    velocity_dispersion : float
        1D velocity dispersion.
    radius : float
        Characteristic radius.

    Returns
    -------
    float
        Crossing time.
    """
    return radius / velocity_dispersion


def relaxation_time(
    n_particles: int,
    crossing_time: float,
) -> float:
    """
    Estimate the two-body relaxation time.

    t_relax ~ (N / ln(N)) * t_cross

    Parameters
    ----------
    n_particles : int
        Number of particles.
    crossing_time : float
        Crossing time.

    Returns
    -------
    float
        Relaxation time.
    """
    return (n_particles / np.log(n_particles)) * crossing_time
