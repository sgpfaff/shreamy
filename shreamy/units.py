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


def to_astropy_units(ro: float = 8.0, vo: float = 220.0):
    """
    Return astropy unit equivalencies for the given unit system.

    Requires astropy to be installed.

    Parameters
    ----------
    ro, vo : float
        Unit scales.

    Returns
    -------
    dict
        Dictionary mapping unit types to astropy units.

    Examples
    --------
    >>> units_dict = to_astropy_units()
    >>> print(units_dict['position'])
    8.0 kpc
    """
    try:
        import astropy.units as u
    except ImportError:
        raise ImportError("astropy is required for this function. Install with: pip install astropy")

    unit_sys = UnitSystem(ro, vo)
    return {
        'position': ro * u.kpc,
        'velocity': vo * u.km / u.s,
        'time': unit_sys.time_in_gyr * u.Gyr,
        'mass': unit_sys.mass_in_msun * u.Msun,
    }


def from_astropy_quantity(quantity, target: str, ro: float = 8.0, vo: float = 220.0) -> np.ndarray:
    """
    Convert an astropy Quantity to shreamy natural units.

    Parameters
    ----------
    quantity : astropy.units.Quantity
        The quantity to convert.
    target : str
        Target unit type: 'position', 'velocity', 'time', 'mass'.
    ro, vo : float
        Unit scales.

    Returns
    -------
    ndarray
        Values in natural units.

    Examples
    --------
    >>> import astropy.units as u
    >>> pos = 16 * u.kpc
    >>> from_astropy_quantity(pos, 'position')
    2.0
    """
    try:
        import astropy.units as u
    except ImportError:
        raise ImportError("astropy is required for this function. Install with: pip install astropy")

    unit_sys = UnitSystem(ro, vo)

    if target == 'position':
        return quantity.to(u.kpc).value / ro
    elif target == 'velocity':
        return quantity.to(u.km / u.s).value / vo
    elif target == 'time':
        return quantity.to(u.Gyr).value / unit_sys.time_in_gyr
    elif target == 'mass':
        return quantity.to(u.Msun).value / unit_sys.mass_in_msun
    else:
        raise ValueError(f"Unknown target: {target}. Use 'position', 'velocity', 'time', or 'mass'.")


# =============================================================================
# Flexible input conversion (float with units flag OR astropy Quantity)
# =============================================================================


def _is_astropy_quantity(value) -> bool:
    """Check if value is an astropy Quantity."""
    try:
        import astropy.units as u
        return isinstance(value, u.Quantity)
    except ImportError:
        return False


def to_natural(value, unit_type: str, physical: bool = False,
               ro: float = 8.0, vo: float = 220.0) -> np.ndarray:
    """
    Convert a value to natural units.

    This is a flexible function that accepts:
    1. A float/array assumed to be in natural units (default)
    2. A float/array in physical units if physical=True
    3. An astropy Quantity (automatically detected and converted)

    Parameters
    ----------
    value : float, array-like, or astropy.units.Quantity
        The value to convert.
    unit_type : str
        Type of unit: 'position', 'velocity', 'time', 'mass'.
    physical : bool, default False
        If True and value is not an astropy Quantity, treat value as
        being in physical units (kpc, km/s, Gyr, Msun).
        Ignored if value is an astropy Quantity.
    ro, vo : float
        Unit scales for conversion.

    Returns
    -------
    ndarray
        Value in natural units.

    Examples
    --------
    >>> # Default: value is already in natural units
    >>> to_natural(2.0, 'position')
    2.0

    >>> # Physical units: need conversion
    >>> to_natural(16.0, 'position', physical=True)  # 16 kpc -> 2.0
    2.0

    >>> # Astropy Quantity: auto-detected
    >>> import astropy.units as u
    >>> to_natural(16 * u.kpc, 'position')
    2.0
    """
    # Check if astropy Quantity
    if _is_astropy_quantity(value):
        return from_astropy_quantity(value, unit_type, ro=ro, vo=vo)

    # Convert to array for consistent handling
    arr = np.atleast_1d(np.asarray(value))

    if not physical:
        # Already in natural units
        return arr if arr.size > 1 else arr[0]

    # Convert from physical units
    unit_sys = UnitSystem(ro, vo)

    if unit_type == 'position':
        result = unit_sys.position_from_physical(arr)
    elif unit_type == 'velocity':
        result = unit_sys.velocity_from_physical(arr)
    elif unit_type == 'time':
        result = unit_sys.time_from_physical(arr)
    elif unit_type == 'mass':
        result = unit_sys.mass_from_physical(arr)
    else:
        raise ValueError(f"Unknown unit_type: {unit_type}. Use 'position', 'velocity', 'time', or 'mass'.")

    return result if result.size > 1 else result.item()


def to_physical(value, unit_type: str, natural: bool = True,
                ro: float = 8.0, vo: float = 220.0) -> np.ndarray:
    """
    Convert a value to physical units.

    This is a flexible function that accepts:
    1. A float/array in natural units (default)
    2. A float/array already in physical units if natural=False
    3. An astropy Quantity (converted to physical, returned as float)

    Parameters
    ----------
    value : float, array-like, or astropy.units.Quantity
        The value to convert.
    unit_type : str
        Type of unit: 'position', 'velocity', 'time', 'mass'.
    natural : bool, default True
        If True and value is not an astropy Quantity, treat value as
        being in natural units and convert to physical.
        Ignored if value is an astropy Quantity.
    ro, vo : float
        Unit scales for conversion.

    Returns
    -------
    float or ndarray
        Value in physical units (kpc, km/s, Gyr, or Msun).

    Examples
    --------
    >>> # Default: convert from natural to physical
    >>> to_physical(2.0, 'position')  # 2.0 natural -> 16.0 kpc
    16.0

    >>> # Already physical: no conversion
    >>> to_physical(16.0, 'position', natural=False)
    16.0

    >>> # Astropy Quantity: extract value in standard physical units
    >>> import astropy.units as u
    >>> to_physical(16000 * u.pc, 'position')  # 16000 pc -> 16.0 kpc
    16.0
    """
    # Check if astropy Quantity - convert to standard physical units
    if _is_astropy_quantity(value):
        try:
            import astropy.units as u
        except ImportError:
            raise ImportError("astropy is required for Quantity input")

        if unit_type == 'position':
            return value.to(u.kpc).value
        elif unit_type == 'velocity':
            return value.to(u.km / u.s).value
        elif unit_type == 'time':
            return value.to(u.Gyr).value
        elif unit_type == 'mass':
            return value.to(u.Msun).value
        else:
            raise ValueError(f"Unknown unit_type: {unit_type}")

    # Convert to array for consistent handling
    arr = np.atleast_1d(np.asarray(value))

    if not natural:
        # Already in physical units
        return arr if arr.size > 1 else arr[0]

    # Convert from natural units
    unit_sys = UnitSystem(ro, vo)

    if unit_type == 'position':
        result = unit_sys.position_to_physical(arr)
    elif unit_type == 'velocity':
        result = unit_sys.velocity_to_physical(arr)
    elif unit_type == 'time':
        result = unit_sys.time_to_physical(arr)
    elif unit_type == 'mass':
        result = unit_sys.mass_to_physical(arr)
    else:
        raise ValueError(f"Unknown unit_type: {unit_type}. Use 'position', 'velocity', 'time', or 'mass'.")

    return result if result.size > 1 else result.item()


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

    Notes
    -----
    Requires galpy to be installed. If the orbit has been integrated,
    this returns the current state. For full orbit history, use
    orbit.getOrbit() directly.
    """
    try:
        # Get Cartesian coordinates in natural units
        # galpy returns values in natural units when use_physical=False
        x = orbit.x(use_physical=False)
        y = orbit.y(use_physical=False)
        z = orbit.z(use_physical=False)
        vx = orbit.vx(use_physical=False)
        vy = orbit.vy(use_physical=False)
        vz = orbit.vz(use_physical=False)

        # Handle both single orbits and arrays
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)
        vx = np.atleast_1d(vx)
        vy = np.atleast_1d(vy)
        vz = np.atleast_1d(vz)

        positions = np.column_stack([x, y, z])
        velocities = np.column_stack([vx, vy, vz])

        return positions, velocities

    except ImportError:
        raise ImportError("galpy is required for this function. Install with: pip install galpy")


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
        Shape (N, 3) or (3,) positions in natural units (Cartesian x, y, z).
    velocities : ndarray
        Shape (N, 3) or (3,) velocities in natural units (Cartesian vx, vy, vz).
    ro, vo : float
        Unit scales.

    Returns
    -------
    galpy.orbit.Orbit
        Orbit object initialized with given phase space coordinates.

    Notes
    -----
    Requires galpy to be installed. Internally converts Cartesian to
    cylindrical coordinates as required by galpy.
    """
    try:
        from galpy.orbit import Orbit
    except ImportError:
        raise ImportError("galpy is required for this function. Install with: pip install galpy")

    positions = np.atleast_2d(positions)
    velocities = np.atleast_2d(velocities)

    n_particles = positions.shape[0]

    orbits = []
    for i in range(n_particles):
        x, y, z = positions[i]
        vx, vy, vz = velocities[i]

        # Convert Cartesian to cylindrical for galpy
        R = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        if R > 0:
            vR = (x * vx + y * vy) / R
            vT = (x * vy - y * vx) / R
        else:
            vR = 0.0
            vT = 0.0

        # galpy expects [R, vR, vT, z, vz, phi]
        vxvv = [R, vR, vT, z, vz, phi]
        orb = Orbit(vxvv=vxvv, ro=ro, vo=vo)
        orbits.append(orb)

    if n_particles == 1:
        return orbits[0]
    return orbits


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
    ro: float = 8.0,
    vo: float = 220.0,
) -> float:
    """
    Compute the dynamical time for a given density.

    t_dyn = sqrt(3 * pi / (16 * G * rho))

    Parameters
    ----------
    density : float
        Mass density. In natural units if units='natural',
        in Msun/kpc^3 if units='physical'.
    units : str, default 'natural'
        'natural' or 'physical'.
    ro, vo : float
        Unit scales (only used when units='physical').

    Returns
    -------
    float
        Dynamical time in the same unit system as input.
    """
    if units == "natural":
        # G = 1 in natural units
        return np.sqrt(3 * np.pi / (16 * density))
    elif units == "physical":
        # Convert density to natural units, compute, then convert back
        unit_sys = UnitSystem(ro, vo)
        # density_natural = density_physical * (ro^3 / mass_unit)
        density_natural = density * (ro**3 / unit_sys.mass_in_msun)
        t_dyn_natural = np.sqrt(3 * np.pi / (16 * density_natural))
        return t_dyn_natural * unit_sys.time_in_gyr
    else:
        raise ValueError(f"Unknown units: {units}. Use 'natural' or 'physical'.")


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
