"""
NFW (Navarro-Frenk-White) potential.

The NFW profile is a universal density profile for dark matter halos
found in cosmological N-body simulations. It has a cuspy center (rho ~ r^{-1})
and falls off as r^{-3} at large radii.
"""

import numpy as np
from typing import Union

from .base import AnalyticPotential


class NFWPotential(AnalyticPotential):
    """
    NFW (Navarro-Frenk-White) potential.

    The density profile is:

    .. math::

        \\rho(r) = \\frac{\\rho_s}{(r/r_s)(1 + r/r_s)^2}

    The corresponding potential is:

    .. math::

        \\Phi(r) = -\\frac{G M_s}{r} \\ln\\left(1 + \\frac{r}{r_s}\\right)

    where M_s = 4 * pi * rho_s * r_s^3.

    Parameters
    ----------
    M_s : float
        Characteristic mass M_s = 4 * pi * rho_s * r_s^3 in natural units.
        This is NOT the virial mass.
    r_s : float
        Scale radius in natural units.
    G : float, default 1.0
        Gravitational constant. Default is 1.0 (natural units).

    Notes
    -----
    Alternative parameterizations exist using virial mass M_vir and
    concentration c = r_vir / r_s. To convert:

        M_s = M_vir / [ln(1 + c) - c/(1 + c)]

    References
    ----------
    Navarro, J. F., Frenk, C. S., & White, S. D. M. (1996). The structure
    of cold dark matter halos. ApJ, 462, 563.

    Examples
    --------
    >>> pot = NFWPotential(M_s=1.0, r_s=0.2)
    >>> x, y, z = np.array([1.0]), np.array([0.0]), np.array([0.0])
    >>> acc = pot.acceleration(x, y, z)
    """

    def __init__(self, M_s: float, r_s: float, G: float = 1.0):
        """Initialize NFW potential."""
        self._M_s = M_s
        self._r_s = r_s
        self._G = G

    @property
    def M_s(self) -> float:
        """Characteristic mass."""
        return self._M_s

    @property
    def r_s(self) -> float:
        """Scale radius."""
        return self._r_s

    @property
    def G(self) -> float:
        """Gravitational constant."""
        return self._G

    @property
    def rho_s(self) -> float:
        """Characteristic density."""
        return self._M_s / (4 * np.pi * self._r_s**3)

    def potential_value(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Compute NFW potential at given positions.

        Parameters
        ----------
        x, y, z : ndarray
            Cartesian positions of shape (N,).
        t : float, default 0.0
            Time (unused, for interface compatibility).

        Returns
        -------
        ndarray
            Potential values of shape (N,).
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)

        r = np.sqrt(x**2 + y**2 + z**2)

        # Handle r=0 case
        r_safe = np.where(r > 0, r, 1e-10)

        # Phi(r) = -G * M_s * ln(1 + r/r_s) / r
        u = r_safe / self._r_s
        phi = -self._G * self._M_s * np.log1p(u) / r_safe

        return phi

    def acceleration(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Compute acceleration from NFW potential at given positions.

        The acceleration is:

        .. math::

            \\vec{a} = -\\nabla \\Phi = -\\frac{G M_s}{r^2} \\left[
                \\frac{\\ln(1 + r/r_s)}{r} - \\frac{1}{r + r_s}
            \\right] \\vec{r}

        which simplifies to:

        .. math::

            \\vec{a} = -\\frac{G M_s \\hat{r}}{r^2} \\left[
                \\ln(1 + r/r_s) - \\frac{r}{r + r_s}
            \\right]

        Parameters
        ----------
        x, y, z : ndarray
            Cartesian positions of shape (N,).
        t : float, default 0.0
            Time (unused, for interface compatibility).

        Returns
        -------
        ndarray
            Accelerations of shape (N, 3) with columns [ax, ay, az].
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)

        r = np.sqrt(x**2 + y**2 + z**2)

        # Handle r=0 case to avoid division by zero
        r_safe = np.where(r > 0, r, 1.0)

        u = r_safe / self._r_s

        # a = -d(Phi)/dr * r_hat
        # Phi = -G * M_s * ln(1 + u) / r
        # d(Phi)/dr = G * M_s * [ln(1+u)/r^2 - 1/(r_s * r * (1 + u))]
        # a = -d(Phi)/dr = -G * M_s * [ln(1+u)/r^2 - 1/(r_s * r * (1 + u))]
        #                = -G * M_s / r^2 * [ln(1+u) - r / (r_s * (1 + u))]
        #                = -G * M_s / r^2 * [ln(1+u) - u / (1 + u)]

        bracket = np.log1p(u) - u / (1 + u)
        factor = -self._G * self._M_s * bracket / (r_safe**3)

        # Set factor to 0 where r=0 (acceleration is 0 at origin by symmetry)
        factor = np.where(r > 0, factor, 0.0)

        ax = factor * x
        ay = factor * y
        az = factor * z

        return np.column_stack([ax, ay, az])

    def density(self, r: np.ndarray) -> np.ndarray:
        """
        Compute density at given radii.

        Parameters
        ----------
        r : ndarray
            Radii.

        Returns
        -------
        ndarray
            Density values.
        """
        r = np.atleast_1d(r)
        # Handle r=0 case
        r_safe = np.where(r > 0, r, 1e-10)
        u = r_safe / self._r_s
        return self.rho_s / (u * (1 + u) ** 2)

    def enclosed_mass(self, r: np.ndarray) -> np.ndarray:
        """
        Compute enclosed mass within given radii.

        Parameters
        ----------
        r : ndarray
            Radii.

        Returns
        -------
        ndarray
            Enclosed mass values.
        """
        r = np.atleast_1d(r)
        u = r / self._r_s
        # M(r) = M_s * [ln(1 + u) - u/(1 + u)]
        return self._M_s * (np.log1p(u) - u / (1 + u))

    def circular_velocity(self, R: np.ndarray, z: float = 0.0) -> np.ndarray:
        """
        Compute circular velocity at given cylindrical radii.

        Parameters
        ----------
        R : ndarray
            Cylindrical radii.
        z : float, default 0.0
            Height above plane.

        Returns
        -------
        ndarray
            Circular velocities.
        """
        R = np.atleast_1d(R)
        r = np.sqrt(R**2 + z**2)

        # Handle r=0 case
        r_safe = np.where(r > 0, r, 1e-10)

        M_enc = self.enclosed_mass(r_safe)
        # v_c^2 = G * M(r) / r
        vc2 = self._G * M_enc / r_safe
        return np.sqrt(np.maximum(vc2, 0))

    def escape_velocity(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        """
        Compute escape velocity at given positions.

        Parameters
        ----------
        x, y, z : ndarray
            Cartesian positions.

        Returns
        -------
        ndarray
            Escape velocities.
        """
        phi = self.potential_value(x, y, z)
        return np.sqrt(-2 * phi)

    def __repr__(self) -> str:
        return f"NFWPotential(M_s={self._M_s}, r_s={self._r_s}, G={self._G})"

    # ----- Class methods for alternative parameterizations -----

    @classmethod
    def from_virial(
        cls, M_vir: float, c: float, G: float = 1.0, Delta: float = 200.0
    ) -> "NFWPotential":
        """
        Create NFWPotential from virial mass and concentration.

        Parameters
        ----------
        M_vir : float
            Virial mass.
        c : float
            Concentration parameter c = r_vir / r_s.
        G : float, default 1.0
            Gravitational constant.
        Delta : float, default 200.0
            Overdensity factor (not used in conversion, but for reference).

        Returns
        -------
        NFWPotential
            Initialized potential.
        """
        # M_vir = M_s * [ln(1 + c) - c/(1 + c)]
        f_c = np.log1p(c) - c / (1 + c)
        M_s = M_vir / f_c

        # We need r_s to define the potential, but we don't have it directly
        # from just M_vir and c. We need the virial radius r_vir.
        # For now, assume r_vir is given implicitly through concentration.
        # This requires additional information (critical density, redshift, etc.)
        # For simplicity, we'll just return with M_s and require user to
        # provide r_s separately.
        raise NotImplementedError(
            "Use from_virial_radius instead, which requires r_vir explicitly."
        )

    @classmethod
    def from_virial_radius(
        cls, M_vir: float, r_vir: float, c: float, G: float = 1.0
    ) -> "NFWPotential":
        """
        Create NFWPotential from virial mass, virial radius, and concentration.

        Parameters
        ----------
        M_vir : float
            Virial mass.
        r_vir : float
            Virial radius.
        c : float
            Concentration parameter c = r_vir / r_s.
        G : float, default 1.0
            Gravitational constant.

        Returns
        -------
        NFWPotential
            Initialized potential.
        """
        r_s = r_vir / c
        f_c = np.log1p(c) - c / (1 + c)
        M_s = M_vir / f_c
        return cls(M_s=M_s, r_s=r_s, G=G)
