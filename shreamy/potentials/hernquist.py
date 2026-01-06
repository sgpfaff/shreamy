"""
Hernquist potential.

The Hernquist potential is a simple, spherically symmetric model that closely
approximates the de Vaucouleurs profile often used for elliptical galaxies
and bulges. It has a finite total mass and the density falls off as r^{-4}
at large radii.
"""

import numpy as np
from typing import Union

from .base import AnalyticPotential


class HernquistPotential(AnalyticPotential):
    """
    Hernquist potential.

    The potential is:

    .. math::

        \\Phi(r) = -\\frac{G M}{r + a}

    and the corresponding density profile is:

    .. math::

        \\rho(r) = \\frac{M}{2 \\pi} \\frac{a}{r (r + a)^3}

    Parameters
    ----------
    M : float
        Total mass in natural units.
    a : float
        Scale radius in natural units.
    G : float, default 1.0
        Gravitational constant. Default is 1.0 (natural units).

    Notes
    -----
    The Hernquist profile has:
    - Half-mass radius: r_h ≈ 1.8153 * a
    - Effective radius (de Vaucouleurs): r_e ≈ 1.8153 * a

    References
    ----------
    Hernquist, L. (1990). An analytical model for spherical galaxies and bulges.
    ApJ, 356, 359.

    Examples
    --------
    >>> pot = HernquistPotential(M=1.0, a=0.1)
    >>> x, y, z = np.array([1.0]), np.array([0.0]), np.array([0.0])
    >>> acc = pot.acceleration(x, y, z)
    """

    def __init__(self, M: float, a: float, G: float = 1.0):
        """Initialize Hernquist potential."""
        self._M = M
        self._a = a
        self._G = G

    @property
    def M(self) -> float:
        """Total mass."""
        return self._M

    @property
    def a(self) -> float:
        """Scale radius."""
        return self._a

    @property
    def G(self) -> float:
        """Gravitational constant."""
        return self._G

    def potential_value(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Compute Hernquist potential at given positions.

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
        return -self._G * self._M / (r + self._a)

    def acceleration(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: float = 0.0,
        check_origin: bool = True,
    ) -> np.ndarray:
        """
        Compute acceleration from Hernquist potential at given positions.

        The acceleration is:

        .. math::

            \\vec{a} = -\\nabla \\Phi = -\\frac{G M \\vec{r}}{r (r + a)^2}

        Parameters
        ----------
        x, y, z : ndarray
            Cartesian positions of shape (N,).
        t : float, default 0.0
            Time (unused, for interface compatibility).
        check_origin : bool, default True
            If True, handle the r=0 edge case safely. Set to False for
            ~30% speedup when you know all particles have r > 0.

        Returns
        -------
        ndarray
            Accelerations of shape (N, 3) with columns [ax, ay, az].
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)

        r = np.sqrt(x**2 + y**2 + z**2)

        if check_origin:
            # Handle r=0 case to avoid division by zero
            r_safe = np.where(r > 0, r, 1.0)
        else:
            # Fast path: assume r > 0 for all particles
            r_safe = r

        # a = -GM * r_hat / (r + a)^2 = -GM * r / (r * (r + a)^2)
        factor = -self._G * self._M / (r_safe * (r_safe + self._a) ** 2)

        if check_origin:
            # Set factor to 0 where r=0 (acceleration is 0 at origin)
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
        rho = (self._M / (2 * np.pi)) * self._a / (r_safe * (r_safe + self._a) ** 3)
        return rho

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
        return self._M * r**2 / (r + self._a) ** 2

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
        # v_c^2 = G * M(r) / r = G * M * r / (r + a)^2
        vc2 = self._G * self._M * r / (r + self._a) ** 2
        return np.sqrt(vc2)

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
        return f"HernquistPotential(M={self._M}, a={self._a}, G={self._G})"
