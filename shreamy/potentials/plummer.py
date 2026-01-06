"""
Plummer sphere potential.

The Plummer potential is a simple, spherically symmetric model commonly used
for globular clusters and dwarf galaxies. It has a finite central density
and the density falls off as r^{-5} at large radii.
"""

import numpy as np
from typing import Union

from .base import AnalyticPotential


class PlummerPotential(AnalyticPotential):
    """
    Plummer sphere potential.

    The potential is:

    .. math::

        \\Phi(r) = -\\frac{G M}{\\sqrt{r^2 + b^2}}

    and the corresponding density profile is:

    .. math::

        \\rho(r) = \\frac{3 M}{4 \\pi b^3} \\left(1 + \\frac{r^2}{b^2}\\right)^{-5/2}

    Parameters
    ----------
    M : float
        Total mass in natural units (or physical units if G is adjusted).
    b : float
        Scale radius (Plummer radius) in natural units.
    G : float, default 1.0
        Gravitational constant. Default is 1.0 (natural units).

    Examples
    --------
    >>> pot = PlummerPotential(M=1.0, b=0.1)
    >>> x, y, z = np.array([1.0]), np.array([0.0]), np.array([0.0])
    >>> acc = pot.acceleration(x, y, z)
    >>> pot_val = pot.potential_value(x, y, z)
    """

    def __init__(self, M: float, b: float, G: float = 1.0):
        """Initialize Plummer potential."""
        self._M = M
        self._b = b
        self._G = G

    @property
    def M(self) -> float:
        """Total mass."""
        return self._M

    @property
    def b(self) -> float:
        """Scale radius (Plummer radius)."""
        return self._b

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
        Compute Plummer potential at given positions.

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

        r2 = x**2 + y**2 + z**2
        return -self._G * self._M / np.sqrt(r2 + self._b**2)

    def acceleration(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Compute acceleration from Plummer potential at given positions.

        The acceleration is:

        .. math::

            \\vec{a} = -\\nabla \\Phi = -\\frac{G M \\vec{r}}{(r^2 + b^2)^{3/2}}

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

        r2 = x**2 + y**2 + z**2
        denom = (r2 + self._b**2) ** 1.5

        # a = -GM * r / (r^2 + b^2)^(3/2)
        factor = -self._G * self._M / denom

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
        return (3 * self._M / (4 * np.pi * self._b**3)) * (
            1 + (r / self._b) ** 2
        ) ** (-2.5)

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
        return self._M * r**3 / (r**2 + self._b**2) ** 1.5

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
        r2 = R**2 + z**2
        # v_c^2 = G * M(r) / r = G * M * r^2 / (r^2 + b^2)^(3/2)
        vc2 = self._G * self._M * r2 / (r2 + self._b**2) ** 1.5
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
        return f"PlummerPotential(M={self._M}, b={self._b}, G={self._G})"
