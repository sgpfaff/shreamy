"""
Base class for host galaxy potentials.
"""

import numpy as np
from abc import ABC, abstractmethod


class HostPotential(ABC):
    """
    Abstract base class for host galaxy potentials.

    A HostPotential computes the gravitational acceleration and potential
    from the host galaxy at given positions.

    This is an abstraction layer that allows shreamy to work with different
    potential implementations (primarily galpy, but extensible to others).
    """

    @abstractmethod
    def acceleration(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Compute acceleration from the host potential at given positions.

        Parameters
        ----------
        x, y, z : ndarray
            Cartesian positions of shape (N,).
        t : float, default 0.0
            Time (for time-dependent potentials).

        Returns
        -------
        ndarray
            Accelerations of shape (N, 3) with columns [ax, ay, az].
        """
        pass

    @abstractmethod
    def potential_value(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Compute potential value at given positions.

        Parameters
        ----------
        x, y, z : ndarray
            Cartesian positions of shape (N,).
        t : float, default 0.0
            Time (for time-dependent potentials).

        Returns
        -------
        ndarray
            Potential values of shape (N,).
        """
        pass

    def circular_velocity(
        self,
        R: np.ndarray,
        z: float = 0.0,
    ) -> np.ndarray:
        """
        Compute circular velocity in the potential.

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
        raise NotImplementedError

    def escape_velocity(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        """
        Compute escape velocity at given positions.
        """
        raise NotImplementedError


class AnalyticPotential(HostPotential):
    """
    Base class for simple analytic potentials.

    Useful for testing and for potentials not available in galpy.
    """

    pass
