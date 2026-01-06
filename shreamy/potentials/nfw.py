"""
NFW (Navarro-Frenk-White) halo potential.
"""

import numpy as np

from .base import AnalyticPotential


class NFWPotential(AnalyticPotential):
    """
    NFW (Navarro-Frenk-White) halo potential.

    rho(r) = rho_0 / (r/r_s) / (1 + r/r_s)^2

    Parameters
    ----------
    M_vir : float
        Virial mass.
    c : float
        Concentration parameter.
    G : float, default 1.0
        Gravitational constant.
    """

    def __init__(self, M_vir: float, c: float, G: float = 1.0):
        """Initialize NFW potential."""
        self._M_vir = M_vir
        self._c = c
        self._G = G

    def acceleration(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """Compute NFW acceleration."""
        raise NotImplementedError

    def potential_value(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """Compute NFW potential."""
        raise NotImplementedError

    @property
    def M_vir(self) -> float:
        """Virial mass."""
        return self._M_vir

    @property
    def c(self) -> float:
        """Concentration parameter."""
        return self._c

    @property
    def r_s(self) -> float:
        """Scale radius (r_vir / c)."""
        raise NotImplementedError

    @property
    def G(self) -> float:
        """Gravitational constant."""
        return self._G
