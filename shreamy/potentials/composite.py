"""
Composite potential from multiple components.
"""

import numpy as np
from typing import List

from .base import HostPotential


class CompositePotential(HostPotential):
    """
    A potential composed of multiple components.

    Parameters
    ----------
    potentials : list of HostPotential
        List of potential components to add together.
    """

    def __init__(self, potentials: List[HostPotential] = None):
        """Initialize composite potential."""
        self._potentials = potentials if potentials is not None else []

    def acceleration(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """Sum accelerations from all components."""
        raise NotImplementedError

    def potential_value(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """Sum potentials from all components."""
        raise NotImplementedError

    def add_potential(self, potential: HostPotential) -> None:
        """Add a potential component."""
        self._potentials.append(potential)

    @property
    def potentials(self) -> List[HostPotential]:
        """List of component potentials."""
        return self._potentials

    def __len__(self) -> int:
        """Number of component potentials."""
        return len(self._potentials)

    def __getitem__(self, index) -> HostPotential:
        """Get component potential by index."""
        return self._potentials[index]
