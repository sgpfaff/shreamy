"""
Barnes-Hut tree code gravity solver.
"""

import numpy as np
from typing import Optional

from .base import GravitySolver


class BarnesHut(GravitySolver):
    """
    Barnes-Hut tree code gravity solver.

    Uses an octree to approximate distant particle groups as single
    particles, achieving O(N log N) complexity. Accuracy is controlled
    by the opening angle parameter.

    Best for large N (> 1000) when speed is important.

    Parameters
    ----------
    softening : float, optional
        Gravitational softening length.
    G : float, default 1.0
        Gravitational constant.
    theta : float, default 0.5
        Opening angle parameter. Smaller values are more accurate but slower.
        Typical values: 0.3-0.7.
    max_depth : int, default 20
        Maximum depth of the octree.
    """

    def __init__(
        self,
        softening: Optional[float] = None,
        G: float = 1.0,
        theta: float = 0.5,
        max_depth: int = 20,
    ):
        """Initialize Barnes-Hut solver."""
        super().__init__(softening=softening, G=G)
        self._theta = theta
        self._max_depth = max_depth
        self._root = None  # Will hold the tree root

    def accelerations(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
    ) -> np.ndarray:
        """Compute accelerations using Barnes-Hut algorithm."""
        raise NotImplementedError

    def potential(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
    ) -> np.ndarray:
        """Compute potential using Barnes-Hut algorithm."""
        raise NotImplementedError

    def build_tree(self, positions: np.ndarray, masses: np.ndarray) -> None:
        """Build the octree from particle data."""
        raise NotImplementedError

    @property
    def theta(self) -> float:
        """Opening angle parameter."""
        return self._theta

    @theta.setter
    def theta(self, value: float) -> None:
        """Set the opening angle."""
        self._theta = value

    @property
    def max_depth(self) -> int:
        """Maximum tree depth."""
        return self._max_depth


class OctreeNode:
    """
    A node in the Barnes-Hut octree.

    Each node represents a cubic region of space and can contain
    either a single particle (leaf) or pointers to 8 child nodes.

    Attributes
    ----------
    center : ndarray
        Center of the node's cubic region.
    size : float
        Side length of the cubic region.
    mass : float
        Total mass contained in this node.
    center_of_mass : ndarray
        Center of mass of particles in this node.
    children : list
        List of 8 child nodes (None if leaf).
    particle_index : int
        Index of particle if this is a leaf node.
    """

    def __init__(
        self,
        center: np.ndarray,
        size: float,
    ):
        """Initialize an octree node."""
        self.center = center
        self.size = size
        self.mass = 0.0
        self.center_of_mass = np.zeros(3)
        self.children = [None] * 8
        self.particle_index = None

    def is_leaf(self) -> bool:
        """Check if this node is a leaf (no children)."""
        return all(child is None for child in self.children)

    def get_octant(self, position: np.ndarray) -> int:
        """
        Determine which octant a position falls into.

        Returns
        -------
        int
            Octant index (0-7).
        """
        raise NotImplementedError

    def insert(
        self,
        position: np.ndarray,
        mass: float,
        index: int,
        depth: int = 0,
        max_depth: int = 20,
    ) -> None:
        """
        Insert a particle into this node.

        Parameters
        ----------
        position : ndarray
            Particle position.
        mass : float
            Particle mass.
        index : int
            Particle index.
        depth : int
            Current tree depth.
        max_depth : int
            Maximum allowed depth.
        """
        raise NotImplementedError

    def compute_moments(self) -> None:
        """Compute mass and center of mass for this node."""
        raise NotImplementedError
