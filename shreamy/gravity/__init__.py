"""
Gravity subpackage for shreamy.

This subpackage provides different methods for computing gravitational
accelerations from particle self-interactions.

Classes
-------
GravitySolver
    Abstract base class for gravity solvers.
DirectSummation
    Direct N^2 summation solver.
BarnesHut
    Barnes-Hut tree code solver.
NoGravity
    Null solver for test particle limit.
"""

from .base import GravitySolver
from .direct import DirectSummation
from .tree import BarnesHut
from .null import NoGravity
from .utils import (
    get_gravity_solver,
    estimate_softening,
    pairwise_distances,
    pairwise_accelerations,
)

__all__ = [
    "GravitySolver",
    "DirectSummation",
    "BarnesHut",
    "NoGravity",
    "get_gravity_solver",
    "estimate_softening",
    "pairwise_distances",
    "pairwise_accelerations",
]
