"""
Potentials subpackage for shreamy.

This subpackage provides wrappers and utilities for interfacing with galpy
potentials, as well as analytic potentials for the host galaxy.

Classes
-------
HostPotential
    Abstract base class for host galaxy potentials.
GalpyPotentialWrapper
    Wrapper around galpy potential objects.
PlummerPotential
    Plummer sphere potential.
HernquistPotential
    Hernquist profile potential.
NFWPotential
    NFW halo potential.
CompositePotential
    Combination of multiple potentials.
TimeDependentPotential
    Time-evolving potential wrapper.
"""

from .base import HostPotential
from .galpy_wrapper import GalpyPotentialWrapper, from_galpy
from .plummer import PlummerPotential
from .hernquist import HernquistPotential
from .nfw import NFWPotential
from .composite import CompositePotential
from .time_dependent import TimeDependentPotential
from .utils import dynamical_time, tidal_radius

__all__ = [
    "HostPotential",
    "GalpyPotentialWrapper",
    "from_galpy",
    "PlummerPotential",
    "HernquistPotential",
    "NFWPotential",
    "CompositePotential",
    "TimeDependentPotential",
    "dynamical_time",
    "tidal_radius",
]
