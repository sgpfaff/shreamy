"""
Satellite subpackage for shreamy.

This subpackage provides classes for initializing satellite galaxies with
self-consistent phase space distributions.

Classes
-------
SatelliteModel
    Abstract base class for satellite models.
PlummerSatellite
    Plummer sphere satellite.
HernquistSatellite
    Hernquist profile satellite.
KingSatellite
    King (lowered isothermal) model.
NFWSatellite
    NFW dark matter halo.
GalpyDFSatellite
    Satellite from galpy distribution function.
CustomSatellite
    User-defined satellite model.
CompositeSatellite
    Multi-component satellite (e.g., dark matter + stellar).
"""

from .base import SatelliteModel
from .plummer import PlummerSatellite
from .hernquist import HernquistSatellite
from .king import KingSatellite
from .nfw import NFWSatellite
from .galpy_df import GalpyDFSatellite
from .custom import CustomSatellite
from .composite import CompositeSatellite
from .utils import (
    rejection_sample_spherical,
    sample_isotropic_velocities,
    hernquist_density,
    plummer_density,
    nfw_density,
    spherical_to_cartesian,
    add_bulk_motion,
    virialize_particles,
    compute_virial_ratio,
    compute_kinetic_energy,
    compute_potential_energy,
    estimate_dynamical_time,
    compute_half_mass_radius,
)

__all__ = [
    "SatelliteModel",
    "PlummerSatellite",
    "HernquistSatellite",
    "KingSatellite",
    "NFWSatellite",
    "GalpyDFSatellite",
    "CustomSatellite",
    "CompositeSatellite",
    "rejection_sample_spherical",
    "sample_isotropic_velocities",
    "hernquist_density",
    "plummer_density",
    "nfw_density",
    "spherical_to_cartesian",
    "add_bulk_motion",
    "virialize_particles",
    "compute_virial_ratio",
    "compute_kinetic_energy",
    "compute_potential_energy",
    "estimate_dynamical_time",
    "compute_half_mass_radius",
]
