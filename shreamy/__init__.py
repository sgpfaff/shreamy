"""
shreamy: A quick, pythonic N-body solver for simulating minor mergers
producing stellar streams and shells.

This package provides user-friendly tools for simulating the tidal
disruption of satellite galaxies in host galaxy potentials, with
full N-body self-gravity between satellite particles.

Main Classes
------------
Shream
    The core particle system class, analogous to galpy's Orbit but for
    a collection of interacting particles.

Modules
-------
particle
    Particle data structures (ParticleSet, ParticleHistory).
gravity
    Self-gravity solvers (DirectSummation, BarnesHut).
integrators
    Numerical integration schemes (Leapfrog, RK4, Hermite).
potentials
    Host galaxy potentials and galpy interface.
satellite
    Satellite galaxy models and initialization.
analysis
    Tools for analyzing streams, shells, and other structures.
io
    Input/output utilities for various file formats.
units
    Unit system handling compatible with galpy.

Examples
--------
Basic usage:

>>> from galpy.potential import MWPotential2014
>>> from shreamy import Shream
>>> from shreamy.satellite import PlummerSatellite
>>>
>>> # Create a satellite galaxy
>>> satellite = PlummerSatellite(mass=1e9, scale_radius=1.0,
...                               position=[50, 0, 0],
...                               velocity=[0, 150, 0])
>>> particles = satellite.sample(n_particles=10000)
>>>
>>> # Initialize the Shream with a host potential
>>> shream = Shream(particles, host_potential=MWPotential2014)
>>>
>>> # Integrate for 5 Gyr
>>> import numpy as np
>>> times = np.linspace(0, 5, 500)  # in galpy natural units
>>> shream.integrate(times)
>>>
>>> # Analyze the results
>>> from shreamy.analysis import classify_bound_unbound
>>> bound, unbound = classify_bound_unbound(shream)
"""

from .version import version as __version__

# Core classes
from .shream import Shream
from .particle import ParticleSet, ParticleHistory
from .gravity import GravitySolver, DirectSummation, BarnesHut, get_gravity_solver
from .integrators import (
    Integrator,
    Leapfrog,
    RungeKutta4,
    Hermite,
    get_integrator,
)
from .potentials import HostPotential, GalpyPotentialWrapper, from_galpy
from .satellite import (
    SatelliteModel,
    PlummerSatellite,
    HernquistSatellite,
    KingSatellite,
    NFWSatellite,
    GalpyDFSatellite,
    CustomSatellite,
    CompositeSatellite,
)
from .units import UnitSystem, get_default_units, set_default_units

__all__ = [
    # Core
    "Shream",
    # Particles
    "ParticleSet",
    "ParticleHistory",
    # Gravity
    "GravitySolver",
    "DirectSummation",
    "BarnesHut",
    "get_gravity_solver",
    # Integrators
    "Integrator",
    "Leapfrog",
    "RungeKutta4",
    "Hermite",
    "get_integrator",
    # Potentials
    "HostPotential",
    "GalpyPotentialWrapper",
    "from_galpy",
    # Satellites
    "SatelliteModel",
    "PlummerSatellite",
    "HernquistSatellite",
    "KingSatellite",
    "NFWSatellite",
    "GalpyDFSatellite",
    "CustomSatellite",
    "CompositeSatellite",
    # Units
    "UnitSystem",
    "get_default_units",
    "set_default_units",
]
