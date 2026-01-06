# shreamy

**A quick, pythonic N-body solver for simulating stellar streams and shells from minor galaxy mergers.**

```{image} _static/stream_banner.svg
:alt: Stellar stream simulation
:class: dark-light
:align: center
:width: 100%
```

---

shreamy provides user-friendly tools for simulating the tidal disruption of satellite galaxies in host galaxy potentials, with full N-body self-gravity between satellite particles. It's designed to feel like a natural extension of [galpy](https://www.galpy.org/) while adding the physics you need for realistic stream and shell formation.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🚀 Quick Start
:link: getting_started/quickstart
:link-type: doc

Get up and running in 5 minutes with your first stellar stream simulation.
:::

:::{grid-item-card} 📖 User Guide
:link: user_guide/index
:link-type: doc

Deep dives into particles, units, potentials, and integration methods.
:::

:::{grid-item-card} 💡 Examples
:link: examples/index
:link-type: doc

Executable notebooks showing streams, shells, and multi-component satellites.
:::

:::{grid-item-card} 🔧 API Reference
:link: api/index
:link-type: doc

Complete API documentation auto-generated from docstrings.
:::
::::

## Why shreamy?

Most N-body codes fall into one of two camps:

1. **Full cosmological simulations** (e.g., Gadget, AREPO) — powerful but overkill for isolated merger studies, with steep learning curves
2. **Test-particle orbit integrators** (e.g., galpy, gala) — elegant and fast, but missing self-gravity

**shreamy fills the gap**: it provides fast, accurate N-body dynamics with self-gravity, wrapped in a clean Python API that feels like galpy. If you can set up a galpy Orbit, you can run a shreamy simulation.

```python
from galpy.potential import MWPotential2014
from shreamy import Shream
from shreamy.satellite import PlummerSatellite

# Create a Plummer sphere satellite
satellite = PlummerSatellite(
    mass=1e9,           # Solar masses  
    scale_radius=1.0,   # kpc
    position=[50, 0, 0],
    velocity=[0, 150, 0]
)

# Sample particles and run
particles = satellite.sample(n_particles=10000)
sim = Shream(particles, host_potential=MWPotential2014)
sim.integrate(t_end=5.0)  # 5 Gyr

# Visualize the stream
sim.plot()
```

## Design Philosophy

shreamy is built on several core principles:

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} galpy-native units
shreamy uses the same natural unit system as galpy ($r_0 = 8$ kpc, $v_0 = 220$ km/s), so potentials, orbits, and shreamy simulations work together seamlessly.
:::

:::{grid-item-card} Progressive complexity  
Start with a simple Plummer sphere, then add dark matter halos, tidal heating, or custom distribution functions—all with the same API.
:::

:::{grid-item-card} Validation-first
Every component has comprehensive tests against analytic solutions and established codes. Trust your results.
:::
::::

Read more in the [Philosophy section](philosophy/index.md).

## Installation

```bash
pip install shreamy
```

Or for the latest development version:

```bash
pip install git+https://github.com/sgpfaff/shreamy.git
```

**Dependencies**: numpy, scipy, galpy. Optional: astropy (for unit handling), matplotlib (for plotting).

## Contents

```{toctree}
:maxdepth: 2
:caption: Getting Started

getting_started/installation
getting_started/quickstart
getting_started/concepts
```

```{toctree}
:maxdepth: 2
:caption: User Guide

user_guide/index
user_guide/particles
user_guide/units
user_guide/potentials
user_guide/satellites
user_guide/gravity
user_guide/integration
```

```{toctree}
:maxdepth: 2
:caption: Examples

examples/index
```

```{toctree}
:maxdepth: 2
:caption: Philosophy

philosophy/index
philosophy/why_shreamy
philosophy/unit_conventions
philosophy/performance
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```

```{toctree}
:maxdepth: 1
:caption: Development

contributing
changelog
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
