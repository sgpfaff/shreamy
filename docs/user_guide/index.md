# User Guide

Welcome to the shreamy User Guide. This section provides in-depth documentation for each component of the library.

## Overview

shreamy is organized into several modules, each handling a specific aspect of N-body simulation:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Particles
:link: particles
:link-type: doc

The `ParticleSet` and `ParticleHistory` classes for managing N-body state.
:::

:::{grid-item-card} Units  
:link: units
:link-type: doc

Unit system compatible with galpy, plus flexible conversion utilities.
:::

:::{grid-item-card} Potentials
:link: potentials
:link-type: doc

Host galaxy potentials and galpy integration.
:::

:::{grid-item-card} Satellites
:link: satellites
:link-type: doc

Satellite galaxy models: Plummer, Hernquist, King, NFW, and composites.
:::

:::{grid-item-card} Gravity
:link: gravity
:link-type: doc

Self-gravity solvers: direct summation and Barnes-Hut tree.
:::

:::{grid-item-card} Integration
:link: integration
:link-type: doc

Time integrators: Leapfrog, RK4, Hermite, Yoshida4.
:::
::::

## Reading Order

If you're new to shreamy, we recommend reading in this order:

1. **[Particles](particles.md)** — Core data structures
2. **[Units](units.md)** — Understanding the unit system
3. **[Satellites](satellites.md)** — Creating initial conditions
4. **[Potentials](potentials.md)** — Host galaxy setup
5. **[Gravity](gravity.md)** — Self-gravity options
6. **[Integration](integration.md)** — Running simulations

Each page includes runnable code examples that you can copy and modify.

```{toctree}
:hidden:
:maxdepth: 2

particles
units
potentials
satellites
gravity
integration
```
