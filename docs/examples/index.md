# 🚧 Examples

```{admonition} Under Construction
:class: warning

This section contains placeholder examples. Full executable notebooks will be added as modules are implemented.
```

## Planned Examples

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🌊 Basic Stellar Stream
:link: basic_stream
:link-type: doc

Simulate a simple Plummer sphere satellite disrupting into a stellar stream.
:::

:::{grid-item-card} 🐚 Shell Formation
:link: shell_formation
:link-type: doc

Radial infall producing shell structures around an elliptical galaxy.
:::

:::{grid-item-card} 🌌 Multi-Component Satellite
:link: multi_component
:link-type: doc

Stars embedded in a dark matter halo, with component-specific analysis.
:::

:::{grid-item-card} 🔗 galpy Integration
:link: galpy_integration
:link-type: doc

Seamless workflow between galpy orbits and shreamy simulations.
:::
::::

## Quick Preview

Once implemented, examples will look like this:

```python
from galpy.potential import MWPotential2014
from shreamy import Shream
from shreamy.satellite import PlummerSatellite

# Create satellite on a plunging orbit
satellite = PlummerSatellite(
    mass=1e9,
    scale_radius=1.0,
    position=[50, 0, 0],
    velocity=[0, 100, 50]
)

# Run simulation
particles = satellite.sample(n_particles=10000)
sim = Shream(particles, host_potential=MWPotential2014)
sim.integrate(t_end=5.0)

# Analyze the stream
sim.plot(projection='xy')
sim.plot_energy_evolution()
```

```{toctree}
:hidden:

basic_stream
shell_formation
multi_component
galpy_integration
```
