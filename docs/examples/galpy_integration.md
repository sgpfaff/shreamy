# 🚧 galpy Integration

```{admonition} Under Construction
:class: warning

This example will be a fully executable notebook once the core modules are implemented.
```

## Overview

shreamy is designed to work seamlessly with galpy. This example demonstrates the workflow between galpy orbit analysis and shreamy N-body simulations.

## What You'll Learn

- Converting galpy Orbits to shreamy initial conditions
- Using galpy potentials as host galaxies
- Comparing test-particle (galpy) vs N-body (shreamy) results
- When self-gravity matters

## Preview Code

```python
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from shreamy import Shream
from shreamy.satellite import PlummerSatellite
from shreamy.units import from_galpy_orbit, to_galpy_orbit

# Start with a galpy orbit for the satellite center
orbit = Orbit([1.0, 0.1, 1.0, 0.0, 0.05, 0.0])  # Cylindrical coords
orbit.integrate(ts, MWPotential2014)

# Get position/velocity at some time
pos, vel = from_galpy_orbit(orbit, t=0.0)

# Create satellite at that location
satellite = PlummerSatellite(
    mass=1e9,
    scale_radius=1.0,
    position=pos,
    velocity=vel
)

# Run N-body simulation
particles = satellite.sample(n_particles=10000)
sim = Shream(particles, host_potential=MWPotential2014)
sim.integrate(t_end=5.0)

# Compare: center of mass trajectory vs galpy orbit
com_trajectory = sim.history.center_of_mass_history()
# Plot comparison...
```

## Coming Soon

This will become a fully executable Jupyter notebook demonstrating:
- The importance of self-gravity for satellite survival
- Mass loss not captured by test particles
- Stream morphology differences
