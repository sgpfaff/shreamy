# 🚧 Basic Stellar Stream

```{admonition} Under Construction
:class: warning

This example will be a fully executable notebook once the core modules are implemented.
```

## Overview

This example demonstrates the simplest use case: a Plummer sphere satellite on a circular orbit, gradually disrupting into leading and trailing tidal tails.

## What You'll Learn

- Creating a satellite galaxy model
- Setting up initial conditions
- Running a simulation with self-gravity
- Visualizing the resulting stream

## Preview Code

```python
import numpy as np
import matplotlib.pyplot as plt
from galpy.potential import MWPotential2014
from shreamy import Shream
from shreamy.satellite import PlummerSatellite

# Satellite on a circular orbit at 50 kpc
satellite = PlummerSatellite(
    mass=1e9,           # 10^9 Msun
    scale_radius=1.0,   # 1 kpc
    position=[50, 0, 0],
    velocity=[0, 150, 0]  # Approximately circular
)

# Sample 10,000 particles
particles = satellite.sample(n_particles=10000, virialize=True)

# Create and run simulation
sim = Shream(
    particles,
    host_potential=MWPotential2014,
    gravity_solver='direct',
    integrator='leapfrog'
)

# Integrate for 5 Gyr
sim.integrate(t_end=5.0, dt=0.01, save_every=10)

# Plot final state
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(sim.particles.positions[:, 0],
           sim.particles.positions[:, 1],
           s=0.5, alpha=0.5)
ax.set_xlabel('X [natural units]')
ax.set_ylabel('Y [natural units]')
ax.set_aspect('equal')
plt.show()
```

## Coming Soon

This will become a fully executable Jupyter notebook with:
- Detailed explanations at each step
- Multiple visualizations
- Energy conservation checks
- Comparison with test-particle results
