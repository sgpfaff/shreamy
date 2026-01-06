# 🚧 Shell Formation

```{admonition} Under Construction
:class: warning

This example will be a fully executable notebook once the core modules are implemented.
```

## Overview

Stellar shells form when a satellite galaxy falls nearly radially into its host. Phase wrapping creates the characteristic arc-like structures seen around many elliptical galaxies.

## What You'll Learn

- Setting up radial infall initial conditions
- Longer integration times needed for shell formation
- Identifying and characterizing shell structures
- Comparing with observations

## Preview Code

```python
from galpy.potential import HernquistPotential
from shreamy import Shream
from shreamy.satellite import PlummerSatellite

# Satellite on nearly radial orbit
satellite = PlummerSatellite(
    mass=5e8,
    scale_radius=0.5,
    position=[100, 0, 0],
    velocity=[-50, 10, 0]  # Nearly radial infall
)

# More particles for shell visibility
particles = satellite.sample(n_particles=50000)

# Elliptical galaxy host
host = HernquistPotential(amp=1e12, a=10)

# Run for longer to see phase wrapping
sim = Shream(particles, host_potential=host)
sim.integrate(t_end=10.0)  # 10 Gyr

# Analyze shell structure
shells = sim.find_shells()
```

## Coming Soon

This will become a fully executable Jupyter notebook.
