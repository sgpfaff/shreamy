# 🚧 Multi-Component Satellite

```{admonition} Under Construction
:class: warning

This example will be a fully executable notebook once the core modules are implemented.
```

## Overview

Real dwarf galaxies have stars embedded in dark matter halos. This example shows how to create and simulate multi-component satellites, tracking each component separately.

## What You'll Learn

- Creating composite satellites with multiple components
- Virialization of multi-component systems
- Component-specific analysis (e.g., stellar stream vs DM debris)
- Mass loss tracking by component

## Preview Code

```python
from shreamy.satellite import (
    PlummerSatellite, 
    NFWSatellite, 
    CompositeSatellite
)

# Stellar component: compact Plummer sphere
stars = PlummerSatellite(
    mass=1e8,           # 10^8 Msun in stars
    scale_radius=0.3,   # 300 pc
    label='stars'
)

# Dark matter component: extended NFW halo
dm = NFWSatellite(
    mass=1e10,          # 10^10 Msun in DM
    scale_radius=5.0,   # 5 kpc
    label='dm'
)

# Combine into composite satellite
satellite = CompositeSatellite(
    components=[stars, dm],
    position=[50, 0, 0],
    velocity=[0, 150, 0]
)

# Sample with proper virialization
particles = satellite.sample(
    n_particles={'stars': 5000, 'dm': 45000},
    virialize=True
)

# Access components after simulation
stellar_particles = sim.particles.get_component('stars')
dm_particles = sim.particles.get_component('dm')
```

## Coming Soon

This will become a fully executable Jupyter notebook.
