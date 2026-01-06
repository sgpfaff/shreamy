# 🚧 Satellites

```{admonition} Under Construction
:class: warning

This page is a placeholder. The satellite module is not yet implemented.
```

## Planned Content

This page will cover:

- **SatelliteModel** — Base class for satellite galaxy models
- **PlummerSatellite** — Plummer sphere model
- **HernquistSatellite** — Hernquist profile model
- **KingSatellite** — King model with tidal truncation
- **NFWSatellite** — NFW dark matter halo
- **CompositeSatellite** — Multi-component satellites (stars + DM)
- **Virialization** — Ensuring dynamical equilibrium

## Preview

```python
from shreamy.satellite import PlummerSatellite, CompositeSatellite, NFWSatellite

# Simple Plummer sphere
satellite = PlummerSatellite(
    mass=1e9,           # Solar masses
    scale_radius=1.0,   # kpc
    position=[50, 0, 0],
    velocity=[0, 150, 0]
)

# Sample particles
particles = satellite.sample(n_particles=10000)

# Multi-component: stars in DM halo
stellar = PlummerSatellite(mass=1e8, scale_radius=0.5, label='stars')
dm_halo = NFWSatellite(mass=1e10, scale_radius=5.0, label='dm')
composite = CompositeSatellite([stellar, dm_halo])

# Sample with virialization
particles = composite.sample(n_particles=50000, virialize=True)
```

## Coming Soon

Check back after the satellite module is implemented, or see the [development roadmap](../contributing.md).
