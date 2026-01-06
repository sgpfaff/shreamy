# Why shreamy?

## The Problem

Stellar streams and shells are powerful probes of galactic structure. The Milky Way alone hosts dozens of known streams from disrupted satellite galaxies, and upcoming surveys like LSST will discover hundreds more. To interpret these observations, we need simulations.

But there's a gap in the available tools:

### Test-Particle Codes (galpy, gala)

**Pros:**
- Fast: integrate millions of orbits in seconds
- Clean Python APIs
- Well-documented, well-tested

**Cons:**
- No self-gravity between particles
- Satellite dissolves instantly (no internal dynamics)
- Can't model satellite survival or mass loss rates

### Full N-body Codes (Gadget, AREPO, REBOUND)

**Pros:**
- Complete gravitational physics
- Handles close encounters, mergers, everything

**Cons:**
- Overkill for single satellite studies
- Complex setup and compilation
- Steep learning curve
- Slow: simulating the host galaxy with particles is expensive

## shreamy's Niche

shreamy fills the gap with a **hybrid approach**:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Host Galaxy: Analytic potential (fast, from galpy)        │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                                                     │   │
│   │   Satellite: N-body with self-gravity               │   │
│   │   (captures disruption physics)                     │   │
│   │                                                     │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

This gives you:
- **Self-gravity** where it matters (the satellite)
- **Speed** from analytic host potentials
- **Ease of use** from a Python API that feels like galpy

## When Self-Gravity Matters

Self-gravity is important when:

1. **Satellite survival time** — A self-gravitating satellite resists tidal disruption longer than test particles predict

2. **Stream gaps** — Perturbations from dark matter subhalos create gaps; self-gravity affects how streams respond

3. **Core vs cusp** — The internal density profile affects disruption; test particles can't capture this

4. **Dynamical friction** — The satellite's gravity on the host (approximated) slows its orbit

5. **Multi-component systems** — Stars and dark matter respond differently to tides

## Design Goals

shreamy is built around these principles:

### 1. galpy Compatibility

If you know galpy, you know shreamy. Same unit conventions, same potential interface, familiar API patterns. You should be able to go from galpy orbits to shreamy simulations in minutes.

### 2. Progressive Complexity

Start simple:
```python
satellite = PlummerSatellite(mass=1e9, scale_radius=1.0)
particles = satellite.sample(10000)
sim = Shream(particles, host_potential=MWPotential2014)
sim.integrate(t_end=5.0)
```

Then add complexity as needed:
```python
satellite = CompositeSatellite([stars, dm_halo])
particles = satellite.sample(50000, virialize=True)
sim = Shream(
    particles, 
    host_potential=MWPotential2014,
    gravity_solver=BarnesHut(theta=0.5),
    integrator=Hermite(eta=0.02)
)
```

### 3. Validation First

Every algorithm is tested against:
- Analytic solutions where they exist
- Established codes (galpy, REBOUND)
- Conservation laws (energy, momentum)

You should trust your results.

### 4. Research Ready

shreamy is designed for publication-quality simulations, not toy problems. Performance is tuned for 10,000-100,000 particles, typical for stream modeling. Larger simulations are possible but may benefit from compiled codes.

## What shreamy is NOT

- **A cosmological code** — We don't do structure formation, just isolated mergers
- **A collisional dynamics code** — We use softening, not regularization; not for globular cluster dynamics
- **A gas dynamics code** — Gravity only, no hydrodynamics

## Getting Started

Ready to try it? Head to the [Quick Start](../getting_started/quickstart.md).
