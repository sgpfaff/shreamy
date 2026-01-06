# Core Concepts

Before diving into the API, it's helpful to understand the key concepts and design decisions behind shreamy.

## The Problem We're Solving

When a small satellite galaxy falls into a larger host galaxy, tidal forces stretch and eventually disrupt it, forming **stellar streams** (elongated tidal tails) or **shells** (phase-wrapped structures). Simulating this process accurately requires:

1. **A host galaxy potential** — The external gravitational field from the Milky Way (or another host)
2. **Self-gravity** — Gravitational interactions between satellite particles
3. **Accurate time integration** — Preserving energy and phase-space structure over Gyr timescales

## shreamy's Approach

### Test Particles + Self-Gravity

Unlike full cosmological N-body codes that simulate both the host and satellite with particles, shreamy uses a **hybrid approach**:

- The **host galaxy** is represented as an analytic potential (fast to evaluate)
- The **satellite** is represented as N-body particles with self-gravity

This is orders of magnitude faster than simulating $10^{11}$ host particles while still capturing the essential physics of satellite disruption.

### galpy Integration

shreamy is designed to work seamlessly with [galpy](https://www.galpy.org/), the popular galactic dynamics library. This means:

- **Use any galpy potential** as your host galaxy
- **Same unit conventions** — no conversion headaches
- **Familiar API** — if you know galpy, shreamy feels natural

## Key Objects

### ParticleSet

A `ParticleSet` holds the state of your N-body system at a single instant:

```python
from shreamy import ParticleSet

particles = ParticleSet(
    positions=pos_array,    # (N, 3) in natural units
    velocities=vel_array,   # (N, 3) in natural units
    masses=mass_array       # (N,) in natural units
)
```

It's immutable by design — operations return new `ParticleSet` objects rather than modifying in place.

### ParticleHistory

A `ParticleHistory` stores snapshots over time, enabling you to:

- Access the system state at any recorded time
- Interpolate between snapshots
- Extract individual particle trajectories

### UnitSystem

shreamy uses **natural units** following galpy's convention:

| Quantity | Natural unit | With $r_0=8$ kpc, $v_0=220$ km/s |
|----------|--------------|----------------------------------|
| Position | $r_0$ | 8 kpc |
| Velocity | $v_0$ | 220 km/s |
| Time | $r_0/v_0$ | ~35.6 Myr |
| Mass | $v_0^2 r_0 / G$ | ~$9 \times 10^{10} M_\odot$ |

The `UnitSystem` class handles conversions:

```python
from shreamy import UnitSystem

units = UnitSystem(ro=8.0, vo=220.0)
pos_kpc = units.position_to_physical(pos_natural)
```

See [Unit Conventions](../philosophy/unit_conventions.md) for the full philosophy.

## The Simulation Loop

At its core, a shreamy simulation does:

```
for each timestep:
    1. Compute host potential forces on all particles
    2. Compute self-gravity forces between particles  
    3. Update positions and velocities (integration step)
    4. Store snapshot if requested
```

The `Shream` class orchestrates this, letting you choose:

- **Gravity solver**: Direct summation, Barnes-Hut tree, or none
- **Integrator**: Leapfrog, RK4, Hermite, or Yoshida4
- **Timestep**: Fixed or adaptive

## What's Next?

- **[Particles](../user_guide/particles.md)** — Working with ParticleSet and ParticleHistory
- **[Units](../user_guide/units.md)** — Deep dive into unit handling
- **[Philosophy](../philosophy/index.md)** — Why we made these design choices
