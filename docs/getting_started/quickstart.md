# Quick Start

This guide will get you running your first shreamy simulation in under 5 minutes.

## Your First Stellar Stream

Let's simulate a small satellite galaxy being tidally disrupted by the Milky Way:

```python
import numpy as np
from galpy.potential import MWPotential2014
from shreamy import Shream, ParticleSet
from shreamy.satellite import PlummerSatellite

# 1. Create a satellite galaxy model
satellite = PlummerSatellite(
    mass=1e9,           # 10^9 solar masses
    scale_radius=1.0,   # 1 kpc scale radius
    position=[50, 0, 0],   # 50 kpc from galactic center
    velocity=[0, 150, 0]   # 150 km/s tangential velocity
)

# 2. Sample particles from the satellite
particles = satellite.sample(n_particles=10000)

# 3. Create the simulation
sim = Shream(
    particles,
    host_potential=MWPotential2014,
    gravity_solver='direct'  # Full N-body self-gravity
)

# 4. Integrate forward in time
sim.integrate(t_end=5.0)  # 5 Gyr

# 5. Access results
final_particles = sim.particles
history = sim.history  # Full time evolution
```

## Visualizing Results

```python
import matplotlib.pyplot as plt

# Plot the final configuration
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# X-Y projection
axes[0].scatter(final_particles.positions[:, 0], 
                final_particles.positions[:, 1], 
                s=0.1, alpha=0.5)
axes[0].set_xlabel('X [kpc]')
axes[0].set_ylabel('Y [kpc]')

# X-Z projection  
axes[1].scatter(final_particles.positions[:, 0],
                final_particles.positions[:, 2],
                s=0.1, alpha=0.5)
axes[1].set_xlabel('X [kpc]')
axes[1].set_ylabel('Z [kpc]')

# Y-Z projection
axes[2].scatter(final_particles.positions[:, 1],
                final_particles.positions[:, 2],
                s=0.1, alpha=0.5)
axes[2].set_xlabel('Y [kpc]')
axes[2].set_ylabel('Z [kpc]')

plt.tight_layout()
plt.show()
```

## Understanding the Output

### ParticleSet

The `ParticleSet` object holds positions, velocities, and masses:

```python
print(f"Number of particles: {final_particles.n_particles}")
print(f"Position shape: {final_particles.positions.shape}")  # (N, 3)
print(f"Velocity shape: {final_particles.velocities.shape}")  # (N, 3)
print(f"Total mass: {final_particles.total_mass}")
```

### ParticleHistory

The `history` object stores snapshots at each time step:

```python
print(f"Number of snapshots: {len(history)}")
print(f"Time array: {history.times}")

# Access a specific snapshot
snapshot_at_2gyr = history.at_time(2.0)

# Get the trajectory of a specific particle
trajectory = history.trajectory(particle_index=0)
```

## Next Steps

Now that you have the basics:

- **[Concepts](concepts.md)** — Understand the physics and design choices
- **[User Guide](../user_guide/index.md)** — Deep dives into each component
- **[Examples](../examples/index.md)** — More complex simulations
