# Particles

The particle module provides two core classes for managing N-body data:

- **`ParticleSet`** — Holds positions, velocities, and masses at a single instant
- **`ParticleHistory`** — Stores snapshots over time for analyzing evolution

## ParticleSet

### Creating a ParticleSet

The most direct way is to provide arrays directly:

```python
import numpy as np
from shreamy import ParticleSet

# Create 1000 particles
N = 1000
particles = ParticleSet(
    positions=np.random.randn(N, 3),    # (N, 3) array
    velocities=np.random.randn(N, 3),   # (N, 3) array  
    masses=np.ones(N)                   # (N,) array
)
```

### Accessing Data

```python
# Basic properties
print(particles.n_particles)  # Number of particles
print(particles.total_mass)   # Sum of all masses

# Arrays (read-only views)
pos = particles.positions     # (N, 3)
vel = particles.velocities    # (N, 3)
m = particles.masses          # (N,)

# Derived quantities
com = particles.center_of_mass          # (3,) array
com_vel = particles.center_of_mass_velocity  # (3,) array
ke = particles.kinetic_energy           # Scalar
```

### Slicing and Indexing

`ParticleSet` supports numpy-style indexing:

```python
# Get a single particle (returns ParticleSet with n=1)
p0 = particles[0]

# Slice a range
first_100 = particles[:100]

# Boolean mask
bound = particles.energies < 0
bound_particles = particles[bound]
```

### Operations

```python
# Combine particle sets
all_particles = particles1 + particles2

# Shift to center of mass frame
centered = particles.center()

# Transform coordinates
rotated = particles.rotate(angle=np.pi/4, axis='z')
shifted = particles.translate(offset=[10, 0, 0])
boosted = particles.boost(velocity=[0, 100, 0])
```

### Component Labels

For multi-component satellites (e.g., stars + dark matter):

```python
# Create with labels
labels = np.array(['star'] * 500 + ['dm'] * 500)
particles = ParticleSet(positions, velocities, masses, labels=labels)

# Access by component
stars = particles.get_component('star')
dark_matter = particles.get_component('dm')

# Check available components
print(particles.component_labels)  # ['star', 'dm']
```

### I/O

```python
# Save to HDF5
particles.to_hdf5('snapshot.h5')

# Load from HDF5
loaded = ParticleSet.from_hdf5('snapshot.h5')

# Save to ASCII (positions, velocities, masses as columns)
particles.to_ascii('snapshot.txt')
```

---

## ParticleHistory

`ParticleHistory` stores the time evolution of a particle system.

### Creating from Snapshots

```python
from shreamy import ParticleHistory

# From a list of (time, ParticleSet) tuples
history = ParticleHistory([
    (0.0, initial_particles),
    (1.0, particles_at_1gyr),
    (2.0, particles_at_2gyr),
])

# Or build incrementally
history = ParticleHistory()
history.add_snapshot(0.0, initial_particles)
history.add_snapshot(1.0, particles_at_1gyr)
```

### Accessing Snapshots

```python
# Number of snapshots
print(len(history))  # 3

# Time array
print(history.times)  # [0.0, 1.0, 2.0]

# Get snapshot at specific time
snap = history.at_time(1.0)  # Returns ParticleSet

# Interpolate between snapshots
snap = history.at_time(1.5, interpolate=True)

# Iterate over all snapshots
for t, particles in history:
    print(f"t={t}: {particles.n_particles} particles")
```

### Particle Trajectories

Extract the trajectory of individual particles:

```python
# Get trajectory of particle 0
traj = history.trajectory(0)
print(traj['times'])       # Time array
print(traj['positions'])   # (n_snapshots, 3) positions
print(traj['velocities'])  # (n_snapshots, 3) velocities

# Get trajectories for multiple particles
trajs = history.trajectories([0, 1, 2])
```

### Analysis

```python
# Center of mass evolution
com_history = history.center_of_mass_history()  # (n_snapshots, 3)

# Total energy evolution (if energies are stored)
energy_history = history.total_energy_history()

# Half-mass radius evolution
rhalf = history.half_mass_radius_history()
```

### I/O

```python
# Save entire history to HDF5
history.to_hdf5('evolution.h5')

# Load
history = ParticleHistory.from_hdf5('evolution.h5')
```

---

## Design Philosophy

### Immutability

`ParticleSet` is designed to be **immutable**. Operations like `translate()`, `rotate()`, and `boost()` return new objects rather than modifying in place. This:

- Prevents accidental data corruption
- Makes it safe to pass `ParticleSet` objects to functions
- Enables easy comparison between states

### Memory Efficiency

For large simulations, `ParticleHistory` can be configured to:

- Store only every Nth snapshot
- Store only a subset of particles
- Write directly to disk (streaming mode)

```python
history = ParticleHistory(
    stride=10,              # Store every 10th step
    particle_stride=100,    # Store every 100th particle
    stream_to='output.h5'   # Write to disk, don't keep in memory
)
```

### Units

All particle data is stored in **natural units** (see [Units](units.md)). Use the `UnitSystem` class for conversions when needed.
