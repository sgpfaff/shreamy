# Units

shreamy uses a **natural unit system** compatible with [galpy](https://www.galpy.org/), making it easy to combine shreamy simulations with galpy potentials and orbits.

## The Natural Unit System

In galpy's convention, units are defined by two scale parameters:

| Parameter | Symbol | Default Value | Meaning |
|-----------|--------|---------------|---------|
| Distance scale | $r_0$ | 8 kpc | Solar galactocentric distance |
| Velocity scale | $v_0$ | 220 km/s | Local circular velocity |

From these, derived units follow naturally:

| Quantity | Natural Unit | Physical Value (default) |
|----------|--------------|--------------------------|
| Position | $r_0$ | 8 kpc |
| Velocity | $v_0$ | 220 km/s |
| Time | $r_0 / v_0$ | ~35.6 Myr |
| Mass | $v_0^2 r_0 / G$ | ~$9.0 \times 10^{10} M_\odot$ |

The key advantage: **G = 1** in natural units, simplifying all gravitational calculations.

## UnitSystem Class

The `UnitSystem` class handles all unit conversions:

```python
from shreamy import UnitSystem

# Create with default galpy parameters
units = UnitSystem(ro=8.0, vo=220.0)

# Check the derived scales
print(f"Time unit: {units.time_in_gyr:.4f} Gyr")
# Output: Time unit: 0.0356 Gyr

print(f"Mass unit: {units.mass_in_msun:.2e} Msun")
# Output: Mass unit: 9.00e+10 Msun
```

### Converting to Physical Units

```python
import numpy as np
from shreamy import UnitSystem

units = UnitSystem()

# Positions: natural -> kpc
pos_natural = np.array([[1.0, 0.0, 0.0], [2.0, 1.0, 0.5]])
pos_kpc = units.position_to_physical(pos_natural)
# Result: [[ 8.  0.  0.]
#          [16.  8.  4.]]

# Velocities: natural -> km/s
vel_natural = np.array([[0.0, 1.0, 0.0]])
vel_kms = units.velocity_to_physical(vel_natural)
# Result: [[  0. 220.   0.]]

# Time: natural -> Gyr
t_natural = 10.0
t_gyr = units.time_to_physical(t_natural)
# Result: 0.36 Gyr

# Mass: natural -> solar masses
m_natural = 0.01
m_msun = units.mass_to_physical(m_natural)
# Result: 9.00e+08 Msun
```

### Converting from Physical Units

```python
from shreamy import UnitSystem

units = UnitSystem()

# kpc -> natural
pos_kpc = 16.0  # 16 kpc
pos_natural = units.position_from_physical(pos_kpc)
# Result: 2.0

# km/s -> natural  
vel_kms = 440.0
vel_natural = units.velocity_from_physical(vel_kms)
# Result: 2.0

# Gyr -> natural
t_gyr = 1.0
t_natural = units.time_from_physical(t_gyr)
# Result: ~28.1 natural time units
```

---

## Flexible Input with `to_natural()` and `to_physical()`

For maximum convenience, shreamy provides flexible conversion functions that accept multiple input types:

1. **Floats in natural units** (default)
2. **Floats in physical units** (with `physical=True`)
3. **Astropy Quantities** (auto-detected)

### to_natural()

```python
from shreamy import to_natural

# Already in natural units (default) - no conversion
pos = to_natural(2.0, 'position')
# Result: 2.0

# Physical units - needs conversion
pos = to_natural(16.0, 'position', physical=True)
# Result: 2.0 (16 kpc -> 2.0 natural units)

# Works for all unit types
vel = to_natural(440.0, 'velocity', physical=True)
# Result: 2.0 (440 km/s -> 2.0 natural units)
```

### to_physical()

```python
from shreamy import to_physical

# From natural to physical (default)
pos_kpc = to_physical(2.0, 'position')
# Result: 16.0 kpc

# Already physical - no conversion
pos_kpc = to_physical(16.0, 'position', natural=False)
# Result: 16.0 kpc
```

### Astropy Quantity Support

If you have [astropy](https://www.astropy.org/) installed, you can pass `Quantity` objects directly:

```python
import astropy.units as u
from shreamy import to_natural, to_physical

# Astropy Quantity -> natural units (auto-detected!)
pos = to_natural(16 * u.kpc, 'position')
# Result: 2.0

# Even with different input units
pos = to_natural(16000 * u.pc, 'position')
# Result: 2.0

# Astropy Quantity -> standard physical units
pos_kpc = to_physical(16000 * u.pc, 'position')
# Result: 16.0 kpc
```

---

## Time Utilities

shreamy provides convenience functions for working with time:

```python
from shreamy.units import gyr_to_natural, natural_to_gyr

# Convert between Gyr and natural time
t_nat = gyr_to_natural(1.0)   # 1 Gyr in natural units
# Result: ~28.1

t_gyr = natural_to_gyr(28.0)  # 28 natural units in Gyr
# Result: ~1.0 Gyr
```

### Dynamical Timescales

Calculate characteristic timescales for gravitational systems:

```python
from shreamy.units import dynamical_time, crossing_time, relaxation_time

# Dynamical time for a system with given density
rho = 0.1  # density in natural units
t_dyn = dynamical_time(rho, units='natural')
# Result: ~5.6 natural units

# Crossing time for a system with given size and velocity dispersion
R = 1.0      # radius in natural units
sigma = 0.5  # velocity dispersion in natural units
t_cross = crossing_time(R, sigma)
# Result: 2.0 natural units

# Two-body relaxation time
N = 10000    # number of particles
t_relax = relaxation_time(N, t_cross)
# Result: ~17,000 natural units
```

---

## galpy Orbit Integration

Convert between shreamy particle data and galpy Orbits:

```python
import numpy as np
from shreamy.units import from_galpy_orbit, to_galpy_orbit
from galpy.orbit import Orbit

# Create a galpy Orbit
o = Orbit([1.0, 0.1, 1.1, 0.0, 0.1, 0.0])  # [R, vR, vT, z, vz, phi]

# Convert to shreamy format (Cartesian positions/velocities)
positions, velocities = from_galpy_orbit(o)
# positions: array with x, y, z in natural units
# velocities: array with vx, vy, vz in natural units

# Convert back to galpy Orbit
orbit = to_galpy_orbit(positions, velocities)
```

---

## Default Units

shreamy maintains a global default unit system:

```python
from shreamy import get_default_units, set_default_units

# Get the current default
units = get_default_units()
print(f"Default: ro={units.ro} kpc, vo={units.vo} km/s")
# Output: Default: ro=8.0 kpc, vo=220.0 km/s

# Change the default (affects all subsequent operations)
set_default_units(ro=8.5, vo=230.0)

# Reset to standard galpy values
set_default_units(ro=8.0, vo=220.0)
```

---

## Physical Constants

For reference, shreamy uses these physical constants (in CGS):

```python
from shreamy.units import G_CGS, KPC_TO_CM, MSUN_TO_G, KM_TO_CM, GYR_TO_S

print(f"G = {G_CGS:.5e} cm³/(g·s²)")      # 6.67430e-08
print(f"1 kpc = {KPC_TO_CM:.6e} cm")      # 3.085678e+21
print(f"1 Msun = {MSUN_TO_G:.5e} g")      # 1.98841e+33
print(f"1 km = {KM_TO_CM:.0e} cm")        # 1e+05
print(f"1 Gyr = {GYR_TO_S:.5e} s")        # 3.15576e+16
```

---

## Summary

| Function | Purpose |
|----------|---------|
| `UnitSystem(ro, vo)` | Create unit system with custom scales |
| `to_natural(value, type)` | Convert to natural units (flexible input) |
| `to_physical(value, type)` | Convert to physical units (flexible input) |
| `get_default_units()` | Get global default unit system |
| `set_default_units(ro, vo)` | Set global default unit system |
| `from_galpy_orbit(orbit)` | Convert galpy Orbit to positions/velocities |
| `to_galpy_orbit(pos, vel)` | Convert positions/velocities to galpy Orbit |

See also: [Unit Conventions Philosophy](../philosophy/unit_conventions.md) for why we made these design choices.
