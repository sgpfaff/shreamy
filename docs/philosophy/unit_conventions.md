# Unit Conventions

This page explains the philosophy behind shreamy's unit system and how to work with it effectively.

## Why Natural Units?

In N-body simulations, we could work entirely in SI or CGS units. But this leads to problems:

1. **Numerical precision**: Galactic masses ($10^{42}$ kg) and timesteps ($10^{14}$ s) span many orders of magnitude

2. **Round-off errors**: Multiplying very large and very small numbers loses precision

3. **G is awkward**: $G = 6.674 \times 10^{-11}$ m³/(kg·s²) is an ugly constant to carry around

**Natural units solve all of this** by choosing scales where typical values are $\mathcal{O}(1)$:

| Quantity | Typical value | In natural units |
|----------|---------------|------------------|
| Galactocentric distance | 8 kpc | 1.0 |
| Circular velocity | 220 km/s | 1.0 |
| Orbital period | ~200 Myr | ~2π |
| Milky Way mass | ~10¹² M☉ | ~10 |

And crucially: **G = 1** in natural units, so gravitational acceleration is just:

$$\vec{a} = -\sum_j \frac{m_j (\vec{r}_i - \vec{r}_j)}{|\vec{r}_i - \vec{r}_j|^3}$$

No constants to track, no unit conversion errors.

## The galpy Convention

galpy established a sensible standard for galactic dynamics:

$$r_0 = 8 \text{ kpc}, \quad v_0 = 220 \text{ km/s}$$

From these, everything else follows:

$$t_0 = \frac{r_0}{v_0} \approx 35.6 \text{ Myr}$$

$$M_0 = \frac{v_0^2 r_0}{G} \approx 9.0 \times 10^{10} M_\odot$$

shreamy adopts this convention exactly, so **galpy potentials work directly in shreamy** with no conversion needed.

## Working with Units in Practice

### Rule 1: Internal calculations use natural units

All positions, velocities, times, and masses inside shreamy are in natural units. This includes:
- Particle arrays in `ParticleSet`
- Integration timesteps
- Potential evaluations

### Rule 2: Convert at the boundaries

When you input data or output results, that's when to convert:

```python
from shreamy import UnitSystem, to_natural, to_physical

# Input: you have physical values
pos_kpc = 50.0
vel_kms = 150.0

# Convert to natural for shreamy
pos = to_natural(pos_kpc, 'position', physical=True)
vel = to_natural(vel_kms, 'velocity', physical=True)

# ... run simulation ...

# Output: convert back for plotting/analysis
final_pos_kpc = to_physical(sim.particles.positions, 'position')
```

### Rule 3: Let shreamy handle it when possible

Many shreamy functions accept physical units directly:

```python
# PlummerSatellite accepts physical units by default
satellite = PlummerSatellite(
    mass=1e9,           # Solar masses (physical)
    scale_radius=1.0,   # kpc (physical)
    position=[50, 0, 0],   # kpc (physical)
    velocity=[0, 150, 0]   # km/s (physical)
)
```

The conversion happens internally.

### Rule 4: astropy integration for complex cases

For complex unit handling, use astropy Quantities:

```python
import astropy.units as u
from shreamy import to_natural

# astropy handles the conversion
distance = 50 * u.kpc
velocity = 150 * u.km / u.s

pos = to_natural(distance, 'position')  # Auto-detected!
vel = to_natural(velocity, 'velocity')
```

This is especially useful when combining data from different sources with different unit conventions.

## Common Gotchas

### Time units are not intuitive

The natural time unit is ~35.6 Myr, so:
- 1 Gyr ≈ 28 natural time units
- 5 Gyr ≈ 140 natural time units

When setting `t_end` or `dt`, think in natural units or use the conversion:

```python
from shreamy.units import gyr_to_natural

t_end = gyr_to_natural(5.0)  # 5 Gyr in natural units
dt = gyr_to_natural(0.001)   # 1 Myr timestep
```

### Mass units are large

The natural mass unit is ~9×10¹⁰ M☉, so:
- A 10⁹ M☉ satellite ≈ 0.011 natural mass units
- A 10⁸ M☉ satellite ≈ 0.0011 natural mass units

This is fine — shreamy handles small masses correctly.

### Energy units

Energy is in units of $v_0^2 = (220 \text{ km/s})^2$:

```python
# Specific energy (per unit mass)
E_natural = -0.5  # Bound orbit
E_physical = units.energy_to_physical(E_natural)  # (km/s)²
```

## Why Not Just Use astropy Units Everywhere?

We considered making shreamy fully astropy-native, but decided against it:

1. **Performance**: Unit checking on every array operation adds overhead
2. **galpy compatibility**: galpy uses dimensionless arrays internally
3. **Simplicity**: Natural units make the physics cleaner

astropy is supported for I/O and conversion, but internal calculations stay dimensionless for speed.

## Summary

| Situation | Recommendation |
|-----------|----------------|
| Inside shreamy | Natural units (dimensionless) |
| User input | Physical units or astropy Quantities |
| Output/plotting | Convert to physical with `to_physical()` |
| galpy interop | Direct compatibility, no conversion needed |

See [Units User Guide](../user_guide/units.md) for the full API reference.
