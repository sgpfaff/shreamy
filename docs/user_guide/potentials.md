# 🚧 Potentials

```{admonition} Under Construction
:class: warning

This page is a placeholder. The potentials module is not yet implemented.
```

## Planned Content

This page will cover:

- **GalpyPotentialWrapper** — Using any galpy potential as a host galaxy
- **Custom potentials** — Defining your own analytic potentials
- **Composite potentials** — Combining multiple potential components
- **Time-dependent potentials** — Evolving host galaxies

## Preview

```python
from galpy.potential import MWPotential2014
from shreamy.potentials import GalpyPotentialWrapper

# Wrap a galpy potential
host = GalpyPotentialWrapper(MWPotential2014)

# Evaluate forces at a position
pos = [8.0, 0.0, 0.0]  # kpc (physical)
force = host.force(pos)  # Returns force in natural units
```

## Coming Soon

Check back after the potentials module is implemented, or see the [development roadmap](../contributing.md).
