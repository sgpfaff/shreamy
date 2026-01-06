# 🚧 Integration

```{admonition} Under Construction
:class: warning

This page is a placeholder. The integrators module is not yet implemented.
```

## Planned Content

This page will cover:

- **Leapfrog** — Second-order symplectic integrator (default)
- **RungeKutta4** — Fourth-order adaptive integrator
- **Hermite** — Fourth-order with jerk, for close encounters
- **Yoshida4** — Fourth-order symplectic
- **Timestep control** — Fixed vs adaptive stepping
- **Energy conservation** — Monitoring integration accuracy

## Preview

```python
from shreamy.integrators import Leapfrog, Hermite

# Simple leapfrog (fast, symplectic)
integrator = Leapfrog(dt=0.01)

# Hermite for close encounters
integrator = Hermite(dt=0.01, eta=0.02)

# Use in simulation
sim = Shream(particles, integrator=integrator)
```

## Integrator Comparison

| Integrator | Order | Symplectic | Adaptive | Best For |
|------------|-------|------------|----------|----------|
| Leapfrog | 2 | ✅ | ❌ | General use |
| RK4 | 4 | ❌ | ✅ | Smooth potentials |
| Hermite | 4 | ❌ | ✅ | Close encounters |
| Yoshida4 | 4 | ✅ | ❌ | Long integrations |

## Coming Soon

Check back after the integrators module is implemented, or see the [development roadmap](../contributing.md).
