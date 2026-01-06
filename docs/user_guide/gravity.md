# 🚧 Gravity Solvers

```{admonition} Under Construction
:class: warning

This page is a placeholder. The gravity module is not yet implemented.
```

## Planned Content

This page will cover:

- **DirectSummation** — $O(N^2)$ pairwise gravity, exact but slow
- **BarnesHut** — $O(N \log N)$ tree code, fast approximation
- **Softening** — Preventing singularities in close encounters
- **Choosing a solver** — When to use which method

## Preview

```python
from shreamy.gravity import DirectSummation, BarnesHut

# Direct summation (exact, for N < 10,000)
gravity = DirectSummation(softening=0.01)

# Barnes-Hut tree (fast, for N > 10,000)
gravity = BarnesHut(theta=0.5, softening=0.01)

# Compute accelerations
accelerations = gravity.compute(particles)
```

## Algorithm Comparison

| Solver | Complexity | Accuracy | Best For |
|--------|------------|----------|----------|
| DirectSummation | $O(N^2)$ | Exact | N < 10,000 |
| BarnesHut | $O(N \log N)$ | θ-dependent | N > 10,000 |

## Coming Soon

Check back after the gravity module is implemented, or see the [development roadmap](../contributing.md).
