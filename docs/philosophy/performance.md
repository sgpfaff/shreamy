# 🚧 Performance vs Flexibility

```{admonition} Under Construction
:class: warning

This page is a placeholder. Content will be added as the performance-critical modules are implemented.
```

## Planned Content

This page will discuss:

### Algorithm Trade-offs

- **Direct summation vs Barnes-Hut**: When is O(N²) acceptable?
- **Softening length**: Balancing force accuracy vs close-encounter handling
- **Timestep selection**: Fixed vs adaptive, and their implications

### Performance Optimization

- **NumPy vectorization**: Why we avoid Python loops
- **Memory layout**: Contiguous arrays for cache efficiency
- **Optional Numba acceleration**: JIT compilation for hot paths
- **Parallelization**: OpenMP for tree codes

### When to Use What

| Particle Count | Gravity Solver | Integrator | Expected Runtime |
|---------------|----------------|------------|------------------|
| < 1,000 | Direct | Leapfrog | Seconds |
| 1,000 - 10,000 | Direct | Leapfrog | Minutes |
| 10,000 - 100,000 | Barnes-Hut | Leapfrog | Minutes to hours |
| > 100,000 | Consider compiled codes | — | — |

### Validation Philosophy

- Every algorithm tested against analytic solutions
- Energy conservation monitored by default
- Comparison with established codes (REBOUND, galpy)

## Coming Soon

Check back after the gravity and integrator modules are implemented.
