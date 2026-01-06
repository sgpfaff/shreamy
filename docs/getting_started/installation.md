# Installation

## Quick Install

The easiest way to install shreamy is via pip:

```bash
pip install shreamy
```

## Development Installation

To install the latest development version from GitHub:

```bash
git clone https://github.com/sgpfaff/shreamy.git
cd shreamy
pip install -e ".[dev]"
```

## Dependencies

### Required

- **Python** ≥ 3.9
- **numpy** — Array operations
- **scipy** — Scientific computing utilities  
- **galpy** — Galactic dynamics library (potentials, orbits)

### Optional

- **astropy** — Unit handling and physical constants
- **matplotlib** — Plotting and visualization
- **h5py** — HDF5 file I/O for snapshots

## Verifying Installation

After installation, verify everything is working:

```python
import shreamy
print(shreamy.__version__)

# Quick sanity check
from shreamy import ParticleSet
import numpy as np

particles = ParticleSet(
    positions=np.random.randn(100, 3),
    velocities=np.random.randn(100, 3),
    masses=np.ones(100)
)
print(f"Created {particles.n_particles} particles")
```

## Troubleshooting

### galpy OpenMP Issues

If you see warnings about OpenMP conflicts when using galpy, this is typically due to mixing pip-installed galpy with conda-installed numpy. Solutions:

1. **Use conda for everything**: `conda install -c conda-forge galpy`
2. **Or use pip for everything**: Install galpy with `pip install galpy --no-binary galpy`

See the [galpy installation docs](https://docs.galpy.org/en/latest/installation.html) for more details.
