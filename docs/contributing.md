# 🚧 Contributing

```{admonition} Under Construction
:class: warning

This page is a placeholder. Full contribution guidelines will be added.
```

## Development Setup

```bash
git clone https://github.com/sgpfaff/shreamy.git
cd shreamy
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest shreamy/tests/ -v
```

With coverage:

```bash
pytest shreamy/tests/ --cov=shreamy --cov-report=html
```

## Code Style

We use:
- **Black** for formatting
- **isort** for import sorting
- **flake8** for linting

## Roadmap

See the [GitHub Issues](https://github.com/sgpfaff/shreamy/issues) for planned features.

### Upcoming Modules

- [ ] Potentials (galpy wrapper)
- [ ] Gravity solvers (direct, Barnes-Hut)
- [ ] Integrators (leapfrog, Hermite)
- [ ] Satellite models (Plummer, NFW, composite)
- [ ] Shream class (main interface)
