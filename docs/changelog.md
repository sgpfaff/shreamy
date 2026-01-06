# Changelog

All notable changes to shreamy will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ParticleSet` class for managing N-body particle data
- `ParticleHistory` class for storing time evolution
- `UnitSystem` class for galpy-compatible unit conversions
- `to_natural()` and `to_physical()` flexible conversion functions
- galpy orbit integration (`from_galpy_orbit`, `to_galpy_orbit`)
- astropy Quantity support in unit conversions
- Sphinx documentation with MyST and Furo theme

### In Progress

- Potentials module (galpy wrapper)
- Gravity solvers (direct summation, Barnes-Hut)
- Integrators (leapfrog, Hermite, RK4)
- Satellite models (Plummer, NFW, composite)
- Shream main simulation class

## [0.1.0] - TBD

Initial release with core functionality.
