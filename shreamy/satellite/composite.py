"""
Composite satellite models combining multiple components.
"""

import numpy as np
from typing import Dict, Optional, Union, Tuple, TYPE_CHECKING

from .base import SatelliteModel

if TYPE_CHECKING:
    from ..particle import ParticleSet, ParticleHistory


class CompositeSatellite(SatelliteModel):
    """
    A satellite composed of multiple components (e.g., dark matter + stellar).

    This class combines multiple SatelliteModel instances into a single
    composite satellite, where each component can have its own density
    profile and sampling strategy. Particles are automatically tagged
    with their component name.

    Parameters
    ----------
    components : dict
        Dictionary mapping component names to SatelliteModel instances.
        Common component names are 'dm' (dark matter) and 'stellar'.
    position : array-like, optional
        Initial center position [x, y, z]. Overrides individual component
        positions.
    velocity : array-like, optional
        Initial center velocity [vx, vy, vz]. Overrides individual component
        velocities.

    Attributes
    ----------
    component_names : list of str
        Names of all components.
    n_components : int
        Number of components.

    Examples
    --------
    >>> from shreamy.satellite import CompositeSatellite, NFWSatellite, PlummerSatellite
    >>>
    >>> # Create a two-component satellite: extended DM halo + compact stellar component
    >>> satellite = CompositeSatellite(
    ...     components={
    ...         'dm': NFWSatellite(mass=1e10, r_s=5.0),
    ...         'stellar': PlummerSatellite(mass=1e8, b=0.5),
    ...     },
    ...     position=[50, 0, 0],
    ...     velocity=[0, 150, 0],
    ... )
    >>>
    >>> # Sample particles from all components
    >>> particles = satellite.sample(n_particles={'dm': 50000, 'stellar': 10000})
    >>>
    >>> # Check component labels
    >>> print(particles.component_names)  # ['dm', 'stellar']

    Notes
    -----
    When sampling, particles from each component are tagged with their
    component name. This allows for component-specific analysis and
    different gravitational softening lengths per component.
    """

    def __init__(
        self,
        components: Dict[str, SatelliteModel],
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
    ):
        """Initialize composite satellite from component models."""
        # Calculate total mass from all components
        total_mass = sum(comp.mass for comp in components.values())
        super().__init__(total_mass, position, velocity)

        self._components = components
        self._component_names = list(components.keys())

    @property
    def components(self) -> Dict[str, SatelliteModel]:
        """Dictionary of component models."""
        return self._components

    @property
    def component_names(self) -> list:
        """List of component names."""
        return self._component_names

    @property
    def n_components(self) -> int:
        """Number of components."""
        return len(self._components)

    def __getitem__(self, component: str) -> SatelliteModel:
        """Access a component by name."""
        return self._components[component]

    def get_component(self, component: str) -> SatelliteModel:
        """
        Get a specific component model.

        Parameters
        ----------
        component : str
            Component name.

        Returns
        -------
        SatelliteModel
            The satellite model for this component.

        Raises
        ------
        KeyError
            If component name is not found.
        """
        return self._components[component]

    def sample(
        self,
        n_particles: Union[int, Dict[str, int]],
        seed: Optional[int] = None,
        virialize: bool = False,
        virialize_time: Optional[float] = None,
        virialize_dt: Optional[float] = None,
        return_virialize_info: bool = False,
    ) -> Union["ParticleSet", Tuple["ParticleSet", "ParticleHistory"]]:
        """
        Sample particles from all components.

        Parameters
        ----------
        n_particles : int or dict
            If int, particles are distributed among components proportionally
            to their masses. If dict, specifies exact number per component,
            e.g., {'dm': 50000, 'stellar': 10000}.
        seed : int, optional
            Random seed for reproducibility.
        virialize : bool, default False
            If True, run a short N-body integration to virialize the
            combined multi-component system. This is done after sampling
            all components and combines them in the composite potential.
        virialize_time : float, optional
            Duration of virialization in dynamical times.
        virialize_dt : float, optional
            Time step for virialization.
        return_virialize_info : bool, default False
            If True and virialize=True, return (ParticleSet, ParticleHistory).

        Returns
        -------
        ParticleSet or tuple
            Combined particle set with component labels, optionally with
            virialization history.

        Examples
        --------
        >>> # Proportional sampling (by mass)
        >>> particles = satellite.sample(n_particles=60000)
        >>>
        >>> # Explicit per-component sampling
        >>> particles = satellite.sample(
        ...     n_particles={'dm': 50000, 'stellar': 10000}
        ... )
        >>>
        >>> # With virialization
        >>> particles, history = satellite.sample(
        ...     n_particles={'dm': 50000, 'stellar': 10000},
        ...     virialize=True,
        ...     return_virialize_info=True,
        ... )
        """
        raise NotImplementedError

    def sample_component(
        self,
        component: str,
        n_particles: int,
        seed: Optional[int] = None,
    ) -> "ParticleSet":
        """
        Sample particles from a single component.

        Parameters
        ----------
        component : str
            Component name.
        n_particles : int
            Number of particles to sample.
        seed : int, optional
            Random seed.

        Returns
        -------
        ParticleSet
            Particles from specified component (with component label).
        """
        raise NotImplementedError

    def density(self, r: np.ndarray, component: Optional[str] = None) -> np.ndarray:
        """
        Compute density at radius r.

        Parameters
        ----------
        r : array-like
            Radii at which to compute density.
        component : str, optional
            If specified, return density of single component.
            If None, return total density (sum of all components).

        Returns
        -------
        array-like
            Density values.
        """
        if component is not None:
            return self._components[component].density(r)

        # Sum densities from all components
        total_density = np.zeros_like(r)
        for comp in self._components.values():
            total_density += comp.density(r)
        return total_density

    def potential(self, r: np.ndarray, component: Optional[str] = None) -> np.ndarray:
        """
        Compute potential at radius r.

        Parameters
        ----------
        r : array-like
            Radii at which to compute potential.
        component : str, optional
            If specified, return potential of single component.
            If None, return total potential (sum of all components).

        Returns
        -------
        array-like
            Potential values.
        """
        if component is not None:
            return self._components[component].potential(r)

        # Sum potentials from all components
        total_potential = np.zeros_like(r)
        for comp in self._components.values():
            total_potential += comp.potential(r)
        return total_potential

    def component_mass(self, component: str) -> float:
        """
        Get the mass of a specific component.

        Parameters
        ----------
        component : str
            Component name.

        Returns
        -------
        float
            Mass of the component.
        """
        return self._components[component].mass

    def component_mass_fraction(self, component: str) -> float:
        """
        Get the mass fraction of a specific component.

        Parameters
        ----------
        component : str
            Component name.

        Returns
        -------
        float
            Fraction of total mass in this component.
        """
        return self._components[component].mass / self.mass

    def get_recommended_softening(self) -> Dict[str, float]:
        """
        Get recommended gravitational softening lengths for each component.

        Returns a dictionary of softening lengths estimated from the
        characteristic scales of each component's density profile.

        Returns
        -------
        dict
            Mapping of component name to recommended softening length.

        Notes
        -----
        For Plummer models, softening ~ b / sqrt(N).
        For NFW models, softening ~ r_s / sqrt(N).
        These are rough estimates; optimal values depend on the science goals.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """String representation."""
        comp_strs = [f"{name}: {comp.__class__.__name__}"
                     for name, comp in self._components.items()]
        return f"CompositeSatellite({', '.join(comp_strs)})"
