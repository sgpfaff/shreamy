"""
Shream: The core particle system class for N-body simulations of minor mergers.

This module provides the main user-facing object for shreamy, analogous to
galpy's Orbit object but for a collection of interacting particles.
"""

import numpy as np
from typing import Optional, Union, Sequence, Dict

# These will be imported once implemented
# from .particle import ParticleSet
# from .integrators import get_integrator
# from .gravity import GravitySolver
# from .potentials import HostPotential


class Shream:
    """
    A collection of N-body particles representing a satellite galaxy undergoing
    a minor merger with a host galaxy.

    This is the main user-facing class in shreamy. It manages:
    - Particle phase space coordinates (positions and velocities)
    - Self-gravity between satellite particles
    - External potential from the host galaxy (via galpy)
    - Time integration of the system

    The name "Shream" evokes both "stream" (stellar streams from tidal stripping)
    and "shell" (shell structures from radial mergers).

    Parameters
    ----------
    particles : ParticleSet or array-like, optional
        Initial particle data. Can be:
        - A ParticleSet object
        - An array of shape (N, 6) with [x, y, z, vx, vy, vz]
        - An array of shape (N, 7) with [x, y, z, vx, vy, vz, mass]
    masses : array-like, optional
        Particle masses. If not provided, assumes equal mass particles.
    host_potential : galpy.potential.Potential, optional
        The gravitational potential of the host galaxy. Must be a galpy
        potential object or a list of galpy potentials.
    self_gravity : bool, default True
        Whether to include gravitational interactions between particles.
    gravity_solver : str or GravitySolver, optional
        Method for computing self-gravity. Options:
        - 'direct' : Direct N^2 summation (default for N < 1000)
        - 'tree' : Barnes-Hut tree code (default for N >= 1000)
        - 'none' : No self-gravity
        - GravitySolver instance : Custom solver
    softening : float or dict, optional
        Gravitational softening length. Can be:
        - float: Same softening for all particles.
        - dict: Per-component softening, e.g., {'dm': 0.1, 'stellar': 0.01}.
        If None, estimated from particle distribution.
    ro : float, optional
        Distance scale for unit conversion (galpy convention). Default: 8 kpc.
    vo : float, optional
        Velocity scale for unit conversion (galpy convention). Default: 220 km/s.

    Examples
    --------
    Basic usage with a galpy potential:

    >>> from galpy.potential import MWPotential2014
    >>> from shreamy import Shream
    >>> from shreamy.satellite import PlummerSatellite
    >>>
    >>> # Create a satellite galaxy
    >>> satellite = PlummerSatellite(mass=1e9, scale_radius=1.0)
    >>> particles = satellite.sample(n_particles=10000)
    >>>
    >>> # Initialize at a position in the host galaxy
    >>> shream = Shream(particles, host_potential=MWPotential2014)
    >>>
    >>> # Integrate for 5 Gyr
    >>> times = np.linspace(0, 5, 1000)  # Gyr
    >>> shream.integrate(times)
    >>>
    >>> # Access the evolved positions
    >>> positions = shream.x(times[-1]), shream.y(times[-1]), shream.z(times[-1])
    """

    def __init__(
        self,
        particles=None,
        masses: Optional[np.ndarray] = None,
        host_potential=None,
        self_gravity: bool = True,
        gravity_solver: Optional[Union[str, "GravitySolver"]] = None,
        softening: Optional[Union[float, Dict[str, float]]] = None,
        ro: float = 8.0,
        vo: float = 220.0,
    ):
        """Initialize the Shream object with particles and potentials."""
        raise NotImplementedError("Shream.__init__ not yet implemented")

    # =========================================================================
    # Integration Methods
    # =========================================================================

    def integrate(
        self,
        t: Union[float, np.ndarray],
        method: str = "leapfrog",
        dt: Optional[float] = None,
        save_every: int = 1,
        progressbar: bool = True,
    ) -> "Shream":
        """
        Integrate the particle system forward (or backward) in time.

        Parameters
        ----------
        t : float or array-like
            Times at which to save the particle positions. If a float,
            interpreted as the final time. If array, saves at each time.
            Times should be in natural units (galpy convention) unless
            use_physical=True in the Shream initialization.
        method : str, default 'leapfrog'
            Integration method. Options:
            - 'leapfrog' : Symplectic leapfrog (2nd order, recommended)
            - 'rk4' : 4th order Runge-Kutta
            - 'hermite' : 4th order Hermite (for high accuracy)
        dt : float, optional
            Time step. If None, automatically determined from the dynamical
            time scale of the system.
        save_every : int, default 1
            Save every N-th integration step. Use to reduce memory for
            long integrations.
        progressbar : bool, default True
            Show a progress bar during integration.

        Returns
        -------
        Shream
            Returns self, allowing method chaining.

        Notes
        -----
        Integration is performed in natural units internally for numerical
        stability. Results can be accessed in physical units using the
        appropriate methods.
        """
        raise NotImplementedError("Shream.integrate not yet implemented")

    def turn_off_self_gravity(self, t: Optional[float] = None) -> None:
        """
        Turn off self-gravity, typically after the satellite is disrupted.

        Parameters
        ----------
        t : float, optional
            Time at which to turn off self-gravity. If None, turns off
            immediately for subsequent integrations.
        """
        raise NotImplementedError

    def turn_on_self_gravity(self) -> None:
        """Turn on self-gravity for subsequent integrations."""
        raise NotImplementedError

    # =========================================================================
    # Phase Space Access Methods (galpy Orbit-like interface)
    # =========================================================================

    def x(self, t: Optional[float] = None, component: Optional[str] = None) -> np.ndarray:
        """
        Return x coordinates of particles.

        Parameters
        ----------
        t : float, optional
            Time at which to return coordinates. If None, returns initial
            positions (t=0) or current positions if integrated.
        component : str, optional
            Component to select (e.g., 'dm', 'stellar'). If None, returns
            all particles.

        Returns
        -------
        ndarray
            x coordinates of shape (N,) or (N, n_times) if multiple times.
        """
        raise NotImplementedError

    def y(self, t: Optional[float] = None, component: Optional[str] = None) -> np.ndarray:
        """Return y coordinates of particles."""
        raise NotImplementedError

    def z(self, t: Optional[float] = None, component: Optional[str] = None) -> np.ndarray:
        """Return z coordinates of particles."""
        raise NotImplementedError

    def vx(self, t: Optional[float] = None, component: Optional[str] = None) -> np.ndarray:
        """Return x velocities of particles."""
        raise NotImplementedError

    def vy(self, t: Optional[float] = None, component: Optional[str] = None) -> np.ndarray:
        """Return y velocities of particles."""
        raise NotImplementedError

    def vz(self, t: Optional[float] = None, component: Optional[str] = None) -> np.ndarray:
        """Return z velocities of particles."""
        raise NotImplementedError

    def R(self, t: Optional[float] = None, component: Optional[str] = None) -> np.ndarray:
        """Return cylindrical radius R = sqrt(x^2 + y^2) of particles."""
        raise NotImplementedError

    def r(self, t: Optional[float] = None, component: Optional[str] = None) -> np.ndarray:
        """Return spherical radius r = sqrt(x^2 + y^2 + z^2) of particles."""
        raise NotImplementedError

    def phi(self, t: Optional[float] = None, component: Optional[str] = None) -> np.ndarray:
        """Return azimuthal angle phi = arctan2(y, x) of particles."""
        raise NotImplementedError

    def theta(self, t: Optional[float] = None, component: Optional[str] = None) -> np.ndarray:
        """Return polar angle theta = arccos(z/r) of particles."""
        raise NotImplementedError

    def pos(self, t: Optional[float] = None, component: Optional[str] = None) -> np.ndarray:
        """
        Return 3D positions of particles.

        Parameters
        ----------
        t : float, optional
            Time at which to return coordinates.
        component : str, optional
            Component to select.

        Returns
        -------
        ndarray
            Positions of shape (N, 3) with columns [x, y, z].
        """
        raise NotImplementedError

    def vel(self, t: Optional[float] = None, component: Optional[str] = None) -> np.ndarray:
        """
        Return 3D velocities of particles.

        Parameters
        ----------
        t : float, optional
            Time at which to return velocities.
        component : str, optional
            Component to select.

        Returns
        -------
        ndarray
            Velocities of shape (N, 3) with columns [vx, vy, vz].
        """
        raise NotImplementedError

    def phase_space(self, t: Optional[float] = None, component: Optional[str] = None) -> np.ndarray:
        """
        Return full 6D phase space coordinates of particles.

        Parameters
        ----------
        t : float, optional
            Time at which to return coordinates.
        component : str, optional
            Component to select.

        Returns
        -------
        ndarray
            Phase space of shape (N, 6) with columns [x, y, z, vx, vy, vz].
        """
        raise NotImplementedError

    # =========================================================================
    # Energy and Angular Momentum
    # =========================================================================

    def E(self, t: Optional[float] = None, component: Optional[str] = None) -> np.ndarray:
        """
        Return total energy of each particle (kinetic + potential).

        Includes both the host potential and self-gravity contributions.

        Parameters
        ----------
        t : float, optional
            Time at which to compute energy.
        component : str, optional
            Component to select.
        """
        raise NotImplementedError

    def Etot(self, t: Optional[float] = None, component: Optional[str] = None) -> float:
        """
        Return total energy of the system or a component.

        Parameters
        ----------
        t : float, optional
            Time at which to compute energy.
        component : str, optional
            Component to select. If None, returns total energy.
        """
        raise NotImplementedError

    def L(self, t: Optional[float] = None, component: Optional[str] = None) -> np.ndarray:
        """
        Return angular momentum vector of each particle.

        Parameters
        ----------
        t : float, optional
            Time at which to compute angular momentum.
        component : str, optional
            Component to select.

        Returns
        -------
        ndarray
            Angular momentum of shape (N, 3) with columns [Lx, Ly, Lz].
        """
        raise NotImplementedError

    def Ltot(self, t: Optional[float] = None, component: Optional[str] = None) -> np.ndarray:
        """
        Return total angular momentum of the system or a component.

        Parameters
        ----------
        t : float, optional
            Time at which to compute angular momentum.
        component : str, optional
            Component to select. If None, returns total angular momentum.
        """
        raise NotImplementedError

    # =========================================================================
    # Center of Mass / Density Center
    # =========================================================================

    def center_of_mass(self, t: Optional[float] = None, component: Optional[str] = None) -> np.ndarray:
        """
        Return the center of mass position.

        Parameters
        ----------
        t : float, optional
            Time at which to compute center of mass.
        component : str, optional
            Component to select. If None, uses all particles.

        Returns
        -------
        ndarray
            Center of mass position [x, y, z].
        """
        raise NotImplementedError

    def center_of_mass_velocity(self, t: Optional[float] = None, component: Optional[str] = None) -> np.ndarray:
        """
        Return the center of mass velocity.

        Parameters
        ----------
        t : float, optional
            Time at which to compute center of mass velocity.
        component : str, optional
            Component to select. If None, uses all particles.

        Returns
        -------
        ndarray
            Center of mass velocity [vx, vy, vz].
        """
        raise NotImplementedError

    def density_center(
        self,
        t: Optional[float] = None,
        method: str = "shrinking_sphere",
        component: Optional[str] = None,
    ) -> np.ndarray:
        """
        Return the density center (more robust than center of mass for
        disrupted satellites).

        Parameters
        ----------
        t : float, optional
            Time at which to compute density center.
        method : str, default 'shrinking_sphere'
            Method for finding density center:
            - 'shrinking_sphere' : Iteratively shrink sphere around densest region
            - 'most_bound' : Use position of most bound particle
        component : str, optional
            Component to select. If None, uses all particles.
        """
        raise NotImplementedError

    # =========================================================================
    # Component Access
    # =========================================================================

    @property
    def component_names(self) -> list:
        """List of unique component names in the particle set."""
        raise NotImplementedError

    @property
    def n_components(self) -> int:
        """Number of unique components."""
        raise NotImplementedError

    def has_component(self, component: str) -> bool:
        """Check if a component exists."""
        raise NotImplementedError

    def n_particles_by_component(self) -> Dict[str, int]:
        """
        Return number of particles in each component.

        Returns
        -------
        dict
            Mapping of component name to particle count.
        """
        raise NotImplementedError

    def mass_by_component(self, t: Optional[float] = None) -> Dict[str, float]:
        """
        Return total mass in each component.

        Parameters
        ----------
        t : float, optional
            Time at which to compute mass. If bound mass changed over time
            due to stripping, this may differ from initial mass.

        Returns
        -------
        dict
            Mapping of component name to total mass.
        """
        raise NotImplementedError

    def get_component(self, component: str) -> "Shream":
        """
        Return a new Shream containing only particles of the specified component.

        Parameters
        ----------
        component : str
            Component name (e.g., 'dm', 'stellar').

        Returns
        -------
        Shream
            A new Shream object with only the selected component.

        Notes
        -----
        This creates a view into the same underlying data where possible,
        but the returned Shream is independent for analysis purposes.
        """
        raise NotImplementedError

    def get_softening(self, component: Optional[str] = None) -> float:
        """
        Get the gravitational softening length.

        Parameters
        ----------
        component : str, optional
            Component name. If None and per-component softening is set,
            returns the minimum softening.

        Returns
        -------
        float
            Softening length.
        """
        raise NotImplementedError

    # =========================================================================
    # Visualization
    # =========================================================================

    def plot(
        self,
        t: Optional[float] = None,
        projection: str = "xy",
        color_by: Optional[str] = None,
        ax=None,
        **kwargs,
    ):
        """
        Plot particle positions.

        Parameters
        ----------
        t : float, optional
            Time at which to plot. Default is the last integrated time.
        projection : str, default 'xy'
            Projection plane: 'xy', 'xz', 'yz', 'Rz' (cylindrical), or '3d'.
        color_by : str, optional
            Color particles by property: 'energy', 'Lz', 'radius', 'mass'.
        ax : matplotlib.axes.Axes, optional
            Axes on which to plot. If None, creates new figure.
        **kwargs
            Additional arguments passed to scatter plot.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        raise NotImplementedError

    def animate(
        self,
        times: Optional[np.ndarray] = None,
        projection: str = "xy",
        filename: Optional[str] = None,
        fps: int = 30,
        **kwargs,
    ):
        """
        Create an animation of the particle evolution.

        Parameters
        ----------
        times : array-like, optional
            Times to include in animation. Default is all saved times.
        projection : str, default 'xy'
            Projection plane for animation.
        filename : str, optional
            If provided, save animation to file (mp4, gif).
        fps : int, default 30
            Frames per second for saved animation.
        """
        raise NotImplementedError

    # =========================================================================
    # I/O Methods
    # =========================================================================

    def save(self, filename: str, format: str = "hdf5") -> None:
        """
        Save the Shream to a file.

        Parameters
        ----------
        filename : str
            Output filename.
        format : str, default 'hdf5'
            File format: 'hdf5', 'npy', 'fits'.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, filename: str) -> "Shream":
        """
        Load a Shream from a file.

        Parameters
        ----------
        filename : str
            Input filename.

        Returns
        -------
        Shream
        """
        raise NotImplementedError

    # =========================================================================
    # Conversion to/from galpy
    # =========================================================================

    def to_orbit(self, t: Optional[float] = None):
        """
        Convert particles to a galpy Orbit object (multi-object Orbit).

        This is useful for analyzing particles in galpy's framework or
        for test particle integrations.

        Returns
        -------
        galpy.orbit.Orbit
        """
        raise NotImplementedError

    @classmethod
    def from_orbit(
        cls,
        orbit,
        masses: Optional[np.ndarray] = None,
        host_potential=None,
        **kwargs,
    ) -> "Shream":
        """
        Create a Shream from a galpy Orbit object.

        Parameters
        ----------
        orbit : galpy.orbit.Orbit
            A galpy Orbit (can contain multiple objects).
        masses : array-like, optional
            Masses for each orbit object.
        host_potential : galpy.potential.Potential, optional
            Host galaxy potential.

        Returns
        -------
        Shream
        """
        raise NotImplementedError

    # =========================================================================
    # Subsetting and Selection
    # =========================================================================

    def __getitem__(self, key) -> "Shream":
        """
        Select a subset of particles by index.

        Returns a new Shream with only the selected particles.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of particles."""
        raise NotImplementedError

    def select(
        self,
        bound: Optional[bool] = None,
        r_min: Optional[float] = None,
        r_max: Optional[float] = None,
        E_max: Optional[float] = None,
    ) -> "Shream":
        """
        Select particles based on criteria.

        Parameters
        ----------
        bound : bool, optional
            If True, select only bound particles. If False, only unbound.
        r_min, r_max : float, optional
            Radial distance cuts (from center of satellite).
        E_max : float, optional
            Maximum energy cut.

        Returns
        -------
        Shream
            New Shream with selected particles.
        """
        raise NotImplementedError

    def bound_particles(self, t: Optional[float] = None) -> "Shream":
        """Return a new Shream with only bound particles."""
        raise NotImplementedError

    def unbound_particles(self, t: Optional[float] = None) -> "Shream":
        """Return a new Shream with only unbound (stripped) particles."""
        raise NotImplementedError

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def n_particles(self) -> int:
        """Number of particles in the Shream."""
        raise NotImplementedError

    @property
    def total_mass(self) -> float:
        """Total mass of all particles."""
        raise NotImplementedError

    @property
    def times(self) -> np.ndarray:
        """Array of times at which the Shream state is saved."""
        raise NotImplementedError

    @property
    def masses(self) -> np.ndarray:
        """Array of particle masses."""
        raise NotImplementedError

    # =========================================================================
    # Dunder Methods
    # =========================================================================

    def __repr__(self) -> str:
        """String representation of the Shream."""
        return f"Shream(n_particles={self.n_particles}, total_mass={self.total_mass})"

    def __add__(self, other: "Shream") -> "Shream":
        """Combine two Shreams (e.g., for multiple satellite systems)."""
        raise NotImplementedError
