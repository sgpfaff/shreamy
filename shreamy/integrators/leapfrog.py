"""
Leapfrog integrators.
"""

import numpy as np
from typing import Tuple, Callable, Optional, List

from .base import Integrator


class Leapfrog(Integrator):
    """
    Leapfrog (kick-drift-kick) integrator.

    A second-order symplectic integrator that is well-suited for
    gravitational N-body simulations. Conserves energy to good
    precision over long timescales.

    This uses the velocity Verlet formulation (KDK):
        v_{n+1/2} = v_n + (dt/2) * a_n
        x_{n+1} = x_n + dt * v_{n+1/2}
        v_{n+1} = v_{n+1/2} + (dt/2) * a_{n+1}

    Parameters
    ----------
    acceleration_func : callable
        Function that computes accelerations given (positions, velocities, t).
        Signature: acceleration_func(pos, vel, t) -> ndarray of shape (N, 3).

    Examples
    --------
    >>> def acc_func(pos, vel, t):
    ...     return -pos  # Simple harmonic oscillator
    >>> integrator = Leapfrog(acc_func)
    >>> pos = np.array([[1.0, 0.0, 0.0]])
    >>> vel = np.array([[0.0, 1.0, 0.0]])
    >>> new_pos, new_vel = integrator.step(pos, vel, t=0.0, dt=0.01)
    """

    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        t: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance using leapfrog KDK scheme.

        Parameters
        ----------
        positions : ndarray
            Current positions of shape (N, 3).
        velocities : ndarray
            Current velocities of shape (N, 3).
        t : float
            Current time.
        dt : float
            Time step.

        Returns
        -------
        new_positions : ndarray
            Updated positions of shape (N, 3).
        new_velocities : ndarray
            Updated velocities of shape (N, 3).
        """
        # Kick: v_{n+1/2} = v_n + (dt/2) * a_n
        acc_n = self._acceleration_func(positions, velocities, t)
        v_half = velocities + 0.5 * dt * acc_n

        # Drift: x_{n+1} = x_n + dt * v_{n+1/2}
        new_positions = positions + dt * v_half

        # Kick: v_{n+1} = v_{n+1/2} + (dt/2) * a_{n+1}
        acc_np1 = self._acceleration_func(new_positions, v_half, t + dt)
        new_velocities = v_half + 0.5 * dt * acc_np1

        return new_positions, new_velocities

    def integrate(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        times: np.ndarray,
        save_every: int = 1,
        progressbar: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Integrate the system over multiple time steps.

        Parameters
        ----------
        positions : ndarray
            Initial positions of shape (N, 3).
        velocities : ndarray
            Initial velocities of shape (N, 3).
        times : ndarray
            Array of times at which to evaluate. Must be monotonic.
            The first element is the initial time.
        save_every : int, default 1
            Save every N-th step.
        progressbar : bool, default True
            Show progress bar (requires tqdm).

        Returns
        -------
        saved_times : ndarray
            Times at which state was saved.
        saved_positions : ndarray
            Positions at each saved time, shape (n_saves, N, 3).
        saved_velocities : ndarray
            Velocities at each saved time, shape (n_saves, N, 3).
        """
        times = np.atleast_1d(times)
        n_steps = len(times) - 1

        if n_steps <= 0:
            return (
                times[:1],
                positions[np.newaxis, :, :],
                velocities[np.newaxis, :, :],
            )

        # Determine save indices
        save_indices = list(range(0, n_steps + 1, save_every))
        if save_indices[-1] != n_steps:
            save_indices.append(n_steps)

        n_saves = len(save_indices)
        N = positions.shape[0]

        # Allocate output arrays
        saved_times = np.zeros(n_saves)
        saved_positions = np.zeros((n_saves, N, 3))
        saved_velocities = np.zeros((n_saves, N, 3))

        # Initialize
        pos = positions.copy()
        vel = velocities.copy()

        # Save initial state
        save_idx = 0
        if 0 in save_indices:
            saved_times[save_idx] = times[0]
            saved_positions[save_idx] = pos.copy()
            saved_velocities[save_idx] = vel.copy()
            save_idx += 1

        # Set up progress bar
        iterator = range(n_steps)
        if progressbar:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Integrating", unit="step")
            except ImportError:
                pass

        # Integrate
        for i in iterator:
            t = times[i]
            dt = times[i + 1] - times[i]

            pos, vel = self.step(pos, vel, t, dt)

            step_num = i + 1
            if step_num in save_indices:
                saved_times[save_idx] = times[step_num]
                saved_positions[save_idx] = pos.copy()
                saved_velocities[save_idx] = vel.copy()
                save_idx += 1

        return saved_times, saved_positions, saved_velocities

    @property
    def order(self) -> int:
        """Order of the integrator."""
        return 2

    @property
    def is_symplectic(self) -> bool:
        """Whether the integrator is symplectic."""
        return True

    def __repr__(self) -> str:
        return "Leapfrog(order=2, symplectic=True)"


class LeapfrogDKD(Integrator):
    """
    Leapfrog (drift-kick-drift) integrator.

    Alternative formulation of leapfrog that drifts positions first.
    Equivalent to KDK for constant time steps but can differ for
    adaptive stepping.

        x_{n+1/2} = x_n + (dt/2) * v_n
        v_{n+1} = v_n + dt * a_{n+1/2}
        x_{n+1} = x_{n+1/2} + (dt/2) * v_{n+1}

    Parameters
    ----------
    acceleration_func : callable
        Function that computes accelerations given (positions, velocities, t).
        Signature: acceleration_func(pos, vel, t) -> ndarray of shape (N, 3).
    """

    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        t: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance using DKD leapfrog scheme.

        Parameters
        ----------
        positions : ndarray
            Current positions of shape (N, 3).
        velocities : ndarray
            Current velocities of shape (N, 3).
        t : float
            Current time.
        dt : float
            Time step.

        Returns
        -------
        new_positions : ndarray
            Updated positions of shape (N, 3).
        new_velocities : ndarray
            Updated velocities of shape (N, 3).
        """
        # Drift: x_{n+1/2} = x_n + (dt/2) * v_n
        x_half = positions + 0.5 * dt * velocities

        # Kick: v_{n+1} = v_n + dt * a_{n+1/2}
        acc_half = self._acceleration_func(x_half, velocities, t + 0.5 * dt)
        new_velocities = velocities + dt * acc_half

        # Drift: x_{n+1} = x_{n+1/2} + (dt/2) * v_{n+1}
        new_positions = x_half + 0.5 * dt * new_velocities

        return new_positions, new_velocities

    def integrate(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        times: np.ndarray,
        save_every: int = 1,
        progressbar: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Integrate the system over multiple time steps.

        Parameters
        ----------
        positions : ndarray
            Initial positions of shape (N, 3).
        velocities : ndarray
            Initial velocities of shape (N, 3).
        times : ndarray
            Array of times at which to evaluate. Must be monotonic.
        save_every : int, default 1
            Save every N-th step.
        progressbar : bool, default True
            Show progress bar.

        Returns
        -------
        saved_times : ndarray
            Times at which state was saved.
        saved_positions : ndarray
            Positions at each saved time, shape (n_saves, N, 3).
        saved_velocities : ndarray
            Velocities at each saved time, shape (n_saves, N, 3).
        """
        times = np.atleast_1d(times)
        n_steps = len(times) - 1

        if n_steps <= 0:
            return (
                times[:1],
                positions[np.newaxis, :, :],
                velocities[np.newaxis, :, :],
            )

        # Determine save indices
        save_indices = list(range(0, n_steps + 1, save_every))
        if save_indices[-1] != n_steps:
            save_indices.append(n_steps)

        n_saves = len(save_indices)
        N = positions.shape[0]

        # Allocate output arrays
        saved_times = np.zeros(n_saves)
        saved_positions = np.zeros((n_saves, N, 3))
        saved_velocities = np.zeros((n_saves, N, 3))

        # Initialize
        pos = positions.copy()
        vel = velocities.copy()

        # Save initial state
        save_idx = 0
        if 0 in save_indices:
            saved_times[save_idx] = times[0]
            saved_positions[save_idx] = pos.copy()
            saved_velocities[save_idx] = vel.copy()
            save_idx += 1

        # Set up progress bar
        iterator = range(n_steps)
        if progressbar:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Integrating", unit="step")
            except ImportError:
                pass

        # Integrate
        for i in iterator:
            t = times[i]
            dt = times[i + 1] - times[i]

            pos, vel = self.step(pos, vel, t, dt)

            step_num = i + 1
            if step_num in save_indices:
                saved_times[save_idx] = times[step_num]
                saved_positions[save_idx] = pos.copy()
                saved_velocities[save_idx] = vel.copy()
                save_idx += 1

        return saved_times, saved_positions, saved_velocities

    @property
    def order(self) -> int:
        """Order of the integrator."""
        return 2

    @property
    def is_symplectic(self) -> bool:
        """Whether the integrator is symplectic."""
        return True

    def __repr__(self) -> str:
        return "LeapfrogDKD(order=2, symplectic=True)"
