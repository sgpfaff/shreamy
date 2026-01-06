"""
Direct N^2 summation gravity solver.
"""

import numpy as np
from typing import Optional

from .base import GravitySolver


class DirectSummation(GravitySolver):
    """
    Direct N^2 summation gravity solver.

    Computes gravitational forces by directly summing over all particle
    pairs. This is O(N^2) in complexity but exact (up to softening).

    Best for small N (< 1000) or when high accuracy is needed.

    Parameters
    ----------
    softening : float, optional
        Gravitational softening length. Plummer softening is used:
        a -> a / (r^2 + softening^2)^(3/2)
    G : float, default 1.0
        Gravitational constant.
    use_vectorized : bool, default True
        Use vectorized numpy operations. Faster for N > ~100.

    Examples
    --------
    >>> solver = DirectSummation(softening=0.01)
    >>> positions = np.array([[0, 0, 0], [1, 0, 0]])
    >>> masses = np.array([1.0, 1.0])
    >>> acc = solver.accelerations(positions, masses)
    """

    def __init__(
        self,
        softening: Optional[float] = None,
        G: float = 1.0,
        use_vectorized: bool = True,
    ):
        """Initialize direct summation solver."""
        super().__init__(softening=softening, G=G)
        self._use_vectorized = use_vectorized

    def accelerations(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
    ) -> np.ndarray:
        """
        Compute accelerations via direct summation.

        Uses Plummer softening:
        a_i = -G * sum_{j != i} m_j * (r_i - r_j) / (|r_i - r_j|^2 + eps^2)^(3/2)

        Parameters
        ----------
        positions : ndarray
            Particle positions of shape (N, 3).
        masses : ndarray
            Particle masses of shape (N,).

        Returns
        -------
        ndarray
            Accelerations of shape (N, 3).
        """
        positions = np.atleast_2d(positions)
        masses = np.atleast_1d(masses)

        N = positions.shape[0]

        if N == 0:
            return np.zeros((0, 3))

        if N == 1:
            return np.zeros((1, 3))

        if self._use_vectorized:
            return self._accelerations_vectorized(positions, masses)
        else:
            return self._accelerations_loop(positions, masses)

    def _accelerations_vectorized(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
    ) -> np.ndarray:
        """Compute accelerations using vectorized operations."""
        N = positions.shape[0]
        eps2 = self._softening**2

        # Compute all pairwise displacement vectors
        # dx[i, j] = positions[j] - positions[i]
        dx = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]  # (N, N, 3)

        # Compute squared distances
        r2 = np.sum(dx**2, axis=2)  # (N, N)

        # Add softening
        r2_soft = r2 + eps2

        # Compute 1 / (r^2 + eps^2)^(3/2)
        # Set diagonal to infinity to avoid self-interaction (will give 0)
        np.fill_diagonal(r2_soft, np.inf)
        inv_r3 = r2_soft ** (-1.5)

        # Acceleration: a_i = G * sum_j m_j * dx[i,j] / |dx|^3
        # Note: dx[i,j] = r_j - r_i, so this gives acceleration toward j
        acc = self._G * np.sum(masses[np.newaxis, :, np.newaxis] * dx * inv_r3[:, :, np.newaxis], axis=1)

        return acc

    def _accelerations_loop(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
    ) -> np.ndarray:
        """Compute accelerations using explicit loops (slower but clearer)."""
        N = positions.shape[0]
        eps2 = self._softening**2
        acc = np.zeros_like(positions)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dx = positions[j] - positions[i]
                r2 = np.sum(dx**2) + eps2
                r3 = r2**1.5
                acc[i] += self._G * masses[j] * dx / r3

        return acc

    def potential(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
    ) -> np.ndarray:
        """
        Compute gravitational potential at each particle position.

        The potential at particle i due to all other particles is:
        Phi_i = -G * sum_{j != i} m_j / sqrt(|r_i - r_j|^2 + eps^2)

        Parameters
        ----------
        positions : ndarray
            Particle positions of shape (N, 3).
        masses : ndarray
            Particle masses of shape (N,).

        Returns
        -------
        ndarray
            Potential values of shape (N,).
        """
        positions = np.atleast_2d(positions)
        masses = np.atleast_1d(masses)

        N = positions.shape[0]

        if N == 0:
            return np.zeros(0)

        if N == 1:
            return np.zeros(1)

        eps2 = self._softening**2

        # Compute all pairwise distances
        dx = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]  # (N, N, 3)
        r2 = np.sum(dx**2, axis=2)  # (N, N)
        r2_soft = r2 + eps2

        # Set diagonal to infinity to avoid self-interaction
        np.fill_diagonal(r2_soft, np.inf)

        inv_r = 1.0 / np.sqrt(r2_soft)

        # Potential at each particle
        phi = -self._G * np.sum(masses[np.newaxis, :] * inv_r, axis=1)

        return phi

    def potential_energy(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
    ) -> float:
        """
        Compute total gravitational potential energy.

        W = -G * sum_{i<j} m_i * m_j / sqrt(|r_i - r_j|^2 + eps^2)

        This counts each pair once (not twice like summing individual potentials).

        Parameters
        ----------
        positions : ndarray
            Particle positions of shape (N, 3).
        masses : ndarray
            Particle masses of shape (N,).

        Returns
        -------
        float
            Total gravitational potential energy (negative).
        """
        positions = np.atleast_2d(positions)
        masses = np.atleast_1d(masses)

        N = positions.shape[0]

        if N < 2:
            return 0.0

        eps2 = self._softening**2

        # Compute all pairwise distances
        dx = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]  # (N, N, 3)
        r2 = np.sum(dx**2, axis=2)  # (N, N)
        r2_soft = r2 + eps2

        # Only count upper triangle (i < j)
        inv_r = 1.0 / np.sqrt(r2_soft)
        np.fill_diagonal(inv_r, 0.0)  # No self-interaction

        # Mass products
        mass_prod = masses[:, np.newaxis] * masses[np.newaxis, :]

        # Sum upper triangle only
        W = -self._G * 0.5 * np.sum(mass_prod * inv_r)

        return W

    def kinetic_energy(
        self,
        velocities: np.ndarray,
        masses: np.ndarray,
    ) -> float:
        """
        Compute total kinetic energy.

        K = 0.5 * sum_i m_i * |v_i|^2

        Parameters
        ----------
        velocities : ndarray
            Particle velocities of shape (N, 3).
        masses : ndarray
            Particle masses of shape (N,).

        Returns
        -------
        float
            Total kinetic energy.
        """
        velocities = np.atleast_2d(velocities)
        masses = np.atleast_1d(masses)

        v2 = np.sum(velocities**2, axis=1)
        return 0.5 * np.sum(masses * v2)

    def total_energy(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
    ) -> float:
        """
        Compute total energy (kinetic + potential).

        Parameters
        ----------
        positions : ndarray
            Particle positions of shape (N, 3).
        velocities : ndarray
            Particle velocities of shape (N, 3).
        masses : ndarray
            Particle masses of shape (N,).

        Returns
        -------
        float
            Total energy.
        """
        K = self.kinetic_energy(velocities, masses)
        W = self.potential_energy(positions, masses)
        return K + W

    @property
    def use_vectorized(self) -> bool:
        """Whether vectorized computation is enabled."""
        return self._use_vectorized

    @property
    def softening(self) -> float:
        """Gravitational softening length."""
        return self._softening

    @property
    def G(self) -> float:
        """Gravitational constant."""
        return self._G

    def __repr__(self) -> str:
        return (
            f"DirectSummation(softening={self._softening}, "
            f"G={self._G}, use_vectorized={self._use_vectorized})"
        )
