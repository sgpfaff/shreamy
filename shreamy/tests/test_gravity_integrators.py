"""
Tests for gravity solvers and integrators.

These tests verify that DirectSummation and Leapfrog implementations
are correct through various sanity checks and physical consistency tests.
"""

import numpy as np
import pytest

from shreamy.gravity.direct import DirectSummation
from shreamy.integrators.leapfrog import Leapfrog, LeapfrogDKD


# ============================================================================
# DirectSummation Tests
# ============================================================================


class TestDirectSummation:
    """Tests for DirectSummation gravity solver."""

    # ---- Initialization Tests ----

    def test_init_default(self):
        """Test default initialization."""
        solver = DirectSummation()
        assert solver.softening == 0.0
        assert solver.G == 1.0
        assert solver.use_vectorized is True

    def test_init_with_softening(self):
        """Test initialization with softening."""
        solver = DirectSummation(softening=0.1)
        assert solver.softening == 0.1

    def test_init_with_G(self):
        """Test initialization with custom G."""
        solver = DirectSummation(G=6.67e-11)
        assert solver.G == 6.67e-11

    def test_repr(self):
        """Test string representation."""
        solver = DirectSummation(softening=0.01, G=1.0)
        assert "DirectSummation" in repr(solver)
        assert "softening=0.01" in repr(solver)

    # ---- Edge Cases ----

    def test_empty_particles(self):
        """Test with no particles."""
        solver = DirectSummation()
        positions = np.zeros((0, 3))
        masses = np.zeros(0)

        acc = solver.accelerations(positions, masses)
        assert acc.shape == (0, 3)

        phi = solver.potential(positions, masses)
        assert phi.shape == (0,)

        W = solver.potential_energy(positions, masses)
        assert W == 0.0

    def test_single_particle(self):
        """Test with single particle (no self-gravity)."""
        solver = DirectSummation()
        positions = np.array([[1.0, 2.0, 3.0]])
        masses = np.array([1.0])

        acc = solver.accelerations(positions, masses)
        np.testing.assert_array_equal(acc, np.zeros((1, 3)))

        phi = solver.potential(positions, masses)
        np.testing.assert_array_equal(phi, np.zeros(1))

        W = solver.potential_energy(positions, masses)
        assert W == 0.0

    # ---- Two-Body Tests ----

    def test_two_body_acceleration_direction(self):
        """Test that particles attract each other."""
        solver = DirectSummation(softening=0.0)
        positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        masses = np.array([1.0, 1.0])

        acc = solver.accelerations(positions, masses)

        # Particle 0 should accelerate toward +x
        assert acc[0, 0] > 0
        assert acc[0, 1] == 0
        assert acc[0, 2] == 0

        # Particle 1 should accelerate toward -x
        assert acc[1, 0] < 0
        assert acc[1, 1] == 0
        assert acc[1, 2] == 0

    def test_two_body_acceleration_magnitude(self):
        """Test acceleration magnitude for two particles."""
        solver = DirectSummation(softening=0.0, G=1.0)
        r = 2.0  # separation
        positions = np.array([[0, 0, 0], [r, 0, 0]], dtype=float)
        masses = np.array([1.0, 1.0])

        acc = solver.accelerations(positions, masses)

        # |a| = G * m / r^2 = 1 * 1 / 4 = 0.25
        expected_acc = 1.0 / r**2
        np.testing.assert_allclose(np.abs(acc[0, 0]), expected_acc, rtol=1e-10)
        np.testing.assert_allclose(np.abs(acc[1, 0]), expected_acc, rtol=1e-10)

    def test_two_body_acceleration_symmetry(self):
        """Test Newton's third law: a_01 = -a_10."""
        solver = DirectSummation(softening=0.01)
        positions = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
        masses = np.array([1.0, 2.0])

        acc = solver.accelerations(positions, masses)

        # m0 * a0 = -m1 * a1 (momentum conservation)
        momentum_0 = masses[0] * acc[0]
        momentum_1 = masses[1] * acc[1]
        np.testing.assert_allclose(momentum_0, -momentum_1, rtol=1e-10)

    def test_two_body_potential(self):
        """Test potential for two particles."""
        solver = DirectSummation(softening=0.0, G=1.0)
        r = 2.0
        positions = np.array([[0, 0, 0], [r, 0, 0]], dtype=float)
        masses = np.array([1.0, 1.0])

        phi = solver.potential(positions, masses)

        # Phi at each particle = -G * m_other / r = -1/2
        expected_phi = -1.0 / r
        np.testing.assert_allclose(phi[0], expected_phi, rtol=1e-10)
        np.testing.assert_allclose(phi[1], expected_phi, rtol=1e-10)

    def test_two_body_potential_energy(self):
        """Test potential energy for two particles."""
        solver = DirectSummation(softening=0.0, G=1.0)
        r = 2.0
        positions = np.array([[0, 0, 0], [r, 0, 0]], dtype=float)
        masses = np.array([1.0, 1.0])

        W = solver.potential_energy(positions, masses)

        # W = -G * m1 * m2 / r = -1/2
        expected_W = -1.0 / r
        np.testing.assert_allclose(W, expected_W, rtol=1e-10)

    # ---- Softening Tests ----

    def test_softening_reduces_force(self):
        """Test that softening reduces force at small separations."""
        positions = np.array([[0, 0, 0], [0.1, 0, 0]], dtype=float)
        masses = np.array([1.0, 1.0])

        solver_no_soft = DirectSummation(softening=0.0)
        solver_soft = DirectSummation(softening=0.1)

        acc_no_soft = solver_no_soft.accelerations(positions, masses)
        acc_soft = solver_soft.accelerations(positions, masses)

        # Softened acceleration should be smaller
        assert np.abs(acc_soft[0, 0]) < np.abs(acc_no_soft[0, 0])

    def test_softening_plummer_formula(self):
        """Test Plummer softening formula."""
        eps = 0.1
        solver = DirectSummation(softening=eps, G=1.0)
        r = 0.5
        positions = np.array([[0, 0, 0], [r, 0, 0]], dtype=float)
        masses = np.array([1.0, 1.0])

        acc = solver.accelerations(positions, masses)

        # |a| = G * m * r / (r^2 + eps^2)^(3/2)
        expected_acc = 1.0 * r / (r**2 + eps**2) ** 1.5
        np.testing.assert_allclose(np.abs(acc[0, 0]), expected_acc, rtol=1e-10)

    # ---- Multi-particle Tests ----

    def test_three_body_total_momentum_zero(self):
        """Test that total force sums to zero (momentum conservation)."""
        solver = DirectSummation(softening=0.01)
        positions = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 0.866, 0],  # Equilateral triangle
        ], dtype=float)
        masses = np.array([1.0, 2.0, 3.0])

        acc = solver.accelerations(positions, masses)

        # Total force = sum(m_i * a_i) should be zero
        total_force = np.sum(masses[:, np.newaxis] * acc, axis=0)
        np.testing.assert_allclose(total_force, np.zeros(3), atol=1e-12)

    def test_vectorized_vs_loop(self):
        """Test that vectorized and loop implementations match."""
        positions = np.random.randn(10, 3)
        masses = np.random.rand(10) + 0.1

        solver_vec = DirectSummation(softening=0.1, use_vectorized=True)
        solver_loop = DirectSummation(softening=0.1, use_vectorized=False)

        acc_vec = solver_vec.accelerations(positions, masses)
        acc_loop = solver_loop.accelerations(positions, masses)

        np.testing.assert_allclose(acc_vec, acc_loop, rtol=1e-10)

    # ---- Energy Tests ----

    def test_kinetic_energy(self):
        """Test kinetic energy calculation."""
        solver = DirectSummation()
        velocities = np.array([[1, 0, 0], [0, 2, 0]], dtype=float)
        masses = np.array([1.0, 2.0])

        K = solver.kinetic_energy(velocities, masses)

        # K = 0.5 * (1 * 1 + 2 * 4) = 0.5 * 9 = 4.5
        expected_K = 0.5 * (1.0 * 1.0 + 2.0 * 4.0)
        np.testing.assert_allclose(K, expected_K, rtol=1e-10)

    def test_total_energy(self):
        """Test total energy calculation."""
        solver = DirectSummation(softening=0.0, G=1.0)
        positions = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        velocities = np.array([[0, 0, 0], [0, 1, 0]], dtype=float)
        masses = np.array([1.0, 1.0])

        E = solver.total_energy(positions, velocities, masses)
        K = solver.kinetic_energy(velocities, masses)
        W = solver.potential_energy(positions, masses)

        np.testing.assert_allclose(E, K + W, rtol=1e-10)


# ============================================================================
# Leapfrog Integrator Tests
# ============================================================================


class TestLeapfrog:
    """Tests for Leapfrog (KDK) integrator."""

    # ---- Initialization Tests ----

    def test_init(self):
        """Test initialization."""
        def acc_func(pos, vel, t):
            return -pos

        integrator = Leapfrog(acc_func)
        assert integrator.order == 2
        assert integrator.is_symplectic is True
        assert integrator.acceleration_func is acc_func

    def test_repr(self):
        """Test string representation."""
        integrator = Leapfrog(lambda p, v, t: -p)
        assert "Leapfrog" in repr(integrator)
        assert "order=2" in repr(integrator)

    # ---- Single Step Tests ----

    def test_step_free_particle(self):
        """Test step for free particle (constant velocity)."""
        def acc_func(pos, vel, t):
            return np.zeros_like(pos)

        integrator = Leapfrog(acc_func)
        pos = np.array([[0.0, 0.0, 0.0]])
        vel = np.array([[1.0, 0.0, 0.0]])

        new_pos, new_vel = integrator.step(pos, vel, t=0.0, dt=0.1)

        # Free particle: x = x0 + v*t
        np.testing.assert_allclose(new_pos, [[0.1, 0.0, 0.0]], rtol=1e-10)
        np.testing.assert_allclose(new_vel, [[1.0, 0.0, 0.0]], rtol=1e-10)

    def test_step_constant_acceleration(self):
        """Test step with constant acceleration."""
        def acc_func(pos, vel, t):
            return np.array([[1.0, 0.0, 0.0]])  # a = 1 in x

        integrator = Leapfrog(acc_func)
        pos = np.array([[0.0, 0.0, 0.0]])
        vel = np.array([[0.0, 0.0, 0.0]])
        dt = 0.1

        new_pos, new_vel = integrator.step(pos, vel, t=0.0, dt=dt)

        # Constant acceleration: v = a*t, x = 0.5*a*t^2
        expected_vel = [[dt, 0.0, 0.0]]
        expected_pos = [[0.5 * dt**2, 0.0, 0.0]]

        np.testing.assert_allclose(new_vel, expected_vel, rtol=1e-10)
        np.testing.assert_allclose(new_pos, expected_pos, rtol=1e-10)

    # ---- Harmonic Oscillator Tests ----

    def test_harmonic_oscillator_period(self):
        """Test that SHO returns to initial position after one period."""
        def acc_func(pos, vel, t):
            return -pos  # omega = 1

        integrator = Leapfrog(acc_func)
        pos = np.array([[1.0, 0.0, 0.0]])
        vel = np.array([[0.0, 0.0, 0.0]])

        # Integrate for one period (T = 2*pi)
        n_steps = 1000
        times = np.linspace(0, 2 * np.pi, n_steps + 1)

        saved_t, saved_pos, saved_vel = integrator.integrate(
            pos, vel, times, save_every=n_steps, progressbar=False
        )

        # Should return close to initial position
        np.testing.assert_allclose(saved_pos[-1, 0, 0], 1.0, atol=0.01)
        np.testing.assert_allclose(saved_pos[-1, 0, 1], 0.0, atol=0.01)
        np.testing.assert_allclose(saved_pos[-1, 0, 2], 0.0, atol=0.01)

    def test_harmonic_oscillator_energy_conservation(self):
        """Test energy conservation for harmonic oscillator."""
        def acc_func(pos, vel, t):
            return -pos

        integrator = Leapfrog(acc_func)
        pos = np.array([[1.0, 0.0, 0.0]])
        vel = np.array([[0.0, 1.0, 0.0]])

        # Total energy E = 0.5*(v^2 + x^2) = 0.5*(1 + 1) = 1
        def total_energy(p, v):
            return 0.5 * (np.sum(v**2) + np.sum(p**2))

        E0 = total_energy(pos, vel)

        # Integrate for many periods
        n_steps = 10000
        times = np.linspace(0, 20 * np.pi, n_steps + 1)

        saved_t, saved_pos, saved_vel = integrator.integrate(
            pos, vel, times, save_every=1000, progressbar=False
        )

        # Check energy at each saved point
        for i in range(len(saved_t)):
            E = total_energy(saved_pos[i], saved_vel[i])
            np.testing.assert_allclose(E, E0, rtol=1e-4)

    # ---- Integration Tests ----

    def test_integrate_saves_correctly(self):
        """Test that integrate saves at correct intervals."""
        def acc_func(pos, vel, t):
            return np.zeros_like(pos)

        integrator = Leapfrog(acc_func)
        pos = np.array([[0.0, 0.0, 0.0]])
        vel = np.array([[1.0, 0.0, 0.0]])

        times = np.linspace(0, 1.0, 101)  # 100 steps

        saved_t, saved_pos, saved_vel = integrator.integrate(
            pos, vel, times, save_every=10, progressbar=False
        )

        # Should save at steps 0, 10, 20, ..., 100 = 11 snapshots
        assert len(saved_t) == 11
        np.testing.assert_allclose(saved_t, np.linspace(0, 1, 11), rtol=1e-10)

    def test_integrate_single_time(self):
        """Test integration with single time point."""
        def acc_func(pos, vel, t):
            return np.zeros_like(pos)

        integrator = Leapfrog(acc_func)
        pos = np.array([[1.0, 0.0, 0.0]])
        vel = np.array([[0.0, 0.0, 0.0]])

        times = np.array([0.0])

        saved_t, saved_pos, saved_vel = integrator.integrate(
            pos, vel, times, progressbar=False
        )

        assert len(saved_t) == 1
        np.testing.assert_array_equal(saved_pos[0], pos)


class TestLeapfrogDKD:
    """Tests for LeapfrogDKD (DKD) integrator."""

    def test_init(self):
        """Test initialization."""
        def acc_func(pos, vel, t):
            return -pos

        integrator = LeapfrogDKD(acc_func)
        assert integrator.order == 2
        assert integrator.is_symplectic is True

    def test_repr(self):
        """Test string representation."""
        integrator = LeapfrogDKD(lambda p, v, t: -p)
        assert "LeapfrogDKD" in repr(integrator)

    def test_step_free_particle(self):
        """Test step for free particle."""
        def acc_func(pos, vel, t):
            return np.zeros_like(pos)

        integrator = LeapfrogDKD(acc_func)
        pos = np.array([[0.0, 0.0, 0.0]])
        vel = np.array([[1.0, 0.0, 0.0]])

        new_pos, new_vel = integrator.step(pos, vel, t=0.0, dt=0.1)

        np.testing.assert_allclose(new_pos, [[0.1, 0.0, 0.0]], rtol=1e-10)
        np.testing.assert_allclose(new_vel, [[1.0, 0.0, 0.0]], rtol=1e-10)

    def test_kdk_vs_dkd_equivalence(self):
        """Test that KDK and DKD give same results for constant dt."""
        def acc_func(pos, vel, t):
            return -pos

        kdk = Leapfrog(acc_func)
        dkd = LeapfrogDKD(acc_func)

        pos = np.array([[1.0, 0.0, 0.0]])
        vel = np.array([[0.0, 1.0, 0.0]])

        n_steps = 100
        times = np.linspace(0, 1.0, n_steps + 1)

        _, pos_kdk, vel_kdk = kdk.integrate(pos, vel, times, progressbar=False)
        _, pos_dkd, vel_dkd = dkd.integrate(pos, vel, times, progressbar=False)

        # Should be very close (not identical due to different orderings)
        np.testing.assert_allclose(pos_kdk[-1], pos_dkd[-1], rtol=1e-3)
        np.testing.assert_allclose(vel_kdk[-1], vel_dkd[-1], rtol=1e-3)


# ============================================================================
# Combined Gravity + Integrator Tests
# ============================================================================


class TestGravityIntegration:
    """Tests combining gravity solver with integrator."""

    def test_two_body_orbit(self):
        """Test two-body Keplerian orbit energy conservation."""
        # Two equal masses - test that energy is conserved during orbit

        solver = DirectSummation(softening=0.01, G=1.0)
        masses = np.array([1.0, 1.0])

        # Initial positions: particles at +/- 1 on x-axis (separation = 2)
        positions = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        # Use a velocity that gives clearly negative total energy (bound orbit)
        # For circular orbit: v = sqrt(G*m/r) where r is separation
        # Use 70% of circular velocity for elliptical orbit with E < 0
        v_circ = np.sqrt(1.0 / 2.0)
        v = 0.7 * v_circ
        velocities = np.array([[0.0, -v, 0.0], [0.0, v, 0.0]])

        E0 = solver.total_energy(positions, velocities, masses)
        assert E0 < 0, "Should be bound (negative energy)"

        def acc_func(pos, vel, t):
            return solver.accelerations(pos, masses)

        integrator = Leapfrog(acc_func)

        # Integrate for some time
        n_steps = 1000
        times = np.linspace(0, 10.0, n_steps + 1)

        saved_t, saved_pos, saved_vel = integrator.integrate(
            positions, velocities, times, save_every=200, progressbar=False
        )

        # Check energy conservation throughout
        for i in range(len(saved_t)):
            E = solver.total_energy(saved_pos[i], saved_vel[i], masses)
            np.testing.assert_allclose(E, E0, rtol=0.01)

    def test_energy_conservation_nbody(self):
        """Test energy conservation for N-body system."""
        np.random.seed(42)

        solver = DirectSummation(softening=0.1, G=1.0)
        N = 10
        masses = np.ones(N)
        positions = np.random.randn(N, 3) * 2
        velocities = np.random.randn(N, 3) * 0.1

        E0 = solver.total_energy(positions, velocities, masses)

        def acc_func(pos, vel, t):
            return solver.accelerations(pos, masses)

        integrator = Leapfrog(acc_func)

        # Integrate with smaller timesteps for better energy conservation
        n_steps = 2000
        times = np.linspace(0, 5.0, n_steps + 1)

        saved_t, saved_pos, saved_vel = integrator.integrate(
            positions, velocities, times, save_every=400, progressbar=False
        )

        # Check energy at each snapshot
        for i in range(len(saved_t)):
            E = solver.total_energy(saved_pos[i], saved_vel[i], masses)
            # Symplectic integrator should conserve energy reasonably well
            np.testing.assert_allclose(E, E0, rtol=0.05)

    def test_momentum_conservation(self):
        """Test momentum conservation for isolated system."""
        solver = DirectSummation(softening=0.1, G=1.0)
        masses = np.array([1.0, 2.0, 3.0])
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        velocities = np.array([
            [0.1, 0.0, 0.0],
            [-0.1, 0.2, 0.0],
            [0.0, -0.1, 0.1],
        ])

        # Initial momentum
        P0 = np.sum(masses[:, np.newaxis] * velocities, axis=0)

        def acc_func(pos, vel, t):
            return solver.accelerations(pos, masses)

        integrator = Leapfrog(acc_func)

        n_steps = 200
        times = np.linspace(0, 5.0, n_steps + 1)

        saved_t, saved_pos, saved_vel = integrator.integrate(
            positions, velocities, times, save_every=50, progressbar=False
        )

        # Check momentum at each snapshot
        for i in range(len(saved_t)):
            P = np.sum(masses[:, np.newaxis] * saved_vel[i], axis=0)
            np.testing.assert_allclose(P, P0, atol=1e-10)

    def test_center_of_mass_stationary(self):
        """Test that center of mass remains stationary."""
        solver = DirectSummation(softening=0.1, G=1.0)
        masses = np.array([1.0, 2.0])
        positions = np.array([[1.0, 0.0, 0.0], [-0.5, 0.0, 0.0]])  # COM at origin
        velocities = np.array([[0.0, 0.5, 0.0], [0.0, -0.25, 0.0]])  # Zero net momentum

        # Verify initial COM at origin
        COM0 = np.sum(masses[:, np.newaxis] * positions, axis=0) / np.sum(masses)
        np.testing.assert_allclose(COM0, [0, 0, 0], atol=1e-10)

        def acc_func(pos, vel, t):
            return solver.accelerations(pos, masses)

        integrator = Leapfrog(acc_func)

        n_steps = 200
        times = np.linspace(0, 5.0, n_steps + 1)

        saved_t, saved_pos, saved_vel = integrator.integrate(
            positions, velocities, times, save_every=50, progressbar=False
        )

        # Check COM at each snapshot
        for i in range(len(saved_t)):
            COM = np.sum(masses[:, np.newaxis] * saved_pos[i], axis=0) / np.sum(masses)
            np.testing.assert_allclose(COM, [0, 0, 0], atol=1e-10)
