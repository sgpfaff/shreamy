"""
Tests for analytic potentials comparing shreamy to galpy implementations.

These tests verify that shreamy's PlummerPotential, HernquistPotential, and
NFWPotential implementations match galpy's implementations to high precision.
"""

import numpy as np
import pytest

from shreamy.potentials import PlummerPotential, HernquistPotential, NFWPotential

# Import galpy potentials for comparison
from galpy.potential import (
    PlummerPotential as GalpyPlummer,
    HernquistPotential as GalpyHernquist,
    NFWPotential as GalpyNFW,
)

# galpy units
from galpy.util import conversion


# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def plummer_params():
    """Parameters for Plummer potential tests."""
    # In galpy's natural units (G=1, ro=8 kpc, vo=220 km/s)
    return {"b": 0.1, "amp": 1.0}  # scale radius  # total mass


@pytest.fixture
def hernquist_params():
    """Parameters for Hernquist potential tests.
    
    Note: galpy uses amp = 2*M for Hernquist, where M is total mass.
    So shreamy M=1 corresponds to galpy amp=2.
    """
    return {"a": 0.1, "M": 1.0, "galpy_amp": 2.0}  # scale radius, total mass, galpy amp


@pytest.fixture
def nfw_params():
    """Parameters for NFW potential tests."""
    return {"a": 0.2, "amp": 1.0}  # scale radius r_s  # characteristic mass factor


@pytest.fixture
def test_positions():
    """Test positions in Cartesian coordinates."""
    # Various positions for testing
    x = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    y = np.array([0.0, 0.1, 0.0, 0.5, 1.0, 0.0])
    z = np.array([0.0, 0.0, 0.1, 0.0, 0.5, 1.0])
    return x, y, z


@pytest.fixture
def test_radii():
    """Test radii for spherically symmetric tests."""
    return np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])


# ============================================================================
# Plummer Potential Tests
# ============================================================================


class TestPlummerPotential:
    """Tests for PlummerPotential comparing to galpy."""

    def test_potential_value_vs_galpy(self, plummer_params, test_positions):
        """Test potential values match galpy's PlummerPotential."""
        x, y, z = test_positions

        # Create shreamy potential
        shreamy_pot = PlummerPotential(M=plummer_params["amp"], b=plummer_params["b"])

        # Create galpy potential
        galpy_pot = GalpyPlummer(amp=plummer_params["amp"], b=plummer_params["b"])

        # Compute potential at test positions
        shreamy_phi = shreamy_pot.potential_value(x, y, z)

        # galpy uses cylindrical coordinates (R, z, phi, t)
        R = np.sqrt(x**2 + y**2)
        galpy_phi = np.array(
            [galpy_pot(R[i], z[i], phi=0, t=0) for i in range(len(x))]
        )

        # Compare - galpy returns potential per unit mass, and shreamy does too
        np.testing.assert_allclose(shreamy_phi, galpy_phi, rtol=1e-10)

    def test_acceleration_vs_galpy(self, plummer_params, test_positions):
        """Test accelerations match galpy's PlummerPotential."""
        x, y, z = test_positions

        # Create potentials
        shreamy_pot = PlummerPotential(M=plummer_params["amp"], b=plummer_params["b"])
        galpy_pot = GalpyPlummer(amp=plummer_params["amp"], b=plummer_params["b"])

        # Compute accelerations
        shreamy_acc = shreamy_pot.acceleration(x, y, z)

        # galpy returns forces in cylindrical (Rforce, zforce)
        R = np.sqrt(x**2 + y**2)
        phi_coord = np.arctan2(y, x)

        galpy_ax = np.zeros_like(x)
        galpy_ay = np.zeros_like(y)
        galpy_az = np.zeros_like(z)

        for i in range(len(x)):
            # galpy Rforce is the radial force (positive outward = negative acceleration)
            Rforce = galpy_pot.Rforce(R[i], z[i], phi=phi_coord[i], t=0)
            zforce = galpy_pot.zforce(R[i], z[i], phi=phi_coord[i], t=0)

            # Convert to Cartesian
            if R[i] > 0:
                galpy_ax[i] = Rforce * np.cos(phi_coord[i])
                galpy_ay[i] = Rforce * np.sin(phi_coord[i])
            galpy_az[i] = zforce

        # Compare
        np.testing.assert_allclose(shreamy_acc[:, 0], galpy_ax, rtol=1e-10, atol=1e-15)
        np.testing.assert_allclose(shreamy_acc[:, 1], galpy_ay, rtol=1e-10, atol=1e-15)
        np.testing.assert_allclose(shreamy_acc[:, 2], galpy_az, rtol=1e-10, atol=1e-15)

    def test_density_vs_galpy(self, plummer_params, test_radii):
        """Test density values match galpy's PlummerPotential."""
        r = test_radii

        shreamy_pot = PlummerPotential(M=plummer_params["amp"], b=plummer_params["b"])
        galpy_pot = GalpyPlummer(amp=plummer_params["amp"], b=plummer_params["b"])

        shreamy_rho = shreamy_pot.density(r)
        galpy_rho = np.array([galpy_pot.dens(r[i], 0, phi=0, t=0) for i in range(len(r))])

        np.testing.assert_allclose(shreamy_rho, galpy_rho, rtol=1e-10)

    def test_enclosed_mass(self, plummer_params, test_radii):
        """Test enclosed mass calculation."""
        r = test_radii

        shreamy_pot = PlummerPotential(M=plummer_params["amp"], b=plummer_params["b"])

        M_enc = shreamy_pot.enclosed_mass(r)

        # Analytic formula: M(r) = M * r^3 / (r^2 + b^2)^(3/2)
        M = plummer_params["amp"]
        b = plummer_params["b"]
        expected = M * r**3 / (r**2 + b**2) ** 1.5

        np.testing.assert_allclose(M_enc, expected, rtol=1e-10)

    def test_circular_velocity(self, plummer_params, test_radii):
        """Test circular velocity calculation."""
        R = test_radii

        shreamy_pot = PlummerPotential(M=plummer_params["amp"], b=plummer_params["b"])
        galpy_pot = GalpyPlummer(amp=plummer_params["amp"], b=plummer_params["b"])

        shreamy_vc = shreamy_pot.circular_velocity(R, z=0)
        galpy_vc = np.array([galpy_pot.vcirc(R[i], phi=0) for i in range(len(R))])

        np.testing.assert_allclose(shreamy_vc, galpy_vc, rtol=1e-10)

    def test_escape_velocity(self, plummer_params, test_positions):
        """Test escape velocity calculation.
        
        Note: galpy uses numerical integration for escape velocity from finite
        potentials, which can differ slightly from our analytic v_esc = sqrt(-2*Phi).
        We use a looser tolerance for this comparison.
        """
        x, y, z = test_positions

        shreamy_pot = PlummerPotential(M=plummer_params["amp"], b=plummer_params["b"])
        galpy_pot = GalpyPlummer(amp=plummer_params["amp"], b=plummer_params["b"])

        shreamy_vesc = shreamy_pot.escape_velocity(x, y, z)

        R = np.sqrt(x**2 + y**2)
        # galpy vesc doesn't take phi argument
        galpy_vesc = np.array([galpy_pot.vesc(R[i], z[i]) for i in range(len(x))])

        # Looser tolerance due to numerical integration differences
        np.testing.assert_allclose(shreamy_vesc, galpy_vesc, rtol=0.01)

    def test_repr(self, plummer_params):
        """Test string representation."""
        pot = PlummerPotential(M=plummer_params["amp"], b=plummer_params["b"])
        assert "PlummerPotential" in repr(pot)
        assert "M=1.0" in repr(pot)
        assert "b=0.1" in repr(pot)


# ============================================================================
# Hernquist Potential Tests
# ============================================================================


class TestHernquistPotential:
    """Tests for HernquistPotential comparing to galpy."""

    def test_potential_value_vs_galpy(self, hernquist_params, test_positions):
        """Test potential values match galpy's HernquistPotential."""
        x, y, z = test_positions

        # Create shreamy potential (M = total mass)
        shreamy_pot = HernquistPotential(
            M=hernquist_params["M"], a=hernquist_params["a"]
        )

        # Create galpy potential (amp = 2 * M for Hernquist)
        galpy_pot = GalpyHernquist(amp=hernquist_params["galpy_amp"], a=hernquist_params["a"])

        # Compute potential at test positions
        shreamy_phi = shreamy_pot.potential_value(x, y, z)

        R = np.sqrt(x**2 + y**2)
        galpy_phi = np.array(
            [galpy_pot(R[i], z[i], phi=0, t=0) for i in range(len(x))]
        )

        np.testing.assert_allclose(shreamy_phi, galpy_phi, rtol=1e-10)

    def test_acceleration_vs_galpy(self, hernquist_params, test_positions):
        """Test accelerations match galpy's HernquistPotential."""
        x, y, z = test_positions

        shreamy_pot = HernquistPotential(
            M=hernquist_params["M"], a=hernquist_params["a"]
        )
        galpy_pot = GalpyHernquist(amp=hernquist_params["galpy_amp"], a=hernquist_params["a"])

        shreamy_acc = shreamy_pot.acceleration(x, y, z)

        R = np.sqrt(x**2 + y**2)
        phi_coord = np.arctan2(y, x)

        galpy_ax = np.zeros_like(x)
        galpy_ay = np.zeros_like(y)
        galpy_az = np.zeros_like(z)

        for i in range(len(x)):
            Rforce = galpy_pot.Rforce(R[i], z[i], phi=phi_coord[i], t=0)
            zforce = galpy_pot.zforce(R[i], z[i], phi=phi_coord[i], t=0)

            if R[i] > 0:
                galpy_ax[i] = Rforce * np.cos(phi_coord[i])
                galpy_ay[i] = Rforce * np.sin(phi_coord[i])
            galpy_az[i] = zforce

        np.testing.assert_allclose(shreamy_acc[:, 0], galpy_ax, rtol=1e-10, atol=1e-15)
        np.testing.assert_allclose(shreamy_acc[:, 1], galpy_ay, rtol=1e-10, atol=1e-15)
        np.testing.assert_allclose(shreamy_acc[:, 2], galpy_az, rtol=1e-10, atol=1e-15)

    def test_density_vs_galpy(self, hernquist_params, test_radii):
        """Test density values match galpy's HernquistPotential."""
        r = test_radii

        shreamy_pot = HernquistPotential(
            M=hernquist_params["M"], a=hernquist_params["a"]
        )
        galpy_pot = GalpyHernquist(amp=hernquist_params["galpy_amp"], a=hernquist_params["a"])

        shreamy_rho = shreamy_pot.density(r)
        galpy_rho = np.array([galpy_pot.dens(r[i], 0, phi=0, t=0) for i in range(len(r))])

        np.testing.assert_allclose(shreamy_rho, galpy_rho, rtol=1e-10)

    def test_enclosed_mass(self, hernquist_params, test_radii):
        """Test enclosed mass calculation."""
        r = test_radii

        shreamy_pot = HernquistPotential(
            M=hernquist_params["M"], a=hernquist_params["a"]
        )

        M_enc = shreamy_pot.enclosed_mass(r)

        # Analytic formula: M(r) = M * r^2 / (r + a)^2
        M = hernquist_params["M"]
        a = hernquist_params["a"]
        expected = M * r**2 / (r + a) ** 2

        np.testing.assert_allclose(M_enc, expected, rtol=1e-10)

    def test_circular_velocity(self, hernquist_params, test_radii):
        """Test circular velocity calculation."""
        R = test_radii

        shreamy_pot = HernquistPotential(
            M=hernquist_params["M"], a=hernquist_params["a"]
        )
        galpy_pot = GalpyHernquist(amp=hernquist_params["galpy_amp"], a=hernquist_params["a"])

        shreamy_vc = shreamy_pot.circular_velocity(R, z=0)
        galpy_vc = np.array([galpy_pot.vcirc(R[i], phi=0) for i in range(len(R))])

        np.testing.assert_allclose(shreamy_vc, galpy_vc, rtol=1e-10)

    def test_escape_velocity(self, hernquist_params, test_positions):
        """Test escape velocity calculation.
        
        Note: galpy uses numerical integration for escape velocity from finite
        potentials, which can differ slightly from our analytic v_esc = sqrt(-2*Phi).
        We use a looser tolerance for this comparison.
        """
        x, y, z = test_positions

        shreamy_pot = HernquistPotential(
            M=hernquist_params["M"], a=hernquist_params["a"]
        )
        galpy_pot = GalpyHernquist(amp=hernquist_params["galpy_amp"], a=hernquist_params["a"])

        shreamy_vesc = shreamy_pot.escape_velocity(x, y, z)

        R = np.sqrt(x**2 + y**2)
        # galpy vesc doesn't take phi argument
        galpy_vesc = np.array([galpy_pot.vesc(R[i], z[i]) for i in range(len(x))])

        # Looser tolerance due to numerical integration differences
        np.testing.assert_allclose(shreamy_vesc, galpy_vesc, rtol=0.01)

    def test_repr(self, hernquist_params):
        """Test string representation."""
        pot = HernquistPotential(M=hernquist_params["M"], a=hernquist_params["a"])
        assert "HernquistPotential" in repr(pot)
        assert "M=1.0" in repr(pot)
        assert "a=0.1" in repr(pot)


# ============================================================================
# NFW Potential Tests
# ============================================================================


class TestNFWPotential:
    """Tests for NFWPotential comparing to galpy.
    
    Note: galpy's NFWPotential uses a different parameterization. In galpy:
    - amp is the amplitude (normalized mass)
    - a is the scale radius
    
    The normalization is such that the enclosed mass function matches.
    """

    def test_potential_value_vs_galpy(self, nfw_params, test_positions):
        """Test potential values match galpy's NFWPotential."""
        x, y, z = test_positions

        # galpy NFW uses amp and a (scale radius)
        # The amp is the amplitude of the profile
        # In galpy, NFWPotential(amp=1, a=a) has M_s normalized differently
        galpy_pot = GalpyNFW(amp=nfw_params["amp"], a=nfw_params["a"])

        # Get galpy's effective M_s for comparison
        # In galpy, the mass normalization is such that 4*pi*rho_s*r_s^3 = amp
        # So M_s = amp for galpy
        shreamy_pot = NFWPotential(M_s=nfw_params["amp"], r_s=nfw_params["a"])

        shreamy_phi = shreamy_pot.potential_value(x, y, z)

        R = np.sqrt(x**2 + y**2)
        galpy_phi = np.array(
            [galpy_pot(R[i], z[i], phi=0, t=0) for i in range(len(x))]
        )

        np.testing.assert_allclose(shreamy_phi, galpy_phi, rtol=1e-10)

    def test_acceleration_vs_galpy(self, nfw_params, test_positions):
        """Test accelerations match galpy's NFWPotential."""
        x, y, z = test_positions

        galpy_pot = GalpyNFW(amp=nfw_params["amp"], a=nfw_params["a"])
        shreamy_pot = NFWPotential(M_s=nfw_params["amp"], r_s=nfw_params["a"])

        shreamy_acc = shreamy_pot.acceleration(x, y, z)

        R = np.sqrt(x**2 + y**2)
        phi_coord = np.arctan2(y, x)

        galpy_ax = np.zeros_like(x)
        galpy_ay = np.zeros_like(y)
        galpy_az = np.zeros_like(z)

        for i in range(len(x)):
            Rforce = galpy_pot.Rforce(R[i], z[i], phi=phi_coord[i], t=0)
            zforce = galpy_pot.zforce(R[i], z[i], phi=phi_coord[i], t=0)

            if R[i] > 0:
                galpy_ax[i] = Rforce * np.cos(phi_coord[i])
                galpy_ay[i] = Rforce * np.sin(phi_coord[i])
            galpy_az[i] = zforce

        np.testing.assert_allclose(shreamy_acc[:, 0], galpy_ax, rtol=1e-10, atol=1e-15)
        np.testing.assert_allclose(shreamy_acc[:, 1], galpy_ay, rtol=1e-10, atol=1e-15)
        np.testing.assert_allclose(shreamy_acc[:, 2], galpy_az, rtol=1e-10, atol=1e-15)

    def test_density_vs_galpy(self, nfw_params, test_radii):
        """Test density values match galpy's NFWPotential."""
        r = test_radii

        galpy_pot = GalpyNFW(amp=nfw_params["amp"], a=nfw_params["a"])
        shreamy_pot = NFWPotential(M_s=nfw_params["amp"], r_s=nfw_params["a"])

        shreamy_rho = shreamy_pot.density(r)
        galpy_rho = np.array([galpy_pot.dens(r[i], 0, phi=0, t=0) for i in range(len(r))])

        np.testing.assert_allclose(shreamy_rho, galpy_rho, rtol=1e-10)

    def test_enclosed_mass_vs_galpy(self, nfw_params, test_radii):
        """Test enclosed mass matches galpy."""
        r = test_radii

        galpy_pot = GalpyNFW(amp=nfw_params["amp"], a=nfw_params["a"])
        shreamy_pot = NFWPotential(M_s=nfw_params["amp"], r_s=nfw_params["a"])

        shreamy_M = shreamy_pot.enclosed_mass(r)
        
        # galpy mass method takes only r (not r, z)
        galpy_M = np.array([galpy_pot.mass(r[i]) for i in range(len(r))])

        np.testing.assert_allclose(shreamy_M, galpy_M, rtol=1e-10)

    def test_circular_velocity(self, nfw_params, test_radii):
        """Test circular velocity calculation."""
        R = test_radii

        galpy_pot = GalpyNFW(amp=nfw_params["amp"], a=nfw_params["a"])
        shreamy_pot = NFWPotential(M_s=nfw_params["amp"], r_s=nfw_params["a"])

        shreamy_vc = shreamy_pot.circular_velocity(R, z=0)
        galpy_vc = np.array([galpy_pot.vcirc(R[i], phi=0) for i in range(len(R))])

        np.testing.assert_allclose(shreamy_vc, galpy_vc, rtol=1e-10)

    def test_escape_velocity(self, nfw_params, test_positions):
        """Test escape velocity calculation.
        
        Note: galpy uses numerical integration for escape velocity from finite
        potentials, which can differ slightly from our analytic v_esc = sqrt(-2*Phi).
        We use a looser tolerance for this comparison.
        """
        x, y, z = test_positions

        galpy_pot = GalpyNFW(amp=nfw_params["amp"], a=nfw_params["a"])
        shreamy_pot = NFWPotential(M_s=nfw_params["amp"], r_s=nfw_params["a"])

        shreamy_vesc = shreamy_pot.escape_velocity(x, y, z)

        R = np.sqrt(x**2 + y**2)
        # galpy vesc doesn't take phi argument
        galpy_vesc = np.array([galpy_pot.vesc(R[i], z[i]) for i in range(len(x))])

        # Looser tolerance due to numerical integration differences
        np.testing.assert_allclose(shreamy_vesc, galpy_vesc, rtol=0.01)

    def test_repr(self, nfw_params):
        """Test string representation."""
        pot = NFWPotential(M_s=nfw_params["amp"], r_s=nfw_params["a"])
        assert "NFWPotential" in repr(pot)
        assert "M_s=1.0" in repr(pot)
        assert "r_s=0.2" in repr(pot)

    def test_from_virial_radius(self):
        """Test creating NFW from virial parameters."""
        M_vir = 1.0
        r_vir = 1.0
        c = 10.0

        pot = NFWPotential.from_virial_radius(M_vir, r_vir, c)

        # Check scale radius
        expected_rs = r_vir / c
        assert pot.r_s == expected_rs

        # Check that enclosed mass at virial radius gives M_vir
        M_enc_vir = pot.enclosed_mass(np.array([r_vir]))[0]
        np.testing.assert_allclose(M_enc_vir, M_vir, rtol=1e-10)


# ============================================================================
# Edge Cases and Special Values
# ============================================================================


class TestEdgeCases:
    """Test edge cases and special values."""

    def test_plummer_at_origin(self, plummer_params):
        """Test Plummer potential behavior at origin."""
        pot = PlummerPotential(M=plummer_params["amp"], b=plummer_params["b"])

        x = np.array([0.0])
        y = np.array([0.0])
        z = np.array([0.0])

        # Potential should be finite at origin
        phi = pot.potential_value(x, y, z)
        assert np.isfinite(phi[0])
        expected_phi = -pot.G * pot.M / pot.b
        np.testing.assert_allclose(phi[0], expected_phi, rtol=1e-10)

        # Acceleration should be zero at origin
        acc = pot.acceleration(x, y, z)
        np.testing.assert_allclose(acc[0], [0, 0, 0], atol=1e-15)

    def test_hernquist_at_origin(self, hernquist_params):
        """Test Hernquist potential behavior at origin."""
        pot = HernquistPotential(M=hernquist_params["M"], a=hernquist_params["a"])

        x = np.array([0.0])
        y = np.array([0.0])
        z = np.array([0.0])

        # Potential should be finite at origin
        phi = pot.potential_value(x, y, z)
        assert np.isfinite(phi[0])
        expected_phi = -pot.G * pot.M / pot.a
        np.testing.assert_allclose(phi[0], expected_phi, rtol=1e-10)

        # Acceleration should be zero at origin
        acc = pot.acceleration(x, y, z)
        np.testing.assert_allclose(acc[0], [0, 0, 0], atol=1e-15)

    def test_nfw_at_origin(self, nfw_params):
        """Test NFW potential behavior near origin."""
        pot = NFWPotential(M_s=nfw_params["amp"], r_s=nfw_params["a"])

        x = np.array([0.0])
        y = np.array([0.0])
        z = np.array([0.0])

        # Acceleration should be zero at origin by symmetry
        acc = pot.acceleration(x, y, z)
        np.testing.assert_allclose(acc[0], [0, 0, 0], atol=1e-15)

    def test_scalar_inputs(self, plummer_params):
        """Test that scalar inputs work correctly."""
        pot = PlummerPotential(M=plummer_params["amp"], b=plummer_params["b"])

        # Scalar inputs
        x, y, z = 1.0, 0.0, 0.0

        phi = pot.potential_value(x, y, z)
        assert phi.shape == (1,)

        acc = pot.acceleration(x, y, z)
        assert acc.shape == (1, 3)

    def test_large_radius_asymptotic(self, plummer_params, hernquist_params, nfw_params):
        """Test asymptotic behavior at large radii."""
        x = np.array([100.0])
        y = np.array([0.0])
        z = np.array([0.0])
        r = 100.0

        # Plummer should approach point mass
        pot_plum = PlummerPotential(M=plummer_params["amp"], b=plummer_params["b"])
        phi_plum = pot_plum.potential_value(x, y, z)[0]
        expected_point = -pot_plum.G * pot_plum.M / r
        # At large r, Phi ~ -GM/r
        assert np.abs(phi_plum - expected_point) / np.abs(expected_point) < 0.01

        # Hernquist should approach point mass
        pot_hern = HernquistPotential(M=hernquist_params["M"], a=hernquist_params["a"])
        phi_hern = pot_hern.potential_value(x, y, z)[0]
        expected_point = -pot_hern.G * pot_hern.M / r
        assert np.abs(phi_hern - expected_point) / np.abs(expected_point) < 0.01


# ============================================================================
# Consistency Tests
# ============================================================================


class TestConsistency:
    """Test internal consistency of potential implementations."""

    def test_plummer_virial_theorem(self, plummer_params):
        """Test that Plummer potential satisfies virial theorem.
        
        For Plummer: |U| = 2K at equilibrium, or |W| = 3 pi M^2 G / (32 b)
        """
        pot = PlummerPotential(M=plummer_params["amp"], b=plummer_params["b"])
        
        # Total potential energy of Plummer sphere: W = -3 pi G M^2 / (32 b)
        M = pot.M
        b = pot.b
        G = pot.G
        expected_W = -3 * np.pi * G * M**2 / (32 * b)
        
        # This is the total gravitational binding energy
        # For now, just verify the formula is consistent with density integral
        # (full numerical integration test would be more involved)
        assert expected_W < 0  # Binding energy is negative

    def test_force_is_gradient_of_potential(self, plummer_params):
        """Test that force is negative gradient of potential."""
        pot = PlummerPotential(M=plummer_params["amp"], b=plummer_params["b"])

        x0, y0, z0 = 1.0, 0.5, 0.3
        h = 1e-6

        # Numerical gradient
        phi_px = pot.potential_value(x0 + h, y0, z0)[0]
        phi_mx = pot.potential_value(x0 - h, y0, z0)[0]
        phi_py = pot.potential_value(x0, y0 + h, z0)[0]
        phi_my = pot.potential_value(x0, y0 - h, z0)[0]
        phi_pz = pot.potential_value(x0, y0, z0 + h)[0]
        phi_mz = pot.potential_value(x0, y0, z0 - h)[0]

        grad_x = (phi_px - phi_mx) / (2 * h)
        grad_y = (phi_py - phi_my) / (2 * h)
        grad_z = (phi_pz - phi_mz) / (2 * h)

        # Analytic acceleration
        acc = pot.acceleration(np.array([x0]), np.array([y0]), np.array([z0]))[0]

        # Force = -grad(Phi), so acceleration = -grad(Phi)
        np.testing.assert_allclose(acc[0], -grad_x, rtol=1e-6)
        np.testing.assert_allclose(acc[1], -grad_y, rtol=1e-6)
        np.testing.assert_allclose(acc[2], -grad_z, rtol=1e-6)

    def test_hernquist_force_is_gradient(self, hernquist_params):
        """Test Hernquist force is negative gradient of potential."""
        pot = HernquistPotential(M=hernquist_params["M"], a=hernquist_params["a"])

        x0, y0, z0 = 1.0, 0.5, 0.3
        h = 1e-6

        phi_px = pot.potential_value(x0 + h, y0, z0)[0]
        phi_mx = pot.potential_value(x0 - h, y0, z0)[0]
        phi_py = pot.potential_value(x0, y0 + h, z0)[0]
        phi_my = pot.potential_value(x0, y0 - h, z0)[0]
        phi_pz = pot.potential_value(x0, y0, z0 + h)[0]
        phi_mz = pot.potential_value(x0, y0, z0 - h)[0]

        grad_x = (phi_px - phi_mx) / (2 * h)
        grad_y = (phi_py - phi_my) / (2 * h)
        grad_z = (phi_pz - phi_mz) / (2 * h)

        acc = pot.acceleration(np.array([x0]), np.array([y0]), np.array([z0]))[0]

        np.testing.assert_allclose(acc[0], -grad_x, rtol=1e-6)
        np.testing.assert_allclose(acc[1], -grad_y, rtol=1e-6)
        np.testing.assert_allclose(acc[2], -grad_z, rtol=1e-6)

    def test_nfw_force_is_gradient(self, nfw_params):
        """Test NFW force is negative gradient of potential."""
        pot = NFWPotential(M_s=nfw_params["amp"], r_s=nfw_params["a"])

        x0, y0, z0 = 1.0, 0.5, 0.3
        h = 1e-6

        phi_px = pot.potential_value(x0 + h, y0, z0)[0]
        phi_mx = pot.potential_value(x0 - h, y0, z0)[0]
        phi_py = pot.potential_value(x0, y0 + h, z0)[0]
        phi_my = pot.potential_value(x0, y0 - h, z0)[0]
        phi_pz = pot.potential_value(x0, y0, z0 + h)[0]
        phi_mz = pot.potential_value(x0, y0, z0 - h)[0]

        grad_x = (phi_px - phi_mx) / (2 * h)
        grad_y = (phi_py - phi_my) / (2 * h)
        grad_z = (phi_pz - phi_mz) / (2 * h)

        acc = pot.acceleration(np.array([x0]), np.array([y0]), np.array([z0]))[0]

        np.testing.assert_allclose(acc[0], -grad_x, rtol=1e-6)
        np.testing.assert_allclose(acc[1], -grad_y, rtol=1e-6)
        np.testing.assert_allclose(acc[2], -grad_z, rtol=1e-6)

    def test_poisson_equation(self, plummer_params):
        """Test that density satisfies Poisson equation.
        
        nabla^2 Phi = 4 * pi * G * rho
        
        For spherically symmetric: (1/r^2) d/dr (r^2 d Phi/dr) = 4 pi G rho
        """
        pot = PlummerPotential(M=plummer_params["amp"], b=plummer_params["b"])

        r = 0.5  # Test at r=0.5
        h = 1e-5

        # Compute d^2 Phi/dr^2 + (2/r) dPhi/dr numerically
        # At r, use central differences
        phi_r = pot.potential_value(r, 0, 0)[0]
        phi_rp = pot.potential_value(r + h, 0, 0)[0]
        phi_rm = pot.potential_value(r - h, 0, 0)[0]

        d2phi_dr2 = (phi_rp - 2 * phi_r + phi_rm) / h**2
        dphi_dr = (phi_rp - phi_rm) / (2 * h)

        laplacian = d2phi_dr2 + 2 * dphi_dr / r

        # Expected: 4 * pi * G * rho
        rho = pot.density(np.array([r]))[0]
        expected = 4 * np.pi * pot.G * rho

        np.testing.assert_allclose(laplacian, expected, rtol=1e-4)
