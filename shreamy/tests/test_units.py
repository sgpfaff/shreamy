"""
Tests for the units module.
"""

import pytest
import numpy as np

from shreamy.units import (
    UnitSystem,
    get_default_units,
    set_default_units,
    gyr_to_natural,
    natural_to_gyr,
    dynamical_time,
    crossing_time,
    relaxation_time,
    from_galpy_orbit,
    to_galpy_orbit,
    to_astropy_units,
    from_astropy_quantity,
    to_natural,
    to_physical,
    G_CGS,
    KPC_TO_CM,
    MSUN_TO_G,
    KM_TO_CM,
    GYR_TO_S,
)


def _galpy_available():
    """Check if galpy is installed."""
    try:
        import galpy
        return True
    except ImportError:
        return False


def _astropy_available():
    """Check if astropy is installed."""
    try:
        import astropy
        return True
    except ImportError:
        return False


class TestUnitSystem:
    """Tests for UnitSystem class."""

    def test_default_units(self):
        """Test default unit system values."""
        units = UnitSystem()
        assert units.ro == 8.0
        assert units.vo == 220.0
        assert units.G == 1.0

    def test_custom_units(self):
        """Test custom unit system values."""
        units = UnitSystem(ro=10.0, vo=250.0)
        assert units.ro == 10.0
        assert units.vo == 250.0

    def test_time_in_gyr(self):
        """Test time unit in Gyr calculation."""
        units = UnitSystem(ro=8.0, vo=220.0)
        # ro/vo in natural units, converted to Gyr
        # 8 kpc / 220 km/s
        # 8 * 3.086e16 km / 220 km/s = 1.12e15 s
        # 1.12e15 s / (3.156e16 s/Gyr) ≈ 0.0355 Gyr
        assert 0.03 < units.time_in_gyr < 0.04  # Rough check

    def test_mass_in_msun(self):
        """Test mass unit in solar masses calculation."""
        units = UnitSystem(ro=8.0, vo=220.0)
        # M = vo^2 * ro / G
        # This should be ~9e10 Msun for galpy defaults
        assert 8e10 < units.mass_in_msun < 1e11

    def test_position_to_physical(self):
        """Test position conversion to physical units."""
        units = UnitSystem(ro=8.0, vo=220.0)
        x_natural = np.array([1.0, 2.0, 0.5])
        x_kpc = units.position_to_physical(x_natural)
        np.testing.assert_allclose(x_kpc, [8.0, 16.0, 4.0])

    def test_velocity_to_physical(self):
        """Test velocity conversion to physical units."""
        units = UnitSystem(ro=8.0, vo=220.0)
        v_natural = np.array([1.0, 0.5])
        v_kms = units.velocity_to_physical(v_natural)
        np.testing.assert_allclose(v_kms, [220.0, 110.0])

    def test_time_to_physical(self):
        """Test time conversion to physical units."""
        units = UnitSystem(ro=8.0, vo=220.0)
        t_natural = 1.0
        t_gyr = units.time_to_physical(t_natural)
        np.testing.assert_allclose(t_gyr, units.time_in_gyr)

    def test_mass_to_physical(self):
        """Test mass conversion to physical units."""
        units = UnitSystem(ro=8.0, vo=220.0)
        m_natural = np.array([1.0, 0.1])
        m_msun = units.mass_to_physical(m_natural)
        np.testing.assert_allclose(m_msun, [units.mass_in_msun, 0.1 * units.mass_in_msun])

    def test_energy_to_physical(self):
        """Test energy conversion to physical units."""
        units = UnitSystem(ro=8.0, vo=220.0)
        E_natural = 1.0
        E_phys = units.energy_to_physical(np.array([E_natural]))
        np.testing.assert_allclose(E_phys, [220.0**2])

    def test_acceleration_to_physical(self):
        """Test acceleration conversion to physical units."""
        units = UnitSystem(ro=8.0, vo=220.0)
        a_natural = np.array([1.0])
        a_phys = units.acceleration_to_physical(a_natural)
        # Result should be positive and have reasonable magnitude
        assert a_phys[0] > 0

    def test_position_from_physical(self):
        """Test position conversion from physical units."""
        units = UnitSystem(ro=8.0, vo=220.0)
        x_kpc = np.array([8.0, 16.0, 4.0])
        x_natural = units.position_from_physical(x_kpc)
        np.testing.assert_allclose(x_natural, [1.0, 2.0, 0.5])

    def test_velocity_from_physical(self):
        """Test velocity conversion from physical units."""
        units = UnitSystem(ro=8.0, vo=220.0)
        v_kms = np.array([220.0, 110.0])
        v_natural = units.velocity_from_physical(v_kms)
        np.testing.assert_allclose(v_natural, [1.0, 0.5])

    def test_time_from_physical(self):
        """Test time conversion from physical units."""
        units = UnitSystem(ro=8.0, vo=220.0)
        t_gyr = units.time_in_gyr
        t_natural = units.time_from_physical(t_gyr)
        np.testing.assert_allclose(t_natural, 1.0)

    def test_mass_from_physical(self):
        """Test mass conversion from physical units."""
        units = UnitSystem(ro=8.0, vo=220.0)
        m_msun = np.array([units.mass_in_msun, 0.1 * units.mass_in_msun])
        m_natural = units.mass_from_physical(m_msun)
        np.testing.assert_allclose(m_natural, [1.0, 0.1])

    def test_roundtrip_position(self):
        """Test position conversion roundtrip."""
        units = UnitSystem(ro=8.0, vo=220.0)
        x_original = np.array([1.5, 2.5, 0.3])
        x_roundtrip = units.position_from_physical(units.position_to_physical(x_original))
        np.testing.assert_allclose(x_roundtrip, x_original)

    def test_roundtrip_velocity(self):
        """Test velocity conversion roundtrip."""
        units = UnitSystem(ro=8.0, vo=220.0)
        v_original = np.array([0.8, 1.2])
        v_roundtrip = units.velocity_from_physical(units.velocity_to_physical(v_original))
        np.testing.assert_allclose(v_roundtrip, v_original)

    def test_roundtrip_time(self):
        """Test time conversion roundtrip."""
        units = UnitSystem(ro=8.0, vo=220.0)
        t_original = 5.0
        t_roundtrip = units.time_from_physical(units.time_to_physical(t_original))
        np.testing.assert_allclose(t_roundtrip, t_original)

    def test_roundtrip_mass(self):
        """Test mass conversion roundtrip."""
        units = UnitSystem(ro=8.0, vo=220.0)
        m_original = np.array([0.5, 2.0])
        m_roundtrip = units.mass_from_physical(units.mass_to_physical(m_original))
        np.testing.assert_allclose(m_roundtrip, m_original)

    def test_repr(self):
        """Test string representation."""
        units = UnitSystem(ro=8.0, vo=220.0)
        repr_str = repr(units)
        assert "ro=8" in repr_str
        assert "vo=220" in repr_str
        assert "Gyr" in repr_str
        assert "Msun" in repr_str


class TestDefaultUnits:
    """Tests for default unit system functions."""

    def test_get_default_units(self):
        """Test getting default units."""
        units = get_default_units()
        assert isinstance(units, UnitSystem)

    def test_set_default_units(self):
        """Test setting default units."""
        # Store original
        original = get_default_units()
        original_ro = original.ro
        
        # Change defaults
        set_default_units(ro=10.0, vo=250.0)
        new_units = get_default_units()
        assert new_units.ro == 10.0
        assert new_units.vo == 250.0
        
        # Restore original
        set_default_units(ro=original_ro, vo=220.0)


class TestTimeUtilities:
    """Tests for time conversion utilities."""

    def test_gyr_to_natural(self):
        """Test Gyr to natural units conversion."""
        units = UnitSystem(ro=8.0, vo=220.0)
        t_gyr = 1.0
        t_natural = gyr_to_natural(t_gyr, ro=8.0, vo=220.0)
        expected = t_gyr / units.time_in_gyr
        np.testing.assert_allclose(t_natural, expected)

    def test_natural_to_gyr(self):
        """Test natural units to Gyr conversion."""
        units = UnitSystem(ro=8.0, vo=220.0)
        t_natural = 1.0
        t_gyr = natural_to_gyr(t_natural, ro=8.0, vo=220.0)
        expected = t_natural * units.time_in_gyr
        np.testing.assert_allclose(t_gyr, expected)

    def test_gyr_natural_roundtrip(self):
        """Test Gyr to natural and back."""
        t_original = 2.5
        t_roundtrip = natural_to_gyr(gyr_to_natural(t_original), ro=8.0, vo=220.0)
        np.testing.assert_allclose(t_roundtrip, t_original)

    def test_dynamical_time_natural(self):
        """Test dynamical time calculation in natural units."""
        # For density = 1 in natural units (G=1)
        # t_dyn = sqrt(3*pi / 16) ≈ 0.767
        density = 1.0
        t_dyn = dynamical_time(density, units="natural")
        expected = np.sqrt(3 * np.pi / 16)
        np.testing.assert_allclose(t_dyn, expected)

    def test_dynamical_time_physical(self):
        """Test dynamical time calculation in physical units."""
        # Use a reasonable density
        density = 1e8  # Msun/kpc^3
        t_dyn = dynamical_time(density, units="physical")
        # Should be a positive number in Gyr, order of ~0.01-1 Gyr
        assert t_dyn > 0
        assert t_dyn < 10  # Should be less than 10 Gyr

    def test_dynamical_time_unknown_units(self):
        """Test dynamical time with unknown units raises error."""
        with pytest.raises(ValueError, match="Unknown units"):
            dynamical_time(1.0, units="invalid")

    def test_crossing_time(self):
        """Test crossing time calculation."""
        velocity_dispersion = 10.0
        radius = 5.0
        t_cross = crossing_time(velocity_dispersion, radius)
        expected = radius / velocity_dispersion
        np.testing.assert_allclose(t_cross, expected)

    def test_relaxation_time(self):
        """Test relaxation time calculation."""
        n_particles = 10000
        t_cross = 1.0
        t_relax = relaxation_time(n_particles, t_cross)
        expected = (n_particles / np.log(n_particles)) * t_cross
        np.testing.assert_allclose(t_relax, expected)


class TestGalpyCompatibility:
    """Tests for galpy unit compatibility."""

    def test_galpy_ro_vo_convention(self):
        """Test that units match galpy conventions."""
        # galpy uses ro=8 kpc, vo=220 km/s as defaults
        units = UnitSystem(ro=8.0, vo=220.0)
        # In galpy natural units, G=1
        assert units.G == 1.0

    @pytest.mark.skipif(
        not _galpy_available(),
        reason="galpy not installed"
    )
    def test_from_galpy_orbit(self):
        """Test extraction of coordinates from galpy Orbit."""
        from galpy.orbit import Orbit
        
        # galpy uses cylindrical coords by default: [R, vR, vT, z, vz, phi]
        # Create orbit at R=1, phi=0 -> x=1, y=0, z=0
        # with vR=0, vT=1, vz=0 -> vx=0, vy=1, vz=0
        orb = Orbit(vxvv=[1.0, 0.0, 1.0, 0.0, 0.0, 0.0], ro=8.0, vo=220.0)
        
        positions, velocities = from_galpy_orbit(orb)
        
        np.testing.assert_allclose(positions[0], [1.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(velocities[0], [0.0, 1.0, 0.0], atol=1e-10)

    @pytest.mark.skipif(
        not _galpy_available(),
        reason="galpy not installed"
    )
    def test_to_galpy_orbit_single(self):
        """Test creation of galpy Orbit from single position/velocity."""
        positions = np.array([1.0, 0.0, 0.5])
        velocities = np.array([0.0, 1.0, 0.0])
        
        orb = to_galpy_orbit(positions, velocities)
        
        # Check that orbit was created
        assert orb is not None
        # Check coordinates match
        np.testing.assert_allclose(orb.x(use_physical=False), 1.0, atol=1e-10)
        np.testing.assert_allclose(orb.y(use_physical=False), 0.0, atol=1e-10)

    @pytest.mark.skipif(
        not _galpy_available(),
        reason="galpy not installed"
    )
    def test_to_galpy_orbit_multiple(self):
        """Test creation of galpy Orbit from multiple positions/velocities."""
        positions = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        velocities = np.array([[0.0, 1.0, 0.0], [0.0, 0.5, 0.0]])
        
        orbits = to_galpy_orbit(positions, velocities)
        
        assert len(orbits) == 2

    def test_to_galpy_orbit_at_origin(self):
        """Test creation of galpy Orbit for particle at R=0 (on z-axis)."""
        # Particle on the z-axis (R=0)
        positions = np.array([[0.0, 0.0, 1.0]])
        velocities = np.array([[0.1, 0.0, 0.0]])
        
        orbit = to_galpy_orbit(positions, velocities)
        
        # Should return a valid orbit (single orbit, not a list)
        # R should be very small (essentially 0)
        assert orbit.R() < 1e-10

    def test_from_galpy_orbit_no_galpy(self):
        """Test that from_galpy_orbit fails gracefully without galpy."""
        # This test only makes sense if galpy is not available
        # We can't easily test this without mocking the import
        pass


class TestAstropyIntegration:
    """Tests for astropy integration."""

    @pytest.mark.skipif(
        not _astropy_available(),
        reason="astropy not installed"
    )
    def test_to_astropy_units(self):
        """Test getting astropy unit equivalencies."""
        units_dict = to_astropy_units(ro=8.0, vo=220.0)
        
        assert 'position' in units_dict
        assert 'velocity' in units_dict
        assert 'time' in units_dict
        assert 'mass' in units_dict

    @pytest.mark.skipif(
        not _astropy_available(),
        reason="astropy not installed"
    )
    def test_from_astropy_quantity_position(self):
        """Test converting astropy position to natural units."""
        import astropy.units as u
        
        pos = 16 * u.kpc
        natural = from_astropy_quantity(pos, 'position', ro=8.0, vo=220.0)
        np.testing.assert_allclose(natural, 2.0)

    @pytest.mark.skipif(
        not _astropy_available(),
        reason="astropy not installed"
    )
    def test_from_astropy_quantity_velocity(self):
        """Test converting astropy velocity to natural units."""
        import astropy.units as u
        
        vel = 440 * u.km / u.s
        natural = from_astropy_quantity(vel, 'velocity', ro=8.0, vo=220.0)
        np.testing.assert_allclose(natural, 2.0)

    @pytest.mark.skipif(
        not _astropy_available(),
        reason="astropy not installed"
    )
    def test_from_astropy_quantity_time(self):
        """Test converting astropy time to natural units."""
        import astropy.units as u
        
        units = UnitSystem(ro=8.0, vo=220.0)
        time_gyr = 2 * units.time_in_gyr * u.Gyr
        natural = from_astropy_quantity(time_gyr, 'time', ro=8.0, vo=220.0)
        np.testing.assert_allclose(natural, 2.0)

    @pytest.mark.skipif(
        not _astropy_available(),
        reason="astropy not installed"
    )
    def test_from_astropy_quantity_mass(self):
        """Test converting astropy mass to natural units."""
        import astropy.units as u
        
        units = UnitSystem(ro=8.0, vo=220.0)
        mass_msun = 2 * units.mass_in_msun * u.Msun
        natural = from_astropy_quantity(mass_msun, 'mass', ro=8.0, vo=220.0)
        np.testing.assert_allclose(natural, 2.0)

    @pytest.mark.skipif(
        not _astropy_available(),
        reason="astropy not installed"
    )
    def test_from_astropy_quantity_unknown_target(self):
        """Test that unknown target raises error."""
        import astropy.units as u
        
        pos = 16 * u.kpc
        with pytest.raises(ValueError, match="Unknown target"):
            from_astropy_quantity(pos, 'invalid')


class TestPhysicalConstants:
    """Tests for physical constants."""

    def test_gravitational_constant(self):
        """Test G value is reasonable."""
        # G = 6.674e-8 cm^3 / (g * s^2)
        assert 6.6e-8 < G_CGS < 6.8e-8

    def test_kpc_to_cm(self):
        """Test kpc to cm conversion factor."""
        # 1 kpc ≈ 3.086e21 cm
        assert 3.0e21 < KPC_TO_CM < 3.2e21

    def test_msun_to_g(self):
        """Test solar mass in grams."""
        # Msun ≈ 1.989e33 g
        assert 1.9e33 < MSUN_TO_G < 2.1e33

    def test_km_to_cm(self):
        """Test km to cm conversion factor."""
        assert KM_TO_CM == 1e5

    def test_gyr_to_s(self):
        """Test Gyr to seconds conversion."""
        # 1 Gyr ≈ 3.156e16 s
        assert 3.1e16 < GYR_TO_S < 3.2e16


class TestFlexibleConversion:
    """Tests for flexible input conversion functions."""

    # =========================================================================
    # to_natural tests
    # =========================================================================

    def test_to_natural_already_natural(self):
        """Test that natural units pass through unchanged."""
        result = to_natural(2.0, 'position')
        np.testing.assert_allclose(result, 2.0)

    def test_to_natural_from_physical_position(self):
        """Test converting physical position to natural."""
        # 16 kpc with ro=8 -> 2.0 natural
        result = to_natural(16.0, 'position', physical=True)
        np.testing.assert_allclose(result, 2.0)

    def test_to_natural_from_physical_velocity(self):
        """Test converting physical velocity to natural."""
        # 440 km/s with vo=220 -> 2.0 natural
        result = to_natural(440.0, 'velocity', physical=True)
        np.testing.assert_allclose(result, 2.0)

    def test_to_natural_from_physical_time(self):
        """Test converting physical time to natural."""
        units = UnitSystem()
        # 2 * time_in_gyr Gyr -> 2.0 natural
        result = to_natural(2 * units.time_in_gyr, 'time', physical=True)
        np.testing.assert_allclose(result, 2.0)

    def test_to_natural_from_physical_mass(self):
        """Test converting physical mass to natural."""
        units = UnitSystem()
        # 2 * mass_in_msun Msun -> 2.0 natural
        result = to_natural(2 * units.mass_in_msun, 'mass', physical=True)
        np.testing.assert_allclose(result, 2.0)

    def test_to_natural_array_input(self):
        """Test with array input."""
        result = to_natural([8.0, 16.0, 24.0], 'position', physical=True)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_to_natural_invalid_unit_type(self):
        """Test that invalid unit type raises error."""
        with pytest.raises(ValueError, match="Unknown unit_type"):
            to_natural(1.0, 'invalid', physical=True)

    @pytest.mark.skipif(
        not _astropy_available(),
        reason="astropy not installed"
    )
    def test_to_natural_astropy_quantity(self):
        """Test auto-detection and conversion of astropy Quantity."""
        import astropy.units as u
        
        pos = 16 * u.kpc
        result = to_natural(pos, 'position')
        np.testing.assert_allclose(result, 2.0)

    @pytest.mark.skipif(
        not _astropy_available(),
        reason="astropy not installed"
    )
    def test_to_natural_astropy_velocity(self):
        """Test astropy velocity conversion."""
        import astropy.units as u
        
        vel = 440 * u.km / u.s
        result = to_natural(vel, 'velocity')
        np.testing.assert_allclose(result, 2.0)

    # =========================================================================
    # to_physical tests
    # =========================================================================

    def test_to_physical_from_natural_position(self):
        """Test converting natural position to physical."""
        result = to_physical(2.0, 'position')
        np.testing.assert_allclose(result, 16.0)

    def test_to_physical_from_natural_velocity(self):
        """Test converting natural velocity to physical."""
        result = to_physical(2.0, 'velocity')
        np.testing.assert_allclose(result, 440.0)

    def test_to_physical_from_natural_time(self):
        """Test converting natural time to physical."""
        units = UnitSystem()
        result = to_physical(2.0, 'time')
        np.testing.assert_allclose(result, 2.0 * units.time_in_gyr)

    def test_to_physical_from_natural_mass(self):
        """Test converting natural mass to physical."""
        units = UnitSystem()
        result = to_physical(2.0, 'mass')
        np.testing.assert_allclose(result, 2.0 * units.mass_in_msun)

    def test_to_physical_already_physical(self):
        """Test that physical units pass through unchanged."""
        result = to_physical(16.0, 'position', natural=False)
        np.testing.assert_allclose(result, 16.0)

    def test_to_physical_array_input(self):
        """Test with array input."""
        result = to_physical([1.0, 2.0, 3.0], 'position')
        np.testing.assert_allclose(result, [8.0, 16.0, 24.0])

    def test_to_physical_invalid_unit_type(self):
        """Test that invalid unit type raises error."""
        with pytest.raises(ValueError, match="Unknown unit_type"):
            to_physical(1.0, 'invalid')

    @pytest.mark.skipif(
        not _astropy_available(),
        reason="astropy not installed"
    )
    def test_to_physical_astropy_quantity(self):
        """Test extracting physical value from astropy Quantity."""
        import astropy.units as u
        
        # 16000 pc should become 16.0 kpc
        pos = 16000 * u.pc
        result = to_physical(pos, 'position')
        np.testing.assert_allclose(result, 16.0)

    @pytest.mark.skipif(
        not _astropy_available(),
        reason="astropy not installed"
    )
    def test_to_physical_astropy_all_types(self):
        """Test astropy conversion for all unit types."""
        import astropy.units as u
        
        # Position
        pos = 16 * u.kpc
        assert np.isclose(to_physical(pos, 'position'), 16.0)
        
        # Velocity
        vel = 440 * u.km / u.s
        assert np.isclose(to_physical(vel, 'velocity'), 440.0)
        
        # Time
        t = 1.0 * u.Gyr
        assert np.isclose(to_physical(t, 'time'), 1.0)
        
        # Mass
        m = 1e10 * u.Msun
        assert np.isclose(to_physical(m, 'mass'), 1e10)

    @pytest.mark.skipif(
        not _astropy_available(),
        reason="astropy not installed"
    )
    def test_to_physical_astropy_invalid_type(self):
        """Test that invalid unit type raises error for astropy."""
        import astropy.units as u
        
        pos = 16 * u.kpc
        with pytest.raises(ValueError, match="Unknown unit_type"):
            to_physical(pos, 'invalid')
