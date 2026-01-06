"""
Tests for the particle module.
"""

import pytest
import numpy as np
import tempfile
import os

from shreamy.particle import ParticleSet, ParticleHistory, Snapshot


class TestParticleSet:
    """Tests for ParticleSet class."""

    def test_init_with_arrays(self):
        """Test ParticleSet initialization with numpy arrays."""
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        velocities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        masses = np.array([1.0, 2.0])
        
        ps = ParticleSet(positions, velocities, masses)
        
        assert ps.n_particles == 2
        np.testing.assert_array_equal(ps.positions, positions)
        np.testing.assert_array_equal(ps.velocities, velocities)
        np.testing.assert_array_equal(ps.masses, masses)
        assert ps.total_mass == 3.0

    def test_init_default_masses(self):
        """Test that default masses are unit masses."""
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        velocities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        ps = ParticleSet(positions, velocities)
        
        np.testing.assert_array_equal(ps.masses, np.ones(2))

    def test_init_default_components(self):
        """Test that default components are 'default'."""
        positions = np.array([[1.0, 2.0, 3.0]])
        velocities = np.array([[0.1, 0.2, 0.3]])
        
        ps = ParticleSet(positions, velocities)
        
        assert ps.component_names == ['default']
        np.testing.assert_array_equal(ps.components, np.array(['default']))

    def test_init_with_components(self):
        """Test ParticleSet initialization with components."""
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        velocities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        components = np.array(['dm', 'stellar'])
        
        ps = ParticleSet(positions, velocities, components=components)
        
        assert set(ps.component_names) == {'dm', 'stellar'}

    def test_init_validation_shape_mismatch(self):
        """Test that mismatched shapes raise ValueError."""
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        velocities = np.array([[0.1, 0.2, 0.3]])  # Wrong shape
        
        with pytest.raises(ValueError):
            ParticleSet(positions, velocities)

    def test_init_validation_wrong_dimensions(self):
        """Test that wrong dimensionality raises ValueError."""
        positions = np.array([[1.0, 2.0]])  # Only 2D
        velocities = np.array([[0.1, 0.2]])
        
        with pytest.raises(ValueError):
            ParticleSet(positions, velocities)

    def test_from_phase_space(self):
        """Test creating ParticleSet from 6D phase space array."""
        phase_space = np.array([
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
            [4.0, 5.0, 6.0, 0.4, 0.5, 0.6]
        ])
        masses = np.array([1.0, 2.0])
        
        ps = ParticleSet.from_phase_space(phase_space, masses)
        
        np.testing.assert_array_equal(ps.positions, phase_space[:, :3])
        np.testing.assert_array_equal(ps.velocities, phase_space[:, 3:])
        np.testing.assert_array_equal(ps.phase_space, phase_space)

    def test_from_cylindrical(self):
        """Test creating ParticleSet from cylindrical coordinates."""
        # Particle at R=1, phi=0, z=0 should be at x=1, y=0, z=0
        R = np.array([1.0])
        phi = np.array([0.0])
        z = np.array([0.0])
        vR = np.array([1.0])
        vT = np.array([0.0])
        vz = np.array([0.0])
        
        ps = ParticleSet.from_cylindrical(R, phi, z, vR, vT, vz)
        
        np.testing.assert_allclose(ps.x, [1.0])
        np.testing.assert_allclose(ps.y, [0.0], atol=1e-10)
        np.testing.assert_allclose(ps.z, [0.0])
        np.testing.assert_allclose(ps.vx, [1.0])
        np.testing.assert_allclose(ps.vy, [0.0], atol=1e-10)

    def test_from_cylindrical_at_phi_90(self):
        """Test cylindrical coordinates at phi=90 degrees."""
        R = np.array([1.0])
        phi = np.array([np.pi / 2])
        z = np.array([0.0])
        vR = np.array([0.0])
        vT = np.array([1.0])
        vz = np.array([0.0])
        
        ps = ParticleSet.from_cylindrical(R, phi, z, vR, vT, vz)
        
        np.testing.assert_allclose(ps.x, [0.0], atol=1e-10)
        np.testing.assert_allclose(ps.y, [1.0])
        # vT points in direction of increasing phi, at phi=90 that's -x direction
        np.testing.assert_allclose(ps.vx, [-1.0])
        np.testing.assert_allclose(ps.vy, [0.0], atol=1e-10)

    def test_from_spherical(self):
        """Test creating ParticleSet from spherical coordinates."""
        # Particle at r=1, theta=90deg, phi=0 should be at x=1, y=0, z=0
        r = np.array([1.0])
        theta = np.array([np.pi / 2])
        phi = np.array([0.0])
        vr = np.array([1.0])
        vtheta = np.array([0.0])
        vphi = np.array([0.0])
        
        ps = ParticleSet.from_spherical(r, theta, phi, vr, vtheta, vphi)
        
        np.testing.assert_allclose(ps.x, [1.0])
        np.testing.assert_allclose(ps.y, [0.0], atol=1e-10)
        np.testing.assert_allclose(ps.z, [0.0], atol=1e-10)

    def test_from_spherical_at_pole(self):
        """Test spherical coordinates at z-pole."""
        r = np.array([1.0])
        theta = np.array([0.0])  # At z-axis
        phi = np.array([0.0])
        vr = np.array([1.0])
        vtheta = np.array([0.0])
        vphi = np.array([0.0])
        
        ps = ParticleSet.from_spherical(r, theta, phi, vr, vtheta, vphi)
        
        np.testing.assert_allclose(ps.x, [0.0], atol=1e-10)
        np.testing.assert_allclose(ps.y, [0.0], atol=1e-10)
        np.testing.assert_allclose(ps.z, [1.0])

    def test_position_properties(self):
        """Test x, y, z property access."""
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        velocities = np.zeros((2, 3))
        
        ps = ParticleSet(positions, velocities)
        
        np.testing.assert_array_equal(ps.x, [1.0, 4.0])
        np.testing.assert_array_equal(ps.y, [2.0, 5.0])
        np.testing.assert_array_equal(ps.z, [3.0, 6.0])

    def test_velocity_properties(self):
        """Test vx, vy, vz property access."""
        positions = np.zeros((2, 3))
        velocities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        ps = ParticleSet(positions, velocities)
        
        np.testing.assert_array_equal(ps.vx, [0.1, 0.4])
        np.testing.assert_array_equal(ps.vy, [0.2, 0.5])
        np.testing.assert_array_equal(ps.vz, [0.3, 0.6])

    def test_derived_coordinates_R(self):
        """Test cylindrical radius R."""
        positions = np.array([[3.0, 4.0, 5.0]])  # R = 5
        velocities = np.zeros((1, 3))
        
        ps = ParticleSet(positions, velocities)
        
        np.testing.assert_allclose(ps.R, [5.0])

    def test_derived_coordinates_r(self):
        """Test spherical radius r."""
        positions = np.array([[3.0, 4.0, 12.0]])  # r = 13
        velocities = np.zeros((1, 3))
        
        ps = ParticleSet(positions, velocities)
        
        np.testing.assert_allclose(ps.r, [13.0])

    def test_derived_coordinates_phi(self):
        """Test azimuthal angle phi."""
        positions = np.array([[1.0, 1.0, 0.0]])  # phi = 45 degrees
        velocities = np.zeros((1, 3))
        
        ps = ParticleSet(positions, velocities)
        
        np.testing.assert_allclose(ps.phi, [np.pi / 4])

    def test_derived_coordinates_theta(self):
        """Test polar angle theta."""
        positions = np.array([[0.0, 0.0, 1.0]])  # theta = 0 (at z-axis)
        velocities = np.zeros((1, 3))
        
        ps = ParticleSet(positions, velocities)
        
        np.testing.assert_allclose(ps.theta, [0.0])

    def test_derived_velocities_vR_vT(self):
        """Test cylindrical velocities vR and vT."""
        # Particle at phi=0 with vx=1, vy=2 should have vR=1, vT=2
        positions = np.array([[1.0, 0.0, 0.0]])
        velocities = np.array([[1.0, 2.0, 0.0]])
        
        ps = ParticleSet(positions, velocities)
        
        np.testing.assert_allclose(ps.vR, [1.0])
        np.testing.assert_allclose(ps.vT, [2.0])

    def test_center_of_mass(self):
        """Test center of mass calculation."""
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        velocities = np.zeros((2, 3))
        masses = np.array([1.0, 1.0])  # Equal masses
        
        ps = ParticleSet(positions, velocities, masses)
        com = ps.center_of_mass()
        
        np.testing.assert_allclose(com, [1.0, 0.0, 0.0])

    def test_center_of_mass_weighted(self):
        """Test center of mass with different masses."""
        positions = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        velocities = np.zeros((2, 3))
        masses = np.array([3.0, 1.0])  # 3:1 mass ratio
        
        ps = ParticleSet(positions, velocities, masses)
        com = ps.center_of_mass()
        
        # COM = (3*0 + 1*4) / 4 = 1.0
        np.testing.assert_allclose(com, [1.0, 0.0, 0.0])

    def test_center_of_mass_velocity(self):
        """Test center of mass velocity calculation."""
        positions = np.zeros((2, 3))
        velocities = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        masses = np.array([1.0, 1.0])
        
        ps = ParticleSet(positions, velocities, masses)
        com_vel = ps.center_of_mass_velocity()
        
        np.testing.assert_allclose(com_vel, [2.0, 0.0, 0.0])

    def test_recenter(self):
        """Test recentering to center of mass."""
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        velocities = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        masses = np.array([1.0, 1.0])
        
        ps = ParticleSet(positions, velocities, masses)
        recentered = ps.recenter()
        
        np.testing.assert_allclose(recentered.center_of_mass(), [0.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(recentered.center_of_mass_velocity(), [0.0, 0.0, 0.0], atol=1e-10)

    def test_shift(self):
        """Test shifting positions and velocities."""
        positions = np.array([[1.0, 2.0, 3.0]])
        velocities = np.array([[0.1, 0.2, 0.3]])
        
        ps = ParticleSet(positions, velocities)
        shifted = ps.shift(dr=[1.0, 0.0, 0.0], dv=[0.0, 1.0, 0.0])
        
        np.testing.assert_allclose(shifted.x, [2.0])
        np.testing.assert_allclose(shifted.vy, [1.2])
        # Original unchanged
        np.testing.assert_allclose(ps.x, [1.0])

    def test_rotate(self):
        """Test rotation transformation."""
        positions = np.array([[1.0, 0.0, 0.0]])
        velocities = np.array([[1.0, 0.0, 0.0]])
        
        # 90 degree rotation around z-axis
        theta = np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        ps = ParticleSet(positions, velocities)
        rotated = ps.rotate(R)
        
        np.testing.assert_allclose(rotated.x, [0.0], atol=1e-10)
        np.testing.assert_allclose(rotated.y, [1.0])
        np.testing.assert_allclose(rotated.vx, [0.0], atol=1e-10)
        np.testing.assert_allclose(rotated.vy, [1.0])

    def test_copy(self):
        """Test deep copy."""
        positions = np.array([[1.0, 2.0, 3.0]])
        velocities = np.array([[0.1, 0.2, 0.3]])
        
        ps = ParticleSet(positions, velocities)
        copied = ps.copy()
        
        # Modify original
        ps._positions[0, 0] = 999.0
        
        # Copy should be unchanged
        np.testing.assert_allclose(copied.x, [1.0])

    def test_getitem_index(self):
        """Test particle selection by index."""
        positions = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        velocities = np.zeros((3, 3))
        
        ps = ParticleSet(positions, velocities)
        subset = ps[1]
        
        assert subset.n_particles == 1
        np.testing.assert_allclose(subset.x, [2.0])

    def test_getitem_slice(self):
        """Test particle selection by slice."""
        positions = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        velocities = np.zeros((3, 3))
        
        ps = ParticleSet(positions, velocities)
        subset = ps[:2]
        
        assert subset.n_particles == 2
        np.testing.assert_allclose(subset.x, [1.0, 2.0])

    def test_getitem_mask(self):
        """Test particle selection by boolean mask."""
        positions = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        velocities = np.zeros((3, 3))
        
        ps = ParticleSet(positions, velocities)
        mask = np.array([True, False, True])
        subset = ps[mask]
        
        assert subset.n_particles == 2
        np.testing.assert_allclose(subset.x, [1.0, 3.0])

    def test_len(self):
        """Test __len__ method."""
        positions = np.zeros((5, 3))
        velocities = np.zeros((5, 3))
        
        ps = ParticleSet(positions, velocities)
        
        assert len(ps) == 5

    def test_repr(self):
        """Test string representation."""
        positions = np.zeros((10, 3))
        velocities = np.zeros((10, 3))
        
        ps = ParticleSet(positions, velocities)
        repr_str = repr(ps)
        
        assert "ParticleSet" in repr_str
        assert "n_particles=10" in repr_str

    def test_get_component(self):
        """Test getting particles by component."""
        positions = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        velocities = np.zeros((3, 3))
        components = np.array(['dm', 'dm', 'stellar'])
        
        ps = ParticleSet(positions, velocities, components=components)
        dm = ps.get_component('dm')
        
        assert dm.n_particles == 2
        np.testing.assert_allclose(dm.x, [1.0, 2.0])

    def test_get_component_not_found(self):
        """Test that getting non-existent component raises error."""
        positions = np.zeros((2, 3))
        velocities = np.zeros((2, 3))
        
        ps = ParticleSet(positions, velocities)
        
        with pytest.raises(ValueError, match="not found"):
            ps.get_component('nonexistent')

    def test_get_component_mask(self):
        """Test getting component mask."""
        positions = np.zeros((3, 3))
        velocities = np.zeros((3, 3))
        components = np.array(['dm', 'stellar', 'dm'])
        
        ps = ParticleSet(positions, velocities, components=components)
        mask = ps.get_component_mask('dm')
        
        np.testing.assert_array_equal(mask, [True, False, True])

    def test_has_component(self):
        """Test checking if component exists."""
        positions = np.zeros((2, 3))
        velocities = np.zeros((2, 3))
        components = np.array(['dm', 'stellar'])
        
        ps = ParticleSet(positions, velocities, components=components)
        
        assert ps.has_component('dm')
        assert ps.has_component('stellar')
        assert not ps.has_component('gas')

    def test_concatenate(self):
        """Test concatenating multiple ParticleSets."""
        ps1 = ParticleSet(
            positions=np.array([[1.0, 0.0, 0.0]]),
            velocities=np.zeros((1, 3)),
            masses=np.array([1.0])
        )
        ps2 = ParticleSet(
            positions=np.array([[2.0, 0.0, 0.0]]),
            velocities=np.zeros((1, 3)),
            masses=np.array([2.0])
        )
        
        combined = ParticleSet.concatenate([ps1, ps2])
        
        assert combined.n_particles == 2
        np.testing.assert_allclose(combined.x, [1.0, 2.0])
        np.testing.assert_allclose(combined.masses, [1.0, 2.0])

    def test_concatenate_with_labels(self):
        """Test concatenating with component labels."""
        ps1 = ParticleSet(
            positions=np.array([[1.0, 0.0, 0.0]]),
            velocities=np.zeros((1, 3)),
        )
        ps2 = ParticleSet(
            positions=np.array([[2.0, 0.0, 0.0]]),
            velocities=np.zeros((1, 3)),
        )
        
        combined = ParticleSet.concatenate([ps1, ps2], component_labels=['dm', 'stellar'])
        
        assert set(combined.component_names) == {'dm', 'stellar'}
        dm = combined.get_component('dm')
        assert dm.n_particles == 1
        np.testing.assert_allclose(dm.x, [1.0])

    def test_to_numpy(self):
        """Test to_numpy method."""
        positions = np.array([[1.0, 2.0, 3.0]])
        velocities = np.array([[0.1, 0.2, 0.3]])
        masses = np.array([5.0])
        
        ps = ParticleSet(positions, velocities, masses)
        pos, vel, m = ps.to_numpy()
        
        np.testing.assert_array_equal(pos, positions)
        np.testing.assert_array_equal(vel, velocities)
        np.testing.assert_array_equal(m, masses)

    def test_to_dict(self):
        """Test to_dict method."""
        positions = np.array([[1.0, 2.0, 3.0]])
        velocities = np.array([[0.1, 0.2, 0.3]])
        
        ps = ParticleSet(positions, velocities)
        data = ps.to_dict()
        
        assert 'positions' in data
        assert 'velocities' in data
        assert 'masses' in data
        np.testing.assert_array_equal(data['positions'], positions)

    def test_save_load_npy(self):
        """Test saving and loading with .npy format."""
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        velocities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        masses = np.array([1.0, 2.0])
        
        ps = ParticleSet(positions, velocities, masses)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            filename = f.name
        
        try:
            ps.save(filename, format='npy')
            loaded = ParticleSet.load(filename)
            
            np.testing.assert_array_equal(loaded.positions, positions)
            np.testing.assert_array_equal(loaded.velocities, velocities)
            np.testing.assert_array_equal(loaded.masses, masses)
        finally:
            os.unlink(filename)

    def test_save_load_npz(self):
        """Test saving and loading with .npz format."""
        positions = np.array([[1.0, 2.0, 3.0]])
        velocities = np.array([[0.1, 0.2, 0.3]])
        
        ps = ParticleSet(positions, velocities)
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            filename = f.name
        
        try:
            ps.save(filename, format='npz')
            loaded = ParticleSet.load(filename)
            
            np.testing.assert_array_equal(loaded.positions, positions)
        finally:
            os.unlink(filename)

    # =========================================================================
    # Edge case tests for full coverage
    # =========================================================================

    def test_init_wrong_mass_length(self):
        """Test that wrong mass array length raises ValueError."""
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        velocities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        masses = np.array([1.0])  # Wrong length
        
        with pytest.raises(ValueError, match="masses must have length"):
            ParticleSet(positions, velocities, masses)

    def test_init_wrong_ids_length(self):
        """Test that wrong ids array length raises ValueError."""
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        velocities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        ids = np.array([0])  # Wrong length
        
        with pytest.raises(ValueError, match="ids must have length"):
            ParticleSet(positions, velocities, ids=ids)

    def test_init_wrong_components_length(self):
        """Test that wrong components array length raises ValueError."""
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        velocities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        components = np.array(['dm'])  # Wrong length
        
        with pytest.raises(ValueError, match="components must have length"):
            ParticleSet(positions, velocities, components=components)

    def test_from_phase_space_wrong_shape(self):
        """Test that from_phase_space with wrong shape raises ValueError."""
        phase_space = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])  # 5 columns, not 6
        
        with pytest.raises(ValueError, match="phase_space must have shape"):
            ParticleSet.from_phase_space(phase_space)

    def test_concatenate_empty_list(self):
        """Test that concatenating empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot concatenate empty"):
            ParticleSet.concatenate([])

    def test_concatenate_mismatched_labels(self):
        """Test that mismatched component_labels length raises ValueError."""
        ps1 = ParticleSet(
            positions=np.array([[1.0, 0.0, 0.0]]),
            velocities=np.zeros((1, 3)),
        )
        ps2 = ParticleSet(
            positions=np.array([[2.0, 0.0, 0.0]]),
            velocities=np.zeros((1, 3)),
        )
        
        with pytest.raises(ValueError, match="component_labels must have same length"):
            ParticleSet.concatenate([ps1, ps2], component_labels=['dm'])  # Only 1 label for 2 sets

    def test_shift_wrong_dr_length(self):
        """Test that shift with wrong dr length raises ValueError."""
        positions = np.array([[1.0, 2.0, 3.0]])
        velocities = np.array([[0.1, 0.2, 0.3]])
        ps = ParticleSet(positions, velocities)
        
        with pytest.raises(ValueError, match="dr must have length 3"):
            ps.shift(dr=[1.0, 2.0])  # Only 2 elements

    def test_shift_wrong_dv_length(self):
        """Test that shift with wrong dv length raises ValueError."""
        positions = np.array([[1.0, 2.0, 3.0]])
        velocities = np.array([[0.1, 0.2, 0.3]])
        ps = ParticleSet(positions, velocities)
        
        with pytest.raises(ValueError, match="dv must have length 3"):
            ps.shift(dv=[1.0, 2.0, 3.0, 4.0])  # 4 elements

    def test_rotate_wrong_matrix_shape(self):
        """Test that rotate with wrong matrix shape raises ValueError."""
        positions = np.array([[1.0, 2.0, 3.0]])
        velocities = np.array([[0.1, 0.2, 0.3]])
        ps = ParticleSet(positions, velocities)
        
        with pytest.raises(ValueError, match="rotation_matrix must have shape"):
            ps.rotate(np.eye(2))  # 2x2 instead of 3x3

    def test_center_of_mass_zero_mass(self):
        """Test center of mass with zero total mass returns zero."""
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        velocities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        masses = np.array([0.0, 0.0])  # Zero masses
        
        ps = ParticleSet(positions, velocities, masses)
        com = ps.center_of_mass()
        
        np.testing.assert_array_equal(com, [0.0, 0.0, 0.0])

    def test_center_of_mass_velocity_zero_mass(self):
        """Test center of mass velocity with zero total mass returns zero."""
        positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        velocities = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        masses = np.array([0.0, 0.0])  # Zero masses
        
        ps = ParticleSet(positions, velocities, masses)
        com_vel = ps.center_of_mass_velocity()
        
        np.testing.assert_array_equal(com_vel, [0.0, 0.0, 0.0])

    def test_save_unknown_format(self):
        """Test that saving with unknown format raises ValueError."""
        positions = np.array([[1.0, 2.0, 3.0]])
        velocities = np.array([[0.1, 0.2, 0.3]])
        ps = ParticleSet(positions, velocities)
        
        with pytest.raises(ValueError, match="Unknown format"):
            ps.save("test.dat", format='unknown')


class TestParticleHistory:
    """Tests for ParticleHistory class."""

    def _make_particles(self, x_offset=0.0, vx_offset=0.0):
        """Helper to create a simple ParticleSet."""
        positions = np.array([[1.0 + x_offset, 0.0, 0.0], [2.0 + x_offset, 0.0, 0.0]])
        velocities = np.array([[0.1 + vx_offset, 0.0, 0.0], [0.2 + vx_offset, 0.0, 0.0]])
        return ParticleSet(positions, velocities)

    def test_init_empty(self):
        """Test empty initialization."""
        history = ParticleHistory()
        
        assert history.n_snapshots == 0
        assert len(history) == 0

    def test_init_with_snapshots(self):
        """Test initialization with snapshots."""
        ps1 = self._make_particles()
        ps2 = self._make_particles(x_offset=1.0)
        snapshots = [Snapshot(t=0.0, particles=ps1), Snapshot(t=1.0, particles=ps2)]
        
        history = ParticleHistory(snapshots)
        
        assert history.n_snapshots == 2
        np.testing.assert_array_equal(history.times, [0.0, 1.0])

    def test_init_sorts_by_time(self):
        """Test that initialization sorts snapshots by time."""
        ps1 = self._make_particles()
        ps2 = self._make_particles()
        snapshots = [Snapshot(t=1.0, particles=ps2), Snapshot(t=0.0, particles=ps1)]
        
        history = ParticleHistory(snapshots)
        
        np.testing.assert_array_equal(history.times, [0.0, 1.0])

    def test_add_snapshot(self):
        """Test adding snapshots."""
        history = ParticleHistory()
        ps = self._make_particles()
        
        history.add_snapshot(0.0, ps)
        history.add_snapshot(1.0, ps)
        
        assert history.n_snapshots == 2
        np.testing.assert_array_equal(history.times, [0.0, 1.0])

    def test_add_snapshot_out_of_order(self):
        """Test adding snapshots out of order maintains sort."""
        history = ParticleHistory()
        ps = self._make_particles()
        
        history.add_snapshot(1.0, ps)
        history.add_snapshot(0.0, ps)
        history.add_snapshot(0.5, ps)
        
        np.testing.assert_array_equal(history.times, [0.0, 0.5, 1.0])

    def test_at_time_exact(self):
        """Test retrieving snapshot at exact saved time."""
        ps0 = self._make_particles(x_offset=0.0)
        ps1 = self._make_particles(x_offset=10.0)
        
        history = ParticleHistory()
        history.add_snapshot(0.0, ps0)
        history.add_snapshot(1.0, ps1)
        
        result = history.at_time(0.0)
        np.testing.assert_allclose(result.x, [1.0, 2.0])
        
        result = history.at_time(1.0)
        np.testing.assert_allclose(result.x, [11.0, 12.0])

    def test_at_time_nearest(self):
        """Test retrieving nearest snapshot."""
        ps0 = self._make_particles(x_offset=0.0)
        ps1 = self._make_particles(x_offset=10.0)
        
        history = ParticleHistory()
        history.add_snapshot(0.0, ps0)
        history.add_snapshot(1.0, ps1)
        
        # Closer to t=0
        result = history.at_time(0.3, interpolate=False)
        np.testing.assert_allclose(result.x, [1.0, 2.0])
        
        # Closer to t=1
        result = history.at_time(0.7, interpolate=False)
        np.testing.assert_allclose(result.x, [11.0, 12.0])

    def test_at_time_interpolate(self):
        """Test interpolated snapshot retrieval."""
        ps0 = self._make_particles(x_offset=0.0)  # x = [1, 2]
        ps1 = self._make_particles(x_offset=10.0)  # x = [11, 12]
        
        history = ParticleHistory()
        history.add_snapshot(0.0, ps0)
        history.add_snapshot(1.0, ps1)
        
        # At t=0.5, should be halfway
        result = history.at_time(0.5, interpolate=True)
        np.testing.assert_allclose(result.x, [6.0, 7.0])
        
        # At t=0.25, should be 25% of the way
        result = history.at_time(0.25, interpolate=True)
        np.testing.assert_allclose(result.x, [3.5, 4.5])

    def test_at_time_before_first(self):
        """Test requesting time before first snapshot."""
        ps = self._make_particles()
        
        history = ParticleHistory()
        history.add_snapshot(1.0, ps)
        
        result = history.at_time(0.0)
        np.testing.assert_allclose(result.x, ps.x)

    def test_at_time_after_last(self):
        """Test requesting time after last snapshot."""
        ps = self._make_particles()
        
        history = ParticleHistory()
        history.add_snapshot(0.0, ps)
        
        result = history.at_time(100.0)
        np.testing.assert_allclose(result.x, ps.x)

    def test_at_time_empty_raises(self):
        """Test that at_time raises error when empty."""
        history = ParticleHistory()
        
        with pytest.raises(ValueError, match="No snapshots"):
            history.at_time(0.0)

    def test_particle_slicing(self):
        """Test slicing history by particle index."""
        ps = self._make_particles()  # 2 particles
        
        history = ParticleHistory()
        history.add_snapshot(0.0, ps)
        history.add_snapshot(1.0, ps)
        
        sliced = history[0]  # Get first particle only
        
        assert sliced.n_snapshots == 2
        particles = sliced.at_time(0.0)
        assert particles.n_particles == 1

    def test_particle_slicing_by_mask(self):
        """Test slicing history by boolean mask."""
        ps = self._make_particles()  # 2 particles
        
        history = ParticleHistory()
        history.add_snapshot(0.0, ps)
        
        mask = np.array([False, True])
        sliced = history[mask]
        
        particles = sliced.at_time(0.0)
        assert particles.n_particles == 1
        np.testing.assert_allclose(particles.x, [2.0])

    def test_get_particle_trajectory(self):
        """Test getting trajectory of single particle."""
        ps0 = self._make_particles(x_offset=0.0)
        ps1 = self._make_particles(x_offset=1.0)
        ps2 = self._make_particles(x_offset=2.0)
        
        history = ParticleHistory()
        history.add_snapshot(0.0, ps0)
        history.add_snapshot(1.0, ps1)
        history.add_snapshot(2.0, ps2)
        
        times, positions, velocities = history.get_particle_trajectory(0)
        
        np.testing.assert_array_equal(times, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(positions[:, 0], [1.0, 2.0, 3.0])  # x positions

    def test_get_center_of_mass_trajectory(self):
        """Test getting center of mass trajectory."""
        ps0 = ParticleSet(
            positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            velocities=np.zeros((2, 3)),
            masses=np.array([1.0, 1.0])
        )
        ps1 = ParticleSet(
            positions=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
            velocities=np.zeros((2, 3)),
            masses=np.array([1.0, 1.0])
        )
        
        history = ParticleHistory()
        history.add_snapshot(0.0, ps0)
        history.add_snapshot(1.0, ps1)
        
        times, positions, velocities = history.get_center_of_mass_trajectory()
        
        np.testing.assert_array_equal(times, [0.0, 1.0])
        np.testing.assert_allclose(positions[:, 0], [1.0, 2.0])  # COM x positions

    def test_get_center_of_mass_trajectory_with_component(self):
        """Test getting center of mass trajectory for specific component."""
        # Create particles with two components
        ps0 = ParticleSet(
            positions=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            velocities=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            masses=np.array([1.0, 1.0, 1.0]),
            components=np.array(['dm', 'dm', 'stellar'])
        )
        ps1 = ParticleSet(
            positions=np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
            velocities=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            masses=np.array([1.0, 1.0, 1.0]),
            components=np.array(['dm', 'dm', 'stellar'])
        )
        
        history = ParticleHistory()
        history.add_snapshot(0.0, ps0)
        history.add_snapshot(1.0, ps1)
        
        # Get COM trajectory for 'dm' component only
        times, positions, velocities = history.get_center_of_mass_trajectory(component='dm')
        
        np.testing.assert_array_equal(times, [0.0, 1.0])
        # DM particles: at t=0, x=[0, 4], COM=2; at t=1, x=[0, 8], COM=4
        np.testing.assert_allclose(positions[:, 0], [2.0, 4.0])

    def test_save_load(self):
        """Test saving and loading ParticleHistory."""
        ps0 = self._make_particles(x_offset=0.0)
        ps1 = self._make_particles(x_offset=5.0)
        
        history = ParticleHistory()
        history.add_snapshot(0.0, ps0)
        history.add_snapshot(1.0, ps1)
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            filename = f.name
        
        try:
            history.save(filename)
            loaded = ParticleHistory.load(filename)
            
            assert loaded.n_snapshots == 2
            np.testing.assert_array_equal(loaded.times, [0.0, 1.0])
            
            p0 = loaded.at_time(0.0)
            np.testing.assert_allclose(p0.x, [1.0, 2.0])
            
            p1 = loaded.at_time(1.0)
            np.testing.assert_allclose(p1.x, [6.0, 7.0])
        finally:
            os.unlink(filename)

    def test_repr(self):
        """Test string representation."""
        ps = self._make_particles()
        
        history = ParticleHistory()
        history.add_snapshot(0.0, ps)
        history.add_snapshot(10.0, ps)
        
        repr_str = repr(history)
        
        assert "ParticleHistory" in repr_str
        assert "n_particles=2" in repr_str
        assert "n_snapshots=2" in repr_str

    def test_repr_empty(self):
        """Test string representation when empty."""
        history = ParticleHistory()
        repr_str = repr(history)
        
        assert "empty" in repr_str
