"""
Tests for the satellite module.
"""

import pytest
import numpy as np


class TestPlummerSatellite:
    """Tests for PlummerSatellite class."""

    def test_density_profile(self):
        """Test that sampled density matches analytic profile."""
        pass

    def test_velocity_dispersion(self):
        """Test that velocity dispersion is correct."""
        pass

    def test_equilibrium(self):
        """Test that sampled system is in equilibrium."""
        pass

    def test_half_mass_radius(self):
        """Test half-mass radius calculation."""
        pass

    def test_sample_reproducibility(self):
        """Test that sampling with same seed is reproducible."""
        pass


class TestHernquistSatellite:
    """Tests for HernquistSatellite class."""

    def test_density_profile(self):
        """Test that sampled density matches analytic profile."""
        pass

    def test_cusp(self):
        """Test central density cusp."""
        pass


class TestKingSatellite:
    """Tests for KingSatellite class."""

    def test_truncation_radius(self):
        """Test that no particles are beyond truncation radius."""
        pass

    def test_central_flattening(self):
        """Test core flattening in velocity distribution."""
        pass


class TestGalpyDFSatellite:
    """Tests for GalpyDFSatellite class."""

    def test_galpy_interface(self):
        """Test interface with galpy distribution functions."""
        pass

    def test_equilibrium(self):
        """Test that sampled system is in equilibrium."""
        pass


class TestSatellitePlacement:
    """Tests for satellite placement utilities."""

    def test_place_on_orbit(self):
        """Test placing satellite at given position/velocity."""
        pass

    def test_from_galpy_orbit(self):
        """Test placement from galpy Orbit."""
        pass
