"""Unit tests for Pressure Point and Pressure Profile models."""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock

from perf_pressure_traverse.models.pressure import PressurePoint, PressureProfile
from perf_pressure_traverse.models.well import WellGeometry


class TestPressurePoint:
    """Tests for PressurePoint model."""
    
    def test_initialization(self):
        """Test basic PressurePoint initialization."""
        point = PressurePoint(
            measured_depth_ft=1000.0,
            true_vertical_depth_ft=900.0,
            pressure_psi=2000.0,
            temperature_f=140.0,
        )
        
        assert point.measured_depth_ft == 1000.0
        assert point.true_vertical_depth_ft == 900.0
        assert point.pressure_psi == 2000.0
        assert point.temperature_f == 140.0
        assert point.flow_regime == "Unknown"
        assert isinstance(point.time, datetime)
    
    def test_complete_initialization(self):
        """Test PressurePoint with all parameters."""
        point = PressurePoint(
            measured_depth_ft=1500.0,
            true_vertical_depth_ft=1200.0,
            pressure_psi=2500.0,
            temperature_f=180.0,
            liquid_holdup_fraction=0.3,
            mixture_velocity_ft_s=15.0,
            quality=0.25,
            flow_regime="Slug",
            time=datetime(2026, 3, 6, 12, 0, 0),
            iteration_number=10,
            residual=0.001,
        )
        
        assert point.liquid_holdup_fraction == 0.3
        assert point.mixture_velocity_ft_s == 15.0
        assert point.quality == 0.25
        assert point.flow_regime == "Slug"
        assert point.time == datetime(2026, 3, 6, 12, 0, 0)
        assert point.iteration_number == 10
        assert abs(point.residual - 0.001) < 1e-9
    
    def test_from_dict(self):
        """Test creating PressurePoint from dictionary."""
        data = {
            'measured_depth_ft': 2000.0,
            'true_vertical_depth_ft': 1800.0,
            'pressure_psi': 3000.0,
            'temperature_f': 200.0,
            'liquid_holdup_fraction': 0.4,
            'mixture_velocity_ft_s': 20.0,
            'quality': 0.3,
            'flow_regime': 'Annular',
            'time': '2026-03-06T12:00:00',
            'iteration_number': 15,
            'residual': 0.0005,
        }
        
        point = PressurePoint.from_dict(data)
        
        assert point.measured_depth_ft == 2000.0
        assert point.true_vertical_depth_ft == 1800.0
        assert point.pressure_psi == 3000.0
        assert point.temperature_f == 200.0
        assert point.liquid_holdup_fraction == 0.4
        assert point.mixture_velocity_ft_s == 20.0
        assert point.quality == 0.3
        assert point.flow_regime == "Annular"
        assert point.time == datetime(2026, 3, 6, 12, 0, 0)
        assert point.iteration_number == 15
        assert abs(point.residual - 0.0005) < 1e-9
    
    def test_to_dict(self):
        """Test converting PressurePoint to dictionary."""
        point = PressurePoint(
            measured_depth_ft=1000.0,
            true_vertical_depth_ft=900.0,
            pressure_psi=2000.0,
            temperature_f=140.0,
        )
        
        result = point.to_dict()
        
        assert result['measured_depth_ft'] == 1000.0
        assert result['true_vertical_depth_ft'] == 900.0
        assert result['pressure_psi'] == 2000.0
        assert result['temperature_f'] == 140.0
        assert isinstance(result['time'], str)
    
    def test_from_dict_extra_fields(self):
        """Test handling extra fields in dictionary."""
        data = {
            'measured_depth_ft': 1000.0,
            'true_vertical_depth_ft': 900.0,
            'pressure_psi': 2000.0,
            'temperature_f': 140.0,
            'extra_field': 'not_used',  # Should be ignored
        }
        
        point = PressurePoint.from_dict(data)
        assert point.measured_depth_ft == 1000.0
    
    def test_invalid_initialization(self):
        """Test that invalid values raise errors."""
        with pytest.raises(TypeError):
            # PressurePoint requires float values for MD/TVD/P/T
            pass


class TestPressureProfile:
    """Tests for PressureProfile model."""
    
    def test_initialization(self):
        """Test PressureProfile initialization."""
        well_geo = WellGeometry(
            measured_depth_ft=2000.0,
            true_vertical_depth_ft=1800.0,
            pipe_inner_diameter_in=5.0,
        )
        
        profile = PressureProfile(well_geo)
        
        assert profile.well_geometry == well_geo
        assert len(profile.points) == 0
    
    def test_add_single_point(self):
        """Test adding pressure points."""
        well_geo = WellGeometry(
            measured_depth_ft=1000.0,
            true_vertical_depth_ft=950.0,
            pipe_inner_diameter_in=4.5,
        )
        
        profile = PressureProfile(well_geo)
        
        point1 = PressurePoint(1000.0, 950.0, 2000.0, 140.0)
        profile.add_point(point1)
        
        assert len(profile.points) == 1
        assert profile.points[0] == point1
    
    def test_add_multiple_points(self):
        """Test adding multiple pressure points."""
        well_geo = WellGeometry(
            measured_depth_ft=3000.0,
            true_vertical_depth_ft=2700.0,
            pipe_inner_diameter_in=6.0,
        )
        
        profile = PressureProfile(well_geo)
        
        points = [
            PressurePoint(300.0, 270.0, 1000.0, 100.0),
            PressurePoint(600.0, 540.0, 1500.0, 120.0),
            PressurePoint(900.0, 810.0, 2000.0, 130.0),
            PressurePoint(1200.0, 1080.0, 2500.0, 150.0),
        ]
        
        for point in points:
            profile.add_point(point)
        
        assert len(profile.points) == 4
        assert profile.points[0].pressure_psi == 1000.0
        assert profile.points[-1].pressure_psi == 2500.0
    
    def test_get_surface_point(self):
        """Test getting surface point."""
        well_geo = WellGeometry(1000.0, 950.0, 4.5)
        profile = PressureProfile(well_geo)
        
        surface_point = PressurePoint(0.0, 0.0, 1000.0, 70.0)
        bottom_point = PressurePoint(1000.0, 950.0, 800.0, 120.0)
        
        profile.add_point(surface_point)
        profile.add_point(bottom_point)
        
        assert profile.get_surface_point() == surface_point
        assert profile.get_surface_point().pressure_psi == 1000.0
    
    def test_get_bottomhole_point(self):
        """Test getting bottomhole point."""
        well_geo = WellGeometry(1000.0, 950.0, 4.5)
        profile = PressureProfile(well_geo)
        
        surface_point = PressurePoint(0.0, 0.0, 1000.0, 70.0)
        bottom_point = PressurePoint(1000.0, 950.0, 800.0, 120.0)
        
        profile.add_point(surface_point)
        profile.add_point(bottom_point)
        
        assert profile.get_bottomhole_point() == bottom_point
        assert profile.get_bottomhole_point().pressure_psi == 800.0
    
    def test_get_empty_arrays(self):
        """Test getting arrays from empty profile."""
        well_geo = WellGeometry(1000.0, 950.0, 4.5)
        profile = PressureProfile(well_geo)
        
        pressure_array = profile.get_pressure_array()
        temperature_array = profile.get_temperature_array()
        
        assert len(pressure_array) == 0
        assert len(temperature_array) == 0
    
    def test_get_arrays_with_data(self):
        """Test getting arrays with data."""
        well_geo = WellGeometry(1000.0, 950.0, 4.5)
        profile = PressureProfile(well_geo)
        
        points = [
            PressurePoint(0.0, 0.0, 1000.0, 70.0),
            PressurePoint(250.0, 237.5, 800.0, 100.0),
            PressurePoint(500.0, 475.0, 600.0, 110.0),
            PressurePoint(750.0, 712.5, 400.0, 125.0),
            PressurePoint(1000.0, 950.0, 200.0, 140.0),
        ]
        
        for point in points:
            profile.add_point(point)
        
        pressure_array = profile.get_pressure_array()
        temperature_array = profile.get_temperature_array()
        
        assert len(pressure_array) == 5
        assert len(temperature_array) == 5
        assert pressure_array[0] == 1000.0
        assert temperature_array[-1] == 140.0
    
    def test_to_dict(self):
        """Test converting profile to dictionary."""
        well_geo = WellGeometry(1000.0, 950.0, 4.5)
        profile = PressureProfile(well_geo)
        
        profile.add_point(PressurePoint(0.0, 0.0, 1000.0, 70.0))
        profile.add_point(PressurePoint(1000.0, 950.0, 800.0, 120.0))
        
        result = profile.to_dict()
        
        assert 'well_geometry' in result
        assert 'points' in result
        assert len(result['points']) == 2
    
    def test_export_to_csv(self, tmp_path):
        """Test exporting profile to CSV."""
        well_geo = WellGeometry(1000.0, 950.0, 4.5)
        profile = PressureProfile(well_geo)
        
        profile.add_point(PressurePoint(0.0, 0.0, 1000.0, 70.0))
        profile.add_point(PressurePoint(250.0, 237.5, 800.0, 100.0))
        profile.add_point(PressurePoint(500.0, 475.0, 600.0, 110.0))
        
        csv_file = tmp_path / "pressure_profile.csv"
        profile.export_to_csv(str(csv_file))
        
        assert csv_file.exists()
        
        with open(csv_file) as f:
            lines = f.readlines()
            assert len(lines) == 4  # Header + 3 points
    
    def test_len(self):
        """Test length operator."""
        well_geo = WellGeometry(1000.0, 950.0, 4.5)
        profile = PressureProfile(well_geo)
        
        assert len(profile) == 0
        
        profile.add_point(PressurePoint(0.0, 0.0, 1000.0, 70.0))
        profile.add_point(PressurePoint(1000.0, 950.0, 800.0, 120.0))
        
        assert len(profile) == 2
    
    def test_getitem(self):
        """Test getting point by index."""
        well_geo = WellGeometry(1000.0, 950.0, 4.5)
        profile = PressureProfile(well_geo)
        
        profile.add_point(PressurePoint(0.0, 0.0, 1000.0, 70.0))
        profile.add_point(PressurePoint(500.0, 475.0, 600.0, 85.0))
        
        assert profile[0].pressure_psi == 1000.0
        assert profile[1].pressure_psi == 600.0
    
    def test_repr(self):
        """Test string representation."""
        well_geo = WellGeometry(1000.0, 950.0, 4.5)
        profile = PressureProfile(well_geo)
        
        repr_str = repr(profile)
        assert "PressureProfile" in repr_str
        assert "points=0" in repr_str