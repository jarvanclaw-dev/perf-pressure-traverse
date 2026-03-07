"""Tests for WellGeometry model."""
import pytest
import numpy as np

from perf_pressure_traverse.models.well import WellGeometry, SectionGeometry, SurfaceCondition


class TestSectionGeometry:
    """Test cases for SectionGeometry class."""
    
    def test_init_with_flat_section(self):
        """Test initializing a flat vertical section."""
        section = SectionGeometry(
            start_depth_ft=0.0,
            end_depth_ft=5000.0,
            inclination_deg=0.0,
            azimuth_deg=0.0
        )
        
        assert section.start_depth_ft == 0.0
        assert section.end_depth_ft == 5000.0
        assert section.inclination_deg == 0.0
        assert section.azimuth_deg == 0.0
        assert section.length_ft == 5000.0
    
    def test_calculate_section_length(self):
        """Test section length calculation with deviation."""
        section = SectionGeometry(
            start_depth_ft=0.0,
            end_depth_ft=10000.0,
            inclination_deg=30.0,
            azimuth_deg=90.0
        )
        
        length = section.length_ft
        
        assert length > 0.0
        assert 10000.0 < length < 11500.0  # ~sqrt(3) correction
    
    def test_calculate_curvature(self):
        """Test curvature calculation."""
        section = SectionGeometry(
            start_depth_ft=0.0,
            end_depth_ft=1000.0,
            inclination_deg=10.0,
            azimuth_deg=45.0
        )
        
        curvature = section.curvature_deg_per_ft
        
        assert curvature > 0.0
        assert curvature < 1.0  # Should be small for 10° over 1000 ft


class TestWellGeometry:
    """Test cases for WellGeometry class."""
    
    @pytest.fixture
    def simple_well(self):
        """Create a simple vertical well."""
        sections = [
            SectionGeometry(
                start_depth_ft=0.0,
                end_depth_ft=1000.0,
                inclination_deg=0.0,
                azimuth_deg=0.0
            ),
            SectionGeometry(
                start_depth_ft=1000.0,
                end_depth_ft=5000.0,
                inclination_deg=0.0,
                azimuth_deg=0.0
            )
        ]
        
        return WellGeometry(
            well_name="Test Well",
            sections=sections,
            surface_pressure_psia=500.0,
            surface_temperature_f=60.0
        )
    
    def test_init_simple_well(self, simple_well):
        """Test initializing a simple well."""
        assert simple_well.well_name == "Test Well"
        assert len(simple_well.sections) == 2
        assert simple_well.surface_pressure_psia == 500.0
    
    def test_total_depth(self, simple_well):
        """Test total well depth calculation."""
        depth = simple_well.total_depth_ft
        
        assert depth == 1000.0  # Last section end depth
    
    def test_calculate_average_inclination(self, simple_well):
        """Test average inclination calculation."""
        avg_inc = simple_well.average_inclination_deg
        
        assert avg_inc == 0.0
    
    def test_get_section_at_depth(self, simple_well):
        """Test getting section at specific depth."""
        section = simple_well.get_section_at_depth(750.0)
        
        assert section is not None
        assert section.start_depth_ft <= 750.0 <= section.end_depth_ft
    
    def test_get_section_at_depth_not_found(self, simple_well):
        """Test getting section when depth is out of range."""
        section = simple_well.get_section_at_depth(1500.0)
        
        assert section is None
    
    def test_section_depth_validation(self):
        """Test section depth validation."""
        # Valid vertical section
        section = SectionGeometry(
            start_depth_ft=0.0,
            end_depth_ft=1000.0,
            inclination_deg=0.0,
            azimuth_deg=0.0
        )
        
        assert section.is_valid()
        
        # Invalid: start > end
        section_invalid = SectionGeometry(
            start_depth_ft=1000.0,
            end_depth_ft=0.0,
            inclination_deg=0.0,
            azimuth_deg=0.0
        )
        
        assert not section_invalid.is_valid()
    
    def test_well_summary(self, simple_well):
        """Test well geometry summary."""
        summary = simple_well.summary()
        
        assert "Test Well" in summary
        assert "Depth" in summary
        assert "Sections" in summary
    
    def test_multiple_section_well(self):
        """Test a well with multiple sections (horizontal well)."""
        sections = [
            SectionGeometry(0.0, 2000.0, 0.0, 0.0),
            SectionGeometry(2000.0, 5000.0, 45.0, 90.0),
            SectionGeometry(5000.0, 8000.0, 90.0, 180.0),
            SectionGeometry(8000.0, 10000.0, 0.0, 0.0)
        ]
        
        well = WellGeometry(
            well_name="Horizontal Test Well",
            sections=sections
        )
        
        # Middle section should be horizontal
        section = well.get_section_at_depth(6500.0)
        assert section.inclination_deg == 90.0


class TestSurfaceCondition:
    """Test cases for SurfaceCondition dataclass."""
    
    def test_surface_condition(self):
        """Test surface condition initialization."""
        condition = SurfaceCondition(
            pressure_psia=500.0,
            temperature_f=60.0,
            choke_orifice=10.0,  # in
            choke_valve=0.4
        )
        
        assert condition.pressure_psia == 500.0
        assert condition.temperature_f == 60.0
        assert condition.choke_orifice == 10.0
