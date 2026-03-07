"""Tests for ParameterValidator utility class."""
import pytest

from perf_pressure_traverse.models.fluid import FluidProperties
from perf_pressure_traverse.models.well import WellGeometry, SectionGeometry
from perf_pressure_traverse.utils.validation import ParameterValidator


class TestParameterValidator:
    """Test cases for ParameterValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a ParameterValidator instance."""
        return ParameterValidator()
    
    def test_fluid_validation_valid(self, validator):
        """Test fluid validation with valid parameters."""
        fluid = FluidProperties(
            oil_gravity=30.0,
            gas_gravity=0.6,
            specific_gravity=0.7,
            water_saturation=0.3
        )
        
        errors = validator.validate_fluid(fluid)
        
        assert len(errors) == 0
    
    def test_fluid_validation_missing_api_gravity(self, validator):
        """Test fluid validation without API gravity."""
        fluid = FluidProperties(
            gas_gravity=0.6,
            specific_gravity=0.7
        )
        
        errors = validator.validate_fluid(fluid)
        
        assert len(errors) > 0
        assert any("API gravity" in str(e).lower() for e in errors)
    
    def test_fluid_validation_invalid_api_gravity(self, validator):
        """Test fluid validation with invalid API gravity."""
        fluid = FluidProperties(oil_gravity=120.0)  # Too high
        errors = validator.validate_fluid(fluid)
        
        assert len(errors) > 0
    
    def test_fluid_water_saturation(self, validator):
        """Test water saturation validation."""
        # Valid
        fluid = FluidProperties(
            oil_gravity=30.0,
            water_saturation=0.4
        )
        errors = validator.validate_fluid(fluid)
        assert len(errors) == 0
        
        # Invalid (negative)
        fluid = FluidProperties(
            oil_gravity=30.0,
            water_saturation=-0.1
        )
        errors = validator.validate_fluid(fluid)
        assert len(errors) > 0
    
    def test_fluid_specific_gravity(self, validator):
        """Test specific gravity validation."""
        # Valid gas
        fluid = FluidProperties(
            oil_gravity=40.0,
            gas_gravity=0.6
        )
        errors = validator.validate_fluid(fluid)
        assert len(errors) == 0
        
        # Valid oil
        fluid = FluidProperties(
            oil_gravity=30.0,
            specific_gravity=0.85
        )
        errors = validator.validate_fluid(fluid)
        assert len(errors) == 0
    
    def test_well_validation_valid(self, validator):
        """Test well geometry validation with valid parameters."""
        sections = [
            SectionGeometry(0.0, 1000.0, 0.0, 0.0),
            SectionGeometry(1000.0, 2000.0, 30.0, 45.0)
        ]
        
        well = WellGeometry(
            well_name="Test Well",
            sections=sections,
            surface_pressure_psia=500.0
        )
        
        errors = validator.validate_well(well)
        
        assert len(errors) == 0
    
    def test_well_validation_no_sections(self, validator):
        """Test well geometry validation without sections."""
        well = WellGeometry(well_name="Test", sections=[])
        
        errors = validator.validate_well(well)
        
        assert len(errors) == 0  # No sections is valid but empty
    
    def test_well_validation_invalid_sections(self, validator):
        """Test well geometry validation with invalid sections."""
        sections = [
            SectionGeometry(1000.0, 0.0, 0.0, 0.0),  # Invalid: start > end
            SectionGeometry(0.0, 1000.0, 120.0, 0.0)  # Invalid: inclination
        ]
        
        well = WellGeometry(
            well_name="Test",
            sections=sections
        )
        
        errors = validator.validate_well(well)
        
        assert len(errors) > 0
    
    def test_well_validation_surface_conditions(self, validator):
        """Test surface condition validation."""
        sections = [
            SectionGeometry(0.0, 1000.0, 0.0, 0.0)
        ]
        
        # Valid surface pressure
        well = WellGeometry(
            well_name="Test",
            sections=sections,
            surface_pressure_psia=500.0
        )
        errors = validator.validate_well(well)
        assert len(errors) == 0
        
        # Invalid surface pressure (negative)
        well = WellGeometry(
            well_name="Test",
            sections=sections,
            surface_pressure_psia=-100.0
        )
        errors = validator.validate_well(well)
        assert len(errors) > 0
    
    def test_validate_parameters_combined(self, validator):
        """Test combined validation."""
        fluid = FluidProperties(
            oil_gravity=30.0,
            specific_gravity=0.7
        )
        
        sections = [
            SectionGeometry(0.0, 1000.0, 0.0, 0.0)
        ]
        
        well = WellGeometry(
            well_name="Test",
            sections=sections
        )
        
        # Combined validation
        errors = validator.validate_parameters(fluid, well)
        
        assert isinstance(errors, list)
