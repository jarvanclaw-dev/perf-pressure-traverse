"""Tests for FluidProperties model."""
import pytest

from perf_pressure_traverse.models.fluid import FluidProperties, FluidType


class TestFluidProperties:
    """Test cases for FluidProperties class."""
    
    @pytest.fixture
    def gas_fluid(self):
        """Create a gas fluid sample."""
        return FluidProperties(
            surface_pressure_psia=500,
            surface_temperature_f=60,
            oil_specific_gravity=0.63,  # Typical gas gravity
            gas_specific_gravity=0.65,
            solution_oil_ratio=20.0,
            gas_solubility=500.0
        )
    
    @pytest.fixture
    def crude_oil_fluid(self):
        """Create crude oil fluid sample."""
        return FluidProperties(
            surface_pressure_psia=500,
            surface_temperature_f=100,
            oil_specific_gravity=0.85,  # Typical oil specific gravity
            gas_oil_ratio=150.0,
            solution_oil_ratio=80.0,
            gas_solubility=300.0,
            gas_specific_gravity=0.6
        )
    
    def test_fluid_initialization(self, gas_fluid):
        """Test fluid object initialization."""
        assert gas_fluid.surface_pressure_psia == 500
        assert gas_fluid.surface_temperature_f == 60
        assert gas_fluid.oil_specific_gravity == 0.63
        assert gas_fluid.gas_specific_gravity == 0.65
        assert gas_fluid.gas_oil_ratio == 20.0
        assert gas_fluid.solution_oil_ratio == 500.0
    
    def test_fluid_properties_validation(self, crude_oil_fluid):
        """Test fluid properties are within valid range."""
        # Specific gravity should be between 0.5 and 1.1
        assert 0.5 < crude_oil_fluid.oil_specific_gravity < 1.1
        assert 0.5 < crude_oil_fluid.gas_specific_gravity < 1.1
    
    def test_fluid_summary_method(self, gas_fluid):
        """Test fluid summary method."""
        summary = gas_fluid.summary()
        
        assert isinstance(summary, str)
        assert "Surface Pressure" in summary or "P" in summary
        assert "Temperature" in summary or "T" in summary
        assert "Oil Specific Gravity" in summary
    
    @pytest.mark.skipif(not hasattr(gas_fluid, 'select_correlation'),
                        reason="select_correlation method not implemented")
    def test_select_correlation_method(self, gas_fluid):
        """Test correlation selection."""
        # Should return a correlation function
        correlation = gas_fluid.select_correlation()
        assert callable(correlation)
    
    @pytest.mark.skipif(not hasattr(gas_fluid, 'calculate_viscosity'),
                        reason="calculate_viscosity method not implemented")
    def test_calculate_viscosity_method(self, gas_fluid):
        """Test viscosity calculation."""
        # Should return a viscosity value
        viscosity = gas_fluid.calculate_viscosity(gas_fluid.surface_pressure_psia, gas_fluid.surface_temperature_f)
        assert isinstance(viscosity, float)
        assert viscosity > 0


class TestFluidType:
    """Test cases for FluidType enum."""
    
    def test_fluid_type_enums(self):
        """Test fluid type values."""
        assert hasattr(FluidType, 'GAS')
        assert hasattr(FluidType, 'CRUDE_OIL')
        assert hasattr(FluidType, 'GAS_CONDENSATE')
        assert hasattr(FluidType, 'OIL')
