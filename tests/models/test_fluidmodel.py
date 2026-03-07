"""Unit tests for Fluid Model."""

import pytest
from unittest.mock import Mock

from perf_pressure_traverse.models.fluidmodel import (
    FluidModel,
    FluidModelFactory,
    FluidType
)
from perf_pressure_traverse.models.fluid import FluidProperties


class TestFluidType(Enum):
    """Alias for FluidType from FluidModel."""
    ...


class TestFluidModel:
    """Tests for FluidModel class."""
    
    def test_initialization(self):
        """Test basic FluidModel initialization."""
        model = FluidModel(
            fluid_type=FluidType.OIL,
            properties=FluidProperties(reservoir_pressure_psia=2000.0),
        )
        
        assert model.fluid_type == FluidType.OIL
        assert model.properties == fluid_properties
        assert model.is_crude_oil is False
    
    def test_initialization_with_composition(self):
        """Test initialization with multiphase composition."""
        model = FluidModel(
            fluid_type=FluidType.OIL_GAS,
            properties=FluidProperties(reservoir_pressure_psia=2500.0),
            composition={'oil': 0.7, 'gas': 0.3},
        )
        
        assert model.composition is not None
        assert model.composition['oil'] == 0.7
    
    def test_from_properties(self):
        """Test creating fluid model from reservoir properties."""
        model = FluidModel.from_properties(
            fluid_type=FluidType.GAS,
            reservoir_pressure_psia=3000.0,
            reservoir_temperature_f=180.0,
            molecular_weight=16.0,
            compressibility_factor_z=0.9,
        )
        
        assert model.fluid_type == FluidType.GAS
        assert model.properties.reservoir_pressure_psia == 3000.0
        assert model.properties.reservoir_temperature_f == 180.0
        assert model.molecular_weight == 16.0
        assert model.compressibility_factor_z == 0.9
        
        # Should have Z-factor in properties
        assert model.properties.formation_volume_factor_gas_res > 0
    
    def test_from_surface_conditions(self):
        """Test creating fluid model from surface conditions."""
        model = FluidModel.from_surface_conditions(
            fluid_type=FluidType.OIL,
            surface_pressure_psia=14.7,
            surface_temperature_f=80.0,
            specific_gravity_oil=0.85,
        )
        
        assert model.fluid_type == FluidType.OIL
        assert model.properties.reservoir_pressure_psia == 14.7
        assert model.properties.reservoir_temperature_f == 80.0
    
    def test_get_gas_density(self):
        """Test gas density calculation."""
        model = FluidModel(
            fluid_type=FluidType.GAS,
            properties=FluidProperties(reservoir_pressure_psia=3000.0),
            molecular_weight=16.0,
            compressibility_factor_z=0.9,
        )
        
        # With Z-factor
        gas_density = model.get_gas_density_lb_ft3()
        assert gas_density > 0
        
        # Without Z-factor, should use properties
        model2 = FluidModel(
            fluid_type=FluidType.GAS,
            properties=FluidProperties(
                reservoir_pressure_psia=3000.0,
                gas_density_lb_ft3=0.8,
            ),
        )
        assert model2.get_gas_density_lb_ft3() == 0.8
    
    def test_get_oil_density(self):
        """Test oil density retrieval."""
        model = FluidModel(
            fluid_type=FluidType.OIL,
            properties=FluidProperties(reservoir_pressure_psia=2000.0),
            specific_gravity_oil=0.85,
        )
        
        # Should return property value
        oil_density = model.get_oil_density_lb_ft3()
        assert oil_density > 0
    
    def test_get_water_density(self):
        """Test water density retrieval."""
        model = FluidModel(
            fluid_type=FluidType.WATER,
            properties=FluidProperties(reservoir_pressure_psia=2000.0),
        )
        
        water_density = model.get_water_density_lb_ft3()
        assert water_density == 62.4  # Standard water density
    
    def test_get_gas_viscosity(self):
        """Test gas viscosity calculation."""
        model = FluidModel(
            fluid_type=FluidType.GAS,
            properties=FluidProperties(reservoir_pressure_psia=2500.0),
            molecular_weight=16.0,
            surface_tension_nt_m=0.07,
        )
        
        viscosity = model.get_gas_viscosity_cP()
        assert viscosity > 0
        
        # With viscosity correction
        model2 = FluidModel(
            fluid_type=FluidType.GAS,
            properties=FluidProperties(reservoir_pressure_psia=2500.0),
            viscosity_corr_factor=1.0,
        )
        viscosity2 = model2.get_gas_viscosity_cP()
        assert viscosity2 == viscosity
    
    def test_get_oil_viscosity(self):
        """Test oil viscosity retrieval."""
        model = FluidModel(
            fluid_type=FluidType.OIL,
            properties=FluidProperties(reservoir_pressure_psia=2000.0, oil_viscosity_cP=1.5),
        )
        
        viscosity = model.get_oil_viscosity_cP()
        assert viscosity == 1.5
    
    def test_get_formation_volume_factor(self):
        """Test formation volume factor retrieval."""
        model = FluidModel(
            fluid_type=FluidType.OIL,
            properties=FluidProperties(reservoir_pressure_psia=2000.0),
        )
        
        # Should use oil FVF
        fvf = model.get_formation_volume_factor()
        assert fvf >= 1.0
    
    def test_get_surface_viscosity(self):
        """Test surface viscosity retrieval."""
        model = FluidModel(
            fluid_type=FluidType.GAS,
            properties=FluidProperties(gas_viscosity_cP=0.02),
        )
        
        viscosity = model.get_surface_viscosity_cP()
        assert viscosity == 0.02
    
    def test_calculate_mixture_density(self):
        """Test mixture density calculation."""
        model = FluidModel(
            fluid_type=FluidType.OIL_GAS,
            properties=FluidProperties(reservoir_pressure_psia=2000.0),
        )
        
        # With quality
        density = model.calculate_mixture_density_lb_ft3(quality=0.25)
        assert density > 0
    
    def test_update_pvt_properties(self):
        """Test updating PVT properties."""
        model = FluidModel(
            fluid_type=FluidType.OIL,
            properties=FluidProperties(reservoir_pressure_psia=2000.0),
        )
        
        model.update_pvt_properties(1500.0, 100.0)
        
        assert model.properties.reservoir_pressure_psia == 1500.0
        assert model.properties.reservoir_temperature_f == 100.0
    
    def test_validate_valid_model(self):
        """Test validation of valid fluid model."""
        model = FluidModel(
            fluid_type=FluidType.OIL,
            properties=FluidProperties(reservoir_pressure_psia=2000.0),
        )
        
        is_valid = model.validate()
        assert is_valid is True
    
    def test_validate_invalid_pressure(self):
        """Test validation fails with invalid pressure."""
        model = FluidModel(
            fluid_type=FluidType.OIL,
            properties=FluidProperties(reservoir_pressure_psia=0.0),
        )
        
        is_valid = model.validate()
        assert is_valid is False
    
    def test_validate_invalid_temperature(self):
        """Test validation fails with negative temperature."""
        model = FluidModel(
            fluid_type=FluidType.OIL,
            properties=FluidProperties(reservoir_pressure_psia=2000.0, reservoir_temperature_f=-100.0),
        )
        
        is_valid = model.validate()
        assert is_valid is False
    
    def test_validate_unusual_temperature(self):
        """Test validation with unusually high temperature."""
        model = FluidModel(
            fluid_type=FluidType.OIL,
            properties=FluidProperties(reservoir_pressure_psia=2000.0, reservoir_temperature_f=350.0),
        )
        
        is_valid = model.validate()
        assert is_valid is True  # Doesn't fail
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        model = FluidModel(
            fluid_type=FluidType.OIL,
            properties=FluidProperties(reservoir_pressure_psia=2000.0),
        )
        
        result = model.to_dict()
        
        assert 'fluid_type' in result
        assert 'reservoir_pressure_psia' in result
        assert 'reservoir_temperature_f' in result
        assert result['fluid_type'] == 'oil'
    
    def test_from_dict(self):
        """Test creating model from dictionary."""
        data = {
            'fluid_type': 'gas',
            'reservoir_pressure_psia': 3000.0,
            'reservoir_temperature_f': 180.0,
            'molecular_weight': 16.0,
            'compressibility_factor_z': 0.9,
        }
        
        model = FluidModel.from_dict(data)
        
        assert model.fluid_type == FluidType.GAS
        assert model.properties.reservoir_pressure_psia == 3000.0
        assert model.molecular_weight == 16.0


class TestFluidModelFactory:
    """Tests for FluidModelFactory class."""
    
    def test_create_regular_oil(self):
        """Test creating regular crude oil model."""
        model = FluidModelFactory.create_regular_oil(reservoir_pressure_psia=2000.0)
        
        assert model.fluid_type == FluidType.OIL_GAS
        assert model.is_crude_oil is True
        assert model.specific_gravity_oil == 0.85
    
    def test_create_gas(self):
        """Test creating gas model."""
        model = FluidModelFactory.create_gas(reservoir_pressure_psia=2500.0)
        
        assert model.fluid_type == FluidType.GAS
        assert model.compressibility_factor_z == 0.9
    
    def test_create_condensate(self):
        """Test creating condensate model."""
        model = FluidModelFactory.create_condensate(reservoir_pressure_psia=3000.0)
        
        assert model.fluid_type == FluidType.GAS
        assert model.is_condensate is True
    
    def test_create_wellstream(self):
        """Test creating wellstream model."""
        model = FluidModelFactory.create_wellstream(flow_rate_oil_gpd=1000.0)
        
        assert model.fluid_type == FluidType.OIL_GAS
        assert model.is_crude_oil is True
        # Should have reasonable properties based on flow rate
        assert model.properties.reservoir_pressure_psia > 0