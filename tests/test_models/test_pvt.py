"""Tests for PVTProperties model."""
import pytest

import numpy as np

from perf_pressure_traverse.models.fluid import FluidProperties
from perf_pressure_traverse.models.pvt_properties import PVTProperties


class TestPVTProperties:
    """Test cases for PVTProperties class."""
    
    def test_init_gas_pvt(self):
        """Test initializing PVT properties for gas."""
        pvt = PVTProperties(
            pressure_psia=3000.0,
            absolute_temperature_rankine=600.0,
            formation_volume_factor=1.2
        )
        
        assert pvt.pressure_psia == 3000.0
        assert pvt.absolute_temperature_rankine == 600.0
        assert pvt.formation_volume_factor == 1.2
        assert pvt.gas_formation_factor == 1.25  # Default
    
    def test_init_oil_pvt(self):
        """Test initializing PVT properties for oil."""
        pvt = PVTProperties(
            fluid_type="CRUDE_OIL",
            pressure_psia=2500.0,
            absolute_temperature_rankine=600.0,
            formation_volume_factor=1.05,
            gas_solubility=300.0
        )
        
        assert pvt.gas_solubility == 300.0
        assert pvt.gas_formation_factor == 1.0
    
    def test_temperature_conversion(self):
        """Test temperature units conversion."""
        temp_f = 60.0
        
        pvt = PVTProperties(
            pressure_psia=3000.0,
            temperature_f=temp_f,
            formation_volume_factor=1.1
        )
        
        # Should convert to Rankine
        expected_rankine = temp_f + 459.67
        assert abs(pvt.true_stock-tank_temperature_rankine - expected_rankine) < 1.0
    
    def test_formation_volume_factor_validation(self):
        """Test formation volume factor validation."""
        # Valid
        pvt = PVTProperties(
            pressure_psia=3000.0,
            absolute_temperature_rankine=600.0,
            formation_volume_factor=1.1
        )
        assert pvt.formation_volume_factor == 1.1
        
        # At boundary
        pvt = PVTProperties(
            pressure_psia=3000.0,
            absolute_temperature_rankine=600.0,
            formation_volume_factor=1.0
        )
        assert pvt.formation_volume_factor == 1.0
    
    @pytest.mark.parametrize("pressure,expected_bvf", [
        (500.0, 1.15),
        (1000.0, 1.12),
        (2000.0, 1.08),
        (3000.0, 1.05),
        (4000.0, 1.02),
    ])
    def test_pressure_bvf_relationship(self, pressure, expected_bvf):
        """Test BVF decreases with increasing pressure."""
        pvt = PVTProperties(
            pressure_psia=pressure,
            absolute_temperature_rankine=600.0
        )
        
        assert pvt.formation_volume_factor == expected_bvf
    
    def test_calculate_compressibility(self):
        """Test isothermal compressibility calculation."""
        pvt = PVTProperties(
            pressure_psia=3000.0,
            absolute_temperature_rankine=600.0,
            formation_volume_factor=1.1
        )
        
        compressibility = pvt.isothermal_compressibility
        
        assert compressibility > 0.0
        assert compressibility < 0.02  # Typical range: 0.5-20 micro-1/psi
    
    def test_boil_up_correction(self):
        """Test boil-up correction for gas reservoirs."""
        fluid = FluidProperties(
            oil_gravity=30.0,
            gas_gravity=0.65,
            specific_gravity=0.7
        )
        
        pvt = PVTProperties(
            pressure_psia=3000.0,
            absolute_temperature_rankine=600.0,
            formation_volume_factor=1.2
        )
        
        bo_correction = pvt.boil_up_correction(fluid)
        
        assert isinstance(bo_correction, float)
        assert bo_correction > 0.0
    
    @pytest.mark.skip("Requires proper fluid properties correlation")
    def test_viscosity_calculation_with_correlation(self):
        """Test viscosity computation with proper correlation."""
        # This test is skipped because the correlation needs to be implemented
        pvt = PVTProperties(
            pressure_psia=3000.0,
            absolute_temperature_rankine=600.0,
            formation_volume_factor=1.1
        )
        
        viscosity = pvt.get_viscosity()
        
        assert isinstance(viscosity, float)
        assert 0.02 < viscosity < 0.5  # Typical gas viscosity in centipoise


class TestPVTPropertiesWithFluid:
    """Integration tests for PVTProperties with FluidProperties."""
    
    def test_pvt_with_gas_fluid(self):
        """Test PVT properties with gas fluid."""
        fluid = FluidProperties(
            oil_gravity=45.0,  # Gas
            gas_gravity=0.6
        )
        
        pvt = PVTProperties(
            pressure_psia=4000.0,
            absolute_temperature_rankine=600.0
        )
        
        # Gas should have higher formation volume factor
        assert pvt.gas_formation_factor > 1.0
        assert pvt.formation_volume_factor > pvt.gas_formation_factor
    
    def test_pvt_with_crude_oil_fluid(self):
        """Test PVT properties with crude oil fluid."""
        fluid = FluidProperties(
            fluid_type="CRUDE_OIL",
            oil_gravity=30.0
        )
        
        pvt = PVTProperties(
            pressure_psia=3000.0,
            absolute_temperature_rankine=600.0
        )
        
        # Oil should have lower formation volume factor
        assert pvt.formation_volume_factor > 1.0 but < 1.5
