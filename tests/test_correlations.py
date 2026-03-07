"""Tests for black oil PVT correlation implementations."""
import pytest

import numpy as np

from perf_pressure_traverse.models.fluid import FluidProperties
from perf_pressure_traverse.models.pvt_properties import PVTProperties
from perf_pressure_traverse.models.implementations import (
    # Oil viscosity correlations
    standing_viscosity_cors,
    beal_viscosity_correlation,
    chew_connally_viscosity_correlation,
    vasquez_beggs_viscosity,
    
    # Solution GOR correlations
    vasquez_beggs_solution_gor,
    beggs_brill_solution_gor,
    standing_solution_gor,
    
    # FVF correlations
    standing_fvf,
    vasquez_beggs_fvf,
    
    # Oil density
    standing_density,
    
    # API temperature corrections
    stanton_temperature_corr,
    api_grade_correction,
    
    # Utilities
    calculate_c1_coefficient,
    validate_pvt_inputs,
    fluid_api_grade,
    
    BlackOilPVT
)


class TestOilViscosityCorrelations:
    """Test oil viscosity correlation implementations."""
    
    def test_standing_viscosity_normal_range(self):
        """Test Standing viscosity for typical reservoir conditions."""
        # Typical reservoir conditions
        mu = standing_viscosity_cors(
            pressure_psia=3000.0,
            temperature_f=150.0,
            rs_scf_stb=500.0,
            sg_oil=0.85,
            sg_gas=0.65,
            viscosity_ref_cP=1.5,
            rs_ref_scf_stb=100.0
        )
        
        assert mu > 0
        assert mu < 10  # Reasonable range for crude oil viscosity
    
    def test_standing_viscosity_low_temperature(self):
        """Test Standing viscosity at low reservoir temperatures."""
        mu = standing_viscosity_cors(
            pressure_psia=3000.0,
            temperature_f=80.0,
            rs_scf_stb=300.0,
            sg_oil=0.8,
            sg_gas=0.6,
            viscosity_ref_cP=0.9,
            rs_ref_scf_stb=50.0
        )
        
        assert mu > 0
        assert mu < 3  # Higher viscosity expected at lower temperature
    
    def test_standing_viscosity_no_gas(self):
        """Test Standing viscosity for dead oil (no gas dissolved)."""
        mu = standing_viscosity_cors(
            pressure_psia=3000.0,
            temperature_f=150.0,
            rs_scf_stb=0.0,
            sg_oil=0.85,
            sg_gas=0.0,
            viscosity_ref_cP=1.5,
            rs_ref_scf_stb=0.0
        )
        
        assert mu == 1.5  # Should return reference viscosity
    
    def test_standing_viscosity_negative_input(self):
        """Test that Standing viscosity raises error for negative inputs."""
        with pytest.raises(ValueError):
            standing_viscosity_cors(
                pressure_psia=-100.0,  # Negative pressure
                temperature_f=150.0,
                rs_scf_stb=500.0,
                sg_oil=0.85,
                sg_gas=0.65,
                viscosity_ref_cP=1.5,
                rs_ref_scf_stb=100.0
            )
    
    def test_beal_viscosity_valid_inputs(self):
        """Test Beal viscosity correlation with valid inputs."""
        # Reference conditions
        viscosity_ref_cP = 1.0
        
        mu = beal_viscosity_correlation(
            pressure_psia=3000.0,
            temperature_f=150.0,
            viscosity_ref_cP=viscosity_ref_cP
        )
        
        assert mu > 0
        assert isinstance(mu, float)
    
    def test_beal_viscosity_surface_pressure(self):
        """Test Beal viscosity at surface pressure (14.7 psi)."""
        mu = beal_viscosity_correlation(
            pressure_psia=14.7,
            temperature_f=150.0,
            viscosity_ref_cP=1.0
        )
        
        # Should be close to reference viscosity
        assert abs(mu - 1.0) < 0.01
    
    def test_beal_viscosity_zero_pressure(self):
        """Test Beal viscosity at zero pressure."""
        with pytest.raises(ValueError):
            beal_viscosity_correlation(
                pressure_psia=0.0,
                temperature_f=150.0,
                viscosity_ref_cP=1.0
            )
    
    def test_chew_connally_light_oil(self):
        """Test Chew-Connally viscosity for light oil."""
        mu = chew_connally_viscosity_correlation(
            rs_scf_stb=200.0,
            sg_oil=0.7,
            temperature_f=120.0
        )
        
        assert mu > 0
        assert mu < 2  # Light oil has lower viscosity
    
    def test_chew_connally_heavy_oil(self):
        """Test Chew-Connally viscosity for heavy oil."""
        mu = chew_connally_viscosity_correlation(
            rs_scf_stb=100.0,
            sg_oil=1.0,
            temperature_f=80.0
        )
        
        assert mu > 0
    
    def test_vasquez_beggs_valid_inputs(self):
        """Test Vasquez-Beggs viscosity with valid inputs."""
        mu = vasquez_beggs_viscosity(
            rs_scf_stb=500.0,
            sg_oil=0.85,
            sg_gas=0.65,
            temperature_f=150.0,
            api_grade=30.0
        )
        
        assert mu > 0
        assert isinstance(mu, float)
    
    def test_vasquez_beggs_no_solution_gas(self):
        """Test Vasquez-Beggs viscosity for dead oil."""
        mu = vasquez_beggs_viscosity(
            rs_scf_stb=0.0,
            sg_oil=0.85,
            sg_gas=0.65,
            temperature_f=150.0,
            api_grade=30.0
        )
        
        # Should use simplified formula for zero Rs
        assert mu > 0
    
    def test_vasquez_beggs_negative_input(self):
        """Test Vasquez-Beggs viscosity with negative inputs."""
        with pytest.raises(ValueError):
            vasquez_beggs_viscosity(
                rs_scf_stb=-100.0,
                sg_oil=0.85,
                sg_gas=0.65,
                temperature_f=150.0
            )


class TestSolutionGORCorrelations:
    """Test solution gas-oil ratio correlation implementations."""
    
    def test_vasquez_beggs_gor_normal_conditions(self):
        """Test Vasquez-Beggs GOR for typical conditions."""
        Rs = vasquez_beggs_solution_gor(
            pressure_psia=3000.0,
            sg_gas=0.65,
            api_grade=30.0,
            temperature_f=80.0
        )
        
        assert Rs > 0
        assert Rs < 5000  # Reasonable GOR range
    
    def test_vasquez_beggs_low_api(self):
        """Test Vasquez-Beggs GOR for heavy oil (API < 20)."""
        Rs = vasquez_beggs_solution_gor(
            pressure_psia=2000.0,
            sg_gas=0.6,
            api_grade=15.0,
            temperature_f=60.0
        )
        
        assert Rs > 0
    
    def test_vasquez_beggs_high_pressure(self):
        """Test Vasquez-Beggs GOR at high reservoir pressure."""
        Rs = vasquez_beggs_solution_gor(
            pressure_psia=8000.0,
            sg_gas=0.65,
            api_grade=30.0,
            temperature_f=60.0
        )
        
        assert Rs > 0
        # At very high pressures, Rs may be capped
        assert Rs < 2000
    
    def test_beggs_brill_gor_valid_input(self):
        """Test Beggs-Brill GOR correlation."""
        Rs = beggs_brill_solution_gor(
            pressure_psia=3000.0,
            sg_gas=0.65,
            api_grade=30.0,
            temperature_f=100.0
        )
        
        assert Rs > 0
    
    def test_beggs_brill_capped_gor(self):
        """Test Beggs-Brill GOR is capped at reasonable value."""
        Rs = beggs_brill_solution_gor(
            pressure_psia=3000.0,
            sg_gas=0.65,
            api_grade=30.0,
            temperature_f=200.0  # High temperature
        )
        
        assert Rs > 0
        assert Rs < 1500  # Should be capped
    
    def test_standing_gor_valid_input(self):
        """Test Standing GOR correlation."""
        Rs = standing_solution_gor(
            pressure_psia=3000.0,
            sg_gas=0.65,
            sg_oil=0.8,
            temperature_f=60.0
        )
        
        assert Rs > 0
    
    def test_gor_zero_negative_input(self):
        """Test that GOR raises error for negative inputs."""
        with pytest.raises(ValueError):
            vasquez_beggs_solution_gor(-100.0, 0.65, 30.0)
    
    def test_gor_absolute_zero_temperature(self):
        """Test GOR at absolute zero temperature."""
        with pytest.raises(ValueError):
            vasquez_beggs_solution_gor(3000.0, 0.65, 30.0, -460.0)


class TestFormationVolumeFactorCorrelations:
    """Test oil formation volume factor correlation implementations."""
    
    def test_standing_fvf_valid_input(self):
        """Test Standing FVF for valid conditions."""
        Bo = standing_fvf(
            pressure_psia=3000.0,
            sg_gas=0.65,
            api_grade=30.0,
            temperature_f=80.0
        )
        
        assert Bo >= 1.0
        assert Bo < 3.0  # Typical FVF range
    
    def test_standing_fvf_surface_conditions(self):
        """Test Standing FVF at surface conditions."""
        Bo = standing_fvf(
            pressure_psia=0.0,  # Near surface
            sg_gas=0.65,
            api_grade=30.0,
            temperature_f=60.0
        )
        
        # Should be close to 1.0
        assert Bo >= 1.0
    
    def test_standing_fvf_temperature_effect(self):
        """Test that FVF increases with temperature."""
        Bo_hot = standing_fvf(
            pressure_psia=3000.0,
            sg_gas=0.65,
            api_grade=30.0,
            temperature_f=180.0
        )
        
        Bo_cold = standing_fvf(
            pressure_psia=3000.0,
            sg_gas=0.65,
            api_grade=30.0,
            temperature_f=80.0
        )
        
        # Higher temperature should give higher FVF
        assert Bo_hot > Bo_cold
    
    def test_vasquez_beggs_fvf_valid_input(self):
        """Test Vasquez-Beggs FVF correlation."""
        Bo = vasquez_beggs_fvf(
            pressure_psia=3000.0,
            sg_gas=0.65,
            sg_oil=0.85,
            api_grade=30.0,
            temperature_f=80.0,
            Rs_scf_stb=500.0
        )
        
        assert Bo >= 1.0
    
    def test_vasquez_beggs_fvf_dry_oil(self):
        """Test Vasquez-Beggs FVF for dry oil (no gas)."""
        Bo = vasquez_beggs_fvf(
            pressure_psia=3000.0,
            sg_gas=0.65,
            sg_oil=0.85,
            api_grade=30.0,
            temperature_f=80.0,
            Rs_scf_stb=0.0
        )
        
        assert Bo >= 1.0


class TestOilDensityCorrelations:
    """Test oil density correlation implementations."""
    
    def test_standing_density_valid_input(self):
        """Test Standing density calculation."""
        density = standing_density(
            pressure_psia=3000.0,
            sg_gas=0.65,
            sg_oil=0.85,
            temperature_f=80.0
        )
        
        assert density > 0
        assert isinstance(density, float)
    
    def test_standing_density_surface_conditions(self):
        """Test density at surface conditions."""
        density = standing_density(
            pressure_psia=14.7,
            sg_gas=0.65,
            sg_oil=0.85,
            temperature_f=60.0
        )
        
        # Surface density = 62.4 * SG
        expected = 62.4 * 0.85
        assert abs(density - expected) < 0.5


class TestAPITemperatureCorrections:
    """Test API temperature correction formulas."""
    
    def test_stanton_temperature_corr_low_to_high(self):
        """Test temperature correction from low to high."""
        mu_initial = 1.5  # cP at 80°F
        mu_final = stanton_temperature_corr(
            initial_temperature_f=80.0,
            final_temperature_f=120.0,
            oil_viscosity_initial_cP=mu_initial
        )
        
        assert mu_final > 0
        # Viscosity should decrease with temperature
        assert mu_final < mu_initial
    
    def test_stanton_temperature_corr_high_to_low(self):
        """Test temperature correction from high to low."""
        mu_initial = 1.2  # cP at 120°F
        mu_final = stanton_temperature_corr(
            initial_temperature_f=120.0,
            final_temperature_f=80.0,
            oil_viscosity_initial_cP=mu_initial
        )
        
        assert mu_final > 0
        # Viscosity should increase as we go to lower temperature
        assert mu_final > mu_initial
    
    def test_stanton_temperature_corr_same_temperature(self):
        """Test temperature correction when temperature is the same."""
        mu = stanton_temperature_corr(
            initial_temperature_f=80.0,
            final_temperature_f=80.0,
            oil_viscosity_initial_cP=1.5
        )
        
        assert mu == 1.5
    
    def test_api_grade_correction(self):
        """Test API grade temperature correction."""
        initial_api = 35.0  # At 60°F
        final_api = api_grade_correction(initial_api, 100.0)
        
        assert final_api > 0
        assert final_api < initial_api  # API decreases with temperature
    
    def test_api_grade_correction_zero_temp(self):
        """Test API grade correction at absolute zero (should be clamped)."""
        initial_api = 35.0
        final_api = api_grade_correction(initial_api, -460.0)
        
        assert final_api >= 0  # Should be non-negative


class TestUtilities:
    """Test utility functions."""
    
    def test_validate_pvt_inputs_valid(self):
        """Test validate_pvt_inputs with valid parameters."""
        assert validate_pvt_inputs(3000.0, 150.0, 500.0, 1.0) is True
    
    def test_validate_pvt_inputs_negative_pressure(self):
        """Test validate_pvt_inputs with negative pressure."""
        with pytest.raises(ValueError):
            validate_pvt_inputs(-100.0, 150.0)
    
    def test_validate_pvt_inputs_absolute_zero_temp(self):
        """Test validate_pvt_inputs at absolute zero."""
        with pytest.raises(ValueError):
            validate_pvt_inputs(3000.0, -460.0)
    
    def test_validate_pvt_inputs_negative_rs(self):
        """Test validate_pvt_inputs with negative Rs."""
        with pytest.raises(ValueError):
            validate_pvt_inputs(3000.0, 150.0, -100.0)
    
    def test_fluid_api_grade(self):
        """Test API grade calculation from specific gravity."""
        sg = 0.85
        api = fluid_api_grade(sg)  # Use FluidProperties mock
        
        # API = 141.5 / (SG + 131.5)
        expected = 141.5 / (0.85 + 131.5)
        assert abs(api - expected) < 0.01
    
    def test_fluid_api_grade_water(self):
        """Test API grade for water (SG = 1.0)."""
        api = fluid_api_grade(1.0)
        assert api == 0.0  # Water has API = 0


class TestBlackOilPVTClass:
    """Test BlackOilPVT calculator class."""
    
    def test_black_oil_pvt_initialization(self):
        """Test BlackOilPVT class initialization."""
        fluid = FluidProperties(
            oil_gravity=30.0,
            gas_gravity=0.65,
            specific_gravity=0.8
        )
        
        pvt = BlackOilPVT(
            fluid=fluid,
            pressure_psia=3000.0,
            temperature_f=150.0
        )
        
        assert pvt.pressure_psia == 3000.0
        assert pvt.temperature_f == 150.0
        assert pvt.fluid == fluid
    
    @pytest.mark.parametrize("correlation", [
        "standing",
        "beal",
        "vasquez_beggs",
        "chew_connally"
    ])
    def test_black_oil_pvt_solution_gor_various_correlations(self, correlation):
        """Test different GOR correlations."""
        fluid = FluidProperties(
            oil_gravity=30.0,
            gas_gravity=0.65
        )
        
        pvt = BlackOilPVT(
            fluid=fluid,
            pressure_psia=3000.0,
            temperature_f=120.0,
            correlation_type=correlation
        )
        
        rs = pvt.calculate_solution_gor()
        
        assert rs > 0
        assert isinstance(rs, float)
    
    def test_black_oil_pvt_oil_viscosity(self):
        """Test oil viscosity calculation."""
        fluid = FluidProperties(
            oil_gravity=30.0,
            gas_gravity=0.65
        )
        
        pvt = BlackOilPVT(
            fluid=fluid,
            pressure_psia=3000.0,
            temperature_f=120.0,
            correlation_type="vasquez_beggs"
        )
        
        mu = pvt.calculate_oil_viscosity()
        
        assert mu > 0
        assert isinstance(mu, float)
    
    def test_black_oil_pvt_fvf(self):
        """Test FVF calculation."""
        fluid = FluidProperties(
            oil_gravity=30.0,
            gas_gravity=0.65
        )
        
        pvt = BlackOilPVT(
            fluid=fluid,
            pressure_psia=3000.0,
            temperature_f=120.0
        )
        
        Bo = pvt.calculate_oil_fvf()
        
        assert Bo >= 1.0
        assert isinstance(Bo, float)
    
    def test_black_oil_pvt_reservoir_properties(self):
        """Test reservoir properties calculation."""
        fluid = FluidProperties(
            oil_gravity=30.0,
            gas_gravity=0.65
        )
        
        pvt = BlackOilPVT(
            fluid=fluid,
            pressure_psia=3000.0,
            temperature_f=120.0
        )
        
        properties = pvt.calculate_reservoir_properties()
        
        assert "viscosity_cP" in properties
        assert "solution_gor_scf_stb" in properties
        assert "formation_volume_factor" in properties
        assert properties["formation_volume_factor"] >= 1.0
    
    def test_black_oil_pvt_invalid_correlation(self):
        """Test BlackOilPVT with invalid correlation type."""
        fluid = FluidProperties(
            oil_gravity=30.0,
            gas_gravity=0.65
        )
        
        pvt = BlackOilPVT(
            fluid=fluid,
            pressure_psia=3000.0,
            temperature_f=120.0,
            correlation_type="invalid"
        )
        
        with pytest.raises(ValueError):
            pvt.calculate_solution_gor()


class TestCorrelationAccuracy:
    """Test correlation accuracy with expected values."""
    
    def test_standing_viscosity_accuracy(self):
        """Test Standing viscosity with expected reference value."""
        # At reference conditions, should return near reference viscosity
        mu = standing_viscosity_cors(
            pressure_psia=14.7,  # Surface pressure
            temperature_f=60.0,  # Std temperature
            rs_scf_stb=0.0,
            sg_oil=0.85,
            sg_gas=0.0,
            viscosity_ref_cP=1.5,  # We'll use this as reference
            rs_ref_scf_stb=0.0
        )
        
        # Should be close to reference viscosity (accounting for slight temperature effect)
        assert 1.4 < mu < 1.6
    
    def test_vasquez_beggs_gor_in_range(self):
        """Test Vasquez-Beggs GOR is within physically reasonable range."""
        Rs = vasquez_beggs_solution_gor(
            pressure_psia=2000.0,
            sg_gas=0.65,
            api_grade=30.0
        )
        
        # Typical reservoir: Rs in range 100-2000 scf/stb
        assert 50 < Rs < 2000
    
    def test_fvf_greater_than_one(self):
        """Test that FVF is always >= 1.0."""
        Bo = vasquez_beggs_fvf(
            pressure_psia=3000.0,
            sg_gas=0.65,
            sg_oil=0.85,
            api_grade=30.0,
            temperature_f=80.0,
            Rs_scf_stb=500.0
        )
        
        assert Bo >= 1.0


class TestCorrelationEdgeCases:
    """Test correlation behavior at edge cases."""
    
    def test_correlation_at_pressure_threshold(self):
        """Test correlations at low pressure threshold."""
        mu = standing_viscosity_cors(
            pressure_psia=100.0,
            temperature_f=80.0,
            rs_scf_stb=300.0,
            sg_oil=0.8,
            sg_gas=0.6,
            viscosity_ref_cP=0.9,
            rs_ref_scf_stb=50.0
        )
        
        assert mu > 0
    
    def test_correlation_at_high_temperature(self):
        """Test correlations at very high temperature."""
        mu = standing_viscosity_cors(
            pressure_psia=3000.0,
            temperature_f=250.0,  # Very high temperature
            rs_scf_stb=500.0,
            sg_oil=0.8,
            sg_gas=0.6,
            viscosity_ref_cP=1.0,
            rs_ref_scf_stb=100.0
        )
        
        assert mu > 0
    
    def test_correlation_with_heavy_oil(self):
        """Test correlations for heavy oil (API < 20)."""
        mu = vasquez_beggs_viscosity(
            rs_scf_stb=100.0,
            sg_oil=1.0,  # Heavy oil
            sg_gas=0.7,
            temperature_f=80.0,
            api_grade=15.0
        )
        
        assert mu > 0
    
    def test_correlation_with_light_oil(self):
        """Test correlations for light oil (API > 40)."""
        mu = vasquez_beggs_viscosity(
            rs_scf_stb=800.0,
            sg_oil=0.65,
            sg_gas=0.6,
            temperature_f=100.0,
            api_grade=45.0
        )
        
        assert mu > 0
