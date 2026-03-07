"""Tests for black oil PVT correlations."""

import pytest
import math
from perf_pressure_traverse.pvt_correlations import BlackOilCorrelations


class TestStandingGasor:
    """Tests for Standing solution GOR correlation."""
    
    @pytest.fixture
    def correlations(self):
        return BlackOilCorrelations
    
    def test_standing_gasor_basic(self, correlations):
        """Test basic Standing GOR calculation."""
        Rs = correlations.get_standing_solution_gor(
            pressure_psia=2000.0,
            temp_f=150.0,
            gas_specific_gravity=0.65,
            oil_specific_gravity=0.8
        )
        
        # Rs should be positive
        assert Rs > 0.0
        # Rs shouldn't be unrealistically high
        assert Rs < 500.0
    
    def test_standing_gasor_zero_pressure(self, correlations):
        """Test GOR at zero pressure."""
        Rs = correlations.get_standing_solution_gor(
            pressure_psia=0.0,
            temp_f=150.0
        )
        
        # Should return 0 or small value
        assert Rs >= 0.0
    
    def test_standing_gasor_zero_temperature(self, correlations):
        """Test GOR at zero temperature."""
        Rs = correlations.get_standing_solution_gor(
            pressure_psia=2000.0,
            temp_f=0.0
        )
        
        # Should be negative but we clip to 0
        assert Rs <= 0.0


class TestBeggsBrillViscosity:
    """Tests for Beggs & Brill oil viscosity correlation."""
    
    @pytest.fixture
    def correlations(self):
        return BlackOilCorrelations
    
    def test_beggs_brill_viscosity_dead_oil(self, correlations):
        """Test dead oil viscosity calculation."""
        viscosity = correlations.get_beggs_brill_oil_viscosity(
            pressure_psia=2000.0,
            temp_f=150.0,
            solution_gor_scf_stb=0.0,
            oil_specific_gravity=0.9
        )
        
        # Viscosity should be positive
        assert viscosity > 0.0
        # Viscosity of crude oil at reservoir conditions should be reasonable
        # Typically 0.5 - 50 cP
        assert 0.1 < viscosity < 50.0
    
    def test_beggs_brill_viscosity_with_gas(self, correlations):
        """Test oil viscosity with gas."""
        viscosity = correlations.get_beggs_brill_oil_viscosity(
            pressure_psia=2000.0,
            temp_f=150.0,
            solution_gor_scf_stb=300.0,
            oil_specific_gravity=0.8,
            gas_specific_gravity=0.65
        )
        
        # With gas, viscosity should be lower than dead oil
        dead_visc = correlations.get_beggs_brill_oil_viscosity(
            pressure_psia=2000.0,
            temp_f=150.0,
            solution_gor_scf_stb=0.0,
            oil_specific_gravity=0.8
        )
        
        assert viscosity < dead_visc
    
    def test_visicosity_lower_bound(self, correlations):
        """Test that viscosity is clipped at minimum 0.1 cP."""
        viscosity = correlations.get_beggs_brill_oil_viscosity(
            pressure_psia=0.0,
            temp_f=60.0,
            solution_gor_scf_stb=0.0
        )
        
        assert viscosity >= 0.1


class TestVasquezBeggsGasor:
    """Tests for Vasquez & Beggs solution GOR correlation."""
    
    @pytest.fixture
    def correlations(self):
        return BlackOilCorrelations
    
    def test_vasquez_beggs_gasor_basic(self, correlations):
        """Test basic Vasquez & Beggs GOR calculation."""
        Rs = correlations.get_vasquez_beggs_solution_gor(
            pressure_psia=2000.0,
            temp_f=150.0,
            oil_specific_gravity=0.8,
            gas_specific_gravity=0.65
        )
        
        # Rs should be positive
        assert Rs > 0.0
    
    def test_vasquez_beggsvacuum_oil_gravity(self, correlations):
        """Test with vacuum oil gravity."""
        Rs = correlations.get_vasquez_beggs_solution_gor(
            pressure_psia=2000.0,
            temp_f=150.0,
            oil_specific_gravity=0.7,
            gas_specific_gravity=0.65
        )
        
        # Higher gravity should give lower Rs
        Rs_high = correlations.get_vasquez_beggs_solution_gor(
            pressure_psia=2000.0,
            temp_f=150.0,
            oil_specific_gravity=0.9,
            gas_specific_gravity=0.65
        )
        
        assert Rs < Rs_high


class TestTemperatureCorrections:
    """Tests for temperature correction correlations."""
    
    @pytest.fixture
    def correlations(self):
        return BlackOilCorrelations
    
    def test_api_temperature_correction(self, correlations):
        """Test API temperature viscosity correction."""
        viscosity = 20.0  # cP at 60°F
        
        corrected = correlations.get_api_temperature_correction(
            oil_viscosity_f=viscosity,
            temp_f=150.0
        )
        
        # Temperature correction should be reasonable
        assert corrected > 0.0
        # Should decrease as temperature increases (viscosity decreases)
        assert corrected < viscosity
    
    def test_api_temperature_correction_absolute_zero(self, correlations):
        """Test with absolute zero temperature."""
        viscosity = 20.0
        
        # Should return minimum 0.1
        corrected = correlations.get_api_temperature_correction(
            oil_viscosity_f=viscosity,
            temp_f=-273.15
        )
        
        assert corrected >= 0.1
    
    def test_katz_temperature_correction(self, correlations):
        """Test Katz temperature correction factor."""
        pressure = 2000.0
        
        factor = correlations.get_katz_temperature_correction(
            pressure_psia=pressure,
            temp_f=150.0
        )
        
        # Should be positive and close to 1.0
        assert 0.9 < factor < 2.0


class TestFormationVolumeFactors:
    """Tests for formation volume factor calculations."""
    
    @pytest.fixture
    def correlations(self):
        return BlackOilCorrelations
    
    def test_bo_beggs_brill(self, correlations):
        """Test Beggs & Brill formation volume factor calculation."""
        Bob = correlations.get_formation_volume_factor_bob(
            pressure_psia=2000.0,
            temp_f=150.0,
            solution_gor_scf_stb=300.0,
            oil_specific_gravity=0.8,
            gas_specific_gravity=0.65
        )
        
        # Bo should be >= 1.0 (oil expands at reservoir conditions)
        assert Bob >= 1.0
        # Typical Bo values range from 1.0 to 3.0
        assert 1.0 < Bob < 3.0
    
    def test_bo_at_reservoir_bubblepoint(self, correlations):
        """Test Bo at bubblepoint pressure."""
        # At solution gas-oil ratio
        Rs = correlations.get_standing_solution_gor(
            pressure_psia=2000.0,
            temp_f=150.0,
            gas_specific_gravity=0.65,
            oil_specific_gravity=0.8
        )
        
        Bob = correlations.get_formation_volume_factor_bob(
            pressure_psia=2000.0,
            temp_f=150.0,
            solution_gor_scf_stb=Rs,
            oil_specific_gravity=0.8,
            gas_specific_gravity=0.65
        )
        
        # Bo should reflect solution gas
        assert Bob > 1.0
    
    def test_gas_oil_ratio_surface(self, correlations):
        """Test surface GOR calculation."""
        Rs_res = 300.0
        Bob = 1.3
        
        Gor = correlations.get_gas_oil_ratio_surface(
            pressure_psia=2000.0,
            temp_f=150.0,
            solution_gor_scf_stb=Rs_res,
            formation_volume_factor_oil=Bob
        )
        
        # Gor should be Rs / Bob (at reservoir)
        expected = Rs_res / Bob
        
        # Account for small numerical tolerance
        assert abs(Gor - expected) < 0.1
        assert Gor > 0.0


class TestUnitConversions:
    """Tests for unit conversions."""
    
    @pytest.fixture
    def correlations(self):
        return BlackOilCorrelations
    
    def test_pressure_units(self, correlations):
        """Test pressure unit conversions."""
        psi_value = 100.0
        
        kpa = correlations.convert_pressure(psi_value, 'psi', 'kPa')
        bar = correlations.convert_pressure(psi_value, 'psi', 'bar')
        atm = correlations.convert_pressure(psi_value, 'psi', 'atm')
        
        # Round trip conversions
        psi_back = correlations.convert_pressure(kpa, 'kPa', 'psi')
        assert abs(psi_back - psi_value) < 0.1
        
        psi_back2 = correlations.convert_pressure(bar, 'bar', 'psi')
        assert abs(psi_back2 - psi_value) < 0.1
        
        psi_back3 = correlations.convert_pressure(atm, 'atm', 'psi')
        assert abs(psi_back3 - psi_value) < 0.1
    
    def test_pressure_negative(self, correlations):
        """Test with negative pressure values."""
        # Should return absolute value
        kpa = correlations.convert_pressure(-100.0, 'psi', 'kPa')
        assert kpa > 0.0
    
    def test_temperature_units(self, correlations):
        """Test temperature unit conversions."""
        f_value = 32.0  # Freezing point
        
        c = correlations.convert_temperature(f_value, 'fahrenheit', 'celsius')
        k = correlations.convert_temperature(f_value, 'fahrenheit', 'kelvin')
        
        # Freezing point in Celsius
        assert abs(c - 0.0) < 0.1
        
        # Freezing point in Kelvin
        assert abs(k - 273.15) < 0.1
    
    def test_temperature_kelvin(self, correlations):
        """Test conversion from Kelvin."""
        k_value = 373.15  # Boiling point
        
        f = correlations.convert_temperature(k_value, 'kelvin', 'fahrenheit')
        
        # Boiling point in Fahrenheit
        assert abs(f - 212.0) < 0.1
    
    def test_temperature_absolute_zero(self, correlations):
        """Test with absolute zero Kelvin."""
        k = correlations.convert_temperature(-273.15, 'fahrenheit', 'kelvin')
        assert k == -273.15  # Should stay at absolute zero


class TestValidation:
    """Tests for correlation input validation."""
    
    @pytest.fixture
    def correlations(self):
        return BlackOilCorrelations
    
    def test_valid_inputs(self, correlations):
        """Test with valid inputs."""
        valid = correlations.validate_correlation_inputs(
            rs=300.0,
            gor=400.0,
            temp=150.0,
            oil_gravity=0.8,
            gas_gravity=0.65
        )
        
        assert valid == True
    
    def test_invalid_oil_gravity_low(self, correlations):
        """Test with oil gravity too low."""
        valid = correlations.validate_correlation_inputs(
            rs=300.0,
            gor=400.0,
            temp=150.0,
            oil_gravity=0.3,  # Too low (should be >= 0.5)
            gas_gravity=0.65
        )
        
        assert valid == False
    
    def test_invalid_oil_gravity_high(self, correlations):
        """Test with oil gravity too high."""
        valid = correlations.validate_correlation_inputs(
            rs=300.0,
            gor=400.0,
            temp=150.0,
            oil_gravity=1.5,  # Too high (should be <= 1.1)
            gas_gravity=0.65
        )
        
        assert valid == False
    
    def test_invalid_gas_gravity_low(self, correlations):
        """Test with gas gravity too low."""
        valid = correlations.validate_correlation_inputs(
            rs=300.0,
            gor=400.0,
            temp=150.0,
            oil_gravity=0.8,
            gas_gravity=0.4  # Too low (should be >= 0.55)
        )
        
        assert valid == False
    
    def test_invalid_gas_gravity_high(self, correlations):
        """Test with gas gravity too high (> 1.0 for air)."""
        valid = correlations.validate_correlation_inputs(
            rs=300.0,
            gor=400.0,
            temp=150.0,
            oil_gravity=0.8,
            gas_gravity=1.2  # Too high (should be <= 1.0)
        )
        
        assert valid == False
    
    def test_invalid_temperature(self, correlations):
        """Test with invalid temperature."""
        valid = correlations.validate_correlation_inputs(
            rs=300.0,
            gor=400.0,
            temp=-300.0,
            oil_gravity=0.8,
            gas_gravity=0.65
        )
        
        assert valid == False
    
    def test_invalid_negative_values(self, correlations):
        """Test with negative Rs/GOR."""
        valid = correlations.validate_correlation_inputs(
            rs=-100.0,
            gor=400.0,
            temp=150.0,
            oil_gravity=0.8,
            gas_gravity=0.65
        )
        
        assert valid == False


class TestMultipleCorrelations:
    """Test comparing multiple correlation methods."""
    
    @pytest.fixture
    def correlations(self):
        return BlackOilCorrelations
    
    def test_correlation_comparison(self, correlations):
        """Test that correlations produce reasonable comparable results."""
        Rs_standing = correlations.get_standing_solution_gor(
            pressure_psia=2000.0,
            temp_f=150.0,
            gas_specific_gravity=0.65,
            oil_specific_gravity=0.8
        )
        
        Rs_vasquez = correlations.get_vasquez_beggs_solution_gor(
            pressure_psia=2000.0,
            temp_f=150.0,
            oil_specific_gravity=0.8,
            gas_specific_gravity=0.65
        )
        
        # Both should produce positive values
        assert Rs_standing > 0.0
        assert Rs_vasquez > 0.0
        
        # Results should be in sensible ranges
        assert Rs_standing < 500.0
        assert Rs_vasquez < 500.0
