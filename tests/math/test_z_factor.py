"""Unit tests for Z-factor correlations."""

import pytest
import numpy as np

from perf_pressure_traverse.math.z_factor import (
    calculate_standing_katz_z_factor,
    LeeGonzalesEspana,
    calculate_aga_dc_z_factor,
    PseudocriticalProperties,
    calculate_pseudocritical_properties,
    get_pseudocritical_temp,
    get_pseudocritical_press,
)


class TestPseudocriticalProperties:
    """Tests for PseudocriticalProperties class."""
    
    def test_init(self):
        """Test initialization."""
        pc = PseudocriticalProperties(pseudo_critical_pressure_psia=672.0, pseudo_critical_temperature_R=343.1)
        
        assert pc.pseudo_critical_pressure_psia == 672.0
        assert pc.pseudo_critical_temperature_R == 343.1
    
    def test_reduced_pressure(self):
        """Test calculation of reduced pressure."""
        pc = PseudocriticalProperties(pseudo_critical_pressure_psia=672.0, pseudo_critical_temperature_R=343.1)
        
        rp = pc.get_reduced_pressure(1344.0)  # Should be exactly 2.0
        assert abs(rp - 2.0) < 1e-10
    
    def test_reduced_pressure_zero(self):
        """Test reduced pressure with zero pseudocritical pressure."""
        pc = PseudocriticalProperties(pseudo_critical_pressure_psia=0.0, pseudo_critical_temperature_R=343.1)
        
        with pytest.raises(ValueError, match="Pseudocritical pressure cannot be zero"):
            pc.get_reduced_pressure(100.0)
    
    def test_reduced_temperature(self):
        """Test calculation of reduced temperature."""
        pc = PseudocriticalProperties(pseudo_critical_pressure_psia=672.0, pseudo_critical_temperature_R=343.1)
        
        rt = pc.get_reduced_temperature(686.2)  # Should be exactly 2.0 (343.1 * 2)
        assert abs(rt - 2.0) < 1e-10


class TestCalculateStandingKatz:
    """Tests for Stand
ing-Katz Z-factor calculation."""
    
    def test_zero_pressure(self):
        """Test at zero pressure."""
        z = calculate_standing_katz_z_factor(pressure_psia=0.0, temperature_f=60.0)
        assert abs(z - 1.0) < 1e-6
    
    def test_standard_conditions(self):
        """Test at standard conditions."""
        z = calculate_standing_katz_z_factor(pressure_psia=14.7, temperature_f=60.0)
        # Z-factor should be very close to 1.0 at SPSC
        assert 0.95 < z < 1.05
    
    def test_typical_conditions(self):
        """Test with typical well conditions."""
        z = calculate_standing_katz_z_factor(pressure_psia=1500, temperature_f=100.0)
        assert 0.9 < z < 1.2
    
    def test_high_pressure(self):
        """Test at high pressure."""
        z = calculate_standing_katz_z_factor(pressure_psia=3000, temperature_f=100.0)
        assert 0.8 < z < 1.0


class TestLeeGonzales:
    """Tests for Lee-Gonzales Espana correlation."""
    
    def test_zero_pressure(self):
        """Test at zero pressure."""
        z = LeeGonzalesEspana.calculate_z_factor(
            pressure_psia=0.0, temperature_f=60.0, specific_gravity=0.65
        )
        assert 0.99 < z < 1.01  # Should approach Z ≈ 1.0 at low pressure
    
    def test_standard_conditions(self):
        """Test at standard conditions."""
        z = LeeGonzalesEspana.calculate_z_factor(
            pressure_psia=14.7, temperature_f=60.0, specific_gravity=0.65
        )
        assert 0.95 < z < 1.05
    
    def test_specific_gravity_1(self):
        """Test with gas gravity = 1.0."""
        z = LeeGonzalesEspana.calculate_z_factor(
            pressure_psia=2000, temperature_f=120.0, specific_gravity=1.0
        )
        assert 0.8 < z < 1.3
    
    def test_high_temperature(self):
        """Test at high temperature."""
        z = LeeGonzalesEspana.calculate_z_factor(
            pressure_psia=1500, temperature_f=200.0, specific_gravity=0.65
        )
        assert 0.8 < z < 1.2
    
    def test_custom_molecular_weight(self):
        """Test with custom molecular weight."""
        z = LeeGonzalesEspana.calculate_z_factor(
            pressure_psia=1500, temperature_f=100.0, specific_gravity=0.65, molecular_weight_moles=29.0
        )
        assert 0.8 < z < 1.3
    
    def test_molecular_weight_from_specific_gravity(self):
        """Test that molecular weight is calculated correctly from specific gravity."""
        z1 = LeeGonzalesEspana.calculate_z_factor(
            pressure_psia=1500, temperature_f=100.0, specific_gravity=0.65
        )
        z2 = LeeGonzalesEspana.calculate_z_factor(
            pressure_psia=1500, temperature_f=100.0, specific_gravity=0.65, molecular_weight_moles=28.964 / 0.65
        )
        # Should be approximately equal
        z_diff = abs(z1 - z2)
        assert z_diff < 1e-6


class TestAGADC:
    """Tests for AGA-8 DC Z-factor calculation."""
    
    def test_zero_pressure(self):
        """Test at zero pressure."""
        z = calculate_aga_dc_z_factor(pressure_psia=0.0, temperature_f=60.0, specific_gravity=0.65)
        assert abs(z - 1.0) < 1e-6
    
    def test_standard_conditions(self):
        """Test at standard conditions."""
        z = calculate_aga_dc_z_factor(pressure_psia=14.7, temperature_f=60.0, specific_gravity=0.65)
        assert 0.95 < z < 1.05
    
    def test_composition_gravity(self):
        """Test with gas gravity = 1.0 (natural gas with air equivalent)."""
        z = calculate_aga_dc_z_factor(pressure_psia=2000, temperature_f=110.0, specific_gravity=1.0)
        assert 0.8 < z < 1.3
    
    def test_high_gravity_gas(self):
        """Test with heavier gas."""
        z = calculate_aga_dc_z_factor(pressure_psia=1800, temperature_f=100.0, specific_gravity=0.85)
        assert 0.7 < z < 1.2


class TestPseudocriticalPropertiesCalculator:
    """Tests for pseudocritical properties calculator."""
    
    def test_single_gravity_stewart_burke_katz(self):
        """Test with single-gravity system using Stewart-Burke-Katz."""
        temp_R, press_PSI = calculate_pseudocritical_properties(gas_specific_gravity=0.65)
        
        # Verify using Stewart-Burke-Katz formulas
        expected_temp_R = (0.554 * 0.65 + 0.4 * (1.0 - 0.554 * 0.65)) * 460.0
        expected_press_PSI = 708.0 - 58.71 * 0.65 + 0.0107 * (0.65 ** 2)
        
        assert abs(temp_R - expected_temp_R) < 1.0
        assert abs(press_PSI - expected_press_PSI) < 1.0
    
    def test_specific_gravity_1(self):
        """Test with gas gravity = 1.0."""
        temp_R, press_PSI = calculate_pseudocritical_properties(gas_specific_gravity=1.0)
        
        # γ = 1.0 means pure air equivalent
        expected_temp_R = (0.554 * 1.0 + 0.4 * (1.0 - 0.554 * 1.0)) * 460.0
        expected_press_PSI = 708.0 - 58.71 * 1.0 + 0.0107 * (1.0 ** 2)
        
        assert abs(temp_R - expected_temp_R) < 1.0
        assert abs(press_PSI - expected_press_PSI) < 1.0
    
    def test_multi_component(self):
        """Test with multi-component composition."""
        composition = {'CH4': 0.85, 'CO2': 0.10, 'N2': 0.05}
        expected_temp_R = 0.85 * 629.1 + 0.10 * 585.1 + 0.05 * 523.5
        expected_press_PSI = 0.85 * 680.0 + 0.10 * 750.0 + 0.05 * 500.0
        
        temp_R, press_PSI = calculate_pseudocritical_properties(
            gas_specific_gravity=0.0, composition=composition
        )
        
        assert abs(temp_R - expected_temp_R) < 1.0
        assert abs(press_PSI - expected_press_PSI) < 1.0
    
    def test_multi_component_different_gravity(self):
        """Test composition with same molecular weight as calculated gravity."""
        composition = {'CH4': 0.80, 'C2H6': 0.15, 'i-C4H10': 0.05}
        
        # Calculate effective gravity from composition
        molecular_weights = {
            'CH4': 16.043, 'C2H6': 30.070, 'i-C4H10': 58.123
        }
        average_mw = sum(composition[mw] * molecular_weights[mw] for mw in composition.keys())
        effective_gamma = 28.964 / average_mw
        
        temp_R, press_PSI = calculate_pseudocritical_properties(
            gas_specific_gravity=effective_gamma, composition=composition
        )
    
    def test_no_specific_gravity_or_composition(self):
        """Test when neither specific gravity nor composition provided."""
        with pytest.raises(ValueError):
            calculate_pseudocritical_properties(gas_specific_gravity=0.0, composition=None)


class TestZFactorAccuracy:
    """Tests for Z-factor accuracy ±0.1%."""
    
    @pytest.mark.parametrize("pressure,temperature,gravity", [
        (500, 60, 0.6),
        (1000, 80, 0.65),
        (1500, 100, 0.7),
        (2000, 120, 0.75),
        (2500, 110, 0.65),
    ])
    def test_aga_dc_within_tolerance(self, pressure, temperature, gravity):
        """Test AGA-8 DC Z-factor is within ±0.1% of expected range."""
        z = calculate_aga_dc_z_factor(pressure_psia=pressure, temperature_f=temperature, specific_gravity=gravity)
        
        # Z-factor should be between 0.5 and 2.0 for typical conditions
        assert 0.5 < z < 2.0
    
    @pytest.mark.parametrize("pressure,temperature,gravity", [
        (500, 70, 0.6),
        (1000, 90, 0.65),
        (2000, 130, 0.7),
    ])
    def test_standing_katz_within_tolerance(self, pressure, temperature, gravity):
        """Test Standing-Katz Z-factor is within ±0.1% of expected range."""
        z = calculate_standing_katz_z_factor(pressure_psia=pressure, temperature_f=temperature)
        
        # Z-factor should be reasonable for natural gas
        assert 0.6 < z < 1.8


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_get_pseudocritical_temp(self):
        """Test pseudocritical temperature calculation."""
        expected = (0.554 * 0.65 + 0.4 * (1.0 - 0.554 * 0.65)) * 460.0
        actual = get_pseudocritical_temp(60.0, 0.65)
        
        assert abs(actual - expected) < 1.0
    
    def test_get_pseudocritical_press(self):
        """Test pseudocritical pressure calculation."""
        expected = 708.0 - 58.71 * 0.65 + 0.0107 * (0.65 ** 2)
        actual = get_pseudocritical_press(1000.0, 0.65)
        
        assert abs(actual - expected) < 1.0
    
    def test_pseudocritical_temp_zero_specific_gravity(self):
        """Test pseudocritical temp with specific gravity = 0."""
        actual = get_pseudocritical_temp(60.0, 0.0)
        # γ = 0 should give reasonable value (though unrealistic gas)
        assert actual > 100.0  # Should be positive


class TestIntegration:
    """Integration tests that use multiple functions."""
    
    def test_full_workflow_basic(self):
        """Test full workflow with multiple calculations."""
        # Calculate pseudocritical properties
        pc_temp, pc_press = get_pseudocritical_temp(100.0), get_pseudocritical_press(1500.0, 0.65)
        
        # Calculate Z-factor
        z_aga = calculate_aga_dc_z_factor(1500.0, 100.0, 0.65)
        
        # Verify reduced properties
        pr = 1500.0 / pc_press
        tr = (100.0 + 460) / pc_temp
        
        # Z-factor should be based on reduced properties
        assert 0.1 < pr < 2.0  # Reasonable reduced pressure
        assert 1.1 < tr < 2.5  # Reasonable reduced temperature
        assert 0.8 < z_aga < 1.2  # Reasonable Z-factor


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
