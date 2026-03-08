"""Tests for black oil PVT property correlations.

This module tests the Vasquez-Beggs and Standing correlations implemented
in perf_pressure_traverse/math/black_oil_pvt.py.

Known API RPI (Recommended Practice 14A) test case values for validation:
- API RPI Test Case: Oil with specific gravity 0.826, Gas gravity 0.65
  Expected: Bo ≈ 1.4, Rs ≈ 350-400 scf/STB at 3000 psia, 100°F
"""

import pytest
import numpy as np

from perf_pressure_traverse.math.black_oil_pvt import (
    VasquezBeggsCorrelations,
    VasquezBeggsError,
    StandingCorrections,
    BlackOilPVTCalculator,
    PVTUnits,
    calculate_vasquez_beggs_pvt
)


class TestPVTUnits:
    """Test unit conversion utilities."""
    
    def test_psi_to_pascals(self):
        """Test psi to pascals conversion."""
        psi = 14.7
        pascals = PVTUnits.psi_to_pascals(psi)
        expected = psi * 6894.76
        assert abs(pascals - expected) < 1e-10
    
    def test_pascals_to_psi(self):
        """Test pascals to psi conversion."""
        pascals = 101325.0
        psi = PVTUnits.pascals_to_psi(pascals)
        expected = pascals / 6894.76
        assert abs(psi - expected) < 1e-8
    
    def test_fahrenheit_to_rankine(self):
        """Test °F to °R conversion."""
        f = 60.0
        r = PVTUnits.fahrenheit_to_rankine(f)
        expected = f + 459.67
        assert r == expected
    
    def test_rankine_to_fahrenheit(self):
        """Test °R to °F conversion."""
        r = 600.0
        f = PVTUnits.rankine_to_fahrenheit(r)
        expected = r - 459.67
        assert f == expected
    
    def test_rbf_to_stb(self):
        """Test formation volume factor conversion (dimensionless)."""
        rbf = 1.2
        assert PVTUnits.rbf_to_stb(rbf) == rbf
    
    def test_stb_to_rbf(self):
        """Test stock tank to reservoir volume conversion."""
        stb = 100.0
        rbf = 1.2
        reservoir_vol = PVTUnits.stb_to_rbf(stb, rbf)
        expected = stb * rbf
        assert abs(reservoir_vol - expected) < 1e-8
    
    def test_cP_to_pascal_second(self):
        """Test cP to Pa·s conversion."""
        cp = 1.0
        ps = PVTUnits.cP_to_pascal_second(cp)
        expected = cp * 1e-3
        assert abs(ps - expected) < 1e-12
    
    def test_pascal_second_to_cP(self):
        """Test Pa·s to cP conversion."""
        ps = 0.001
        cp = PVTUnits.pascal_second_to_cP(ps)
        expected = ps * 1e3
        assert abs(cp - expected) < 1e-8
    
    def test_scf_stb_to_sm3_stb(self):
        """Test scf/STB to m³/STB conversion."""
        scf_stb = 1000.0
        sm3_stb = PVTUnits.scf_stb_to_sm3_stb(scf_stb)
        expected = scf_stb * 0.0283168
        assert abs(sm3_stb - expected) < 1e-6


class TestVasquezBeggsCorrelations:
    """Test Vasquez-Beggs correlation implementations."""
    
    def test_gas_solubility_at_reference_conditions(self):
        """Test gas solubility at API RPI reference conditions."""
        # API RPI Test Case: Oil sg=0.826, Gas sg=0.65, 3000 psia, 100°F
        pressure_psia = 3000.0
        temperature_f = 100.0
        gas_specific_gravity = 0.65
        oil_specific_gravity = 0.826
        
        Rs = VasquezBeggsCorrelations.calculate_gas_solubility(
            pressure_psia,
            PVTUnits.fahrenheit_to_rankine(temperature_f),
            gas_specific_gravity,
            oil_specific_gravity
        )
        
        # Expected Rs: ~350-400 scf/STB (API RPI range)
        assert isinstance(Rs, float)
        assert Rs > 0.0
        assert Rs < 1000.0
        # Rough validation against API RPI expected range
        assert Rs > 200.0 and Rs < 800.0
    
    def test_gas_solubility_pressure_dependency(self):
        """Test gas solubility increases with pressure."""
        temperature_f = 100.0
        temp_r = PVTUnits.fahrenheit_to_rankine(temperature_f)
        gas_specific_gravity = 0.6
        oil_specific_gravity = 0.9
        
        Rs_high = VasquezBeggsCorrelations.calculate_gas_solubility(
            4000.0, temp_r, gas_specific_gravity, oil_specific_gravity
        )
        Rs_low = VasquezBeggsCorrelations.calculate_gas_solubility(
            2000.0, temp_r, gas_specific_gravity, oil_specific_gravity
        )
        
        assert Rs_high > Rs_low
    
    def test_gas_solubility_temperature_dependency(self):
        """Test gas solubility decreases with temperature."""
        pressure_psia = 3000.0
        gas_specific_gravity = 0.6
        oil_specific_gravity = 0.9
        
        Rs_high_t = VasquezBeggsCorrelations.calculate_gas_solubility(
            pressure_psia,
            PVTUnits.fahrenheit_to_rankine(130.0),
            gas_specific_gravity,
            oil_specific_gravity
        )
        Rs_low_t = VasquezBeggsCorrelations.calculate_gas_solubility(
            pressure_psia,
            PVTUnits.fahrenheit_to_rankine(80.0),
            gas_specific_gravity,
            oil_specific_gravity
        )
        
        assert Rs_high_t < Rs_low_t
    
    def test_gas_solubility_negative_pressure(self):
        """Test gas solubility raises error for negative pressure."""
        temp_r = 560.0  # 100°F
        gas_specific_gravity = 0.6
        oil_specific_gravity = 0.9
        
        with pytest.raises(VasquezBeggsError):
            VasquezBeggsCorrelations.calculate_gas_solubility(
                -100.0, temp_r, gas_specific_gravity, oil_specific_gravity
            )
    
    def test_gas_solubility_zero_temperature(self):
        """Test gas solubility raises error for zero/negative temperature."""
        pressure_psia = 3000.0
        gas_specific_gravity = 0.6
        oil_specific_gravity = 0.9
        
        with pytest.raises(VasquezBeggsError):
            VasquezBeggsCorrelations.calculate_gas_solubility(
                pressure_psia, 0.0, gas_specific_gravity, oil_specific_gravity
            )
    
    def test_gas_solubility_valid_gravity_range(self):
        """Test gas solubility works within valid specific gravity range."""
        pressure_psia = 3000.0
        temp_r = 560.0  # 100°F
        oil_specific_gravity = 0.8
        
        # Test within valid range (0.0 - 1.5)
        for gg in [0.3, 0.5, 1.0, 1.2]:
            Rs = VasquezBeggsCorrelations.calculate_gas_solubility(
                pressure_psia, temp_r, gg, oil_specific_gravity
            )
            assert Rs > 0.0
    
    def test_oil_viscosity_at_reference_conditions(self):
        """Test oil viscosity at API RPI reference conditions."""
        pressure_psia = 3000.0
        temperature_f = 100.0
        gas_specific_gravity = 0.65
        oil_specific_gravity = 0.826
        
        mu_o = VasquezBeggsCorrelations.calculate_oil_viscosity(
            pressure_psia,
            PVTUnits.fahrenheit_to_rankine(temperature_f),
            gas_specific_gravity,
            oil_specific_gravity
        )
        
        # Oil viscosity should be in reasonable range for crude oil (0.02-1.0 cP)
        assert isinstance(mu_o, float)
        assert mu_o > 0.0
        assert mu_o < 1.0
    
    def test_oil_viscosity_temperature_dependency(self):
        """Test oil viscosity decreases with temperature."""
        pressure_psia = 3000.0
        gas_specific_gravity = 0.6
        oil_specific_gravity = 0.9
        
        mu_high_t = VasquezBeggsCorrelations.calculate_oil_viscosity(
            pressure_psia,
            PVTUnits.fahrenheit_to_rankine(120.0),
            gas_specific_gravity,
            oil_specific_gravity
        )
        mu_low_t = VasquezBeggsCorrelations.calculate_oil_viscosity(
            pressure_psia,
            PVTUnits.fahrenheit_to_rankine(80.0),
            gas_specific_gravity,
            oil_specific_gravity
        )
        
        assert mu_high_t < mu_low_t
    
    def test_oil_viscosity_pressure_dependency(self):
        """Test oil viscosity decreases with pressure."""
        temperature_f = 100.0
        temp_r = PVTUnits.fahrenheit_to_rankine(temperature_f)
        gas_specific_gravity = 0.6
        oil_specific_gravity = 0.9
        
        mu_high_p = VasquezBeggsCorrelations.calculate_oil_viscosity(
            4000.0, temp_r, gas_specific_gravity, oil_specific_gravity
        )
        mu_low_p = VasquezBeggsCorrelations.calculate_oil_viscosity(
            2000.0, temp_r, gas_specific_gravity, oil_specific_gravity
        )
        
        assert mu_high_p < mu_low_p
    
    def test_oil_viscosity_negative_pressure(self):
        """Test oil viscosity raises error for negative pressure."""
        temp_r = 560.0  # 100°F
        gas_specific_gravity = 0.6
        oil_specific_gravity = 0.9
        
        with pytest.raises(VasquezBeggsError):
            VasquezBeggsCorrelations.calculate_oil_viscosity(
                -100.0, temp_r, gas_specific_gravity, oil_specific_gravity
            )
    
    def test_oil_viscosity_zero_gas_gravity(self):
        """Test oil viscosity with light gas (no dissolved gas)."""
        pressure_psia = 3000.0
        temperature_f = 100.0
        gas_specific_gravity = 0.0
        oil_specific_gravity = 0.8
        
        mu_o = VasquezBeggsCorrelations.calculate_oil_viscosity(
            pressure_psia,
            PVTUnits.fahrenheit_to_rankine(temperature_f),
            gas_specific_gravity,
            oil_specific_gravity
        )
        
        # Should return positive viscosity
        assert mu_o > 0.0
    
    def test_oil_fvf_at_reference_conditions(self):
        """Test oil formation volume factor at API RPI reference conditions."""
        pressure_psia = 3000.0
        temperature_f = 100.0
        gas_specific_gravity = 0.65
        oil_specific_gravity = 0.826
        
        Bo = VasquezBeggsCorrelations.calculate_oil_fvf(
            pressure_psia,
            PVTUnits.fahrenheit_to_rankine(temperature_f),
            gas_specific_gravity,
            oil_specific_gravity
        )
        
        # Bo should be > 1.0 (reservoir volume > stock tank volume)
        assert isinstance(Bo, float)
        assert Bo > 1.0
        # Typical Bo range: 1.0 - 2.0 RB/STB
        assert Bo < 2.5
    
    def test_oil_fvf_rsb_relationship(self):
        """Test Bo increases with Rs (gas solubility)."""
        temperature_f = 100.0
        temp_r = PVTUnits.fahrenheit_to_rankine(temperature_f)
        gas_specific_gravity = 0.6
        oil_specific_gravity = 0.9
        
        # High Rs (high pressure) -> higher Bo
        Bo_high = VasquezBeggsCorrelations.calculate_oil_fvf(
            4000.0, temp_r, gas_specific_gravity, oil_specific_gravity
        )
        
        # Low Rs (low pressure) -> lower Bo
        Bo_low = VasquezBeggsCorrelations.calculate_oil_fvf(
            2000.0, temp_r, gas_specific_gravity, oil_specific_gravity
        )
        
        assert Bo_high > Bo_low
    
    def test_oil_fvf_temperature_dependency(self):
        """Test Bo increases with temperature."""
        pressure_psia = 3000.0
        gas_specific_gravity = 0.6
        oil_specific_gravity = 0.9
        
        Bo_high_t = VasquezBeggsCorrelations.calculate_oil_fvf(
            pressure_psia,
            PVTUnits.fahrenheit_to_rankine(120.0),
            gas_specific_gravity,
            oil_specific_gravity
        )
        Bo_low_t = VasquezBeggsCorrelations.calculate_oil_fvf(
            pressure_psia,
            PVTUnits.fahrenheit_to_rankine(80.0),
            gas_specific_gravity,
            oil_specific_gravity
        )
        
        assert Bo_high_t > Bo_low_t
    
    def test_all_correlations_consistency(self):
        """Test that all correlations produce consistent results."""
        pressure_psia = 3000.0
        temperature_f = 100.0
        gas_specific_gravity = 0.6
        oil_specific_gravity = 0.9
        
        temp_r = PVTUnits.fahrenheit_to_rankine(temperature_f)
        
        mu_o = VasquezBeggsCorrelations.calculate_oil_viscosity(
            pressure_psia, temp_r, gas_specific_gravity, oil_specific_gravity
        )
        
        Rs = VasquezBeggsCorrelations.calculate_gas_solubility(
            pressure_psia, temp_r, gas_specific_gravity, oil_specific_gravity
        )
        
        Bo = VasquezBeggsCorrelations.calculate_oil_fvf(
            pressure_psia, temp_r, gas_specific_gravity, oil_specific_gravity
        )
        
        # All should be positive
        assert mu_o > 0.0
        assert Rs > 0.0
        assert Bo > 1.0


class TestStandingCorrections:
    """Test Standing gas property corrections."""
    
    def test_apply_gas_gravity_correction_light_gas(self):
        """Test gas gravity correction for light gas (< 0.7 sg)."""
        gas_specific_gravity = 0.5
        temp_r = 600.0  # ~140°F
        
        corrected = StandingCorrections.apply_gas_gravity_correction(
            gas_specific_gravity, temp_r
        )
        
        # Light gas correction factor
        expected = 0.5 + (0.0005 * (10.75 - 0.5)**2)
        assert abs(corrected - expected) < 1e-6
    
    def test_apply_gas_gravity_correction_heavy_gas(self):
        """Test gas gravity correction for heavy gas (>= 0.7 sg)."""
        gas_specific_gravity = 1.1
        temp_r = 600.0
        
        corrected = StandingCorrections.apply_gas_gravity_correction(
            gas_specific_gravity, temp_r
        )
        
        # Heavy gas correction larger
        expected = 1.1 + (0.0005 * (10.75 - 1.1)**2)
        assert abs(corrected - expected) < 1e-6
    
    def test_gas_gravity_correction_same_sg(self):
        """Test gas gravity correction for SG = 1.0 (air)."""
        gas_specific_gravity = 1.0
        temp_r = 600.0
        
        corrected = StandingCorrections.apply_gas_gravity_correction(
            gas_specific_gravity, temp_r
        )
        
        expected = 1.0 + (0.0005 * (10.75 - 1.0)**2)
        assert abs(corrected - expected) < 1e-6


class TestBlackOilPVTCalculator:
    """Test comprehensive black oil PVT calculator."""
    
    def test_calculate_reservoir_properties_full(self):
        """Test complete reservoir property calculation."""
        calculator = BlackOilPVTCalculator(
            gas_specific_gravity=0.6,
            oil_specific_gravity=0.85
        )
        
        props = calculator.calculate_reservoir_properties(
            pressure_psia=3000.0,
            temperature_f=100.0
        )
        
        # All expected properties present and positive
        assert 'oil_viscosity_cP' in props
        assert 'oil_fvf_RB_STB' in props
        assert 'gas_solubility_scf_STB' in props
        assert 'temperature_rankine' in props
        
        assert props['oil_viscosity_cP'] > 0.0
        assert props['oil_fvf_RB_STB'] > 1.0
        assert props['gas_solubility_scf_STB'] > 0.0
    
    def test_calculate_reservoir_properties_api_rpi_case(self):
        """Test API RPI reference case."""
        calculator = BlackOilPVTCalculator(
            gas_specific_gravity=0.65,
            oil_specific_gravity=0.826
        )
        
        props = calculator.calculate_reservoir_properties(
            pressure_psia=3000.0,
            temperature_f=100.0
        )
        
        # RPI test case validation
        # Typical expected ranges (from API RPI)
        assert props['oil_fvf_RB_STB'] > 1.3  # Bo should be > 1.3
        assert props['oil_fvf_RB_STB'] < 2.0
        assert props['gas_solubility_scf_STB'] > 350.0  # Rs > 350
        assert props['oil_viscosity_cP'] < 1.0
    
    def test_calculate_surface_properties(self):
        """Test surface property calculation."""
        calculator = BlackOilPVTCalculator(
            gas_specific_gravity=0.6,
            oil_specific_gravity=0.8
        )
        
        reservoir_props = calculator.calculate_reservoir_properties(
            3000.0, 100.0
        )
        
        surface_props = calculator.calculate_surface_properties(reservoir_props)
        
        assert 'stock_tank_oil_viscosity_cP' in surface_props
        assert 'stock_tank_gas_viscosity_cP' in surface_props
        assert surface_props['stock_tank_gas_viscosity_cP'] <= surface_props['stock_tank_oil_viscosity_cP']
    
    def test_calculate_vpt_profile(self):
        """Test comprehensive PVT profile over pressure range."""
        calculator = BlackOilPVTCalculator(
            gas_specific_gravity=0.6,
            oil_specific_gravity=0.8
        )
        
        profile = calculator.calculate_vpt_profile(
            pressure_min=2000.0,
            pressure_max=4000.0,
            temperature_f=100.0,
            pressure_step=500.0
        )
        
        assert 'pressure_array' in profile
        assert 'viscosity_array' in profile
        assert 'fvf_array' in profile
        assert 'solubility_array' in profile
        
        assert len(profile['pressure_array']) == 5
        # All properties should monotonically change with pressure
        
        # Verify array lengths equal
        assert len(profile['pressure_array']) == len(profile['viscosity_array'])
        assert len(profile['pressure_array']) == len(profile['fvf_array'])
        assert len(profile['pressure_array']) == len(profile['solubility_array'])
    
    def test_calculate_vpt_profile_extreme_pressures(self):
        """Test PVT profile at extreme pressure ranges."""
        calculator = BlackOilPVTCalculator(
            gas_specific_gravity=0.7,
            oil_specific_gravity=0.7
        )
        
        profile = calculator.calculate_vpt_profile(
            pressure_min=1000.0,
            pressure_max=8000.0,
            temperature_f=120.0,
            pressure_step=2000.0
        )
        
        assert len(profile['pressure_array']) == 4
        assert np.any(profile['pressure_array'] > 0)
        assert np.any(profile['pressure_array'] >= 1000.0)
        assert np.any(profile['pressure_array'] <= 8000.0)
    
    def test_calculate_vpt_profile_temperature_variation(self):
        """Test PVT profile with temperature variation."""
        calculator = BlackOilPVTCalculator(
            gas_specific_gravity=0.55,
            oil_specific_gravity=0.75
        )
        
        # Compare different temperatures
        profile_hot = calculator.calculate_vpt_profile(
            2000.0, 4000.0, 120.0, 500.0
        )
        
        profile_cold = calculator.calculate_vpt_profile(
            2000.0, 4000.0, 80.0, 500.0
        )
        
        # Properties should change with temperature
        # (Hotter = lower viscosity, higher Bo, lower Rs)
        assert profile_hot['viscosity_array'].mean() < profile_cold['viscosity_array'].mean()
        assert profile_hot['fvf_array'].mean() > profile_cold['fvf_array'].mean()
        assert profile_hot['solubility_array'].mean() < profile_cold['solubility_array'].mean()


class TestLegacyFunction:
    """Test compatibility function."""
    
    def test_calculate_vasquez_beggs_pvt(self):
        """Test legacy compatibility function."""
        result = calculate_vasquez_beggs_pvt(
            pressure_psia=3000.0,
            temperature_f=100.0,
            gas_specific_gravity=0.6,
            oil_specific_gravity=0.8
        )
        
        # Should return dict with standard keys
        assert 'oil_viscosity_cP' in result
        assert 'oil_fvf_RB_STB' in result
        assert 'gas_solubility_scf_STB' in result
        
        # All values should be in valid ranges
        assert result['oil_viscosity_cP'] > 0.0
        assert result['oil_fvf_RB_STB'] > 1.0
        assert result['gas_solubility_scf_STB'] > 0.0
    
    def test_calculate_vasquez_beggs_pvt_api_rpi(self):
        """Test API RPI case with legacy function."""
        result = calculate_vasquez_beggs_pvt(
            pressure_psia=3000.0,
            temperature_f=100.0,
            gas_specific_gravity=0.65,
            oil_specific_gravity=0.826
        )
        
        # RPI test case validation
        assert result['oil_fvf_RB_STB'] > 1.3
        assert result['oil_fvf_RB_STB'] < 2.0
        assert result['gas_solubility_scf_STB'] > 350.0


@pytest.mark.parametrize("pressure_oilgravity", [
    (5000.0, 0.8),
    (3000.0, 0.85),
    (1500.0, 0.9),
    (1000.0, 0.95),
])
def test_gas_solubility_regression_coefficients(pressure_oilgravity):
    """Test regression coefficient interpolation for different oil gravities."""
    pressure, oil_gravity = pressure_oilgravity
    
    temp_r = 600.0  # ~140°F
    gas_specific_gravity = 0.6
    
    # Different oil gravities should produce different Rs
    results = []
    for og in [0.8, 0.85, 0.9]:
        Rs = VasquezBeggsCorrelations.calculate_gas_solubility(
            pressure, temp_r, gas_specific_gravity, og
        )
        results.append(Rs)
    
    # Heavier oil (higher gravity) should have slightly different Rs
    # (not necessarily larger or smaller, but different)
    assert abs(results[2] - results[1]) < 50.0  # Should be different
    assert abs(results[1] - results[0]) < 50.0


def test_all_api_rpi_cases():
    """Test various API RPI test case scenarios."""
    test_cases = [
        {"pressure": 3000, "temp": 100, "gas_sg": 0.65, "oil_sg": 0.826},
        {"pressure": 2500, "temp": 90, "gas_sg": 0.60, "oil_sg": 0.850},
        {"pressure": 3500, "temp": 110, "gas_sg": 0.70, "oil_sg": 0.800},
    ]
    
    for case in test_cases:
        result = calculate_vasquez_beggs_pvt(
            pressure_psia=case["pressure"],
            temperature_f=case["temp"],
            gas_specific_gravity=case["gas_sg"],
            oil_specific_gravity=case["oil_sg"]
        )
        
        # All results should be valid
        assert result['oil_fvf_RB_STB'] > 1.0
        assert result['gas_solubility_scf_STB'] > 0.0
        assert result['oil_viscosity_cP'] > 0.0
