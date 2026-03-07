"""Unit tests for unit conversion utilities."""

import pytest
from perf_pressure_traverse.utils.units import (
    ft_to_m,
    m_to_ft,
    psi_to_pa,
    pa_to_psi,
    fahrenheit_to_kelvin,
    kelvin_to_fahrenheit,
    rankine_to_kelvin,
    kelvin_to_rankine,
    convert_length,
    convert_pressure,
    convert_temperature,
    fahrenheit_to_celsius,
    celsius_to_fahrenheit,
    celsius_to_kelvin,
    kelvin_to_celsius,
    UnitConversionError,
    validate_unit_pair,
)


class TestLengthConversions:
    """Test length conversions between feet and meters."""
    
    @pytest.mark.parametrize("ft,m", [
        (1.0, 0.3048),
        (0.0, 0.0),
        (100.0, 30.48),
        (10.5, 3.2004),
        (123.456, 37.6007728),
        (0.3048, 0.1),
    ])
    def test_ft_to_m_accurate(self, ft, m):
        """Test ft to m conversion is physically accurate."""
        result = ft_to_m(ft)
        # Allow 0.01% accuracy
        tolerance = abs(m) * 0.0001
        assert abs(result - m) <= tolerance or (
            abs(m) < 1e-6 and abs(result) < 1e-10
        ), f"ft_to_m({ft}) = {result}, expected {m}"
    
    @pytest.mark.parametrize("m,ft", [
        (1.0, 3.280839895),
        (0.0, 0.0),
        (30.48, 100.0),
        (3.2004, 10.5),
        (37.6007728, 123.456),
        (0.1, 0.32808399),
    ])
    def test_m_to_ft_accurate(self, m, ft):
        """Test m to ft conversion is physically accurate."""
        result = m_to_ft(m)
        # Allow 0.01% accuracy
        tolerance = abs(ft) * 0.0001
        assert abs(result - ft) <= tolerance or (
            abs(ft) < 1e-6 and abs(result) < 1e-10
        ), f"m_to_ft({m}) = {result}, expected {ft}"
    
    @pytest.mark.parametrize("value, from_unit, to_unit", [
        (1.0, 'ft', 'm'),
        (1.0, 'm', 'ft'),
        (100.0, 'ft', 'm'),
        (30.48, 'm', 'ft'),
    ])
    def test_convert_length(self, value, from_unit, to_unit):
        """Test convert_length function."""
        result = convert_length(value, from_unit, to_unit)
        
        if from_unit == 'ft' and to_unit == 'm':
            expected = ft_to_m(value)
        elif from_unit == 'm' and to_unit == 'ft':
            expected = m_to_ft(value)
        else:
            expected = value
        
        assert abs(result - expected) < 1e-9, \
            f"convert_length failed: got {result}, expected {expected}"
    
    def test_convert_length_same_unit(self):
        """Test convert_length with same units."""
        assert convert_length(10.0, 'ft', 'ft') == 10.0
        assert convert_length(20.0, 'm', 'm') == 20.0
    
    def test_convert_length_invalid_unit(self):
        """Test convert_length with invalid units."""
        with pytest.raises(ValueError, match="Unknown length unit"):
            convert_length(1.0, 'km', 'm')
        
        with pytest.raises(ValueError, match="Unknown length unit"):
            convert_length(1.0, 'ft', 'km')
    
    def test_convert_length_invalid_pair(self):
        """Test convert_length with incompatible units."""
        with pytest.raises(UnitConversionError, match="Incompatible units"):
            convert_length(1.0, 'ft', 'psi')
    
    def test_length_negative_values(self):
        """Test ft_to_m and m_to_ft with negative values."""
        with pytest.raises(ValueError, match="cannot be negative"):
            ft_to_m(-1.0)
        
        with pytest.raises(ValueError, match="cannot be negative"):
            m_to_ft(-1.0)


class TestPressureConversions:
    """Test pressure conversions between psi and pascals."""
    
    @pytest.mark.parametrize("psi,pa", [
        (1.0, 6894.75729),
        (0.0, 0.0),
        (14.7, 101325.24),
        (100.0, 689475.729),
        (1000.0, 6894757.29),
        (6894.75729, 100000.0),
    ])
    def test_psi_to_pa_accurate(self, psi, pa):
        """Test psi to pa conversion is physically accurate."""
        result = psi_to_pa(psi)
        # Allow 0.01% accuracy
        tolerance = abs(pa) * 0.0001
        assert abs(result - pa) <= tolerance or (
            abs(pa) < 1e-6 and abs(result) < 1e-10
        ), f"psi_to_pa({psi}) = {result}, expected {pa}"
    
    @pytest.mark.parametrize("pa,psi", [
        (6894.75729, 1.0),
        (0.0, 0.0),
        (101325.24, 14.7),
        (689475.729, 100.0),
        (6894757.29, 1000.0),
    ])
    def test_pa_to_psi_accurate(self, pa, psi):
        """Test pa to psi conversion is physically accurate."""
        result = pa_to_psi(pa)
        # Allow 0.01% accuracy
        tolerance = abs(psi) * 0.0001
        assert abs(result - psi) <= tolerance or (
            abs(psi) < 1e-6 and abs(result) < 1e-10
        ), f"pa_to_psi({pa}) = {result}, expected {psi}"
    
    @pytest.mark.parametrize("value, from_unit, to_unit", [
        (1.0, 'psi', 'Pa'),
        (1.0, 'psi', 'Pascal'),
        (1.0, 'Pa', 'psi'),
        (1.0, 'Pa', 'Pascal'),
        (100.0, 'psi', 'Pa'),
        (14.7, 'Pa', 'psi'),
    ])
    def test_convert_pressure(self, value, from_unit, to_unit):
        """Test convert_pressure function."""
        result = convert_pressure(value, from_unit, to_unit)
        
        if from_unit == 'psi' and to_unit in ['Pa', 'Pascal']:
            expected = psi_to_pa(value)
        elif from_unit in ['Pa', 'Pascal'] and to_unit == 'psi':
            expected = pa_to_psi(value)
        else:
            expected = value
        
        assert abs(result - expected) < 0.01, \
            f"convert_pressure failed: got {result}, expected {expected}"
    
    def test_convert_pressure_same_unit(self):
        """Test convert_pressure with same units."""
        assert convert_pressure(100.0, 'psi', 'psi') == 100.0
        assert convert_pressure(200.0, 'Pa', 'Pa') == 200.0
    
    def test_convert_pressure_invalid_unit(self):
        """Test convert_pressure with invalid units."""
        with pytest.raises(ValueError, match="Unknown pressure unit"):
            convert_pressure(1.0, 'bar', 'psi')
        
        with pytest.raises(ValueError, match="Unknown pressure unit"):
            convert_pressure(1.0, 'psi', 'bar')
    
    def test_convert_pressure_invalid_pair(self):
        """Test convert_pressure with incompatible units."""
        with pytest.raises(UnitConversionError, match="Incompatible units"):
            convert_pressure(1.0, 'psi', 'ft')
    
    def test_pressure_negative_values(self):
        """Test psi_to_pa and pa_to_psi with negative values."""
        with pytest.raises(ValueError, match="cannot be negative"):
            psi_to_pa(-1.0)
        
        with pytest.raises(ValueError, match="cannot be negative"):
            pa_to_psi(-1.0)
    
    def test_pressure_zero(self):
        """Test conversion with zero pressure."""
        assert psi_to_pa(0.0) == 0.0
        assert pa_to_psi(0.0) == 0.0


class TestTemperatureConversions:
    """Test temperature conversions."""
    
    def test_fahrenheit_to_kelvin_accurate(self):
        """Test °F to K conversion is physically accurate."""
        # Example: 32°F = 0°C = 273.15K
        self.assertAlmostEqual_with_tolerance(
            fahrenheit_to_kelvin(32.0), 273.15, 0.0001
        )
        # Example: 212°F = 100°C = 373.15K
        self.assertAlmostEqual_with_tolerance(
            fahrenheit_to_kelvin(212.0), 373.15, 0.0001
        )
        # Example: 68°F = 20°C = 293.15K
        self.assertAlmostEqual_with_tolerance(
            fahrenheit_to_kelvin(68.0), 293.15, 0.0001
        )
    
    def test_kelvin_to_fahrenheit_accurate(self):
        """Test K to °F conversion is physically accurate."""
        # Example: 273.15K = 0°C = 32°F
        self.assertAlmostEqual_with_tolerance(
            kelvin_to_fahrenheit(273.15), 32.0, 0.0001
        )
        # Example: 373.15K = 100°C = 212°F
        self.assertAlmostEqual_with_tolerance(
            kelvin_to_fahrenheit(373.15), 212.0, 0.0001
        )
        # Example: 293.15K = 20°C = 68°F
        self.assertAlmostEqual_with_tolerance(
            kelvin_to_fahrenheit(293.15), 68.0, 0.0001
        )
    
    def test_rankine_to_kelvin_accurate(self):
        """Test °R to K conversion is physically accurate."""
        # Freezing water: 491.67°R = 273.15K
        self.assertAlmostEqual_with_tolerance(
            rankine_to_kelvin(491.67), 273.15, 0.0001
        )
        # Boiling water: 671.67°R = 373.15K
        self.assertAlmostEqual_with_tolerance(
            rankine_to_kelvin(671.67), 373.15, 0.0001
        )
    
    def test_kelvin_to_rankine_accurate(self):
        """Test K to °R conversion is physically accurate."""
        # Freezing water: 273.15K = 491.67°R
        self.assertAlmostEqual_with_tolerance(
            kelvin_to_rankine(273.15), 491.67, 0.0001
        )
        # Boiling water: 373.15K = 671.67°R
        self.assertAlmostEqual_with_tolerance(
            kelvin_to_rankine(373.15), 671.67, 0.0001
        )
    
    @pytest.mark.parametrize("value, from_unit, to_unit", [
        # °F to °C conversions
        (32.0, '°F', '°C'),
        (68.0, '°F', '°C'),
        (212.0, '°F', '°C'),
        # °C to °F conversions
        (0.0, '°C', '°F'),
        (20.0, '°C', '°F'),
        (100.0, '°C', '°F'),
        # °F to K conversions
        (32.0, '°F', 'K'),
        (68.0, '°F', 'K'),
        # K to °F conversions
        (273.15, 'K', '°F'),
        (293.15, 'K', '°F'),
        # K to °C conversions
        (273.15, 'K', '°C'),
        (293.15, 'K', '°C'),
        # °C to K conversions
        (0.0, '°C', 'K'),
        (20.0, '°C', 'K'),
        # °R to K conversions
        (491.67, '°R', 'K'),
        (671.67, '°R', 'K'),
        # K to °R conversions
        (273.15, 'K', '°R'),
        (373.15, 'K', '°R'),
    ])
    def test_convert_temperature(self, value, from_unit, to_unit):
        """Test convert_temperature function."""
        result = convert_temperature(value, from_unit, to_unit)
        
        # Re-convert back to verify roundtrip
        back = convert_temperature(result, to_unit, from_unit)
        
        assert abs(result - value) <= 1e-9, \
            f"convert_temperature failed: got {result}, expected {value}"
        assert abs(back - value) <= 1e-9, \
            f"Roundtrip conversion failed: {to_unit} to {from_unit}: got {back}, expected {value}"
        
        # Check accuracy
        back = convert_temperature(result, to_unit, from_unit)
        assert abs(result - value) <= abs(value) * 0.0001, \
            f"Accuracy violation: {value} {from_unit} -> {result} {to_unit}"
    
    def test_convert_temperature_with_aliases(self):
        """Test convert_temperature with common unit aliases."""
        # Test °F variations
        self.assertAlmostEqual_with_tolerance(
            convert_temperature(32.0, 'f', 'fahrenheit'), 32.0, 1e-9
        )
        # Test K variations
        self.assertAlmostEqual_with_tolerance(
            convert_temperature(273.15, 'k', 'kelvin'), 273.15, 1e-9
        )
        # Test °R variations
        self.assertAlmostEqual_with_tolerance(
            convert_temperature(491.67, 'r', 'rankine'), 491.67, 1e-9
        )
    
    def test_convert_temperature_same_unit(self):
        """Test convert_temperature with same units."""
        assert convert_temperature(100.0, '°F', '°F') == 100.0
        assert convert_temperature(200.0, 'K', 'K') == 200.0
        assert convert_temperature(300.0, '°R', '°R') == 300.0
    
    def test_convert_temperature_invalid_unit(self):
        """Test convert_temperature with invalid units."""
        with pytest.raises(ValueError, match="Unknown temperature unit"):
            convert_temperature(100.0, 'deg', '°F')
        
        with pytest.raises(ValueError, match="Unknown temperature unit"):
            convert_temperature(100.0, '°F', 'deg')
    
    def test_temperature_negative_values(self):
        """Test temperature conversions with negative values."""
        # Below absolute zero should raise error
        with pytest.raises(ValueError, match="cannot be below absolute zero"):
            fahrenheit_to_kelvin(-500.0)
        
        with pytest.raises(ValueError, match="cannot be below absolute zero"):
            kelvin_to_fahrenheit(-1.0)
        
        with pytest.raises(ValueError, match="cannot be below absolute zero"):
            rankine_to_kelvin(-100.0)
    
    def test_temperature_absolute_zero(self):
        """Test exact conversion at absolute zero."""
        self.assertAlmostEqual_with_tolerance(
            fahrenheit_to_kelvin(-459.67), 0.0, 1e-9
        )
        self.assertAlmostEqual_with_tolerance(
            kelvin_to_fahrenheit(0.0), -459.67, 1e-9
        )
        self.assertAlmostEqual_with_tolerance(
            rankine_to_kelvin(0.0), 0.0, 1e-9
        )
        self.assertAlmostEqual_with_tolerance(
            kelvin_to_rankine(0.0), 0.0, 1e-9
        )
    
    @staticmethod
    def assertAlmostEqual_with_tolerance(actual, expected, tolerance):
        """Helper assertion method with 0.01% tolerance."""
        assert abs(actual - expected) <= abs(expected) * 0.0001, \
            f"Expected {expected}, got {actual} (tolerance: {abs(expected) * 0.0001})"


class TestUnitConversionError:
    """Test UnitConversionError."""
    
    def test_unit_conversion_error(self):
        """Test UnitConversionError can be raised."""
        with pytest.raises(UnitConversionError, match="Incompatible units"):
            validate_unit_pair('ft', 'psi', [('ft', 'm')])
    
    def test_error_message(self):
        """Test UnitConversionError message."""
        with pytest.raises(UnitConversionError, match="ft and psi cannot be converted"):
            validate_unit_pair('ft', 'psi', [])


class TestAdditionalHelpers:
    """Test helper temperature conversion functions."""
    
    def test_fahrenheit_to_celsius(self):
        """Test °F to °C conversion."""
        # Water freezing
        self.assertAlmostEqual_with_tolerance(fahrenheit_to_celsius(32.0), 0.0, 0.0001)
        # Water boiling
        self.assertAlmostEqual_with_tolerance(fahrenheit_to_celsius(212.0), 100.0, 0.0001)
        # Room temp
        self.assertAlmostEqual_with_tolerance(fahrenheit_to_celsius(68.0), 20.0, 0.0001)
    
    def test_celsius_to_fahrenheit(self):
        """Test °C to °F conversion."""
        # Water freezing
        self.assertAlmostEqual_with_tolerance(celsius_to_fahrenheit(0.0), 32.0, 0.0001)
        # Water boiling
        self.assertAlmostEqual_with_tolerance(celsius_to_fahrenheit(100.0), 212.0, 0.0001)
        # Room temp
        self.assertAlmostEqual_with_tolerance(celsius_to_fahrenheit(20.0), 68.0, 0.0001)
    
    def test_celsius_to_kelvin(self):
        """Test °C to K conversion."""
        self.assertAlmostEqual_with_tolerance(celsius_to_kelvin(0.0), 273.15, 0.0001)
        self.assertAlmostEqual_with_tolerance(celsius_to_kelvin(100.0), 373.15, 0.0001)
    
    def test_kelvin_to_celsius(self):
        """Test K to °C conversion."""
        self.assertAlmostEqual_with_tolerance(kelvin_to_celsius(273.15), 0.0, 0.0001)
        self.assertAlmostEqual_with_tolerance(kelvin_to_celsius(373.15), 100.0, 0.0001)
