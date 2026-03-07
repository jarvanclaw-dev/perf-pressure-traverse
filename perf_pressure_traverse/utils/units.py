"""Unit conversion utilities for API and SI units."""

from __future__ import annotations

from typing import Union
import math


# Physical constants
FT_TO_M = 0.3048  # 1 foot = 0.3048 meters
PSI_TO_PA = 6894.75729  # 1 psi = 6894.75729 pascals
RANKINE_TO_KELVIN = 5.0 / 9.0  # 1 Rankine = 0.555555... K
RANKINE_TO_CELSIUS = 491.67  # Freezing point of water in Rankine (0°C = 491.67°R)
KELVIN_TO_RANKINE = RANKINE_TO_KELVIN  # Direct inverse


class UnitConversionError(ValueError):
    """Raised when converting between incompatible units."""
    pass


def validate_unit_pair(
    unit1: str,
    unit2: str,
    allowed_pairs: list[tuple[str, str]]
) -> None:
    """
    Validate that two units can be converted between each other.
    
    Parameters
    ----------
    unit1 : str
        First unit label.
    unit2 : str
        Second unit label.
    allowed_pairs : list of tuples
        List of valid pairs (unit1, unit2) that allow conversion.
        Each unit in a pair can convert to both directions.
    
    Raises
    ------
    UnitConversionError
        If the units are not compatible.
    
    Examples
    --------
    >>> validate_unit_pair('ft', 'm', [('ft', 'm')])
    # No error
    
    >>> validate_unit_pair('ft', 'psi', [('ft', 'm')])
    UnitConversionError: Incompatible units: ft and psi cannot be converted
    """
    if (unit1, unit2) not in allowed_pairs and (unit2, unit1) not in allowed_pairs:
        raise UnitConversionError(
            f"Incompatible units: {unit1} and {unit2} cannot be converted"
        )


def ft_to_m(feet: float) -> float:
    """
    Convert length from API units (feet) to SI units (meters).
    
    Parameters
    ----------
    feet : float
        Length in feet.
    
    Returns
    -------
    float
        Length in meters.
    
    Examples
    --------
    >>> ft_to_m(1.0)
    0.3048
    >>> ft_to_m(100.0)
    30.48
    """
    if feet < 0:
        raise ValueError("Length cannot be negative")
    return feet * FT_TO_M


def m_to_ft(meters: float) -> float:
    """
    Convert length from SI units (meters) to API units (feet).
    
    Parameters
    ----------
    meters : float
        Length in meters.
    
    Returns
    -------
    float
        Length in feet.
    
    Examples
    --------
    >>> m_to_ft(0.3048)
    1.0
    >>> m_to_ft(30.48)
    100.0
    """
    if meters < 0:
        raise ValueError("Length cannot be negative")
    return meters / FT_TO_M


def psi_to_pa(psi: float) -> float:
    """
    Convert pressure from API units (psi) to SI units (pascals).
    
    Parameters
    ----------
    psi : float
        Pressure in pounds per square inch.
    
    Returns
    -------
    float
        Pressure in pascals.
    
    Examples
    --------
    >>> psi_to_pa(1.0)
    6894.75729
    >>> round(psi_to_pa(14.7), 2)
    101325.24
    """
    if psi < 0:
        raise ValueError("Pressure cannot be negative")
    return psi * PSI_TO_PA


def pa_to_psi(pascals: float) -> float:
    """
    Convert pressure from SI units (pascals) to API units (psi).
    
    Parameters
    ----------
    pascals : float
        Pressure in pascals.
    
    Returns
    -------
    float
        Pressure in pounds per square inch.
    
    Examples
    --------
    >>> round(pa_to_psi(6894.75729), 6)
    1.0
    >>> round(pa_to_psi(101325.24), 2)
    14.7
    """
    if pascals < 0:
        raise ValueError("Pressure cannot be negative")
    return pascals / PSI_TO_PA


def fahrenheit_to_kelvin(fahrenheit: float) -> float:
    """
    Convert temperature from API units (°F) to SI units (K).
    
    The formula is: K = (°F - 32) * 5/9 + 273.15
    
    Parameters
    ----------
    fahrenheit : float
        Temperature in degrees Fahrenheit.
    
    Returns
    -------
    float
        Temperature in kelvin.
    
    Examples
    --------
    >>> round(fahrenheit_to_kelvin(32.0), 2)
    273.15
    >>> round(fahrenheit_to_kelvin(212.0), 2)
    373.15
    """
    if fahrenheit < -459.67:
        raise ValueError("Temperature cannot be below absolute zero (-459.67°F)")
    return (fahrenheit - 32.0) * (5.0 / 9.0) + 273.15


def kelvin_to_fahrenheit(kelvin: float) -> float:
    """
    Convert temperature from SI units (K) to API units (°F).
    
    The formula is: °F = K * 9/5 - 459.67
    
    Parameters
    ----------
    kelvin : float
        Temperature in kelvin.
    
    Returns
    -------
    float
        Temperature in degrees Fahrenheit.
    
    Examples
    --------
    >>> round(kelvin_to_fahrenheit(273.15), 2)
    32.0
    >>> round(kelvin_to_fahrenheit(373.15), 2)
    212.0
    """
    if kelvin < 0:
        raise ValueError("Temperature cannot be below absolute zero (0 K)")
    return kelvin * (9.0 / 5.0) - 459.67


def rankine_to_kelvin(rankine: float) -> float:
    """
    Convert temperature from API absolute units (°R) to SI units (K).
    
    The formula is: K = °R * 5/9
    
    Parameters
    ----------
    rankine : float
        Temperature in degrees Rankine.
    
    Returns
    -------
    float
        Temperature in kelvin.
    
    Examples
    --------
    >>> round(rankine_to_kelvin(491.67), 2)
    273.15
    >>> round(rankine_to_kelvin(671.67), 2)
    373.15
    """
    if rankine < 0:
        raise ValueError("Temperature cannot be below absolute zero (0°R)")
    return rankine * RANKINE_TO_KELVIN


def kelvin_to_rankine(kelvin: float) -> float:
    """
    Convert temperature from SI units (K) to API absolute units (°R).
    
    The formula is: °R = K * 9/5
    
    Parameters
    ----------
    kelvin : float
        Temperature in kelvin.
    
    Returns
    -------
    float
        Temperature in degrees Rankine.
    
    Examples
    --------
    >>> round(kelvin_to_rankine(273.15), 2)
    491.67
    >>> round(kelvin_to_rankine(373.15), 2)
    671.67
    """
    if kelvin < 0:
        raise ValueError("Temperature cannot be below absolute zero (0 K)")
    return kelvin * (9.0 / 5.0)


def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """
    Convert temperature from API units (°F) to API units (°C).
    
    The formula is: °C = (°F - 32) * 5/9
    
    Parameters
    ----------
    fahrenheit : float
        Temperature in degrees Fahrenheit.
    
    Returns
    -------
    float
        Temperature in degrees Celsius.
    
    Examples
    --------
    >>> round(fahrenheit_to_celsius(32.0), 2)
    0.0
    >>> round(fahrenheit_to_celsius(212.0), 2)
    100.0
    """
    if fahrenheit < -459.67:
        raise ValueError("Temperature cannot be below absolute zero (-459.67°F)")
    return (fahrenheit - 32.0) * (5.0 / 9.0)


def celsius_to_fahrenheit(celsius: float) -> float:
    """
    Convert temperature from API units (°C) to API units (°F).
    
    The formula is: °F = (°C * 9/5) + 32
    
    Parameters
    ----------
    celsius : float
        Temperature in degrees Celsius.
    
    Returns
    -------
    float
        Temperature in degrees Fahrenheit.
    
    Examples
    --------
    >>> round(celsius_to_fahrenheit(0.0), 2)
    32.0
    >>> round(celsius_to_fahrenheit(100.0), 2)
    212.0
    """
    if celsius < -273.15:
        raise ValueError("Temperature cannot be below absolute zero (-273.15°C)")
    return (celsius * 9.0 / 5.0) + 32.0


def celsius_to_kelvin(celsius: float) -> float:
    """
    Convert temperature from API units (°C) to SI units (K).
    
    The formula is: K = °C + 273.15
    
    Parameters
    ----------
    celsius : float
        Temperature in degrees Celsius.
    
    Returns
    -------
    float
        Temperature in kelvin.
    
    Examples
    --------
    >>> celsius_to_kelvin(0.0)
    273.15
    >>> celsius_to_kelvin(100.0)
    373.15
    """
    if celsius < -273.15:
        raise ValueError("Temperature cannot be below absolute zero (-273.15°C)")
    return celsius + 273.15


def kelvin_to_celsius(kelvin: float) -> float:
    """
    Convert temperature from SI units (K) to API units (°C).
    
    The formula is: °C = K - 273.15
    
    Parameters
    ----------
    kelvin : float
        Temperature in kelvin.
    
    Returns
    -------
    float
        Temperature in degrees Celsius.
    
    Examples
    --------
    >>> round(kelvin_to_celsius(273.15), 2)
    0.0
    >>> round(kelvin_to_celsius(373.15), 2)
    100.0
    """
    if kelvin < 0:
        raise ValueError("Temperature cannot be below absolute zero (0 K)")
    return kelvin - 273.15


def fahrenheit_to_rankine(fahrenheit: float) -> float:
    """
    Convert temperature from API units (°F) to API absolute units (°R).
    
    The formula is: °R = °F + 459.67
    
    Parameters
    ----------
    fahrenheit : float
        Temperature in degrees Fahrenheit.
    
    Returns
    -------
    float
        Temperature in degrees Rankine.
    
    Examples
    --------
    >>> round(fahrenheit_to_rankine(32.0), 2)
    491.67
    >>> round(fahrenheit_to_rankine(212.0), 2)
    671.67
    """
    if fahrenheit < -459.67:
        raise ValueError("Temperature cannot be below absolute zero (-459.67°F)")
    return fahrenheit + RANKINE_TO_CELSIUS


def rankine_to_fahrenheit(rankine: float) -> float:
    """
    Convert temperature from API absolute units (°R) to API units (°F).
    
    The formula is: °F = °R - 459.67
    
    Parameters
    ----------
    rankine : float
        Temperature in degrees Rankine.
    
    Returns
    -------
    float
        Temperature in degrees Fahrenheit.
    
    Examples
    --------
    >>> round(rankine_to_fahrenheit(491.67), 2)
    32.0
    >>> round(rankine_to_fahrenheit(671.67), 2)
    212.0
    """
    if rankine < 0:
        raise ValueError("Temperature cannot be below absolute zero (0°R)")
    return rankine - RANKINE_TO_CELSIUS


def celsius_to_rankine(celsius: float) -> float:
    """
    Convert temperature from API units (°C) to API absolute units (°R).
    
    The formula is: °R = (°C + 273.15) * 9/5
    
    Parameters
    ----------
    celsius : float
        Temperature in degrees Celsius.
    
    Returns
    -------
    float
        Temperature in degrees Rankine.
    
    Examples
    --------
    >>> round(celsius_to_rankine(0.0), 2)
    491.67
    >>> round(celsius_to_rankine(100.0), 2)
    671.67
    """
    if celsius < -273.15:
        raise ValueError("Temperature cannot be below absolute zero (-273.15°C)")
    return (celsius + 273.15) * (9.0 / 5.0)


def rankine_to_celsius(rankine: float) -> float:
    """
    Convert temperature from API absolute units (°R) to API units (°C).
    
    The formula is: °C = (°R - 491.67) * 5/9
    
    Parameters
    ----------
    rankine : float
        Temperature in degrees Rankine.
    
    Returns
    -------
    float
        Temperature in degrees Celsius.
    
    Examples
    --------
    >>> round(rankine_to_celsius(491.67), 2)
    0.0
    >>> round(rankine_to_celsius(671.67), 2)
    100.0
    """
    if rankine < 0:
        raise ValueError("Temperature cannot be below absolute zero (0°R)")
    return (rankine - RANKINE_TO_CELSIUS) * (5.0 / 9.0)


def convert_length(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert a quantity between length units.
    
    Supported units: 'ft', 'm', 'meter', 'meters', 'feet', 'foot'.
    
    Parameters
    ----------
    value : float
        Value to convert.
    from_unit : str
        Source unit label.
    to_unit : str
        Target unit label.
    
    Returns
    -------
    float
        Converted value.
    
    Raises
    ------
    ValueError
        If unit is not recognized.
    UnitConversionError
        If units are incompatible.
    """
    # Normalize unit labels
    from_unit_clean = from_unit.lower().strip()
    to_unit_clean = to_unit.lower().strip()
    
    # Convert to 'ft' and 'm' internally
    if from_unit_clean in ('ft', 'foot', 'feet'):
        if value < 0:
            raise ValueError("Length cannot be negative")
        value_ft = value
    elif from_unit_clean in ('m', 'meter', 'meters'):
        if value < 0:
            raise ValueError("Length cannot be negative")
        value_ft = m_to_ft(value)
    else:
        raise ValueError(f"Unknown length unit: {from_unit}")
    
    result_ft = 0.0
    if to_unit_clean in ('ft', 'foot', 'feet'):
        result_ft = value_ft
    elif to_unit_clean in ('m', 'meter', 'meters'):
        result_ft = ft_to_m(value_ft)
    else:
        raise ValueError(f"Unknown length unit: {to_unit}")
    
    return result_ft


def convert_pressure(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert a quantity between pressure units.
    
    Supported units: 'psi', 'Pa', 'Pascal', 'pascal', 'psia', 'psig'.
    
    Parameters
    ----------
    value : float
        Value to convert.
    from_unit : str
        Source unit label.
    to_unit : str
        Target unit label.
    
    Returns
    -------
    float
        Converted value.
    
    Raises
    ------
    ValueError
        If unit is not recognized.
    UnitConversionError
        If units are incompatible.
    """
    # Normalize unit labels
    from_unit_clean = from_unit.lower().strip()
    to_unit_clean = to_unit.lower().strip()
    
    # Convert to 'psi' and 'Pa' internally
    if from_unit_clean in ('psi', 'psia', 'psig'):
        if value < 0:
            raise ValueError("Pressure cannot be negative")
        value_psi = value
    elif from_unit_clean in ('pa', 'pascal'):
        if value < 0:
            raise ValueError("Pressure cannot be negative")
        value_psi = pa_to_psi(value)
    else:
        raise ValueError(f"Unknown pressure unit: {from_unit}")
    
    result_psi = 0.0
    if to_unit_clean in ('psi', 'psia', 'psig'):
        result_psi = value_psi
    elif to_unit_clean in ('pa', 'pascal'):
        result_psi = pa_to_psi(value_psi)
    else:
        raise ValueError(f"Unknown pressure unit: {to_unit}")
    
    return result_psi


def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert a quantity between temperature units.
    
    Supported units: 
    - 'f', 'F', 'fahrenheit', 'Fahrenheit', '°F', '^°F'
    - 'c', 'C', 'celsius', 'Celsius', '°C', '^°C'
    - 'k', 'K', 'kelvin', 'Kelvin', 'abs', 'abskelvin'
    - 'r', 'R', 'rankine', 'Rankine', '°R', '^°R', 'abs', 'absrankine'
    
    Parameters
    ----------
    value : float
        Value to convert.
    from_unit : str
        Source unit label.
    to_unit : str
        Target unit label.
    
    Returns
    -------
    float
        Converted value.
    
    Raises
    ------
    ValueError
        If unit is not recognized or value is below absolute zero.
    UnitConversionError
        If units are incompatible.
    """
    def normalize_temp_unit(unit: str) -> str:
        """Normalize temperature unit string to canonical form."""
        unit = unit.lower().strip()
        # Remove degree symbol if present
        unit = unit.replace('°', '').replace('^', '').strip()
        # Common aliases
        if unit in ('abs', 'abskelvin', 'absolute', 'abskelvindeg'):
            return 'K'
        if unit in ('absrankine', 'absrankinedeg'):
            return 'R'
        return unit
    
    from_unit_norm = normalize_temp_unit(from_unit)
    to_unit_norm = normalize_temp_unit(to_unit)
    
    # Convert to Celsius/Kelvin internally
    if from_unit_norm == 'f':
        if value < -459.67:
            raise ValueError("Temperature cannot be below absolute zero (-459.67°F)")
        temp_c = fahrenheit_to_celsius(value)
    elif from_unit_norm == 'c':
        if value < -273.15:
            raise ValueError("Temperature cannot be below absolute zero (-273.15°C)")
        temp_c = value
    elif from_unit_norm == 'k':
        if value < 0:
            raise ValueError("Temperature cannot be below absolute zero (0 K)")
        temp_c = kelvin_to_celsius(value)
    elif from_unit_norm == 'r':
        if value < 0:
            raise ValueError("Temperature cannot be below absolute zero (0°R)")
        temp_c = rankine_to_celsius(value)
    else:
        raise ValueError(f"Unknown temperature unit: {from_unit}")
    
    # Convert to target unit
    if to_unit_norm == 'f':
        result = celsius_to_fahrenheit(temp_c)
    elif to_unit_norm == 'c':
        result = temp_c
    elif to_unit_norm == 'k':
        result = celsius_to_kelvin(temp_c)
    elif to_unit_norm == 'r':
        result = celsius_to_rankine(temp_c)
    else:
        raise ValueError(f"Unknown temperature unit: {to_unit}")
    
    return result


__all__ = [
    'validate_unit_pair',
    'ft_to_m',
    'm_to_ft',
    'psi_to_pa',
    'pa_to_psi',
    'fahrenheit_to_kelvin',
    'kelvin_to_fahrenheit',
    'rankine_to_kelvin',
    'kelvin_to_rankine',
    'fahrenheit_to_celsius',
    'celsius_to_fahrenheit',
    'celsius_to_kelvin',
    'kelvin_to_celsius',
    'fahrenheit_to_rankine',
    'rankine_to_fahrenheit',
    'celsius_to_rankine',
    'rankine_to_celsius',
    'convert_length',
    'convert_pressure',
    'convert_temperature',
    'UnitConversionError',
]
