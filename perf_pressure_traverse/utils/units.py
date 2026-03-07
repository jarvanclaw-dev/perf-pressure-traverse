"""Unit conversion utilities for API and SI units."""

from __future__ import annotations

from typing import Union
import math


# Physical constants
FT_TO_M = 0.3048  # 1 foot = 0.3048 meters
PSI_TO_PA = 6894.75729  # 1 psi = 6894.75729 pascals
RANKINE_TO_KELVIN = 5.0 / 9.0  # 1 Rankine = 0.555555... K
RANKINE_TO_CELSIUS = 459.67  # Freezing point of water in Rankine (0°C = 491.67°R)
KELVIN_TO_RANKINE = RANKINE_TO_KELVIN  # Direct inverse


class UnitConversionError(ValueError):
    """Raised when converting between incompatible units."""
    pass


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


# Conversion shortcuts
def convert_length(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert length between API and SI units.
    
    Supported units: 'ft' (feet), 'm' (meters)
    
    Parameters
    ----------
    value : float
        Value to convert.
    from_unit : str
        Source unit label ('ft' or 'm').
    to_unit : str
        Target unit label ('ft' or 'm').
    
    Returns
    -------
    float
        Converted value.
    
    Raises
    ------
    UnitConversionError
        If units are incompatible.
    ValueError
        If units are not recognized.
    """
    units = ['ft', 'm']
    
    if from_unit not in units:
        raise ValueError(f"Unknown length unit: {from_unit}")
    if to_unit not in units:
        raise ValueError(f"Unknown length unit: {to_unit}")
    if from_unit == to_unit:
        return value
    
    validate_unit_pair(from_unit, to_unit, [('ft', 'm')])
    
    if from_unit == 'ft':
        return ft_to_m(value)
    else:
        return m_to_ft(value)


def convert_pressure(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert pressure between API and SI units.
    
    Supported units: 'psi', 'Pa', 'Pascal'
    
    Parameters
    ----------
    value : float
        Value to convert.
    from_unit : str
        Source unit label ('psi', 'Pa', 'Pascal').
    to_unit : str
        Target unit label ('psi', 'Pa', 'Pascal').
    
    Returns
    -------
    float
        Converted value.
    
    Raises
    ------
    UnitConversionError
        If units are incompatible.
    ValueError
        If units are not recognized.
    """
    # Alias Pa to Pascal for easier use
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    
    # Normalize Pa to Pascal
    if from_unit == 'pa':
        from_unit = 'pascal'
    if to_unit == 'pa':
        to_unit = 'pascal'
    
    units = ['psi', 'pascal']
    
    if from_unit not in units:
        raise ValueError(f"Unknown pressure unit: {from_unit}")
    if to_unit not in units:
        raise ValueError(f"Unknown pressure unit: {to_unit}")
    if from_unit == to_unit:
        return value
    
    validate_unit_pair(from_unit, to_unit, [('psi', 'pascal')])
    
    if from_unit == 'psi':
        return psi_to_pa(value)
    else:
        return pa_to_psi(value)


def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert temperature between API relative units (°F), SI relative units (°C),
    and absolute units (K, °R).
    
    Supported relative units: '°F', 'F', 'fahrenheit', 'celsius', 'C', '°C'
    Supported absolute units: 'K', 'kelvin', 'rankine', '°R', 'R', 'RANKINE'
    
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
    UnitConversionError
        If units are incompatible.
    ValueError
        If units are not recognized.
    """
    # Normalize relative units
    from_unit_lower = from_unit.lower()
    to_unit_lower = to_unit.lower()
    
    # Handle °F variations
    if from_unit_lower in ['f', 'fahrenheit']:
        from_unit = '°F'
    elif from_unit_lower in ['c', 'celsius', '°c']:
        from_unit = '°C'
    # Handle absolute units
    elif from_unit_lower in ['k', 'kelvin']:
        from_unit = 'K'
    elif from_unit_lower in ['r', 'rankine', '°r']:
        from_unit = '°R'
    else:
        raise ValueError(f"Unknown temperature unit: {from_unit}")
    
    if to_unit_lower in ['f', 'fahrenheit']:
        to_unit = '°F'
    elif to_unit_lower in ['c', 'celsius', '°c']:
        to_unit = '°C'
    elif to_unit_lower in ['k', 'kelvin']:
        to_unit = 'K'
    elif to_unit_lower in ['r', 'rankine', '°r']:
        to_unit = '°R'
    else:
        raise ValueError(f"Unknown temperature unit: {to_unit}")
    
    if from_unit == to_unit:
        return value
    
    # Relative units (°F ↔ °C)
    if set([from_unit, to_unit]) == {'°F', '°C'}:
        if from_unit == '°F':
            return fahrenheit_to_celsius(value)
        else:
            return celsius_to_fahrenheit(value)
    
    # Absolute units (K ↔ °R)
    elif set([from_unit, to_unit]) == {'K', '°R'}:
        if from_unit == 'K':
            return kelvin_to_rankine(value)
        else:
            return rankine_to_kelvin(value)
    
    # Cross-conversions require intermediate conversion
    else:
        if from_unit == '°F':
            kelvin = fahrenheit_to_kelvin(value)
            if to_unit == '°C':
                return kelvin_to_celsius(kelvin)
            else:  # to_unit is '°R'
                return kelvin_to_rankine(kelvin)
        elif from_unit == '°C':
            kelvin = celsius_to_kelvin(value)
            if to_unit == '°F':
                return kelvin_to_fahrenheit(kelvin)
            else:  # to_unit is '°R'
                return kelvin_to_rankine(kelvin)
        elif from_unit == 'K':
            if to_unit == '°F':
                return kelvin_to_fahrenheit(value)
            else:  # to_unit is '°C' or '°R'
                rankine = kelvin_to_rankine(value)
                if to_unit == '°C':
                    return rankine_to_celsius(rankine)
                else:  # to_unit is '°R'
                    return rankine
        elif from_unit == '°R':
            kelvin = rankine_to_kelvin(value)
            if to_unit == '°F':
                return kelvin_to_fahrenheit(kelvin)
            else:  # to_unit is '°C'
                return rankine_to_celsius(kelvin)


def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """Convert °F to °C."""
    return (fahrenheit - 32.0) * (5.0 / 9.0)


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert °C to °F."""
    return (celsius * 9.0 / 5.0) + 32.0


def celsius_to_kelvin(celsius: float) -> float:
    """Convert °C to K."""
    return celsius + 273.15


def kelvin_to_celsius(kelvin: float) -> float:
    """Convert K to °C."""
    return kelvin - 273.15


__all__ = [
    'ft_to_m',
    'm_to_ft',
    'psi_to_pa',
    'pa_to_psi',
    'fahrenheit_to_kelvin',
    'kelvin_to_fahrenheit',
    'rankine_to_kelvin',
    'kelvin_to_rankine',
    'convert_length',
    'convert_pressure',
    'convert_temperature',
    'UnitConversionError',
]
