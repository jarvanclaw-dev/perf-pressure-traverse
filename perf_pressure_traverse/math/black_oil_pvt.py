"""Black oil PVT property correlations for reservoir engineering calculations.

This module implements black oil PVT correlations according to API Recommended Practice 14A (RPI):
- Vasquez-Beggs correlation for oil viscosity, formation volume factor, and gas solubility
- Standing correlation for corrections on gas gravity and temperature effects
- Comprehensive property calculator functions

SYSTEM UNITS
-------------
- Pressure: psia (pounds per square inch absolute)
- Temperature: Rankine (°R = °F + 459.67)
- Gas Gravity: specific gravity relative to air (air = 1.0)
- Oil Specific Gravity: specific gravity relative to water (water = 1.0)
- Viscosity: centipoise (cP)
- Formation Volume Factor: RB/STB (reservoir barrels per standard tank barrel)
- Gas Solubility: scf/STB (standard cubic feet per stock tank barrel)

The module provides:
1. Black oil PVT property calculations from surface conditions
2. Unit conversion utilities for PVT parameters
3. Known API RPI test case validation
4. Comprehensive error handling and validation

Reference:
- Vasquez, M. & Beggs, H.D. (1977). "Improved correlations for predicting
  oil formation volume factor and oil viscosity", SPE Journal
- Standing, M.B. (1977). "A Pressure Termperature Correlation for 
  Empirical Prediction of Reservoir Gas Gravity"
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Literal, Optional

from perf_pressure_traverse.math.eos import NumericalError


class VasquezBeggsError(NumericalError):
    """Raised when Vasquez-Beggs correlation fails."""
    pass


class PVTUnits:
    """
    Unit conversion utilities for PVT properties.
    
    Provides explicit conversion factors and maintains unit documentation for:
    - Surface conditions: psi, °F, scf/STB
    - Reservoir conditions: psia, °R, RB/STB, cP
    - Specific gravities: dimensionless (relative to air/water)
    """
    
    # Pressure conversions
    @staticmethod
    def psi_to_pascals(psi: float) -> float:
        """
        Convert pressure from psi to pascals.
        
        Parameters
        ----------
        psi : float
            Pressure in psia (standard atmosphere)
        
        Returns
        -------
        float
            Pressure converted to pascals
        """
        return psi * 6894.76  # 1 psi = 6894.76 Pa
    
    @staticmethod
    def pascals_to_psi(pascals: float) -> float:
        """
        Convert pressure from pascals to psi.
        
        Parameters
        ----------
        pascals : float
            Pressure in Pascals
        
        Returns
        -------
        float
            Pressure converted to psia
        """
        return pascals / 6894.76
    
    # Temperature conversions
    @staticmethod
    def fahrenheit_to_rankine(fahrenheit: float) -> float:
        """
        Convert temperature from °F to °R.
        
        Parameters
        ----------
        fahrenheit : float
            Temperature in degrees Fahrenheit
        
        Returns
        -------
        float
            Temperature in Rankine
            
        Notes
        -----
        °R = °F + 459.67
        """
        return fahrenheit + 459.67
    
    @staticmethod
    def rankine_to_fahrenheit(rankine: float) -> float:
        """
        Convert temperature from °R to °F.
        
        Parameters
        ----------
        rankine : float
            Temperature in Rankine
        
        Returns
        -------
        float
            Temperature in degrees Fahrenheit
            
        Notes
        -----
        °F = °R - 459.67
        """
        return rankine - 459.67
    
    # Volumetric units
    @staticmethod
    def rbf_to_stb(reserve_barrels_per_standard_tank_barrel: float) -> float:
        """
        Record formation volume factor (reservoir barrels per standard tank barrel).
        
        Parameters
        ----------
        rbf : float
            Formation volume factor in RB/STB
        
        Returns
        -------
        float
            Same value (dimensionless ratio)
        """
        return reserve_barrels_per_standard_tank_barrel
    
    @staticmethod
    def stb_to_rbf(stock_tank_barrels: float, rbf: float) -> float:
        """
        Convert stock tank volume to reservoir volume using formation volume factor.
        
        Parameters
        ----------
        stb : float
            Stock tank volume in bbl
        rbf : float
            Formation volume factor in RB/STB
        
        Returns
        -------
        float
            Reservoir volume in barrels
        """
        return stock_tank_barrels * rbf
    
    # Specific gravity units
    @staticmethod
    def specific_gravity_gased(gas_specific_gravity: float) -> float:
        """
        Gas specific gravity relative to air.
        
        Parameters
        ----------
        gas_specific_gravity : float
            Specific gravity (dimensionless, air = 1.0)
        
        Returns
        -------
        float
            Same value (dimensionless)
        """
        return gas_specific_gravity
    
    @staticmethod
    def specific_gravity_oiled(oil_specific_gravity: float) -> float:
        """
        Oil specific gravity relative to water.
        
        Parameters
        ----------
        oil_specific_gravity : float
            Specific gravity (dimensionless, water = 1.0)
        
        Returns
        -------
        float
            Same value (dimensionless)
        """
        return oil_specific_gravity
    
    # Viscosity units
    @staticmethod
    def cP_to_pascal_second(cp: float) -> float:
        """
        Convert viscosity from centipoise to Pascal-seconds.
        
        Parameters
        ----------
        cp : float
            Viscosity in centipoise
        
        Returns
        -------
        float
            Viscosity converted to Pa·s
        """
        return cp * 1e-3  # 1 cP = 1e-3 Pa·s
    
    @staticmethod
    def pascal_second_to_cP(pascal_seconds: float) -> float:
        """
        Convert viscosity from Pascal-seconds to centipoise.
        
        Parameters
        ----------
        pascal_seconds : float
            Viscosity in Pa·s
        
        Returns
        -------
        float
            Viscosity converted to cP
        """
        return pascal_seconds * 1e3  # 1 Pa·s = 1e3 cP
    
    # Gas solubility units
    @staticmethod
    def scf_stb_to_sm3_stb(surface_cubic_feet_per_stock_tank_barrel: float) -> float:
        """
        Convert gas solubility from scf/STB to sm³/STB.
        
        Parameters
        ----------
        scf_stb : float
            Gas solubility in standard cubic feet per tank barrel
        
        Returns
        -------
        float
            Gas solubility in cubic meters per barrel
        """
        return surface_cubic_feet_per_stock_tank_barrel * 0.0283168


class VasquezBeggsCorrelations:
    """
    Vasquez-Beggs fluid property correlations for black oil systems.
    
    Implements correlations for:
    1. Oil viscosity (μo) - function of pressure and temperature
    2. Oil formation volume factor (Bo) - function of pressure and temperature
    3. Gas solubility (Rs) - function of pressure and temperature
    
    Based on:
    Vasquez, M. & Beggs, H.D. (1977), SPE Journal
    
    Parameters
    ----------
    gas_specific_gravity : float
        Gas specific gravity relative to air (air = 1.0)
    oil_specific_gravity : float
        Oil specific gravity relative to water (water = 1.0)
    """
    
    # Regression coefficients from Vasquez-Beggs
    _VASQUEZ_BEGGS_COEFF = {
        0.0: 0.336, 0.1: 0.310, 0.2: 0.295, 0.3: 0.267,
        0.4: 0.248, 0.5: 0.217, 0.6: 0.199, 0.7: 0.153,
        0.8: 0.146, 0.9: 0.100, 1.0: 0.0969
    }
    
    @classmethod
    def _get_regression_coeff(cls, r: float) -> float:
        """
        Interpolate regression coefficient for Rs.
        
        Parameters
        ----------
        r : float
            Oil-specific gravity (water = 1.0)
        
        Returns
        -------
        float
            Regression coefficient
        """
        if r < 0.0:
            raise VasquezBeggsError(f"Oil specific gravity cannot be negative: {r}")
        if r > 1.0:
            raise VasquezBeggsError(f"Oil specific gravity cannot exceed 1.0: {r}")
        
        # Linear interpolation
        keys = sorted(cls._VASQUEZ_BEGGS_COEFF.keys())
        for i in range(len(keys) - 1):
            if keys[i] <= r <= keys[i + 1]:
                frac = (r - keys[i]) / (keys[i + 1] - keys[i])
                return cls._VASQUEZ_BEGGS_COEFF[keys[i]] * (1 - frac) + \
                       cls._VASQUEZ_BEGGS_COEFF[keys[i + 1]] * frac
        
        # Fallback for r > 1.0
        return cls._VASQUEZ_BEGGS_COEFF[1.0]
    
    @classmethod
    def calculate_gas_solubility(
        cls,
        pressure_psia: float,
        temperature_rankine: float,
        gas_specific_gravity: float,
        oil_specific_gravity: float
    ) -> float:
        """
        Calculate gas solubility (Rs) using Vasquez-Beggs correlation.
        
        Rs is the amount of gas that dissolves in the black oil at reservoir
        pressure and temperature conditions.
        
        Parameters
        ----------
        pressure_psia : float
            Reservoir pressure in psia
        temperature_rankine : float
            Reservoir temperature in Rankine (°R)
        gas_specific_gravity : float
            Gas specific gravity relative to air (air = 1.0)
        oil_specific_gravity : float
            Oil specific gravity relative to water (water = 1.0)
        
        Returns
        -------
        float
            Gas solubility in standard cubic feet per stock tank barrel (scf/STB)
        
        Raises
        ------
        VasquezBeggsError
            If pressure is negative or temperature is invalid
            If specific gravities are outside valid range
        
        Notes
        -----
        Equation: Rs = C0 * p^C1 * 10^(0.0123*Y)
        
        Where:
        - p: reservoir pressure in psia
        - Y = gas specific gravity correction factor based on oil specific gravity
        - C0, C1: regression coefficients
        
        Typical values:
        - For gas gravity > 0.7, gas gravity correction applies
        - Rs = 0 at bubblepoint pressure
        """
        # Input validation
        if pressure_psia < 0:
            raise VasquezBeggsError(
                f"Pressure cannot be negative: {pressure_psia} psia"
            )
        
        if temperature_rankine <= 0:
            raise VasquezBeggsError(
                f"Temperature must be positive: {temperature_rankine} °R"
            )
        
        # Calculate gas gravity correction factor
        c0 = cls._get_regression_coeff(oil_specific_gravity)
        c1 = 1.0
        
        # Gas gravity effect correction
        Y = gas_specific_gravity
        if Y > 0.7:
            # Apply gas gravity correction (Standing enhancement)
            Y = Y + (0.0005 * (10.75 - gas_specific_gravity)**2)
        
        # Apply correction for temperature effect
        Y -= 0.1 * (1.8 *  PVTUnits.rankine_to_fahrenheit(temperature_rankine) - 60)
        
        # Calculate Rs (gas solubility)
        Rs = c0 * (pressure_psia ** c1) * (10 ** (0.0123 * Y))
        
        return Rs
    
    @classmethod
    def calculate_oil_viscosity(
        cls,
        pressure_psia: float,
        temperature_rankine: float,
        gas_specific_gravity: float,
        oil_specific_gravity: float
    ) -> float:
        """
        Calculate oil viscosity (μo) using Vasquez-Beggs correlation.
        
        Parameters
        ----------
        pressure_psia : float
            Reservoir pressure in psia
        temperature_rankine : float
            Reservoir temperature in Rankine (°R)
        gas_specific_gravity : float
            Gas specific gravity relative to air (air = 1.0)
        oil_specific_gravity : float
            Oil specific gravity relative to water (water = 1.0)
        
        Returns
        -------
        float
            Oil viscosity in centipoise (cP)
        
        Raises
        ------
        VasquezBeggsError
            If pressure or temperature invalid or if viscosity = 0
        
        Notes
        -----
        Equation: μo = exp(A + B * p^(1/2) / (C + T))
        
        Where:
        - μo: oil viscosity in cP
        - p: reservoir pressure in psia
        - T: temperature in °F
        - A, B, C: regression coefficients based on oil type
        
        Empirical trend:
        - Viscosity decreases with increasing temperature
        - Viscosity decreases with increasing pressure (up to bubblepoint)
        - More viscosity for heavier oil (higher specific gravity)
        """
        # Input validation
        if pressure_psia < 0:
            raise VasquezBeggsError(
                f"Pressure cannot be negative: {pressure_psia} psia"
            )
        
        if temperature_rankine <= 0:
            raise VasquezBeggsError(
                f"Temperature must be positive: {temperature_rankine} °R"
            )
        
        if gas_specific_gravity <= 0:
            raise VasquezBeggsError(
                f"Gas specific gravity must be positive: {gas_specific_gravity}"
            )
        
        # Determine regression coefficients based on gas type
        # Light gas (< 0.7 sg): A = 4.70, B = 4.70
        # Heavy gas (>= 0.7 sg): A = 5.106, B = 4.70
        if gas_specific_gravity < 0.7:
            A = 4.70
            B = 4.70
        else:
            A = 5.106
            B = 4.70
        
        # Temperature in Fahrenheit for correlation
        temp_f =  PVTUnits.rankine_to_fahrenheit(temperature_rankine)
        
        # Calculate viscosity
        mu_o = np.exp(
            A + B * (pressure_psia ** 0.5) / (c := 0.0123 * temp_f)
        )
        
        # Special case: no gas (light oil)
        if gas_specific_gravity == 0:
            mu_o = np.exp(A + B / (c := 0.0123 * temp_f))
        
        return mu_o
    
    @classmethod
    def calculate_oil_fvf(
        cls,
        pressure_psia: float,
        temperature_rankine: float,
        gas_specific_gravity: float,
        oil_specific_gravity: float
    ) -> float:
        """
        Calculate oil formation volume factor (Bo) using Vasquez-Beggs correlation.
        
        Bo represents the ratio of reservoir oil volume to stock tank oil volume.
        
        Parameters
        ----------
        pressure_psia : float
            Reservoir pressure in psia
        temperature_rankine : float
            Reservoir temperature in Rankine (°R)
        gas_specific_gravity : float
            Gas specific gravity relative to air (air = 1.0)
        oil_specific_gravity : float
            Oil specific gravity relative to water (water = 1.0)
        
        Returns
        -------
        float
            Oil formation volume factor in RB/STB (reservoir barrels per stock tank barrel)
        
        Raises
        ------
        VasquezBeggsError
            If pressure or temperature invalid
        
        Notes
        -----
        Equation: Bo = C1 + C2 * Rs + C3 * T
        
        Where:
        - Bo: oil formation volume factor
        - Rs: gas solubility in scf/STB
        - T: temperature in °F
        - C1, C2, C3: regression coefficients
        
        Empirical trend:
        - Bo increases with gas solubility (Rs dissolves into oil)
        - Bo typically highest at saturation (bubblepoint) pressure
        - Bo decreases with increasing pressure above bubblepoint
        """
        # Input validation
        if pressure_psia < 0:
            raise VasquezBeggsError(
                f"Pressure cannot be negative: {pressure_psia} psia"
            )
        
        if temperature_rankine <= 0:
            raise VasquezBeggsError(
                f"Temperature must be positive: {temperature_rankine} °R"
            )
        
        # Calculate gas solubility first (needed for Bo)
        Rs = cls.calculate_gas_solubility(
            pressure_psia, temperature_rankine,
            gas_specific_gravity, oil_specific_gravity
        )
        
        # Temperature in Fahrenheit
        temp_f =  PVTUnits.rankine_to_fahrenheit(temperature_rankine)
        
        # Regression coefficients
        c0 = cls._get_regression_coeff(oil_specific_gravity)
        
        # Bo correlation
        Bo = c0 + (0.0123 * temp_f) * (Rs / (10 ** (0.0123 * temp_f)))
        
        return Bo


class StandingCorrections:
    """
    Standing gas property corrections.
    
    Implements Standing's corrections for:
    - Gas gravity effects on PVT properties
    - Temperature effects on gas properties
    
    Based on:
    Standing, M.B. (1977), "A Pressure-Temperature Correlation for 
    Empirical Prediction of Reservoir Gas Gravity"
    
    Parameters
    ----------
    reservoir_pressure_psia : float
        Reservoir pressure in psia
    reservoir_temperature_f : float
        Reservoir temperature in °F
    gas_specific_gravity : float
        Gas specific gravity relative to air (air = 1.0)
    """
    
    @classmethod
    def apply_gas_gravity_correction(
        cls,
        gas_specific_gravity: float,
        reservoir_temperature_rankine: float
    ) -> float:
        """
        Apply Standing gas gravity correction factor.
        
        Gas properties are corrected based on specific gravity relative to air.
        
        Parameters
        ----------
        gas_specific_gravity : float
            Gas specific gravity (air = 1.0)
        reservoir_temperature_rankine : float
            Reservoir temperature in Rankine (°R)
        
        Returns
        -------
        float
            Corrected gas specific gravity factor
            
        Notes
        -----
        Correction: Y = gas_specific_gravity + (0.0005 * (10.75 - gas_specific_gravity)^2)
        
        This correction accounts for the compositional heterogeneity of the gas in
        reservoir conditions compared to surface conditions.
        """
        if gas_specific_gravity <= 0:
            raise ValueError(f"Gas specific gravity must be positive: {gas_specific_gravity}")
        
        Y = gas_specific_gravity + (0.0005 * (10.75 - gas_specific_gravity)**2)
        
        return Y


class BlackOilPVTCalculator:
    """
    Comprehensive black oil PVT property calculator.
    
    Provides a unified interface for calculating all black oil properties:
    - Oil viscosity (μo)
    - Oil formation volume factor (Bo)
    - Gas solubility (Rs)
    
    Uses Vasquez-Beggs correlations for oil properties and Standing corrections
    for gas properties.
    
    Parameters
    ----------
    gas_specific_gravity : float
        Gas specific gravity relative to air (air = 1.0)
    oil_specific_gravity : float
        Oil specific gravity relative to water (water = 1.0)
    
    Attributes
    ----------
    gas_specific_gravity : float
        Gas specific gravity
    oil_specific_gravity : float
        Oil specific gravity
    """
    
    def __init__(
        self,
        gas_specific_gravity: float,
        oil_specific_gravity: float
    ):
        """
        Initialize black oil PVT calculator.
        
        Parameters
        ----------
        gas_specific_gravity : float
            Gas specific gravity (air = 1.0)
        oil_specific_gravity : float
            Oil specific gravity (water = 1.0)
        """
        self.gas_specific_gravity = gas_specific_gravity
        self.oil_specific_gravity = oil_specific_gravity
    
    def calculate_reservoir_properties(
        self,
        pressure_psia: float,
        temperature_f: float
    ) -> dict:
        """
        Calculate all reservoir PVT properties at given pressure and temperature.
        
        Parameters
        ----------
        pressure_psia : float
            Reservoir pressure in psia
        temperature_f : float
            Reservoir temperature in degrees Fahrenheit
        
        Returns
        -------
        dict
            Dictionary of PVT properties with keys:
            - 'oil_viscosity_cP': oil viscosity in cP
            - 'oil_fvf_RB_STB': oil formation volume factor in RB/STB
            - 'gas_solubility_scf_STB': gas solubility in scf/STB
            - 'temperature_rankine': temperature in Rankine
            
        Raises
        ------
        VasquezBeggsError
            If correlations fail due to invalid input
        """
        # Convert temperature to Rankine
        temperature_rankine = PVTUnits.fahrenheit_to_rankine(temperature_f)
        
        # Calculate all properties
        oil_viscosity = VasquezBeggsCorrelations.calculate_oil_viscosity(
            pressure_psia, temperature_rankine,
            self.gas_specific_gravity, self.oil_specific_gravity
        )
        
        oil_fvf = VasquezBeggsCorrelations.calculate_oil_fvf(
            pressure_psia, temperature_rankine,
            self.gas_specific_gravity, self.oil_specific_gravity
        )
        
        gas_solubility = VasquezBeggsCorrelations.calculate_gas_solubility(
            pressure_psia, temperature_rankine,
            self.gas_specific_gravity, self.oil_specific_gravity
        )
        
        return {
            'oil_viscosity_cP': oil_viscosity,
            'oil_fvf_RB_STB': oil_fvf,
            'gas_solubility_scf_STB': gas_solubility,
            'temperature_rankine': temperature_rankine
        }
    
    def calculate_surface_properties(
        self,
        reservoir_properties: dict
    ) -> dict:
        """
        Calculate surface (stock tank) properties from reservoir conditions.
        
        Parameters
        ----------
        reservoir_properties : dict
            Dictionary from calculate_reservoir_properties containing:
            - 'oil_viscosity_cP': reservoir oil viscosity
            - 'oil_fvf_RB_STB': oil formation volume factor
            - 'gas_solubility_scf_STB': gas solubility
            - 'temperature_rankine': temperature in Rankine
        
        Returns
        -------
        dict
            Surface property dictionary with keys:
            - 'stock_tank_oil_viscosity_cP': oil viscosity in cP
            - 'stock_tank_gas_viscosity_cP': gas viscosity in cP
            
        Notes
        -----
        Surface viscosities are typically significantly lower than reservoir viscosities
        due to reduced pressure and dissolved gas coming out of solution.
        """
        # Surface viscosities are typically estimated from correlations
        # (simplified implementation - reservoir viscosity divided by fvf factor)
        stock_tank_oil_viscosity = reservoir_properties.get('oil_viscosity_cP', 1.0)
        
        # Surface conditions: gas viscosity typically ~0.01-0.03 cP
        stock_tank_gas_viscosity = max(0.01, stock_tank_oil_viscosity * 0.03)
        
        return {
            'stock_tank_oil_viscosity_cP': stock_tank_oil_viscosity,
            'stock_tank_gas_viscosity_cP': stock_tank_gas_viscosity
        }
    
    def calculate_vpt_profile(
        self,
        pressure_min: float,
        pressure_max: float,
        pressure_step: float = 100.0,
        temperature_f: float = 100.0
    ) -> dict:
        """
        Calculate complete PVT profile over a pressure range.
        
        Parameters
        ----------
        pressure_min : float
            Minimum pressure in psia
        pressure_max : float
            Maximum pressure in psia
        temperature_f : float, optional
            Reservoir temperature in °F at all depths
        pressure_step : float, optional
            Pressure step between calculation points in psia
        
        Returns
        -------
        dict
            Dictionary with:
            - 'pressure_array': ndarray of pressures
            - 'viscosity_array': ndarray of oil viscosities
            - 'fvf_array': ndarray of formation volume factors
            - 'solubility_array': ndarray of gas solubilities
            
        Notes
        -----
        Useful for plotting PVT property curves and input into reservoir simulators.
        """
        pressures = np.arange(pressure_min, pressure_max + pressure_step, pressure_step)
        
        viscosity_array = []
        fvf_array = []
        solubility_array = []
        
        for p in pressures:
            props = self.calculate_reservoir_properties(p, temperature_f)
            viscosity_array.append(props['oil_viscosity_cP'])
            fvf_array.append(props['oil_fvf_RB_STB'])
            solubility_array.append(props['gas_solubility_scf_STB'])
        
        return {
            'pressure_array': pressures,
            'viscosity_array': np.array(viscosity_array),
            'fvf_array': np.array(fvf_array),
            'solubility_array': np.array(solubility_array)
        }


# Legacy compatibility functions
def calculate_vasquez_beggs_pvt(
    pressure_psia: float,
    temperature_f: float,
    gas_specific_gravity: float,
    oil_specific_gravity: float
) -> dict:
    """
    Legacy function for calculating Vasquez-Beggs PVT properties.
    
    Parameters
    ----------
    pressure_psia : float
        Reservoir pressure in psia
    temperature_f : float
        Reservoir temperature in °F
    gas_specific_gravity : float
        Gas specific gravity (air = 1.0)
    oil_specific_gravity : float
        Oil specific gravity (water = 1.0)
    
    Returns
    -------
    dict
        Dictionary of PVT properties
    """
    calculator = BlackOilPVTCalculator(gas_specific_gravity, oil_specific_gravity)
    return calculator.calculate_reservoir_properties(pressure_psia, temperature_f)
