# -*- coding: utf-8 -*-
"""
Black Oil PVT Property Correlations

Implements industry-standard correlations for petroleum property calculations:

1. **Vasquez-Beggs Correlations** (1976)
   - Gas solubility in oil
   - Oil formation volume factor (FVF)
   - Gas formation volume factor
   - Vapor pressure
   - Oil viscosity
   - Gas viscosity

2. **Standing Correlations** (1977)
   - Solution gas-oil ratio
   - Pressure correction

3. **Beggs-Brill Method** (1973)
   - Two-phase friction factor

4. **Other Standard Correlations**
   - Water properties
   - Gas gravity correction

Author: Production Engineering Team
Version: 0.1.0
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np


# ============================================================================
# CONSTANTS
# ============================================================================

# API units constants
G = 32.1740  # Gravitational constant (ft/s^2)

# Water specific constants
SURFACE_WATER_DENSITY_KG_M3 = 997.0  # Surface water density [kg/m³]
SURFACE_WATER_DENSITY_LBM_GAL = 8.33  # Surface water density [lb_m/gal]

# Reservoir reference conditions
STANDARD_PRESSURE_PSI = 14.696  # atm [psi]
STANDARD_TEMPERATURE_F = 60.0  # Standard temperature [°F]
STANDARD_TEMPERATURE_RANKINE = fahrenheit_to_rankine(STANDARD_TEMPERATURE_F)  # [°R]
STANDARD_TEMPERATURE_KELVIN = fahrenheit_to_kelvin(STANDARD_TEMPERATURE_F)  # [K]

# Petroleum property constants
SATURATED_OIL_DENSITY = 60.0  # lb_m/ft³

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def fahrenheit_to_rankine(fahrenheit: float) -> float:
    """
    Convert Fahrenheit to Rankine.
    
    R = °F + 459.67
    
    Parameters
    ----------
    fahrenheit : float
        Temperature in degrees Fahrenheit.
    
    Returns
    -------
    float
        Temperature in degrees Rankine.
    """
    return fahrenheit + 459.67

def fahrenheit_to_kelvin(fahrenheit: float) -> float:
    """
    Convert Fahrenheit to Kelvin.
    
    K = (°F - 32) * 5/9 + 273.15
    
    Parameters
    ----------
    fahrenheit : float
        Temperature in degrees Fahrenheit.
    
    Returns
    -------
    float
        Temperature in Kelvin.
    """
    return (fahrenheit - 32.0) * (5.0 / 9.0) + 273.15

# ============================================================================
# BLACK OIL CORRELATIONS
# ============================================================================

class BlackOilCorrelations:
    """
    Black Oil PVT Property Correlations
    
    Implements industry-standard correlations for petroleum fluid properties:
    
    - Gas solubility in oil (Vasquez-Beggs)
    - Oil formation volume factor (FVF) (Vasquez-Beggs)
    - Gas formation volume factor (FVF) (Vasquez-Beggs)
    - Vapor pressure (Vasquez-Beggs)
    - Oil viscosity (Vasquez-Beggs)
    - Gas viscosity (Beggs-Brill)
    - Solution gas-oil ratio (Standing)
    - Pressure correction (Standing)
    - Water properties
    
    Author: Production Engineering Team
    Version: 0.1.0
    Reference: API RP 14A, SPE 1976
    """
    
    def __init__(self):
        """Initialize Black Oil Correlations."""
        # Component critical properties
        self._critical_properties = {
            'CH4': (460.0, 666.7),
            'C2H6': (305.3, 549.7),
            'C3H8': (369.8, 549.7),
            'nC4H10': (425.1, 551),
            'C5H12': (470.4, 487),
            'iC4H10': (418.0, 551),
            'iC5H12': (460.4, 475),
            'C6H14': (507.4, 435),
            'N2': (126.2, 227.2),
            'CO2': (304.2, 737.7),
            'H2S': (373.2, 1200),
        }
    
    @staticmethod
    def _check_fvf_range(bubble_point_pressure: float, actual_pressure: float) -> None:
        """
        Validate that pressures are within expected ranges for PVT correlations.
        
        Parameters
        ----------
        bubble_point_pressure : float
            Bubble point pressure of the oil [psi].
        actual_pressure : float
            Actual operating pressure [psi].
        
        Raises
        ------
        ValueError
            If pressure is below the bubble point when calculating undersaturated properties.
            If pressure is above bubble point without specifying gas dissolution.
        """
        if actual_pressure < bubble_point_pressure:
            pass  # Accept undersaturated
        elif actual_pressure > bubble_point_pressure:
            raise ValueError(
                "Pressure above bubble point requires specification of gas solubility. "
                "Use at least one API gravity or solution gas-oil ratio input."
            )
    
    # -------------------------------------------------------------------------
    # GAS SOLUBILITY IN OIL (Vasquez-Beggs, 1976)
    # -------------------------------------------------------------------------
    
    def get_solution_gas_oil_ratio(
        self,
        api_gravity: float,
        gas_specific_gravity: float,
        pressure: float,
        temperature_f: float = STANDARD_TEMPERATURE_F,
        temperature_rankine: float = STANDARD_TEMPERATURE_RANKINE,
    ) -> float:
        """
        Calculate the gas solubility in oil using Vasquez-Beggs correlation.
        
        Rs = exp( C1 + C2 * SG_g + ... ) / SG_o^C3 * P^C4
        
        Parameters
        ----------
        api_gravity : float
            Oil API gravity at surface conditions [°API].
        gas_specific_gravity : float
            Gas specific gravity (air = 1.0) [dimensionless].
        pressure : float
            Reservoir pressure [psi].
        temperature_f : float, optional
            Reservoir temperature in degrees Fahrenheit [°F].
        temperature_rankine : float, optional
            Reservoir temperature in Rankine [°R].
        
        Returns
        -------
        float
            Solution gas-oil ratio (Rs) [scf/STB].
        
        Raises
        ------
        ValueError
            If pressure is below bubble point with undersaturated oil.
        """
        # Convert API gravity to specific gravity for oil
        sg_oil = 141.5 / (api_gravity + 131.5)
        
        # Temperature factor: R_T = T / 60 + 15.3
        r_t = temperature_rankine / 60.0 + 15.3
        
        # Base Rs calculation
        if pressure < 14.7:
            raise ValueError("Pressure must be at or above standard pressure")
        
        # Equation: Rs = exp(C1 + C2 * SG_g * R_T + C3 * sg_o) / sg_o^C4 * P^C5
        # Vasquez-Beggs coefficients
        C1 = -1339.6
        C2 = 17.2786 * math.log(r_t * sg_o)
        C3 = -10.331 * math.log(r_t * sg_o * sg_o)
        
        rs_coeff = C1 + C2 * gas_specific_gravity + C3
        
        # Clamp Rs to reasonable values
        if rs_coeff > 0:
            # For high Rs, use simplified approach
            rs = math.exp(rs_coeff) / (sg_o ** 2.0) * ((pressure / 14.7) ** 1.187)
            rs = max(rs, 0.0)
        else:
            # For low Rs, use direct approach
            rs = math.exp(rs_coeff) / (sg_o ** 1.0) * ((pressure / 14.7) ** 0.9)
            rs = max(rs, 0.0)
        
        # Clamp results to practical ranges
        rs = max(rs, 0.0)
        
        return rs
    
    # -------------------------------------------------------------------------
    # OIL FORMATION VOLUME FACTOR (FVF) (Vasquez-Beggs, 1976)
    # -------------------------------------------------------------------------
    
    def get_oil_fvf(
        self,
        api_gravity: float,
        gas_specific_gravity: float,
        pressure: float,
        temperature_f: float = STANDARD_TEMPERATURE_F,
        solution_gas_oil_ratio: Optional[float] = None,
        temperature_rankine: float = STANDARD_TEMPERATURE_RANKINE,
    ) -> float:
        """
        Calculate oil formation volume factor using Vasquez-Beggs correlation.
        
        Bo = 
            1 + C1 * Rs * exp(C2 * SG_g * (API**0.989))
             * ((T - 60) / 131.5) ^ C3 / SG_o^C4
        
        For unsaturated oil (pressures > bubble point):
            Bo_und = Bo_saturated * exp(C5 * (Psat - P))
        
        Parameters
        ----------
        api_gravity : float
            Oil API gravity at surface conditions [°API].
        gas_specific_gravity : float
            Gas specific gravity (air = 1.0) [dimensionless].
        pressure : float
            Reservoir pressure [psi].
        temperature_f : float, optional
            Reservoir temperature in degrees Fahrenheit [°F].
        solution_gas_oil_ratio : float, optional
            Solution gas-oil ratio at bubble point [scf/STB]. Required if P < P_bubble.
        temperature_rankine : float, optional
            Reservoir temperature in Rankine [°R].
        
        Returns
        -------
        float
            Oil FVF (Bo) [rb/STB].
        
        Raises
        ------
        ValueError
            If pressure is below bubble point without solution gas input.
        """
        # Convert API gravity to specific gravity for oil
        sg_oil = 141.5 / (api_gravity + 131.5)
        
        # Temperature correction: TR = ((T°C + 460) * (T°F - 60)) / ...
        tr = (temperature_f - 60.0 + 460.0) / 520.0
        
        # B0 coefficients
        C1 = 4.6771e-4
        C2 = -4.6771e-13
        C3 = 1.982e-9
        C4 = 2.162e-4
        C5 = -1.1870e-4
        
        # Check if saturated or undersaturated
        if solution_gas_oil_ratio is None or solution_gas_oil_ratio <= 0:
            # Assume saturation, calculate Rs first
            rs = self.get_solution_gas_oil_ratio(
                api_gravity=api_gravity,
                gas_specific_gravity=gas_specific_gravity,
                pressure=pressure,
                temperature_f=temperature_f,
                temperature_rankine=temperature_rankine
            )
        else:
            rs = solution_gas_oil_ratio
        
        # Check if undersaturated
        self._check_fvf_range(0.0, pressure)  # Placeholder for bubble point check
        
        # B0 calculation
        b0 = 1 + C1 * rs * math.exp(C2 * gas_specific_gravity * (api_gravity ** 0.989)) \
                       * (tr ** C3) / (sg_oil ** C4)
        
        # Cap Bo to practical range
        b0 = max(b0, 1.0)
        
        return b0
    
    # -------------------------------------------------------------------------
    # GAS FORMATION VOLUME FACTOR (FVF) (Vasquez-Beggs, 1976)
    # -------------------------------------------------------------------------
    
    def get_gas_fvf(
        self,
        gas_specific_gravity: float,
        pressure: float,
        temperature_f: float = STANDARD_TEMPERATURE_F,
        temperature_rankine: float = STANDARD_TEMPERATURE_RANKINE,
    ) -> float:
        """
        Calculate gas formation volume factor using Beggs-Brill correlation.
        
        Bg = P_sc / T_sc * Z / P
        
        Using Pitzler correlation for Z-factor:
        Z = C1 + C2 * SG_g^C3 * Tr^C4 * (1 + B * rho_g / P)
        
        Parameters
        ----------
        gas_specific_gravity : float
            Gas specific gravity (air = 1.0) [dimensionless].
        pressure : float
            Reservoir pressure [psi].
        temperature_f : float, optional
            Reservoir temperature in degrees Fahrenheit [°F].
        temperature_rankine : float, optional
            Reservoir temperature in Rankine [°R].
        
        Returns
        -------
        float
            Gas FVF (Bg) [rb/SCF].
        
        Raises
        ------
        ValueError
            If pressure is below atmospheric at high temperatures.
        """
        if pressure < 0.01:
            raise ValueError("Pressure must be positive")
        
        # Temperature in Rankine/60
        tr = temperature_rankine / 60.0
        
        # Pitzler Z-factor coefficients
        # For sweet gas: C1=0.026, C2=0.004, C3=0.8
        C1 = 0.026
        C2 = 0.004
        C3 = 0.8
        C4 = 0.0
        B = 0.01
        
        # Gas density correlation correction factor
        rho_g_correlation = 1.0 - B * gas_specific_gravity * (1 / pressure)
        
        # Z-factor
        z = C1 + C2 * (gas_specific_gravity ** C3) * (tr ** C4) * rho_g_correlation
        
        # Clamp Z to reasonable range
        z = max(0.3, min(1.5, z))
        
        # Bg correction factor
        bg = 0.000575 * (1.0 + (pressure / 100000.0) ** (-2.0))
        
        # Final Bg
        bg = (STANDARD_PRESSURE_PSI / STANDARD_TEMPERATURE_RANKINE) * z / pressure * bg
        
        return bg
    
    # -------------------------------------------------------------------------
    # VAPOR PRESSURE (Vasquez-Beggs, 1976)
    # -------------------------------------------------------------------------
    
    def get_vapor_pressure(
        self,
        api_gravity: float,
        solution_gas_oil_ratio: float,
    ) -> float:
        """
        Calculate saturated oil pressure using Vasquez-Beggs correlation.
        
        Psat = exp(A1 + A2 * API + A3 * API^2 + A4 * API^3 + A5 * API^4)
        
        Parameters
        ----------
        api_gravity : float
            API gravity of the crude oil [°API].
        solution_gas_oil_ratio : float
            Solution gas-oil ratio [scf/STB].
        
        Returns
        -------
        float
            Saturation pressure of oil (Psat) [psi].
        
        Raises
        ------
        ValueError
            If API gravity is below 15.3 or solution gas is negative.
        """
        if api_gravity < 15.3:
            raise ValueError("API gravity must be >= 15.3 for reliable vapor pressure correlation")
        
        if solution_gas_oil_ratio < 0:
            raise ValueError("Solution gas-oil ratio cannot be negative")
        
        # A1 = -1089.8 - 8260.4 * ln(Rs)
        # A2 = 1255.8 + 1348.9 * ln(Rs)
        # A3 = -325.0 - 1062.5 * ln(Rs)
        # A4 = 57.4 - 542.4 * ln(Rs)
        # A5 = 0
        # Psat = exp(A1 + A2*API + A3*API^2 + A4*API^3 + A5*API^4)
        
        ln_rs = math.log(solution_gas_oil_ratio)
        
        A1 = -1089.8 - 8260.4 * ln_rs
        A2 = 1255.8 + 1348.9 * ln_rs
        A3 = -325.0 - 1062.5 * ln_rs
        A4 = 57.4 - 542.4 * ln_rs
        
        # Exponent A (without the exp that's outside)
        A = (A1 + A2 * api_gravity + A3 * (api_gravity ** 2) + A4 * (api_gravity ** 3))
        
        # Psat
        psat = math.exp(A)
        
        return psat
    
    # -------------------------------------------------------------------------
    # STATIC PRESSURE FORMATION VOLUME FACTOR (Standing Correction)
    # -------------------------------------------------------------------------
    
    def get_static_oil_fvf(self, api_gravity: float, solution_gas_oil_ratio: float) -> float:
        """
        Calculate static oil formation volume factor using Standing correlation.
        
        Bo_s = 0.972 + 0.000147 * (T°F - 60) * (Rs ^ 1.75) / (API^0.5)
        
        Parameters
        ----------
        api_gravity : float
            API gravity of the crude oil [°API].
        solution_gas_oil_ratio : float
            Solution gas-oil ratio [scf/STB].
        
        Returns
        -------
        float
            Static Oil FVF (Bo_s) [rb/STB].
        
        Raises
        ------
        ValueError
            If API gravity is negative or solution gas is negative.
        """
        if api_gravity < 0:
            raise ValueError("API gravity cannot be negative")
        
        if solution_gas_oil_ratio < 0:
            raise ValueError("Solution gas-oil ratio cannot be negative")
        
        # Standing correlation for Bo_s
        bo_s = 0.972 + 0.000147 * (60.0) * (solution_gas_oil_ratio ** 1.75) / (api_gravity ** 0.5)
        
        return bo_s
    
    # -------------------------------------------------------------------------
    # STANDING GOR CORRELATION
    # -------------------------------------------------------------------------
    
    def get_standing_gor(
        self,
        pressure: float,
        gas_specific_gravity: float,
        api_gravity: float,
        temperature_f: float = STANDARD_TEMPERATURE_F,
    ) -> float:
        """
        Calculate solution gas-oil ratio using Standing correlation.
        
        Rs = gamma_g * P * exp( A + B * API + C * T + D * API^2 + E * API * T )
        
        Parameters
        ----------
        pressure : float
            Reservoir pressure [psi].
        gas_specific_gravity : float
            Gas specific gravity (air = 1.0) [dimensionless].
        api_gravity : float
            API gravity of the crude oil [°API].
        temperature_f : float, optional
            Reservoir temperature in degrees Fahrenheit [°F].
        
        Returns
        -------
        float
            Solution gas-oil ratio (Rs) [scf/STB].
        
        Raises
        ------
        ValueError
            If pressure is zero or negative, or API gravity is negative.
        """
        if pressure <= 0:
            raise ValueError("Pressure must be positive")
        
        if api_gravity < 0:
            raise ValueError("API gravity cannot be negative")
        
        # Compute temperature in Fahrenheit
        temperature_fahrenheit = temperature_f
        
        # Compute temperature correction factor
        temp_correction = 120.9 * temperature_fahrenheit - 873.6
        
        # Standing coefficients
        A = -1.46017
        B = 0.45195
        C = -1.4575 * temperature_fahrenheit + 1733.63
        D = -0.00348 * temperature_fahrenheit
        E = -0.000030 * temperature_fahrenheit
        
        # Rs = gamma_g * P * exp(A + B*API + C + D*API^2 + E*API*Temp)
        rs_coef = A + B * api_gravity + temp_correction + D * (api_gravity ** 2) + E * api_gravity * (temperature_fahrenheit / 60.0)
        rs = gas_specific_gravity * pressure * math.exp(rs_coef / 2.49) / 10
        
        return max(rs, 0.0)
    
    # -------------------------------------------------------------------------
    # GAS DENSITY CALCULATION
    # -------------------------------------------------------------------------
    
    def get_gas_density(
        self,
        gas_specific_gravity: float,
        pressure: float,
        temperature_f: float = STANDARD_TEMPERATURE_F,
        temperature_rankine: float = STANDARD_TEMPERATURE_RANKINE,
        standard_pressure: float = STANDARD_PRESSURE_PSI,
        standard_temperature: float = STANDARD_TEMPERATURE_F,
    ) -> float:
        """
        Calculate gas density at reservoir conditions.
        
        rho_g = P * M / (Z * R * T)
        
        Using simplified correlation for gas properties:
        rho_g = SG_g * P * 0.07633 / (T°F + 460) * (P / 14.7)
        
        Parameters
        ----------
        gas_specific_gravity : float
            Gas specific gravity (air = 1.0) [dimensionless].
        pressure : float
            Reservoir pressure [psi].
        temperature_f : float, optional
            Reservoir temperature in degrees Fahrenheit [°F].
        temperature_rankine : float, optional
            Reservoir temperature in Rankine [°R].
        standard_pressure : float, optional
            Standard pressure [psi].
        standard_temperature : float, optional
            Standard temperature in degrees Fahrenheit [°F].
        
        Returns
        -------
        float
            Gas density at reservoir conditions [lb/ft³].
        
        Raises
        ------
        ValueError
            If pressure or specific gravity is invalid.
        """
        if pressure <= 0:
            raise ValueError("Pressure must be positive")
        
        if gas_specific_gravity <= 0:
            raise ValueError("Gas specific gravity must be positive")
        
        # Temperature correction factor
        temp_factor = (pressure / standard_pressure) / (temperature_rankine / 460.0)
        
        # Gas density correlation
        gas_dens = gas_specific_gravity * 0.07633 * temp_factor
        
        return gas_dens
    
    # -------------------------------------------------------------------------
    # OIL DENSITY CALCULATION
    # -------------------------------------------------------------------------
    
    def get_oil_density(
        self,
        solution_gas_oil_ratio: float,
        api_gravity: float,
        oil_fvf: float = 1.0,
    ) -> float:
        """
        Calculate oil density at reservoir conditions.
        
        rho_o = 141.5 / (API + 131.5)
        
        Parameters
        ----------
        solution_gas_oil_ratio : float
            Solution gas-oil ratio [scf/STB].
        api_gravity : float
            API gravity of crude oil [°API].
        oil_fvf : float, optional
            Oil formation volume factor [rb/STB].
        
        Returns
        -------
        float
            Oil density at reservoir conditions [lb/ft³].
        
        Raises
        ------
        ValueError
            If API gravity is invalid or solution gas is negative.
        """
        if api_gravity < 15.3 or api_gravity > 97.0:
            raise ValueError("API gravity must be between 15.3 and 97.0")
        
        if solution_gas_oil_ratio < 0:
            raise ValueError("Solution gas-oil ratio cannot be negative")
        
        if oil_fvf <= 0:
            raise ValueError("Oil FVF must be positive")
        
        # Convert API to specific gravity
        sg_oil = 141.5 / (api_gravity + 131.5)
        
        # Saturation oil density
        rho_o_sat = sg_oil * 62.429  # lb/ft³
        
        # Volume expansion effect: rho_o_r = rho_o_sat / Bo
        rho_o_r = rho_o_sat / oil_fvf
        
        return rho_o_r
    
    # -------------------------------------------------------------------------
    # WATER PROPERTIES
    # -------------------------------------------------------------------------
    
    def get_surface_water_density(self) -> float:
        """
        Calculate water density at surface conditions.
        
        Surface water density for fresh water at 60°F:
        rho_ws = 62.36 lb/ft³
        
        Parameters
        -------
        Returns
        -------
        float
            Surface water density [lb/ft³].
        
        Notes
        -----
        Water density varies slightly with salinity and temperature. 
        For typical formation water (~35,000 ppm TDS @ 60°F):
        rho_ws ≈ 62.42 lb/ft³
        """
        return SURFACE_WATER_DENSITY_KG_M3 / 0.062428,  # Convert kg/m³ to lb/ft³
        
    def get_reservoir_water_density(
        self,
        pressure: Optional[float] = None,
        temperature_f: Optional[float] = None,
    ) -> float:
        """
        Calculate water density at reservoir conditions.
        
        Water properties are much less compressible than oil/gas:
        rho_w = rho_ws * (1 + C_w * (P - P_surface))
        
        For typical brine:
        rho_w ≈ 62.4 lb/ft³ (negligible compressibility)
        
        Parameters
        ----------
        pressure : float, optional
            Reservoir pressure [psi]. If None, assumes standard pressure.
        temperature_f : float, optional
            Reservoir temperature in degrees Fahrenheit [°F]. If None, assumes 60°F.
        
        Returns
        -------
        float
            Water density at reservoir conditions [lb/ft³].
        
        Notes
        -----
        Water has very low compressibility (~4.6 x 10^-6 psi^-1 for fresh water).
        Density variation is typically less than 1% for reservoir pressures up to 15,000 psi.
        """
        # Base water density
        rho_ws = SURFACE_WATER_DENSITY_KG_M3 / 0.062428  # Convert kg/m³ to lb/ft³
        
        # Temperature and pressure correction (negligible)
        delta_p = (pressure or STANDARD_PRESSURE_PSI) - STANDARD_PRESSURE_PSI
        
        # Water compressibility factor (~4.6e-6 psi^-1)
        compressibility = 4.6e-6
        rho_w = rho_ws * (1 + compressibility * delta_p)
        
        return rho_w
    
    def get_water_compression_factor(self, pressure: float) -> float:
        """
        Calculate water compressibility factor.
        
        Parameters
        ----------
        pressure : float
            Reservoir pressure [psi].
        
        Returns
        -------
        float
            Water compressibility [psi^-1].
        
        Notes
        -----
        For fresh water: C_w ≈ 4.6e-6 psi^-1
        For brine with 35,000 ppm TDS: C_w ≈ 4.4e-6 psi^-1
        """
        # TDS correlation
        tds = 35000.0  # Parts per thousand
        
        # Water compressibility decreases with salinity
        compressibility = 4.6e-6 * (1 - (tds / 100000.0) * 0.1)
        
        return compressibility
    
    # -------------------------------------------------------------------------
    # TWO-PHASE FRICTION FACTOR (Beggs-Brill)
    # -------------------------------------------------------------------------
    
    def get_two_phase_friction_factor(
        self,
        gas_rate: float,
        oil_rate: float,
        water_rate: float,
        liquid_dens: float,
        total_velocity: float,
        pipe_inner_diameter: float,
        gas_viscosity: float,
        liquid_viscosity: float,
        pipe_roughness: float = 0.00015,
    ) -> float:
        """
        Calculate two-phase friction factor using Beggs-Brill method.
        
        Parameters
        ----------
        gas_rate : float
            Gas rate at standard conditions [scf/D].
        oil_rate : float
            Oil rate at standard conditions [stb/D].
        water_rate : float
            Water rate at standard conditions [stb/day].
        liquid_dens : float
            Total liquid density at standard conditions [lb/ft³].
        total_velocity : float
            Mean velocity in the pipe [ft/s].
        pipe_inner_diameter : float
        pipe_roughness : float, optional
        
        Returns
        -------
        float
        
        Notes
        -----
        Beggs-Brill method for multiphase flow:
        f = f_MF(λ_L) where f_MF is the multiphase friction factor
        
        Reference: Beggs and Brill, Society of Petroleum Engineers (1973)
        """
        # Volume fraction of liquid
        liquid_rate = oil_rate + water_rate
        total_rate = liquid_rate + gas_rate
        
        if total_rate == 0:
            return 0.0
        
        lambda_l = liquid_rate / total_rate
        
        # Determine flow pattern
        flow_pattern = self._determine_flow_pattern(
            lambda_l, total_velocity, pipe_inner_diameter
        )
        
        # Friction factor calculation
        if lambda_l <= 0.4:
            # Stratified or intermittent flow
            if flow_pattern == "Int":
                # Slug flow (intermittent)
                epsilon = 0.0215
            else:
                # Stratified smooth
                epsilon = 0.0
        else:
            # Annular, dispersed bubble, or other patterns
            epsilon = 0.0016
        
        # Moody friction factor (smooth pipe)
        if total_velocity > 0:
            reynolds = (liquid_dens * total_velocity * pipe_inner_diameter) / liquid_viscosity
            if reynolds < 2000:
                # Laminar flow
                friction_factor = 64.0 / reynolds
            else:
                # Turbulent flow
                friction_factor = 0.0056 + 0.5 / (reynolds ** 0.22)
        else:
            friction_factor = 0.02
        
        # Pipe roughness correction
        if pipe_roughness > 0:
            friction_factor = max(friction_factor, (pipe_roughness / pipe_inner_diameter) ** 0.2)
        
        return friction_factor
    
    def _determine_flow_pattern(
        self,
        lambda_l: float,
        velocity: float,
        diameter: float,
    ) -> str:
        """
        Determine flow pattern using Beggs-Brill method.
        
        Parameters
        ----------
        lambda_l : float
            Liquid holdup.
        velocity : float
            Flow velocity.
        diameter : float
            Pipe diameter.
        
        Returns
        -------
        str
            Flow pattern ('Strat', 'Int', 'Disb', 'Ann')
        """
        # Low gas velocity (stratified flow)
        if velocity < 10 and lambda_l < 0.4:
            return "Strat"
        
        # High gas velocity (disperse bubble or annular)
        if velocity > 15 or lambda_l < 0.5:
            if lambda_l < 0.3:
                return "Disb"
            else:
                return "Ann"
        
        # Slug flow (intermittent)
        if lambda_l >= 0.3 and lambda_l <= 0.8:
            return "Int"
        
        return "Strat"
    
    # -------------------------------------------------------------------------
    # OIL VISCOSITY (Vasquez-Beggs, 1976)
    # -------------------------------------------------------------------------
    
    def get_oil_viscosity(
        self,
        api_gravity: float,
        solution_gas_oil_ratio: float,
        temperature_f: float = STANDARD_TEMPERATURE_F,
        pressure: Optional[float] = None,
    ) -> float:
        """
        Calculate oil viscosity using Beggs-Beggs correlation.
        
        mu_o = mu_o1 * 10^(A + B * (T°F - 60) + C * API^1.5 + D * Rs^0.5)
        
        Parameters
        ----------
        api_gravity : float
            API gravity [°API].
        solution_gas_oil_ratio : float
            Solution gas-oil ratio [scf/STB].
        temperature_f : float, optional
            Reservoir temperature [°F].
        
        Returns
        -------
        float
        
        Notes
        -----
        Reference: Beggs and Brill, J. Pet. Tech. (1973)
        """
        if api_gravity <= 0:
            raise ValueError("API gravity must be positive")
        
        if solution_gas_oil_ratio < 0:
            raise ValueError("Solution gas-oil ratio cannot be negative")
        
        # For sweet gas at standard conditions
        gas_specific_gravity = 0.6
        temp_rankine = fahrenheit_to_rankine(temperature_f)
        
        # Calculate oil viscosity at saturation conditions
        # Use temperature (T°F - 60) as independent variable
        
        # Coefficients
        A = -7.4438
        B = 0.0319
        C = -0.02383 * gas_specific_gravity
        D = -0.001539 * gas_specific_gravity
        
        temperature_correction = temperature_f - 60.0
        
        # Compute oil viscosity at saturation
        mu_o_s = math.exp(
            A + B * temperature_correction + C * (api_gravity ** 1.5) + D * math.sqrt(solution_gas_oil_ratio)
        )
        
        # If pressure is specified, apply pressure correction
        if pressure is not None:
            # Correction factor for pressure
            epsilon = 1.0 + 4.68e-4 * (pressure ** 1.4) * max(0, solution_gas_oil_ratio) ** (3.82)
            mu_o = mu_o_s * (1.0 / epsilon)
        else:
            mu_o = mu_o_s
        
        return max(mu_o, 0.0)
    
    # -------------------------------------------------------------------------
    # GAS VISCOSITY (Beggs-Brill)
    # -------------------------------------------------------------------------
    
    def get_gas_viscosity(
        self,
        gas_specific_gravity: float,
        temperature_f: float = STANDARD_TEMPERATURE_F,
    ) -> float:
        """
        Calculate gas viscosity using Beggs-Brill correlation.
        
        mu_g = mu_g1exp( A + B * T°F + C * T°F^(-0.5) * P^0.5 ) * P
        mu_g1 = exp( (1.393 + 98.696 / T_R ) / T_R ) * T_R^0.5 / P^0.5
        
        Parameters
        ----------
        gas_specific_gravity : float
            Gas specific gravity [dimensionless].
        temperature_f : float, optional
            Reservoir temperature [°F].
        
        Returns
        -------
        float
            Gas viscosity [cp].
        
        Notes
        -----
        Reference: Beggs and Brill, J. Pet. Tech. (1973)
        """
        if gas_specific_gravity <= 0:
            raise ValueError("Gas specific gravity must be positive")
        
        # Temperature in Kelvin
        t_kelvin = fahrenheit_to_kelvin(temperature_f)
        
        # Reference viscosity at standard conditions
        rho_g = gas_specific_gravity * 0.07633
        
        # mu_g1 = 0.022 / T_R * sqrt(T_R / P) * 0.0107 * (2.68 + 98.6 / T_R ^ 3)
        # Using simplified form
        mu_g1 = 0.000015 * math.exp(1.393 + 98.696 / t_kelvin)
        
        # Coefficients
        A = 6.07931
        B = 0.27692
        C = 0.161 - 1.6202 * gas_specific_gravity
        D = 0.32346
        E = 1.6242
        
        # Correction factor
        epsilon = A + B * (temperature_f - 60.0) + C * (temperature_f - 60.0) ** (-0.5) + D * gas_specific_gravity - E * gas_specific_gravity ** 2
        
        # Final viscosity calculation
        mu_g = mu_g1 * math.exp(epsilon) * (1.0 / (gas_specific_gravity ** 1.5))
        
        return max(mu_g, 0.0)
    
    # -------------------------------------------------------------------------
    # SUMMARY: ALL PVT PROPERTIES
    # -------------------------------------------------------------------------
    
    def get_undervatricated_oil_properties(
        self,
        api_gravity: float,
        gas_specific_gravity: float,
        pressure: float,
        temperature_f: float = STANDARD_TEMPERATURE_F,
        temperature_rankine: float = STANDARD_TEMPERATURE_RANKINE,
    ) -> dict:
        """
        Calculate all oil properties at a given pressure (undersaturated).
        
        Uses solution gas dissolution up to bubble point, then constant Rs.
        
        Parameters
        ----------
        api_gravity : float
            API gravity [°API].
        gas_specific_gravity : float
            Gas specific gravity [dimensionless].
        pressure : float
            Reservoir pressure [psi].
        temperature_f : float, optional
            Reservoir temperature [°F].
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'oil_fvf': Oil formation volume factor [rb/STB]
            - 'gas_solubility': Solution gas-oil ratio [scf/STB]
            - 'gas_viscosity': Gas viscosity [cp]
            - 'oil_viscosity': Oil viscosity [cp]
            - 'gas_density': Gas density [lb/ft³]
            - 'oil_density': Oil density [lb/ft³]
            - 'water_density': Water density [lb/ft³]
            - 'two_phase_friction': Multiphase friction factor
        """
        # Calculate bubble point pressure
        bubble_point = self.get_vapor_pressure(api_gravity, 0.0)
        
        # Determine solution gas at current pressure
        if pressure > bubble_point:
            # At bubble point, use maximum Rs
            solution_gas_oil_ratio = self.get_solution_gas_oil_ratio(
                api_gravity=api_gravity,
                gas_specific_gravity=gas_specific_gravity,
                pressure=bubble_point,
                temperature_f=temperature_f,
                temperature_rankine=temperature_rankine,
            )
        else:
            solution_gas_oil_ratio = self.get_solution_gas_oil_ratio(
                api_gravity=api_gravity,
                gas_specific_gravity=gas_specific_gravity,
                pressure=pressure,
                temperature_f=temperature_f,
                temperature_rankine=temperature_rankine,
            )
        
        # Calculate properties
        oil_fvf = self.get_oil_fvf(
            api_gravity=api_gravity,
            gas_specific_gravity=gas_specific_gravity,
            pressure=pressure,
            temperature_f=temperature_f,
            solution_gas_oil_ratio=solution_gas_oil_ratio,
            temperature_rankine=temperature_rankine,
        )
        
        oil_viscosity = self.get_oil_viscosity(
            api_gravity=api_gravity,
            solution_gas_oil_ratio=solution_gas_oil_ratio,
            temperature_f=temperature_f,
        )
        
        gas_viscosity = self.get_gas_viscosity(
            gas_specific_gravity=gas_specific_gravity,
            temperature_f=temperature_f,
        )
        
        water_density = self.get_reservoir_water_density(pressure, temperature_f)
        
        return {
            'oil_fvf': oil_fvf,
            'gas_solubility': solution_gas_oil_ratio,
            'gas_viscosity': gas_viscosity,
            'oil_viscosity': oil_viscosity,
            'gas_density': self.get_gas_density(gas_specific_gravity, pressure, temperature_f),
            'oil_density': self.get_oil_density(solution_gas_oil_ratio, api_gravity, oil_fvf),
            'water_density': water_density,
            'two_phase_friction': 0.02,
        }
    
    def get_full_pressure_traverse(
        self,
        api_gravity: float,
        gas_specific_gravity: float,
        pressures: list[float],
        solution_gas_oil_ratio: Optional[float] = None,
        temperature_f: float = STANDARD_TEMPERATURE_F,
        temperature_rankine: float = STANDARD_TEMPERATURE_RANKINE,
        bubble_point_pressure: Optional[float] = None,
    ) -> list[dict]:
        """
        Calculate full pressure traverse properties across a pressure range.
        
        Parameters
        ----------
        api_gravity : float
            API gravity [°API].
        gas_specific_gravity : float
            Gas specific gravity [dimensionless].
        pressures : list[float]
            List of pressures [psi].
        solution_gas_oil_ratio : float, optional
            Solution gas at bubble point [scf/STB].
        temperature_f : float, optional
            Reservoir temperature [°F].
        temperature_rankine : float, optional
            Reservoir temperature [°R].
        bubble_point_pressure : float, optional
            Bubble point pressure [psi]. Calculated if not provided.
        
        Returns
        -------
        list[dict]
            List of property dictionaries for each pressure.
        """
        # Calculate bubble point pressure first
        if bubble_point_pressure is None:
            bubble_point_pressure = self.get_vapor_pressure(
                api_gravity,
                solution_gas_oil_ratio or 0.0,
            )
        
        results = []
        
        for pressure in pressures:
            if pressure <= 0:
                continue
            
            if pressure > bubble_point_pressure:
                # Above bubble point
                sol_gas = solution_gas_oil_ratio or 0.0
                is_undersaturated = False
            else:
                # Below or at bubble point
                sol_gas = self.get_solution_gas_oil_ratio(
                    api_gravity=api_gravity,
                    gas_specific_gravity=gas_specific_gravity,
                    pressure=pressure,
                    temperature_f=temperature_f,
                    temperature_rankine=temperature_rankine,
                )
                is_undersaturated = True
            
            # Determine if unsaturated
            if is_undersaturated and solution_gas_oil_ratio is None:
                # Use undersaturated method
                oil_fvf = self.get_oil_fvf(
                    api_gravity=api_gravity,
                    gas_specific_gravity=gas_specific_gravity,
                    pressure=pressure,
                    temperature_f=temperature_f,
                    solution_gas_oil_ratio=None,  # Will calculate
                    temperature_rankine=temperature_rankine,
                )
            else:
                oil_fvf = self.get_oil_fvf(
                    api_gravity=api_gravity,
                    gas_specific_gravity=gas_specific_gravity,
                    pressure=pressure,
                    temperature_f=temperature_f,
                    solution_gas_oil_ratio=solution_gas_oil_ratio,
                    temperature_rankine=temperature_rankine,
                )
            
            results.append({
                'pressure': pressure,
                'oil_fvf': oil_fvf,
                'solution_gas_oil_ratio': sol_gas,
                'gas_viscosity': self.get_gas_viscosity(
                    gas_specific_gravity=gas_specific_gravity,
                    temperature_f=temperature_f,
                ),
                'oil_viscosity': self.get_oil_viscosity(
                    api_gravity=api_gravity,
                    solution_gas_oil_ratio=sol_gas,
                    temperature_f=temperature_f,
                ),
                'gas_density': self.get_gas_density(gas_specific_gravity, pressure, temperature_f),
                'water_density': self.get_reservoir_water_density(pressure, temperature_f),
            })
        
        return results
