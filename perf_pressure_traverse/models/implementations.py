"""
Black Oil PVT Property Correlations.

This module implements various black oil PVT correlations including:
- Oil viscosity correlations (Standing, Beal, Chew-Connally, Vasquez-Beggs)
- Solution GOR correlations (Vasquez-Beggs, Beggs-Brill, Standing)
- Formation Volume Factor (FVF) correlations
- Oil density correlations
- API Temperature correction formulas
"""

from __future__ import annotations

import math
from typing import Tuple

from perf_pressure_traverse.models.fluid import FluidProperties
from perf_pressure_traverse.models.pvt_properties import PVTProperties
from perf_pressure_traverse.utils.units import (
    convert_temperature,
    fahrenheit_to_celsius
)


class PVTCorrelationError(ValueError):
    """Exception raised when PVT correlation calculation fails."""
    pass


# ============================================================================
# OIL VISCOSITY CORRELATIONS
# ============================================================================

def standing_viscosity_cors(
    pressure_psia: float,
    temperature_f: float,
    rs_scf_stb: float,
    sg_oil: float,
    sg_gas: float,
    viscosity_ref_cP: float,
    rs_ref_scf_stb: float
) -> float:
    """
    Standing (1977) oil viscosity correlation.
    
    Parameters
    ----------
    pressure_psia : float
        Pressure in psi.
    temperature_f : float
        Temperature in °F.
    rs_scf_stb : float
        Solution gas-oil ratio in scf/stb or scf/bbl.
    sg_oil : float
        Oil surface gravity (specific gravity relative to water = 1.0).
    sg_gas : float
        Gas specific gravity (air = 1.0).
    viscosity_ref_cP : float
        Reference viscosity at reference conditions in cP.
    rs_ref_scf_stb : float
        Reference solution GOR at initial conditions in scf/stb.
    
    Returns
    -------
    float
        Oil viscosity at reservoir conditions in centipoise.
    
    Notes
    -----
    Standing viscosity correlation is valid for light to medium crude oils.
    The correlation includes correction for solution gas effect and temperature effect.
    """
    if rs_scf_stb < 0 or pressure_psia < 0 or temperature_f < -459.67:
        raise PVTCorrelationError("Input values must be physically valid")
    
    # Base viscosity at reference conditions (at reservoir temperature)
    # Actually, Standing uses viscosity at surface pressure as base
    # Let's compute the correction factor
    
    # Correction for solution gas effect (decreases viscosity)
    if rs_ref_scf_stb > 0:
        a = math.exp(0.048 * rs_scf_stb)
    else:
        a = 1.0
    
    # Temperature correction (decreases viscosity with temperature)
    t_ref = temperature_f
    t_current = temperature_f
    
    # Adjust to a reference temperature (typically 60°F)
    t_ref_std = 60.0
    
    # First correction for temperature difference (relative to reference)
    u = 0.047 * (t_ref_std - t_ref)
    
    if rs_ref_scf_stb > 0:
        y = 0.24 * rs_scf_stb / u
    else:
        y = 0.24 * rs_scf_stb
    
    # Correction factor
    if viscosity_ref_cP < 0.1:
        # Light oil - use different method
        mu_ref = viscosity_ref_cP
    else:
        mu_ref = viscosity_ref_cP
    
    # Standing's method
    mu = mu_ref * (a ** (0.047 * y ** 0.323))
    
    return mu


def beal_viscosity_correlation(
    pressure_psia: float,
    temperature_f: float,
    viscosity_ref_cP: float
) -> float:
    """
    Beal (1946) oil viscosity correlation.
    
    Parameters
    ----------
    pressure_psia : float
        Pressure in psi.
    temperature_f : float
        Temperature in °F.
    viscosity_ref_cP : float
        Viscosity at reference conditions in cP.
    
    Returns
    -------
    float
        Oil viscosity at reservoir conditions in centipoise.
    
    Notes
    -----
    Beal's correlation is specific to pressure effects, temperature effect 
    needs to be applied separately.
    
    Formula: μ = μ_r * (P/P_r)^(-0.045 * (log(T/T_r))^2)
    """
    if pressure_psia <= 0:
        raise PVTCorrelationError("Pressure must be positive")
    
    # Reference conditions (usually at API temperature and 14.7 psi)
    p_ref = 14.7  # psi
    t_ref = temperature_f
    
    # Calculate correction factor
    factor = (pressure_psia / p_ref) ** (-0.045 * (math.log(temperature_f / t_ref) ** 2))
    
    mu = viscosity_ref_cP * factor
    
    return mu


def chew_connally_viscosity_correlation(
    rs_scf_stb: float,
    sg_oil: float,
    temperature_f: float
) -> float:
    """
    Chew-Connally (1969) oil viscosity correlation.
    
    Parameters
    ----------
    rs_scf_stb : float
        Solution gas-oil ratio in scf/stb.
    sg_oil : float
        Oil surface gravity.
    temperature_f : float
        Temperature in °F.
    
    Returns
    -------
    float
        Oil viscosity in centipoise.
    
    Notes
    -----
    This correlation is for saturated oils at reservoir conditions.
    """
    # Convert temperature to Rankine for calculation
    t_rankine = temperature_f + 459.67
    
    # Correction factor for solution gas
    rs_corr = rs_scf_stb * (1.0 - (sg_oil ** 0.896) / (sg_oil ** 0.576))
    rs_corr = max(0, rs_corr)
    
    # Viscosity calculation
    if rs_corr < 0:
        raise PVTCorrelationError("Invalid gas correction")
    
    # Chew-Connally uses an iterative approach
    y = 0.0
    
    if rs_scf_stb > 0:
        y = -1 + 0.0014 * (rs_scf_stb ** 1.5)
    
    # Viscosity exponent
    if temperature_f > 100:
        # Heavy oil correlation
        lambda_val = 0.033 * (sg_oil ** 1.015)
    else:
        lambda_val = 0.043 * (sg_oil ** 0.749)
    
    # Beal correlation as base
    mu_base = 1.0 + 0.045 * (sg_oil - 1.0) * (temperature_f / 100) ** (3.4 + 0.007 * rs_scf_stb)
    
    # Chew-Connally correction
    mu = mu_base * (1.0 - y) ** (0.453 * lambda_val)
    
    return mu


def vasquez_beggs_viscosity(
    rs_scf_stb: float,
    sg_oil: float,
    sg_gas: float,
    temperature_f: float,
    api_grade: float = 30.0
) -> float:
    """
    Vasquez-Beggs (1976) oil viscosity correlation.
    
    Parameters
    ----------
    rs_scf_stb : float
        Solution gas-oil ratio in scf/stb.
    sg_oil : float
        Oil surface gravity.
    sg_gas : float
        Gas specific gravity.
    temperature_f : float
        Temperature in °F.
    api_grade : float, optional
        API gravity of oil (defaults to 30.0).
    
    Returns
    -------
    float
        Oil viscosity at reservoir conditions in centipoise.
    
    Notes
    -----
    Vasquez-Beggs correlation is widely used for crude oils. It's valid for 
    pressures above bubble point pressure.
    
    For undersaturated (P > Pb), use:
    μ = A * exp(B * rs_scf_stb) * P^(-D)
    
    Where A, B, D depend on temperature and oil gravity.
    """
    def viscosity_correlation(
        pressure_psia: float,
        rs_scf_stb: float,
        temperature_f: float,
        sg_oil: float
    ) -> Tuple[float, float, float]:
        """
        Compute coefficients A, B, D for undersaturated oils.
        
        Returns
        -------
        A, B, D : tuple of floats
            Coefficients for μ = A * exp(B * rs) * P^(-D)
        """
        # Convert temperature to Rankine
        t_rankine = temperature_f + 459.67
        
        # Base A coefficient (depends on oil gravity and temperature)
        if temperature_f < 133:
            y = 0.0
            A = 0.986 + 4.70e-4 * sg_oil + 0.0364 * rs_scf_stb - 4.68e-14 * (rs_scf_stb ** 4.56)
        else:
            y = (temperature_f - 133) / 611
            A = 1.0966 - 1.62e-5 * sg_oil - 1.5383*y + 1.2737y**2 - 3.6872y**3 + 28.7427y**4 - 43.8984y**5 + 25.7853y**6
        
        # B coefficient
        if rs_scf_stb <= 0:
            B = 0
        else:
            B = 11.72 + 0.02719 * sg_oil + 4.708e-4 * (1.0 - rs_scf_stb**0.546)
        
        # D coefficient
        if rs_scf_stb <= 0:
            D = 0.0364 * sg_oil
        else:
            D = 0.00339 * sg_oil + 7.434e-5 * rs_scf_stb - 0.000061 * rs_scf_stb**2
        
        return A, B, D
    
    # Determine if undersaturated (P > Pb) or saturated (P <= Pb)
    # We'll assume saturated by default and use solution gas effect
    # For undersaturated, the above coefficients already account for pressure
    
    A, B, D = viscosity_correlation(pressure_psia=14.7, rs_scf_stb=rs_scf_stb, 
                                     temperature_f=temperature_f, sg_oil=sg_oil)
    
    # Viscosity calculation for pressure > Pb
    mu_undersaturated = (A * math.exp(B * rs_scf_stb) * 
                        max(pressure_psia, 1.0) ** (-D))
    
    # Saturated case (simplified)
    # Temperature effect correction
    if rs_scf_stb <= 0:
        mu = 0.9996 * temperature_f ** -1.716 / sg_oil ** 0.239
    elif rs_scf_stb < 50:
        mu = 0.3243 * temperature_f ** -1.916 / sg_oil ** 0.396
    elif rs_scf_stb < 1100:
        mu = 0.0459 * temperature_f ** -3.031 / sg_oil ** 0.619
    else:
        mu = 0.000002 * temperature_f ** -3.319 / sg_oil ** 1.01
    
    # For simplicity in this implementation, we'll use the undersaturated formula
    # and assume reservoir pressure is above bubble point
    
    mu = mu_undersaturated
    
    return mu


# ============================================================================
# SOLUTION GOR CORRELATIONS
# ============================================================================

def vasquez_beggs_solution_gor(
    pressure_psia: float,
    sg_gas: float,
    api_grade: float,
    temperature_f: float = 80.0
) -> float:
    """
    Vasquez-Beggs (1976) solution GOR correlation.
    
    Parameters
    ----------
    pressure_psia : float
        Pressure in psi.
    sg_gas : float
        Gas specific gravity.
    api_grade : float
        API gravity of oil.
    temperature_f : float, optional
        Temperature in °F (defaults to 80°F).
    
    Returns
    -------
    float
        Solution gas-oil ratio in scf/stb.
    
    Notes
    -----
    This correlation calculates Rs (solution gas) at bubble point pressure.
    For pressures below bubble point, Rs = Rs_at_Pb (no more gas can dissolve).
    
    Formula:
    Rs = 107.75 * P^0.183 * exp(0.00512 * Y * G)
    
    Where Y = API^(-0.25) and G = sg_gas
    """
    # Coefficients for different API grades
    # Reference API = 30, sg_gas = 0.7
    
    # Calculate parameters
    Y = api_grade ** (-0.25)
    G = sg_gas
    
    # Coefficients for Rs calculation
    if temperature_f <= 60:
        A = 0.027
        B = 3.332
    else:
        A = 0.0362
        B = 1.096
    
    if api_grade > 30:
        mu = 0.3934 - 0.000282 * (api_grade - 30) / sg_gas
        A1 = A / mu
        B1 = B / (mu ** 4.42)
    else:
        mu = 1.0
        A1 = A
        B1 = B
    
    # Rs calculation
    if pressure_psia >= 7683.0:
        # For very high pressures, Rs may saturate
        Rs = 1500 + (pressure_psia - 7683) * 0.525
    else:
        Rs = A1 * (pressure_psia / 14.7) ** B1 * math.exp(0.00512 * Y * G)
    
    return Rs


def beggs_brill_solution_gor(
    pressure_psia: float,
    sg_gas: float,
    api_grade: float,
    temperature_f: float = 60.0
) -> float:
    """
    Beggs-Brill (1973) solution GOR correlation.
    
    Parameters
    ----------
    pressure_psia : float
        Pressure in psi.
    sg_gas : float
        Gas specific gravity.
    api_grade : float
        API gravity of oil.
    temperature_f : float, optional
        Temperature in °F.
    
    Returns
    -------
    float
        Solution gas-oil ratio in scf/stb.
    
    Notes
    -----
    Beggs-Brill correlation accounts for temperature effects on gas solubility.
    
    Formula derived from Standing's correlation:
    Rs = 3500 * P^0.173 * 10^(-0.00018 * API) * exp(0.00476 * 273.15)
    
    Simplified form:
    Rs = 30 * P^0.331 * exp(0.0362 * (1.073/sg_oil - 1) * (1/Y - 1))
    """
    Y = api_grade ** 0.9774
    
    # Coefficients depend on temperature
    if temperature_f <= 100:
        a = 0.331 * (temperature_f - 60) / 40.0
        b = 0.018 * (temperature_f - 60) / 40.0
    else:
        a = 0.331
        b = 0.018
    
    # Rs calculation
    Rs = 30.0 * (pressure_psia / 14.7) ** (a + b * sg_gas) * \
         math.exp(0.0362 * (1.073 / sg_gas - 1) * (1.0 / Y - 1.0))
    
    # Clamp Rs to reasonable values
    Rs = max(0, min(Rs, 1500))
    
    return Rs


def standing_solution_gor(
    pressure_psia: float,
    sg_gas: float,
    sg_oil: float,
    temperature_f: float = 60.0
) -> float:
    """
    Standing (1977) solution GOR correlation.
    
    Parameters
    ----------
    pressure_psia : float
        Pressure in psi.
    sg_gas : float
        Gas specific gravity.
    sg_oil : float
        Oil specific gravity.
    temperature_f : float, optional
        Temperature in °F.
    
    Returns
    -------
    float
        Solution gas-oil ratio in scf/stb.
    
    Notes
    -----
    Standing's correlation is widely used for solution GOR calculation.
    
    Formula: Rs = γ_g * Rs^o * P^0.73 * [1/(1.8*T)]^(2.7)
    
    Where Rs^o = 0.0362 * P^0.173 * 10^(-0.00018 * API)
    """
    # Convert specific gravity to API
    # sg_oil = 141.5 / (API + 131.5)
    api_grade = 141.5 / (sg_oil + 131.5)
    
    # Reference GOR at reference conditions (°F)
    Rs_ref = standing_viscosity_cors[0]  # We need a different approach
    
    # Actually, Standing's correlation:
    Rs = 0.1347 * (sg_gas ** 0.83) * (P / 14.7 ** 0.515) * \
         math.exp[0.000115 * (1.8 * T - 60)]
    
    # Simplified Standing correlation:
    # Rs = γ_g * 28.4 * 10^(0.0007 * API - 0.006 * T)
    
    T_standard = 60.0  # Std temperature
    
    Rs = 28.4 * sg_gas * (pressure_psia / 14.7) ** 0.515 * \
         math.exp(0.000115 * (1.8 * temperature_f - 60))
    
    return Rs


# ============================================================================
# FORMATION VOLUME FACTOR (FVF) CORRELATIONS
# ============================================================================

def standing_fvf(
    pressure_psia: float,
    sg_gas: float,
    api_grade: float,
    temperature_f: float = 60.0
) -> float:
    """
    Standing (1977) oil formation volume factor correlation.
    
    Parameters
    ----------
    pressure_psia : float
        Pressure in psi.
    sg_gas : float
        Gas specific gravity.
    api_grade : float
        API gravity of oil.
    temperature_f : float, optional
        Temperature in °F.
    
    Returns
    -------
    float
        Oil formation volume factor (reservoir conditions/surface conditions).
    
    Notes
    -----
    FVF >= 1.0. At surface conditions, FVF = 1.0.
    
    Formula: Bo = Rs * γ_g * 0.02 / γ_o + 0.972 + 1.47 * 10^-4 * Rs
    """
    # Temperature correction
    # The effect of temperature on FVF
    T_rel = temperature_f / 60.0
    
    # Standing's correlation:
    # Bo = 0.975949 + 1.610734e-5 * Rs + 1.096978e-5 * Rs * (1.8 * T - 60) + 1.43e-13 * Rs^2 * (1.8 * T - 60)
    
    # Simplified form for this implementation
    Bo = 0.972 + 0.015 * (pressure_psia / 14.7) ** -0.5
    
    # Correction for temperature and gas solubility
    Bo = Bo * (1.0 - 0.00015 * (1.8 * temperature_f - 60) * (pressure_psia / 14.7) ** -0.5)
    
    return Bo


def vasquez_beggs_fvf(
    pressure_psia: float,
    SG_g: float,
    SG_o: float,
    API_grade: float,
    temperature_f: float = 60.0,
    Rs_scf_stb: float = 0.0
) -> float:
    """
    Vasquez-Beggs (1976) oil formation volume factor correlation.
    
    Parameters
    ----------
    pressure_psia : float
        Pressure in psi.
    SG_g : float
        Gas specific gravity.
    SG_o : float
        Oil surface gravity.
    API_grade : float
        API gravity of oil.
    temperature_f : float, optional
        Temperature in °F.
    Rs_scf_stb : float, optional
        Solution gas-oil ratio in scf/stb.
    
    Returns
    -------
    float
        Oil formation volume factor (reservoir conditions/surface conditions).
    
    Notes
    -----
    Vasquez-Beggs FVF correlation is commonly used for crude oils.
    
    For saturated conditions (P <= Pb): FVF is primarily a function of temperature
    For undersaturated conditions (P > Pb): FVF decreases with pressure.
    
    Bo = 1 / (1 - 0.00045 * RS / Y)
    
    Where Y = 3500 * P^0.183 * exp(0.00512 * Y * SG_g)
    """
    # Convert SG_o to API grade
    # API = 141.5 / (SG_o + 131.5)
    
    # Coefficients
    a = 0.460 * Rs_scf_stb / (SG_o ** -1.3) if SG_o > 0 else 0
    
    Bo1 = 1.0 - (0.00072 * Rs_scf_stb) / (SG_o ** -1.25)
    Bo2 = 0.0014 * Rs_scf_stb
    
    # Temperature correction
    T_factor = 0.00018 * (1.8 * temperature_f - 60)
    
    # Final Bo calculation
    Bo = Bo1 / (1.0 + Bo2 * T_factor)
    
    return Bo


# ============================================================================
# OIL DENSITY CORRELATIONS
# ============================================================================

def standing_density(
    pressure_psia: float,
    sg_gas: float,
    sg_oil: float,
    temperature_f: float = 60.0
) -> float:
    """
    Standing oil density correlation.
    
    Parameters
    ----------
    pressure_psia : float
        Pressure in psi.
    sg_gas : float
        Gas specific gravity.
    sg_oil : float
        Oil surface gravity.
    temperature_f : float, optional
        Temperature in °F.
    
    Returns
    -------
    float
        Oil density at reservoir conditions in lb/ft³ (API units).
    
    Notes
    -----
    Density decreases with temperature and increases with pressure.
    
    ρ_o = SG_o - 62.4 * (1 - Bo) * (1 - Rs * SG_g)
    """
    # FVF correction
    Bo = standing_fvf(pressure_psia, sg_gas, sg_oil, temperature_f)
    
    # Density calculation
    # Surface density = 62.4 * SG_o (lb/ft³)
    density_surface = 62.4 * sg_o
    
    # Volume change with pressure (Bo)
    density = density_surface / Bo
    
    return density


# ============================================================================
# API TEMPERATURE CORRECTION FORMULAS
# ============================================================================

def stanton_temperature_corr(
    initial_temperature_f: float,
    final_temperature_f: float,
    oil_viscosity_initial_cP: float,
    oil_viscosity_units: str = "cP"
) -> float:
    """
    Stanton (1985) API temperature correction for oil properties.
    
    Parameters
    ----------
    initial_temperature_f : float
        Initial temperature in °F.
    final_temperature_f : float
        Final temperature in °F.
    oil_viscosity_initial_cP : float
        Oil viscosity at initial temperature in cP.
    
    Returns
    -------
    float
        Oil viscosity at final temperature in cP.
    
    Notes
    -----
    API standard for temperature correction of oil properties.
    Formula: μ_2 = μ_1 * (T_1 + C)^a * (T_2 + C)^(-a)
    
    Where C ≈ -273.15 (adjusted for °F)
    """
    # Convert temperatures to Rankine
    T1_rankine = initial_temperature_f + 459.67
    T2_rankine = final_temperature_f + 459.67
    
    # API temperature correction exponent a
    a = 0.25  # For viscosity
    
    # Compute temperature correction factor
    if T2_rankine > 0 and T1_rankine > 0:
        factor = (T1_rankine ** a) / (T2_rankine ** a)
    else:
        factor = 0.0
    
    mu_final = oil_viscosity_initial_cP * factor
    
    return mu_final


def api_grade_correction(initial_api_grade: float, final_temperature_f: float) -> float:
    """
    API grade temperature correction.
    
    Parameters
    ----------
    initial_api_grade : float
        Initial API grade at reference temperature.
    final_temperature_f : float
        Final temperature in °F.
    
    Returns
    -------
    float
        Corrected API grade at final temperature.
    
    Notes
    -----
    API gravity typically decreases with temperature.
    
    API_T = API_ref * (1 - 0.0005 * (T - T_ref))
    """
    T_ref = 60.0  # Standard reference temperature
    correction = initial_api_grade * (1.0 - 0.0005 * (final_temperature_f - T_ref))
    
    return max(0, correction)


# ============================================================================
# UTILITY FUNCTIONS FOR PVT CALCULATIONS
# ============================================================================

def calculate_c1_coefficient(
    gas_composition: dict,
    z_factor: float,
    pressure_psia: float,
    temperature_rankine: float
) -> float:
    """
    Calculate composition parameter c1 for PVT correlations.
    
    Parameters
    ----------
    gas_composition : dict
        Dictionary of gas component names and mole fractions.
    z_factor : float
        Gas compressibility factor (z).
    pressure_psia : float
        Pressure in psi.
    temperature_rankine : float
        Temperature in °R.
    
    Returns
    -------
    float
        Composition parameter c1.
    
    Notes
    -----
    c1 = ∑ (x_i * f_i)
    Where f_i is a function of gas composition and conditions.
    """
    # Simplified implementation
    # In real application, this would require detailed gas composition
    c1 = z_factor * (pressure_psia / 14.7)**0.25
    
    return c1


def validate_pvt_inputs(
    pressure_psia: float,
    temperature_f: float,
    rs_scf_stb: float = 0.0,
    viscosity_cP: float = 0.0
) -> bool:
    """
    Validate PVT input parameters.
    
    Parameters
    ----------
    pressure_psia : float
        Pressure in psi.
    temperature_f : float
        Temperature in °F.
    rs_scf_stb : float, optional
        Solution GOR in scf/stb.
    viscosity_cP : float, optional
        Viscosity in cP.
    
    Returns
    -------
    bool
        True if inputs are valid, False otherwise.
    
    Raises
    ------
    PVTCorrelationError
        If any input parameter is invalid.
    """
    if pressure_psia < 0:
        raise PVTCorrelationError("Pressure must be non-negative")
    
    if temperature_f < -459.67:
        raise PVTCorrelationError("Temperature below absolute zero")
    
    if rs_scf_stb < 0:
        raise PVTCorrelationError("Solution GOR must be non-negative")
    
    if viscosity_cP < 0:
        raise PVTCorrelationError("Viscosity must be non-negative")
    
    return True


# ============================================================================
# CLASS-BASED PVT CALCULATOR
# ============================================================================

class BlackOilPVT:
    """
    Black oil PVT calculator using correlations.
    
    This class provides a unified interface for calculating black oil PVT
    properties using various correlations from the industry.
    
    Parameters
    ----------
    fluid : FluidProperties
        Surface fluid properties.
    pressure_psia : float
        Reservoir pressure in psi.
    temperature_f : float
        Reservoir temperature in °F.
    correlation_type : str, optional
        Correlation type to use. Supported: 'standing', 'beal', 
        'vasquez_beggs', 'chew_connally'. Defaults to 'vasquez_beggs'.
    """
    
    def __init__(
        self,
        fluid: FluidProperties,
        pressure_psia: float,
        temperature_f: float,
        correlation_type: str = "vasquez_beggs"
    ):
        self.fluid = fluid
        self.pressure_psia = pressure_psia
        self.temperature_f = temperature_f
        self.correlation_type = correlation_type.lower()
        
        # Store reservoir conditions for reference
        self.reservoir_temperature_rankine = temperature_f + 459.67
    
    def calculate_solution_gor(self) -> float:
        """
        Calculate solution gas-oil ratio using selected correlation.
        
        Returns
        -------
        float
            Solution GOR in scf/stb.
        """
        if self.correlation_type == "vasquez_beggs":
            return vasquez_beggs_solution_gor(
                self.pressure_psia,
                self.fluid.gas_gravity_sg,
                fluid_api_grade(fluid)
            )
        elif self.correlation_type == "beggs_brill":
            return beggs_brill_solution_gor(
                self.pressure_psia,
                self.fluid.gas_gravity_sg,
                fluid_api_grade(fluid)
            )
        elif self.correlation_type == "standing":
            return standing_solution_gor(
                self.pressure_psia,
                self.fluid.gas_gravity_sg,
                self.fluid.oil_gravity_sg,
                self.temperature_f
            )
        else:
            raise ValueError(f"Unknown correlation type: {self.correlation_type}")
    
    def calculate_oil_viscosity(self) -> float:
        """
        Calculate oil viscosity using selected correlation.
        
        Returns
        -------
        float
            Oil viscosity in cP.
        """
        rs_scf_stb = self.fluid.solution_oil_ratio
        
        if self.correlation_type == "standing":
            return standing_viscosity_cors(
                self.pressure_psia,
                self.temperature_f,
                rs_scf_stb,
                self.fluid.oil_gravity_sg,
                self.fluid.gas_gravity_sg,
                self.fluid.pvts.viscosity_ref_cP if hasattr(self.fluid, 'pvts') else 0.1,
                rs_scf_stb
            )
        elif self.correlation_type == "beal":
            return beal_viscosity_correlation(
                self.pressure_psia,
                self.temperature_f,
                self.fluid.oil_gravity_sg
            )
        elif self.correlation_type == "vasquez_beggs":
            return vasquez_beggs_viscosity(
                rs_scf_stb,
                self.fluid.oil_gravity_sg,
                self.fluid.gas_gravity_sg,
                self.temperature_f,
                fluid_api_grade(fluid)
            )
        elif self.correlation_type == "chew_connally":
            return chew_connally_viscosity_correlation(
                rs_scf_stb,
                self.fluid.oil_gravity_sg,
                self.temperature_f
            )
        else:
            raise ValueError(f"Unknown correlation type: {self.correlation_type}")
    
    def calculate_oil_fvf(self) -> float:
        """
        Calculate oil formation volume factor using selected correlation.
        
        Returns
        -------
        float
            Oil FVF (dimensionless).
        """
        if self.correlation_type == "standing":
            return standing_fvf(
                self.pressure_psia,
                self.fluid.gas_gravity_sg,
                fluid_api_grade(fluid),
                self.temperature_f
            )
        elif self.correlation_type == "vasquez_beggs":
            return vasquez_beggs_fvf(
                self.pressure_psia,
                self.fluid.gas_gravity_sg,
                self.fluid.oil_gravity_sg,
                fluid_api_grade(fluid),
                self.temperature_f,
                self.calculate_solution_gor()
            )
        else:
            raise ValueError(f"Unknown correlation type: {self.correlation_type}")
    
    def calculate_reservoir_properties(self) -> dict:
        """
        Calculate all reservoir PVT properties.
        
        Returns
        -------
        dict
            Dictionary containing:
            - viscoity_cP: Oil viscosity
            - solution_gor_scf_stb: Solution GOR
            - formation_volume_factor: Oil FVF
            - rs_scf_stb: Solution gas per STB
        """
        self.fluid.solution_oil_ratio = self.calculate_solution_gor()
        
        properties = {
            "viscosity_cP": self.calculate_oil_viscosity(),
            "solution_gor_scf_stb": self.calculate_solution_gor(),
            "formation_volume_factor": self.calculate_oil_fvf(),
            "rs_scf_stb": self.calculate_solution_gor()  # Alias
        }
        
        return properties


def fluid_api_grade(fluid: FluidProperties) -> float:
    """
    Calculate API grade from specific gravity.
    
    Parameters
    ----------
    fluid : FluidProperties
        Fluid properties containing specific gravity.
    
    Returns
    -------
    float
        API grade.
    
    Formula: API = 141.5 / (SG + 131.5)
    """
    return 141.5 / (fluid.oil_gravity_sg + 131.5)


__all__ = [
    'PVTCorrelationError',
    
    # Oil viscosity correlations
    'standing_viscosity_cors',
    'beal_viscosity_correlation',
    'chew_connally_viscosity_correlation',
    'vasquez_beggs_viscosity',
    
    # Solution GOR correlations
    'vasquez_beggs_solution_gor',
    'beggs_brill_solution_gor',
    'standing_solution_gor',
    
    # FVF correlations
    'standing_fvf',
    'vasquez_beggs_fvf',
    
    # Oil density
    'standing_density',
    
    # API temperature corrections
    'stanton_temperature_corr',
    'api_grade_correction',
    
    # Utilities
    'calculate_c1_coefficient',
    'validate_pvt_inputs',
    
    # Calculator class
    'BlackOilPVT',
    'fluid_api_grade',
]
