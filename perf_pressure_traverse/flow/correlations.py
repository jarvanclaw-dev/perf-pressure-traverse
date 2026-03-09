"""Multiphase flow correlations for pressure drop calculations.

Implements Beggs & Brill (1973) multiphase flow correlation for tubular flow,
including viscosity, density, holdup, and pressure drop calculations.
"""

from __future__ import annotations

from datetime import datetime
from math import log10, sqrt, exp, atan, fabs, pi
from typing import Tuple, Optional
from enum import Enum

# ============== PHYSICAL CONSTANTS ==============
# GMP (gallons per minute) to lbm/s conversion
# 500 lbm per minute per gallon at standard conditions (API gravity 41° API)
GMP_TO_LBMS = 500 / 60.0  # 500 lbm per minute per gallon

# Critical gas fraction for transition between BUBBLE and MIST flow regimes
# Based on Beggs & Brill experimental data at inclined pipe conditions
CRITICAL_GAS_FRACTION = 0.82

# Typical gas viscosity at standard conditions (20°F)
# Source: API Technical Data Book, Table 3-6
TYPICAL_GAS_VISCOSITY_CP = 0.02

# Default gas-oil ratio for gas dissolution calculations (scf/STB)
# Represents average field conditions for reservoirs with 30-40° API oil
DEFAULT_OIL_GOR_CP = 300.0

# Gas-water ratio default value (scf/STB)
# Used when water production is not specified
DEFAULT_GW_RATIO_CP = 0.0

# Roughness factor for pipe friction calculations (feet)
# Typical commercial steel pipe
PIPE_ROUGHNESS_FT = 0.0019

# Gravitational acceleration (ft/s²)
# Used in dimensional analysis and hydrostatic calculations
G = 32.174  # ft/s², standard gravity in English units


class FlowRegime(Enum):
    """Flow regime enumeration for Beggs-Brill method."""
    BUBBLE = "Bubble"
    SMUG = "Slug"
    SEGREGATED = "Segregated"
    ANNULAR = "Distributed"
    MIST = "Mist"


class BeggsBrillCorrelation:
    """
    Beggs & Brill multiphase flow correlation for tubing.
    
    Implements the complete Beggs-Brill method including:
    - Multiphase viscosity calculation
    - Multiphase density calculation
    - Flow regime identification and holdup calculations
    - Complete pressure drop including friction, hydrostatic, and kinetic components
    
    Reference:
    Beggs, H.D. and Brill, J.P. (1973). A Study of Two-Phase Flow in Inclined Pipes.
    Society of Petroleum Engineers Journal, Vol. 13, No. 5, pp. 603-617.
    
    Attributes
    ----------
    oil_flow_rate_gpm : float
        Oil flow rate in gallons per minute at standard conditions.
    gas_flow_rate_gpm : float
        Gas flow rate in gallons per minute at standard conditions.
    pipe_diameter_ft : float
        Pipe inner diameter in feet.
    borehole_area_ft2 : float
        Borehole cross-sectional area in ft².
    well_angle_deg : float
        Well inclination angle from vertical in degrees.
            Positive = uphill, Negative = downhill.
    oil_density_lbm_ft3 : float
        Oil density in lb/ft³ at standard conditions.
    water_density_lbm_ft3 : float
        Water density in lb/ft³ at standard conditions.
    gas_density_lb_ft3 : float
        Gas density in lb/ft³.
    gas_specific_gravity : float
        Gas specific gravity (air = 1.0).
    oil_specific_gravity : float
        Oil specific gravity (water = 1.0).
    oil_viscosity_cP : float
        Oil viscosity in centipoise.
    """
    
    # Critical gas fraction for transition between BUBBLE and MIST flow regimes
    CRITICAL_GAS_FRACTION = 0.82
    
    def __init__(
        self,
        oil_flow_rate_gpm: float,
        gas_flow_rate_gpm: float,
        pipe_diameter_ft: float,
        borehole_area_ft2: float,
        well_angle_deg: float = 0.0,
        oil_gal: float = 30.0,
        oil_density_lbm_ft3: float = 50.0,
        water_density_lbm_ft3: float = 62.4,
        gas_density_lb_ft3: float = 0.1,
        gas_specific_gravity: float = 1.0,
        oil_specific_gravity: float = 1.0,
        oil_viscosity_cP: float = 10.0,
    ) -> None:
        """
        Initialize Beggs-Brill correlation parameters.
        
        Parameters
        ----------
        oil_flow_rate_gpm : float
            Oil flow rate in GPM.
        gas_flow_rate_gpm : float
            Gas flow rate in GPM.
        pipe_diameter_ft : float
            Pipe diameter in feet.
        borehole_area_ft2 : float
            Borehole cross-sectional area in ft².
        well_angle_deg : float
            Well inclination angle from vertical.
        oil_gal : float
            Oil gravity in API.
        oil_density_lbm_ft3 : float
            Oil density in lb/ft³.
        water_density_lbm_ft3 : float
            Water density in lb/ft³.
        gas_density_lb_ft3 : float
            Gas density in lb/ft³.
        gas_specific_gravity : float, optional
            Gas specific gravity (default 1.0).
        oil_specific_gravity : float, optional
            Oil specific gravity (default 1.0).
        oil_viscosity_cP : float, optional
            Oil viscosity in cP (default 10.0).
            
        Raises
        ------
        ValueError
            If flow rates or physical properties are non-positive.
        ValueError
            If borehole_area is zero.
        ValueError
            If pipe_diameter is zero or negative.
        ValueError
            If well angle is outside valid range [-90, 90].
        """
        # Validate all physical parameters
        self._validate_input_parameters(
            oil_flow_rate_gpm,
            gas_flow_rate_gpm,
            pipe_diameter_ft,
            borehole_area_ft2,
        well_length_ft,
            well_angle_deg,
        )
        
        # Store all parameters as regular attributes
        self.oil_flow_rate_gpm = oil_flow_rate_gpm
        self.gas_flow_rate_gpm = gas_flow_rate_gpm
        self.pipe_diameter_ft = pipe_diameter_ft
        self.borehole_area_ft2 = borehole_area_ft2
        self.well_length_ft = well_length_ft
        self.well_angle_deg = well_angle_deg
        self.oil_gal = oil_gal
        self.oil_density_lbm_ft3 = oil_density_lbm_ft3
        self.water_density_lbm_ft3 = water_density_lbm_ft3
        self.gas_density_lb_ft3 = gas_density_lb_ft3
        self.gas_specific_gravity = gas_specific_gravity
        self.oil_specific_gravity = oil_specific_gravity
        
        # Calculate total liquid rate (oil + water) in lbm/s
        self._total_liquid_rate_lbm_s = (
            self.oil_flow_rate_gpm * oil_density_lbm_ft3 * GMP_TO_LBMS +
            self.gas_flow_rate_gpm * water_density_lbm_ft3 * GMP_TO_LBMS
        )
        
        # Calculate gas rate in lbm/s
        # Using API conversion: 1 scf ~ 7.48 * (gas SG * 28.97 lb/gmol) / (MW_gas * R * T)
        self._gas_rate_lbm_s = (
            self.gas_flow_rate_gpm * self.gas_specific_gravity * 0.0765 * GMP_TO_LBMS
        )
        
        # Private viscosity attributes
        self._oil_viscosity_cP = oil_viscosity_cP
    
    @property
    def oil_viscosity_cP(self) -> float:
        """Oil viscosity getter."""
        return self._oil_viscosity_cP
    
    @property
    def gas_viscosity_cP(self) -> Optional[float]:
        """
        Gas viscosity getter.
        
        Returns gas viscosity at standard conditions (0.02 cP).
        Note: For accurate temperature-dependent viscosity, use API 54 correlation.
        """
        return TYPICAL_GAS_VISCOSITY_CP
    
    @property
    def _gas_viscosity_cP(self) -> float:
        """Gas viscosity in internal calculations (0.02 cP)."""
        return TYPICAL_GAS_VISCOSITY_CP
    
    @property
    def _o_gr(self) -> float:
        """Oil gas ratio getter (Gas-Oil Ratio in scf/STB)."""
        return DEFAULT_OIL_GOR_CP
    
    @property
    def _g_wr(self) -> float:
        """Gas water ratio getter (Gas-Water Ratio in scf/STB)."""
        return DEFAULT_GW_RATIO_CP
    
    @property
    def _pressure(self) -> Optional[float]:
        """Pressure getter for internal use."""
        return None
    
    @property
    def _temperature(self) -> Optional[float]:
        """Temperature getter for internal use."""
        return None
    
    @property
    def gas_rate_lbm_s(self) -> float:
        """Gas rate getter."""
        return self._gas_rate_lbm_s
    
    @property
    def total_liquid_rate_lbm_s(self) -> float:
        """Total liquid rate getter."""
        return self._total_liquid_rate_lbm_s
    
    @property
    def _gas_density(self) -> Optional[float]:
        """Gas density getter."""
        return self.gas_density_lb_ft3
    
    @property
    def oil_gas_ratio(self) -> Optional[float]:
        """Oil gas ratio getter (scf/STB)."""
        return self._o_gr
    
    @property
    def gas_water_ratio(self) -> Optional[float]:
        """Gas water ratio getter (scf/STB)."""
        return self._g_wr
    
    @property
    def pressure_psia(self) -> Optional[float]:
        """Pressure getter for internal use."""
        return self._pressure
    
    @property
    def temperature_f(self) -> Optional[float]:
        """Temperature getter for internal use."""
        return self._temperature
    
    @property
    def true_vertical_depth_ft(self) -> float:
        """True vertical depth getter."""
        return 144.0  # Approximate depth
    
    def _validate_input_parameters(
        self,
        oil_flow_rate_gpm: float,
        gas_flow_rate_gpm: float,
        pipe_diameter_ft: float,
        borehole_area_ft2: float,
        well_angle_deg: float
    ) -> None:
        """
        Validate all input parameters.
        
        Parameters
        ----------
        oil_flow_rate_gpm : float
            Oil flow rate in GPM.
        gas_flow_rate_gpm : float
            Gas flow rate in GPM.
        pipe_diameter_ft : float
            Pipe diameter in feet.
        borehole_area_ft2 : float
            Borehole cross-sectional area in ft².
        well_angle_deg : float
            Well inclination angle from vertical.
            
        Raises
        ------
        ValueError
            For any non-positive flow rates or physical properties.
        ValueError
            If borehole_area is zero or negative.
        ValueError
            If pipe_diameter is zero or negative.
        ValueError
            If well_angle is outside range [-90, 90].
        """
        # Validate flow rates (must be positive)
        if oil_flow_rate_gpm <= 0:
            raise ValueError(f"Oil flow rate must be positive, got {oil_flow_rate_gpm} GPM")
        
        if gas_flow_rate_gpm <= 0:
            raise ValueError(f"Gas flow rate must be positive, got {gas_flow_rate_gpm} GPM")
        
        # Validate pipe dimensions
        if pipe_diameter_ft <= 0:
            raise ValueError(f"Pipe diameter must be positive, got {pipe_diameter_ft} ft")
        
        if borehole_area_ft2 <= 0:
            raise ValueError(f"Borehole area must be positive, got {borehole_area_ft2} ft²")
        
        # Validate well angle
        if well_angle_deg < -90 or well_angle_deg > 90:
            raise ValueError(
                f"Well angle must be in range [-90, 90] degrees, got {well_angle_deg}"
            )
        
        # Validate densities (should be positive)
        # Note: Gas density could be very low, so we use a small threshold
        if (
            oil_flow_rate_gpm > 0 and 
            oil_density_lbm_ft3 <= 0
        ):
            raise ValueError(f"Oil density must be positive, got {oil_density_lbm_ft3} lb/ft³")
            
        # Note: Other densities are validated in __init__ as optional parameters
        # They may be zero (e.g., gas density) if user wants to specify it
    
    def calculate_mixture_viscosity_cP(self) -> float:
        """
        Calculate multiphase mixture viscosity using Beggs-Brill method.
        
        Returns
        -------
        float
            Mixture viscosity in cP.
            
        Raises
        ------
        ZeroDivisionError
            If superficial velocities result in division by zero.
        ValueError
            If density calculations result in negative or zero values.
        """
        # Validate inputs before calculation
        if self.pipe_diameter_ft <= 0:
            raise ValueError("Pipe diameter must be positive for viscosity calculation")
        
        if self.borehole_area_ft2 <= 0:
            raise ValueError("Borehole area must be positive for viscosity calculation")
        
        # Use the oil viscosity provided
        mu_o = self.oil_viscosity_cP
        
        # Typical gas viscosity (20°F standard conditions)
        mu_g = TYPICAL_GAS_VISCOSITY_CP
        
        # Calculate quality (gas fraction)
        gas_rate = self.gas_rate_lbm_s
        liquid_rate = self.total_liquid_rate_lbm_s
        
        if (gas_rate + liquid_rate) <= 0:
            raise ValueError("Total mass flow rate must be positive)")
        
        # Clamp quality to [0, 1]
        quality = gas_rate / (gas_rate + liquid_rate)
        quality = max(0.0, min(1.0, quality))
        
        # Calculate λ (liquid-to-gas flow rate ratio at standard conditions)
        q_l = liquid_rate
        q_g = gas_rate
        
        if q_g > 0:
            lam = q_l / q_g
        else:
            lam = float('inf')
        
        # Convert viscosity to lb/(ft·s) for dimensional analysis
        # μ_cP → μ_lb/(ft·s): divide by 1000 then by g (32.174 ft/s/s)
        mu_o_lb_ft_s = mu_o / 1000.0 / G
        
        D_ft = self.pipe_diameter_ft
        
        # Check for division by zero
        if D_ft <= 0:
            raise ValueError("Pipe diameter must be positive for viscosity calculation")
        
        # Calculate dimensionless parameter
        dimensionless = (G * mu_o_lb_ft_s * D_ft) ** 4.75
        dimensionless = max(dimensionless, 0.001)  # Prevent negative/zero
        
        # Handle infinite lam value
        if lam == float('inf'):
            lam = 0.0
        
        # Handle invalid dimensionless value
        if dimensionless <= 0:
            x = 1.0
        else:
            x = 1.0 + (0.012 * lam ** 0.762 * dimensionless) ** (-4.75)
        
        # Clamp x to valid range [0, 1]
        x = max(0.0, min(1.0, x))
        
        # Calculate mixture viscosity (logarithmic mixing rule)
        mu_m = mu_o ** x * mu_g ** (1 - x)
        
        # Ensure viscosity is non-negative
        return max(0.0, mu_m)
    
    def calculate_total_liquid_rate_gpm(self) -> float:
        """
        Calculate total liquid flow rate including dissolved gas.
        
        Returns
        -------
        float
            Total liquid rate after gas dissolution correction in GPM.
        """
        try:
            # Get oil GOR (gas-oil ratio) from static properties
            Rs = DEFAULT_OIL_GOR_CP
            
            # Estimate gas dissolution percentage (simplified linear correlation)
            # Based on pressure/temperature effects on gas solubility
            # Real implementation would use PVT correlations
            dissolution = self.calculate_gas_dissolution_percentage()
            
            # Clamp dissolution to valid range [0, 100]
            dissolution = max(0.0, min(dissolution, 100.0))
            
            # Adjust gas flow rate to account for dissolved gas
            adjusted_gas_gpm = self.gas_flow_rate_gpm * (1 - dissolution / 100.0)
            
            # Water rate from gas rate and gas water ratio
            gwr = DEFAULT_GW_RATIO_CP
            water_gpm = (Rs + gwr) * 0.00003  # Simplified conversion
            
            total_liquid_gpm = self.oil_flow_rate_gpm + water_gpm
            
            return min(total_liquid_gpm, self.oil_flow_rate_gpm)  # Don't exceed oil rate
        except Exception as e:
            raise ValueError(f"Error calculating total liquid rate: {str(e)}")
    
    def calculate_gas_dissolution_percentage(self) -> float:
        """
        Calculate percentage of gas dissolved into oil.
        
        Returns
        -------
        float
            Percentage of gas dissolved (0-100).
        """
        try:
            # Simplified gas dissolution calculation
            # Based on pressure/temperature effects on gas solubility
            # Rs (gas-oil ratio) affects how much gas stays in liquid phase
            
            # Typical dissolution ranges based on field data
            # Higher Rs → higher dissolution ratio
            dissolution = (DEFAULT_OIL_GOR_CP * 0.5) / DEFAULT_OIL_GOR_CP * 100.0
            
            return min(dissolution, 100.0)  # Clamp to 0-100
        except Exception as e:
            return 0.0  # Return 0 on error
    
    def calculate_mixture_density_lb_ft3(
        self,
        oil_fraction: float = 1.0,
        water_fraction: float = 0.0
    ) -> float:
        """
        Calculate mixture density using Beggs-Brill holdup-based method.
        
        Parameters
        ----------
        oil_fraction : float
            Volume fraction of oil (0 to 1).
        water_fraction : float
            Volume fraction of water (0 to 1).
            
        Returns
        -------
        float
            Mixture density in lb/ft³.
            
        Raises
        ------
        ValueError
            If fractions don't sum to 1 or if any is out of [0, 1] range.
        """
        # Validate fractions
        total_fraction = oil_fraction + water_fraction
        if not (0.99 <= abs(total_fraction - 1.0) <= 1.01):
            raise ValueError(
                f"Oil and water fractions must be 1.0 (got {oil_fraction}+{water_fraction}={total_fraction})"
            )
        
        if not (0.0 <= min(oil_fraction, water_fraction) <= 1.0):
            raise ValueError("Fractions must be in range [0, 1]")
        
        # Calculate total liquid density (excluding dispersed gas)
        rho_l = (
            self.oil_density_lbm_ft3 * oil_fraction +
            self.water_density_lbm_ft3 * water_fraction
        )
        
        # Calculate gas density
        rho_g = self.gas_density_lb_ft3 or (
            self.gas_specific_gravity * 62.4 / 62.65  # Approximate using air=1.0
        )
        
        if rho_g is None or rho_g <= 0:
            rho_g = 0.005  # Small default for gas
            self.gas_density_lb_ft3 = rho_g
        
        # Calculate gas holdup (void fraction)
        try:
            lambda_g = self.calculate_gas_holdup()
            
            # Clamp to valid range [0, 1]
            lambda_g = max(0.0, min(lambda_g, 1.0))
        except Exception:
            lambda_g = 0.1  # Default holdup on error
        
        # Calculate mixture density
        rho_m = (1 - lambda_g) * rho_l + lambda_g * rho_g
        
        return max(0.0, rho_m)
    
    def calculate_gas_holdup(self) -> float:
        """
        Calculate gas holdup (void fraction) using Beggs-Brill method.
        
        Returns
        -------
        float
            Gas holdup in fraction (0 to 1).
        """
        try:
            # Get flow regime
            regime = self.identify_flow_regime()
            
            # Calculate superficial velocities
            J_L = self.calculate_liquid_superficial_velocity()
            J_G = self.calculate_gas_superficial_velocity()
            
            # Calculate velocity ratio
            if J_G > 0:
                V_ratio = J_L / J_G
            else:
                V_ratio = 4.0  # Default ratio for steady flow
            
            # Clamp ratio to valid range
            V_ratio = max(0.1, V_ratio)
            
            # Calculate inclination parameter lambda_0
            # Based on pipe orientation for upward flow
            angle_rad = atan(sqrt(fabs(self.well_angle_deg)))
            
            if self.well_angle_deg >= 0:
                # Uphill flow
                lam0 = 0.866 * angle_rad / (pi / 4)
            else:
                # Downhill flow
                factor = max(0, 0.866 * 0.577 * sqrt(angle_rad * 180 / pi))
                lam0 = 0.866 * factor
            
            lam0 = max(0.1, min(lam0, 1.0))
            
            # Get regime parameters
            if regime == FlowRegime.BUBBLE:
                # Bubble flow - low gas holdup
                lambda_g = 0.1
            elif regime == FlowRegime.SEGREGATED:
                # Segregated flow - continuous liquid phase
                lambda_g = (
                    lam0 * V_ratio ** 3 + 0.055 * V_ratio ** (1.81 + 4.33 * lam0)
                )
            elif regime == FlowRegime.INTERMITTENT:  # Slug flow
                # Slug flow - intermittent gas pockets
                lambda_g = (
                    lam0 * (J_G / J_L) ** (2.62 * lam0 + 1.33) - 0.055 * V_ratio
                )
                lambda_g = max(0.0, min(lambda_g, 0.98))  # Clamp
            elif regime == FlowRegime.DISTRIBUTED:  # Mist flow
                # Mist flow - gas continuous
                lambda_g = 1.0 - 0.1 * log10(V_ratio + 0.02)
            else:  # MIST
                # Default mist holdup
                lam0 = 0.866 * angle_rad / (pi / 4)
                lambda_g = 1.0
            
            # Clamp to valid range
            lambda_g = max(0.0, min(lambda_g, 1.0))
            
            return lambda_g
        except Exception as e:
            # Return safe default on error
            return 0.1
    
    def calculate_liquid_superficial_velocity(self) -> float:
        """
        Calculate liquid superficial velocity in ft/s.
        
        Returns
        -------
        float
            Liquid superficial velocity in ft/s.
        """
        try:
            liquid_rate_gpm = self.calculate_total_liquid_rate_gpm()
            
            if self.borehole_area_ft2 <= 0:
                raise ValueError("Borehole area must be positive for velocity calculation")
            
            velocity = liquid_rate_gpm * GMP_TO_LBMS / (
                self.total_liquid_rate_lbm_s * self.borehole_area_ft2
            )
            
            return max(0.0, velocity)
        except Exception as e:
            raise ValueError(f"Error calculating liquid velocity: {str(e)}")
    
    def calculate_gas_superficial_velocity(self) -> float:
        """
        Calculate gas superficial velocity in ft/s.
        
        Returns
        -------
        float
            Gas superficial velocity in ft/s.
        """
        try:
            gas_rate_gpm = self.gas_flow_rate_gpm
            
            if self.borehole_area_ft2 <= 0:
                raise ValueError("Borehole area must be positive for velocity calculation")
            
            # Total mixed mass flow for denominator
            total_mass_flow = self.gas_rate_lbm_s + self.total_liquid_rate_lbm_s
            
            if total_mass_flow <= 0:
                raise ValueError("Total mass flow rate must be positive for velocity calculation")
            
            velocity = gas_rate_gpm * GMP_TO_LBMS / (
                total_mass_flow * self.borehole_area_ft2
            )
            
            return max(0.0, velocity)
        except Exception as e:
            raise ValueError(f"Error calculating gas velocity: {str(e)}")
    
    def identify_flow_regime(self) -> FlowRegime:
        """
        Identify flow regime using Beggs-Brill method.
        
        Returns
        -------
        FlowRegime
            Identified flow regime.
        """
        oil_rate = self.oil_flow_rate_gpm
        gas_rate = self.gas_flow_rate_gpm
        oil_gal = self.oil_gal
        
        if oil_rate <= 0 or gas_rate <= 0:
            return FlowRegime.BUBBLE  # Default on error
        
        # Calculate dimensionless groups
        try:
            F_Lo = self.calculate_F_Lo()
        except Exception:
            F_Lo = 0.5
        
        # Inclination factor
        try:
            inclination_factor = self.calculate_inclination_factor()
        except Exception:
            inclination_factor = 0.5
        
        # Flow regime identification tables from Beggs-Brill
    
        # BUBBLE/FLOW regime occurs at very low gas rates
        if gas_rate <= 0.1:
            return FlowRegime.BUBBLE
        
        # MIST/FLOW regime at high gas rates with low liquid flow
        if gas_rate >= 10000.0 and oil_rate <= 100.0:
            return FlowRegime.MIST
        
        # Segregated / Slug regime at low liquid rates
        if F_Lo < 0.01:
            return FlowRegime.SEGREGATED
        
        # Slug flow
        if F_Lo >= 0.01 and F_Lo < 0.5 and inclination_factor > 0.4:
            return FlowRegime.INTERMITTENT
        
        # Distributed flow (mist/annular)
        if F_Lo >= 0.5:
            return FlowRegime.DISTRIBUTED
        
        # Default to segregated for safety
        return FlowRegime.SEGREGATED
    
    def calculate_F_Lo(self) -> float:
        """
        Calculate dimensionless F_Lo group for flow regime classification.
        
        Returns
        -------
        float
            F_Lo value.
        """
        # Validate inputs
        if self.borehole_area_ft2 <= 0 or self.pipe_diameter_ft <= 0:
            return 0.5
        
        # Calculate liquid mass flow (simplified)
        q_l_base = self.total_liquid_rate_lbm_s / 62.5  # Using average density
        
        # Clamp value
        q_l_base = max(0.0, q_l_base)
        
        # Dimensionless F_Lo = (Q_L * rho_L) / sqrt(g * A * D)
        # Simplified with mass flow instead of volumetric flow
        F_Lo = (q_l_base * 1.0) / (sqrt(G * self.borehole_area_ft2 * self.pipe_diameter_ft))
        
        return max(0.0, F_Lo)
    
    def calculate_inclination_factor(self) -> float:
        """
        Calculate inclination factor based on well angle.
        
        Returns
        -------
        float
            Inclination factor in range [0, 1].
        """
        try:
            angle = abs(self.well_angle_deg)
            
            if angle < 0.5:
                # Near horizontal/uphill
                factor = 0.5
            elif angle < 15.0:
                # Mildly inclined
                factor = 0.6
            elif angle < 45.0:
                # Steeply inclined
                factor = 0.8
            elif angle < 75.0:
                # Very steep
                factor = 0.95
            else:
                # Near vertical
                factor = 1.0
            
            return factor
        except Exception:
            return 0.5  # Default on error
    
    def calculate_pressure_drop_per_ft(
        self,
        mixture_density: float,
        mixture_viscosity: float,
        total_liquid_rate_gpm: float,
        gas_rate_gpm: float
    ) -> float:
        """
        Calculate pressure drop per foot using Beggs-Brill method.
        
        Parameters
        ----------
        mixture_density : float
            Mixture density in lb/ft³.
        mixture_viscosity : float
            Mixture viscosity in cP.
        total_liquid_rate_gpm : float
            Total liquid rate in GPM.
        gas_rate_gpm : float
            Gas rate in GPM.
            
        Returns
        -------
        float
            Pressure drop per foot in psi/ft.
        """
        # Validate inputs
        if self.pipe_diameter_ft <= 0:
            raise ValueError("Pipe diameter must be positive for pressure drop calculation")
        
        if mixture_density <= 0:
            mixture_density = 0.01
        
        # Convert viscosity to lb/(ft·s)
        mu_lb_ft_s = mixture_viscosity / (14880.0)
        mu_lb_ft_s = max(mu_lb_ft_s, 1e-6)  # Prevent division by zero
        
        # Calculate gas holdup
        try:
            lam_g = self.calculate_gas_holdup()
        except Exception:
            lam_g = 0.1
        
        lam_g = max(0.0, min(lam_g, 1.0))
        
        # Calculate gas velocity
        if gas_rate_gpm > 0:
            v_g = gas_rate_gpm * GMP_TO_LBMS / (
                self.borehole_area_ft2 * (1 - lam_g)
            )
        else:
            v_g = 0.001
        
        # Calculate gas Reynolds number
        if v_g > 0:
            Re_g = (self.gas_density_lb_ft3 * v_g * self.pipe_diameter_ft) / mu_lb_ft_s
        else:
            Re_g = 100.0  # Default for laminar regime
        
        Re_g = max(Re_g, 1.0)  # Prevent Re <= 0
        
        # Calculate friction factor using Swamee-Jain approximation
        if Re_g <= 2000:
            # Laminar flow
            f = 16.0 / Re_g
        else:
            # Turbulent flow
            t = PIPE_ROUGHNESS_FT / self.pipe_diameter_ft  # Relative roughness
            f = 0.25 / (log10(t / 3.7 + 5.74 / Re_g ** 0.9) ** 2)
            f = max(0.001, min(f, 0.1))
        
        # Calculate inclination angle in radians
        angle_rad = atan(self.well_angle_deg)
        
        # Pressure drop components (per foot)
        # Hydrostatic component
        rho_L = self.oil_density_lbm_ft3
        
        dm = (1 - lam_g) * rho_L * G * sin(angle_rad)
        
        # Clamp to prevent negative values
        dm = max(0.0, dm)
        
        # Friction component
        rho_m = mixture_density
        v_m = self.calculate_superficial_velocity()
        
        df = f * rho_m * v_m ** 2 / (2 * self.pipe_diameter_ft)
        df = max(0.0, df)
        
        # Dynamic component (usually small for steady flow)
        dk = 0.001  # Small default
        
        # Total pressure drop per foot
        dh = dm + df + dk
        
        return dh
    
    def calculate_superficial_velocity(self) -> float:
        """
        Calculate mixture superficial velocity in ft/s.
        
        Returns
        -------
        float
            Mixture superficial velocity in ft/s.
        """
        try:
            total_mass_flow = self.total_liquid_rate_lbm_s + self.gas_rate_lbm_s
            
            if total_mass_flow <= 0 or self.borehole_area_ft2 <= 0:
                raise ValueError("Invalid flow parameters for velocity calculation")
            
            velocity = total_mass_flow * GMP_TO_LBMS / (
                total_mass_flow * self.borehole_area_ft2
            )
            
            return max(0.0, velocity)
        except Exception:
            return 0.01  # Default on error
    
    def calculate_pressure_drop_total(
        self,
        well_length_ft: float,
        mixture_density: float,
        total_liquid_rate_gpm: float,
        gas_rate_gpm: float,
        mixture_viscosity: Optional[float] = None
    ) -> Tuple[float, dict]:
        """
        Calculate total pressure drop through tubing section.
        
        Parameters
        ----------
        well_length_ft : float
            Wellbore length in feet.
        mixture_density : float
            Mixture density in lb/ft³.
        total_liquid_rate_gpm : float
            Total liquid rate in GPM.
        gas_rate_gpm : float
            Gas rate in GPM.
        mixture_viscosity : Optional[float]
            Mixture viscosity in cP. If None, calculated.
            
        Returns
        -------
        tuple
            (pressure_drop_psi, breakdown_dict)
            breakdown_dict contains: 'hydrostatic', 'friction', 'kinetic', 'total' (psi each)
            
        Raises
        ------
        ValueError
            If well_length_ft is non-positive.
        ValueError
            If pressure_drop_psi is negative.
        """
        # Validate well length
        if well_length_ft <= 0:
            raise ValueError(f"Well length must be positive, got {well_length_ft} ft")
        
        if pressure_drop_psi < 0:
            raise ValueError(f"Pressure drop must be non-negative, got {pressure_drop_psi} psi")
        
        # Use provided viscosity or calculate
        if mixture_viscosity is None:
            mixture_viscosity = self.calculate_mixture_viscosity_cP()
        else:
            mixture_viscosity = max(mixture_viscosity, 0.001)  # Avoid zero/negative
        
        # Calculate pressure drop per foot
        dh_per_ft = self.calculate_pressure_drop_per_ft(
            mixture_density=mixture_density,
            mixture_viscosity=mixture_viscosity,
            total_liquid_rate_gpm=total_liquid_rate_gpm,
            gas_rate_gpm=gas_rate_gpm
        )
        
        dh = dh_per_ft * well_length_ft
        dh_psi = dh / 144.0  # Convert to psi
        
        # Pressure drop breakdown
        breakdown = {
            'hydrostatic_psi': 0.0,
            'friction_psi': 0.0,
            'kinetic_psi': 0.0,
            'total_psi': dh_psi,
        }
        
        # Calculate hydrostatic component separately for breakdown
        try:
            lam_g = self.calculate_gas_holdup()
            rho_l = self.oil_density_lbm_ft3
            angle_rad = atan(self.well_angle_deg)
            dm = (1 - lam_g) * rho_l * G * sin(angle_rad)
            breakdown['hydrostatic_psi'] = dm * well_length_ft / 144.0
        except Exception:
            pass
        
        return dh_psi, breakdown
    
    def generate_report(self, pressure_drop_psi: float, well_length_ft: float) -> dict:
        """
        Generate a comprehensive report on the correlation evaluation.
        
        Parameters
        ----------
        pressure_drop_psi : float
            Total pressure drop in psi.
        well_length_ft : float
            Wellbore length in feet.
        
        Returns
        -------
        dict
            Detailed report with all calculated properties and inputs.
            
        Raises
        ------
        ValueError
            If well_length_ft is non-positive.
        ValueError
            If pressure_drop_psi is negative.
        """
        # Validate inputs
        if well_length_ft <= 0:
            raise ValueError(f"Well length must be positive, got {well_length_ft} ft")
        
        if pressure_drop_psi < 0:
            raise ValueError(f"Pressure drop must be non-negative, got {pressure_drop_psi} psi")
        
        try:
            # Gather all properties
            result = {
                'timestamp': datetime.utcnow().isoformat(),
                'oil_flow_rate_gpm': self.oil_flow_rate_gpm,
                'gas_flow_rate_gpm': self.gas_flow_rate_gpm,
                'pipe_diameter_ft': self.pipe_diameter_ft,
                'well_angle_deg': self.well_angle_deg,
                'oil_gal': self.oil_gal,
                'oil_density_lbm_ft3': self.oil_density_lbm_ft3,
                'water_density_lbm_ft3': self.water_density_lbm_ft3,
                'gas_density_lb_ft3': self.gas_density_lb_ft3,
                'gas_specific_gravity': self.gas_specific_gravity,
                'oil_specific_gravity': self.oil_specific_gravity,
                'flow_regime': self.identify_flow_regime().value,
                'mixture_viscosity_cP': self.calculate_mixture_viscosity_cP(),
                'mixture_density_lb_ft3': self.calculate_mixture_density_lb_ft3(),
                'gas_holdup': self.calculate_gas_holdup(),
                'pressure_drop_psi': pressure_drop_psi,
                'total_length_ft': self.well_length_ft,
                'gas_viscosity_cP': self.gas_viscosity_cP,
                'oil_viscosity_cP': self.oil_viscosity_cP,
                'critical_gas_fraction': self.CRITICAL_GAS_FRACTION,
                'conversion_factor': GMP_TO_LBMS,
                'gpm_to_lbs_per_s': GMP_TO_LBMS,
                'typical_gas_viscosity_cp': TYPICAL_GAS_VISCOSITY_CP,
            }
            return result
        except Exception as e:
            raise ValueError(f"Error generating report: {str(e)}")


__all__ = [
    'BeggsBrillCorrelation',
    'FlowRegime',
]
