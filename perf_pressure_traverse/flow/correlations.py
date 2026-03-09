"""Multiphase flow correlations for pressure drop calculations.

Implements Beggs & Brill (1973) multiphase flow correlation for tubular flow,
including viscosity, density, holdup, and pressure drop calculations.
"""

from __future__ import annotations

from datetime import datetime
from math import log10, sqrt, exp, atan, fabs, pi
from typing import Tuple, Optional
from enum import Enum

# Convert GPM to lbm/s
GPM_TO_LBMS = 500 / 60.0  # 500 lbm per minute per gallon

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
    oil_gal : float
        Oil gravity in API.
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
    
    # Critical gas fraction for transition between BUBBLE and MIST
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
        """
        # Store all parameters as regular attributes
        self.oil_flow_rate_gpm = oil_flow_rate_gpm
        self.gas_flow_rate_gpm = gas_flow_rate_gpm
        self.pipe_diameter_ft = pipe_diameter_ft
        self.borehole_area_ft2 = borehole_area_ft2
        self.well_angle_deg = well_angle_deg
        self.oil_gal = oil_gal
        self.oil_density_lbm_ft3 = oil_density_lbm_ft3
        self.water_density_lbm_ft3 = water_density_lbm_ft3
        self.gas_density_lb_ft3 = gas_density_lb_ft3
        self.gas_specific_gravity = gas_specific_gravity
        self.oil_specific_gravity = oil_specific_gravity
        
        # Calculate total liquid rate (oil + water) in lbm/s
        self._total_liquid_rate_lbm_s = (
            self.oil_flow_rate_gpm * oil_density_lbm_ft3 * GPM_TO_LBMS +
            self.gas_flow_rate_gpm * water_density_lbm_ft3 * GPM_TO_LBMS
        )
        
        # Calculate gas rate in lbm/s
        self._gas_rate_lbm_s = (
            self.gas_flow_rate_gpm * self.gas_specific_gravity * 0.0765 * GPM_TO_LBMS
        )
        
        # Private viscosity
        self._oil_viscosity_cP = oil_viscosity_cP
    
    @property
    def oil_viscosity_cP(self) -> float:
        """Oil viscosity getter."""
        return self._oil_viscosity_cP
    
    @property
    def gas_viscosity_cP(self) -> Optional[float]:
        """Gas viscosity getter."""
        return 0.02
    
    @property
    def _gas_viscosity_cP(self) -> float:
        """Gas viscosity getter."""
        return 0.02
    
    @property
    def _o_gr(self) -> float:
        """Oil gas ratio getter."""
        return 300.0
    
    @property
    def _g_wr(self) -> float:
        """Gas water ratio getter."""
        return 0.0
    
    @property
    def _pressure(self) -> Optional[float]:
        """Pressure getter."""
        return None
    
    @property
    def _temperature(self) -> Optional[float]:
        """Temperature getter."""
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
        """Oil gas ratio getter."""
        return self._o_gr
    
    @property
    def gas_water_ratio(self) -> Optional[float]:
        """Gas water ratio getter."""
        return self._g_wr
    
    @property
    def pressure_psia(self) -> Optional[float]:
        """Pressure getter."""
        return self._pressure
    
    @property
    def temperature_f(self) -> Optional[float]:
        """Temperature getter."""
        return self._temperature
    
    @property
    def true_vertical_depth_ft(self) -> float:
        """True vertical depth getter."""
        return 144.0  # Approximate
    
    def calculate_mixture_viscosity_cP(self) -> float:
        """
        Calculate multiphase mixture viscosity using Beggs-Brill method.
        
        Returns
        -------
        float
            Mixture viscosity in cP.
        """
        # Use the oil viscosity provided
        mu_o = self.oil_viscosity_cP
        
        # Typical gas viscosity
        mu_g = 0.02
        
        # Calculate quality (gas fraction)
        if self.gas_rate_lbm_s > 0:
            quality = self.gas_rate_lbm_s / (self.gas_rate_lbm_s + self.total_liquid_rate_lbm_s)
        else:
            quality = 0
        
        # Calculate λ (liquid-to-gas flow rate ratio at standard conditions)
        q_l = self.total_liquid_rate_lbm_s
        q_g = self.gas_rate_lbm_s
        
        if q_g > 0:
            lam = q_l / q_g
        else:
            lam = float('inf')
        
        # Calculate exponent x
        # This accounts for interaction between oil and gas phases
        
        # Calculate dimensionless parameter
        # μ_o in lb/ft·s = μ_o_cP / 1000 / 1488
        
        mu_o_lb_ft_s = mu_o / 1000.0 / 1488.0
        D_ft = self.pipe_diameter_ft
        
        # Dimensionless parameter
        if D_ft > 0:
            dimensionless = (9.81 * mu_o_lb_ft_s * D_ft) ** 4.75
        else:
            dimensionless = 0.001
        
        if lam == float('inf'):
            lam = 0
        
        if dimensionless <= 0:
            x = 1.0
        else:
            x = 1.0 + (0.012 * lam ** 0.762 * dimensionless) ** (-4.75)
        
        # Clamp x to valid range [0, 1]
        x = max(0.0, min(1.0, x))
        
        # Calculate mixture viscosity
        mu_m = mu_o ** x * mu_g ** (1 - x)
        
        return mu_m
    
    def calculate_total_liquid_rate_gpm(self) -> float:
        """
        Calculate total liquid flow rate including dissolved gas.
        
        Returns
        -------
        float
            Total liquid rate after gas dissolution correction in GPM.
        """
        # Get oil GOR (gas-oil ratio) from fluid properties
        # This would typically come from a PVT model
        # For simplicity, we'll use a typical value
        Rs = 300.0  # scf/STB
        
        # Gas dissolution factor
        dissolution = self.calculate_gas_dissolution_percentage()
        
        # Adjust gas flow rate
        adjusted_gas_gpm = self.gas_flow_rate_gpm * (1 - dissolution / 100.0)
        
        # Total liquid rate = oil rate + water rate
        # Water rate calculated from gas rate and gas water ratio
        gwr = 0.0  # scf/STB
        
        water_gpm = (Rs + gwr) * 0.00003  # Simplified conversion
        
        total_liquid_gpm = self.oil_flow_rate_gpm + water_gpm
        
        return total_liquid_gpm
    
    def calculate_gas_dissolution_percentage(self) -> float:
        """
        Calculate percentage of gas dissolved into oil.
        
        Returns
        -------
        float
            Percentage of gas dissolved.
        """
        # Simplified
        return 0.0
    
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
        """
        # Calculate total liquid density (excluding dispersed gas)
        rho_l = (
            self.oil_density_lbm_ft3 * oil_fraction +
            self.water_density_lbm_ft3 * water_fraction
        )
        
        # Calculate gas density
        rho_g = self.gas_density_lb_ft3 or (
            self.gas_specific_gravity * 62.4 / 62.5  # Approximate
        )
        
        # Calculate gas holdup (void fraction)
        lambda_g = self.calculate_gas_holdup()
        
        # Clamp values to valid range [0, 1]
        lambda_g = max(0.0, min(1.0, lambda_g))
        
        # Calculate mixture density
        rho_m = (1 - lambda_g) * rho_l + lambda_g * rho_g
        
        return rho_m
    
    def calculate_gas_holdup(self) -> float:
        """
        Calculate gas holdup (void fraction) using Beggs-Brill method.
        
        Returns
        -------
        float
            Gas holdup in fraction (0 to 1).
        """
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
        
        # Calculate inclination parameter λ_0
        # For vertical flow, λ_0 = 1.0
        # For horizontal flow, λ_0 = 0.2
        
        angle_rad = atan(sqrt(fabs(self.well_angle_deg)))
        
        if self.well_angle_deg >= 0:
            # Uphill flow
            lam0 = 0.866 if self.well_angle_deg == 0 else \
                   0.866 * angle_rad / (pi / 4)
        else:
            # Downhill flow
            factor = max(0, 0.866 * 0.577 * sqrt(angle_rad * 180 / pi))
            lam0 = 0.866 * factor
        
        lam0 = max(0.1, min(lam0, 1.0))
        
        # Get regime parameters
        if regime == FlowRegime.BUBBLE:
            # Bubble flow
            lambda_g = 0.1
        elif regime == FlowRegime.SEGREGATED:
            # Segregated flow
            lambda_g = (
                lam0 * V_ratio ** 3 + 0.055 * V_ratio ** (1.81 + 4.33 * lam0)
            )
        elif regime == FlowRegime.INTERMITTENT:  # Slug flow
            # Slug flow
            lambda_g = (
                lam0 * (J_G / J_L) ** (2.62 * lam0 + 1.33) - 0.055 * V_ratio
            )
            lambda_g = max(0.0, min(lambda_g, 0.98))  # Clamp
        elif regime == FlowRegime.DISTRIBUTED:  # Mist flow
            # Mist flow
            lambda_g = 1.0 - 0.1 * log10(V_ratio + 0.02)
        else:  # MIST
            # Mist flow
            lam0 = 0.866 * angle_rad / (pi / 4)
            lambda_g = 1.0
            
        # Clamp to valid range
        lambda_g = max(0.0, min(lambda_g, 1.0))
        
        return lambda_g
    
    def calculate_liquid_superficial_velocity(self) -> float:
        """
        Calculate liquid superficial velocity.
        
        Returns
        -------
        float
            Liquid superficial velocity in ft/s.
        """
        liquid_rate_gpm = self.calculate_total_liquid_rate_gpm()
        
        velocity = liquid_rate_gpm * GPM_TO_LBMS / (
            self.total_liquid_rate_lbm_s * self.borehole_area_ft2
        )
        
        return max(0.0, velocity)
    
    def calculate_gas_superficial_velocity(self) -> float:
        """
        Calculate gas superficial velocity.
        
        Returns
        -------
        float
            Gas superficial velocity in ft/s.
        """
        gas_rate_gpm = self.gas_flow_rate_gpm
        
        velocity = gas_rate_gpm * GPM_TO_LBMS / (
            (self.gas_rate_lbm_s + self.total_liquid_rate_lbm_s) * self.borehole_area_ft2
        )
        
        return max(0.0, velocity)
    
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
        
        # Calculate dimensionless groups
        F_Lo = self.calculate_F_Lo()
        
        # Inclination factor
        inclination_factor = self.calculate_inclination_factor()
        
        # Flow regime identification tables from Beggs-Brill
        
        # BUBBLE/FLOW regime occurs at low gas rates
        if gas_rate <= 0.1:
            return FlowRegime.BUBBLE
        
        # MIST/FLOW regime at very high gas rates
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
        
        # Default to segregated
        return FlowRegime.SEGREGATED
    
    def calculate_F_Lo(self) -> float:
        """
        Calculate dimensionless F_Lo group.
        
        Returns
        -------
        float
            F_Lo value.
        """
        # Simplified calculation
        q_l_base = self.total_liquid_rate_lbm_s / 62.4  # Simplified
        
        g = 32.2  # ft/s²
        A = self.borehole_area_ft2
        D = self.pipe_diameter_ft
        
        if A <= 0 or D <= 0:
            return 0.0
        
        F_Lo = (q_l_base * 1.0) / (sqrt(g * A * D))
        
        return F_Lo
    
    def calculate_inclination_factor(self) -> float:
        """
        Calculate inclination factor based on well angle.
        
        Returns
        -------
        float
            Inclination factor in range [0, 1].
        """
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
        # Convert viscosity to lb/(ft·s)
        mu_lb_ft_s = mixture_viscosity / 14880.0
        
        # Calculate gas holdup
        lam_g = self.calculate_gas_holdup()
        
        # Calculate gas velocity
        if gas_rate_gpm > 0:
            v_g = gas_rate_gpm * GPM_TO_LBMS / (
                self.borehole_area_ft2 * (1 - lam_g)
            )
        else:
            v_g = 0
        
        # Calculate gas Reynolds number
        if v_g > 0:
            Re_g = (self.gas_density_lb_ft3 * v_g * self.pipe_diameter_ft) / mu_lb_ft_s
        else:
            Re_g = 0
        
        # Calculate friction factor
        if Re_g <= 2000:
            # Laminar
            f = 16.0 / Re_g if Re_g > 0 else 0.001
        else:
            # Turbulent - use Swamee-Jain approximation
            t = 0.005 * 0.0019 / self.pipe_diameter_ft  # Roughness
            f = 0.25 / (log10(t / 3.7 + 5.74 / Re_g ** 0.9) ** 2)
            f = max(0.001, min(f, 0.1))
        
        # Calculate inclination angle in radians for sine term
        angle_rad = atan(self.well_angle_deg)
        
        # Pressure drop components (per foot)
        # Hydrostatic component
        rho_L = self.oil_density_lbm_ft3
        
        dm = (1 - lam_g) * rho_L * 32.2 * sin(angle_rad)
        
        # Friction component
        rho_m = mixture_density
        v_m = self.calculate_superficial_velocity()
        
        df = f * rho_m * v_m ** 2 / (2 * self.pipe_diameter_ft)
        df = max(0.0, df)
        
        # Dynamic component (usually small for steady flow)
        dk = 0.0
        
        # Total pressure drop per foot
        dh = dm + df + dk
        
        return dh
    
    def calculate_superficial_velocity(self) -> float:
        """
        Calculate mixture superficial velocity.
        
        Returns
        -------
        float
            Mixture superficial velocity in ft/s.
        """
        total_mass_flow = self.total_liquid_rate_lbm_s + self.gas_rate_lbm_s
        velocity = total_mass_flow * GPM_TO_LBMS / (
            total_mass_flow * self.borehole_area_ft2
        )
        return max(0.0, velocity)
    
    def calculate_pressure_drop_total(
        self,
        well_length_ft: float,
        mixture_density: float,
        total_liquid_rate_gpm: float,
        gas_rate_gpm: float
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
        
        Returns
        -------
        tuple
            (pressure_drop_psi, breakdown_dict)
            breakdown_dict contains: 'hydrostatic', 'friction', 'kinetic'
        """
        dh_per_ft = self.calculate_pressure_drop_per_ft(
            mixture_density=mixture_density,
            mixture_viscosity=self.calculate_mixture_viscosity_cP(),
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
        
        return dh_psi, breakdown
    
    def generate_report(self, pressure_drop_psi: float) -> dict:
        """
        Generate a comprehensive report on the correlation evaluation.
        
        Parameters
        ----------
        pressure_drop_psi : float
            Total pressure drop in psi.
        
        Returns
        -------
        dict
            Detailed report with all calculated properties and inputs.
        """
        return {
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
            'total_length_ft': well_length_ft,
        }


__all__ = [
    'BeggsBrillCorrelation',
    'FlowRegime',
]
