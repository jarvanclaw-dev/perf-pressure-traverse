"""Fluid model representing fluid properties and behavior."""

from __future__ import annotations

from typing import Optional, List, Dict, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum

from perf_pressure_traverse.models.fluid import FluidProperties
from perf_pressure_traverse.math.z_factor import calculate_z_factor_aga_dc


class FluidType(Enum):
    """Fluid type enumeration."""
    OIL = "oil"
    GAS = "gas"
    WATER = "water"
    MIST = "mist"
    OIL_GAS = "oil_gas"  # Two-phase


@dataclass
class FluidModel:
    """
    Fluid model representing fluid properties and behavior.
    
    Attributes
    ----------
    fluid_type : FluidType
        Type of fluid.
    properties : FluidProperties
        Base fluid properties.
    molecular_weight : Optional[float]
        Molecular weight of the fluid.
    specific_gravity_gas : float
        Specific gravity of gas (air = 1.0).
    specific_gravity_oil : float
        Specific gravity of oil (water = 1.0).
    composition : Optional[Dict[str, float]]
        Fractional composition of multiphase fluid.
    surface_tension_nt_m : Optional[float]
        Surface tension at surface conditions in N/m.
        For gas-liquid interfaces.
    compressibility_factor_z : Optional[float]
        Gas compressibility factor.
        Z-factor for gas volumetrics.
    viscosity_corr_factor : Optional[float]
        Viscosity correction factor.
    is_gasoline : bool
        Whether fluid is gasoline or condensate.
    is_crude_oil : bool
        Whether fluid is crude oil.
    is_condensate : bool
        Whether fluid is light gas condensate.
    """
    
    fluid_type: FluidType
    properties: FluidProperties
    molecular_weight: Optional[float] = None
    specific_gravity_gas: float = 1.0
    specific_gravity_oil: float = 1.0
    composition: Optional[Dict[str, float]] = None
    surface_tension_nt_m: Optional[float] = None
    compressibility_factor_z: Optional[float] = None
    viscosity_corr_factor: Optional[float] = None
    is_gasoline: bool = False
    is_crude_oil: bool = False
    is_condensate: bool = False
    calculated_z_factor: Optional[float] = None
    
    @classmethod
    def from_properties(
        cls, 
        fluid_type: FluidType,
        reservoir_pressure_psia: float,
        reservoir_temperature_f: float,
        **kwargs
    ) -> 'FluidModel':
        """
        Create fluid model from reservoir PVT properties.
        
        Parameters
        ----------
        fluid_type : FluidType
            Fluid type.
        reservoir_pressure_psia : float
            Reservoir pressure.
        reservoir_temperature_f : float
            Reservoir temperature.
        **kwargs
            Additional properties and parameters.
        
        Returns
        -------
        FluidModel
            Fluid model instance.
        """
        return cls(
            fluid_type=fluid_type,
            properties=FluidProperties(
                reservoir_pressure_psia=reservoir_pressure_psia,
                reservoir_temperature_f=reservoir_temperature_f,
            ),
            **kwargs
        )
    
    @classmethod
    def from_surface_conditions(
        cls, 
        fluid_type: FluidType,
        surface_pressure_psia: float,
        surface_temperature_f: float,
        **kwargs
    ) -> 'FluidModel':
        """
        Create fluid model from surface conditions.
        
        Parameters
        ----------
        fluid_type : FluidType
            Fluid type.
        surface_pressure_psia : float
            Surface pressure.
        surface_temperature_f : float
            Surface temperature.
        **kwargs
            Additional properties and parameters.
        
        Returns
        -------
        FluidModel
            Fluid model instance.
        """
        return cls(
            fluid_type=fluid_type,
            properties=FluidProperties(
                reservoir_pressure_psia=surface_pressure_psia,
                reservoir_temperature_f=surface_temperature_f,
            ),
            **kwargs
        )
    
    def get_gas_density_lb_ft3(self) -> float:
        """
        Calculate gas density using ideal gas law and Z-factor.
        
        Returns
        -------
        float
            Gas density in lb/ft³.
        """
        if self.properties.gas_density_lb_ft3 > 0:
            return self.properties.gas_density_lb_ft3
    
        # Use ideal gas law: rho = PG / (Z * R * T)
        # R = 10.73 (psi * ft³ / lb-mol * °R)
        T_rankine = self.properties.reservoir_temperature_f + 459.67
        R = 10.73
        
        if self.compressibility_factor_z and self.molecular_weight and T_rankine > 0:
            gas_density = (
                self.properties.reservoir_pressure_psia * self.molecular_weight
                / (self.compressibility_factor_z * R * T_rankine * 144)
            )
            return gas_density
        return 0.0
    
    def get_oil_density_lb_ft3(self) -> float:
        """
        Calculate or retrieve oil density.
        
        Returns
        -------
        float
            Oil density in lb/ft³.
        """
        return self.properties.oil_density_lb_ft3
    
    def get_water_density_lb_ft3(self) -> float:
        """
        Calculate or retrieve water density.
        
        Returns
        -------
        float
            Water density in lb/ft³.
        """
        return self.properties.water_density_lb_ft3
    
    def get_gas_viscosity_cP(self) -> float:
        """
        Calculate gas viscosity using theoretical correlation.
        
        Returns
        -------
        float
            Gas viscosity in cP.
        """
        if self.properties.gas_viscosity_cP > 0:
            return self.properties.gas_viscosity_cP
        
        # Lee-Gonzalez-Eakin correlation
        # mu_g = (10^(-4) * exp(A * rho_g^B)) / S^D
        # where rho_g in lb/ft³ (mixture density), S = surface tension (N/m)
        
        gas_rho = self.get_gas_density_lb_ft3()
        if gas_rho <= 0 or not self.surface_tension_nt_m:
            return 0.0
        
        # Calculate S factor (N/m to lb/ft)
        S = self.surface_tension_nt_m * 0.06896
        
        # Simplified Lee-Gonzalez-Eakin coefficients
        A = 1.0  # Simplified
        B = 2.0  # Simplified
        D = 1.0  # Simplified
        
        mu_g = (1e-4) * (A * (gas_rho ** B)) / (S ** D)
        
        if viscosity_corr_factor:
            mu_g *= self.viscosity_corr_factor
        
        return mu_g
    
    def get_oil_viscosity_cP(self) -> float:
        """
        Calculate or retrieve oil viscosity.
        
        Returns
        -------
        float
            Oil viscosity in cP.
        """
        return self.properties.oil_viscosity_cP
    
    def get_formation_volume_factor(self) -> float:
        """
        Calculate formation volume factor for the fluid.
        
        Returns
        -------
        float
            Formation volume factor (FVs or FVB).
        """
        if self.fluid_type == FluidType.GAS:
            return self.properties.formation_volume_factor_gas_res
        else:
            return self.properties.formation_volume_factor_oil_res
    
    def get_surface_viscosity_cP(self) -> float:
        """
        Get viscosity at surface conditions from properties.
        
        Returns
        -------
        float
            Viscosity at surface in cP.
        """
        if self.fluid_type == FluidType.GAS:
            return self.properties.gas_viscosity_cP or 0.0
        else:
            return self.properties.oil_viscosity_cP or 0.0
    
    def calculate_mixture_density_lb_ft3(
        self,
        quality: float = 0.0,
        oil_fraction: float = 1.0,
        water_fraction: float = 0.0
    ) -> float:
        """
        Calculate mixture density based on quality and fractions.
        
        Parameters
        ----------
        quality : float, optional
            Gas quality (x) - fraction of gas.
        oil_fraction : float, optional
            Oil fraction.
        water_fraction : float, optional
            Water fraction.
        
        Returns
        -------
        float
            Mixture density in lb/ft³.
        """
        rho_g = self.get_gas_density_lb_ft3()
        rho_o = self.get_oil_density_lb_ft3()
        rho_w = self.get_water_density_lb_ft3()
        
        # Mass weighted density: rho_m = x * rho_g + (1-x) * rho_o
        # Assuming water fraction is small and can be approximated as oil
        rho_m = (quality * rho_g) + ((oil_fraction + water_fraction - quality) * rho_o)
        
        return rho_m
    
    def update_pvt_properties(
        self,
        pressure_psia: float,
        temperature_f: float
    ) -> None:
        """
        Update PVT properties for new conditions.
        
        This is a placeholder method for dynamic PVT updates.
        
        Parameters
        ----------
        pressure_psia : float
            New pressure.
        temperature_f : float
            New temperature.
        """
        self.properties.reservoir_pressure_psia = pressure_psia
        self.properties.reservoir_temperature_f = temperature_f
    
    def validate(self) -> bool:
        """
        Validate fluid model parameters.
        
        Returns
        -------
        bool
            True if model is valid.
        """
        errors = []

        if self.properties.reservoir_pressure_psia <= 0:
            errors.append("Pressure must be positive")
        if self.properties.reservoir_temperature_f < -273.15:
            errors.append("Temperature must be non-negative")
        if self.properties.reservoir_temperature_f > 300:
            errors.append("Temperature is unusually high for reservoir conditions")
        
        if self.fluid_type in (FluidType.OIL, FluidType.OIL_GAS, FluidType.MIST):
            if self.properties.oil_viscosity_cP <= 0:
                errors.append("Oil viscosity must be positive")
        
        if self.fluid_type == FluidType.GAS:
            if self.compressibility_factor_z is None:
                errors.append("Compressibility factor (Z-factor) is recommended")
        
        return len(errors) == 0
    
    def to_dict(self) -> Dict:
        """
        Convert to dictionary.
        
        Returns
        -------
        dict
            Dictionary representation.
        """
        return {
            'fluid_type': self.fluid_type.value,
            'reservoir_pressure_psia': self.properties.reservoir_pressure_psia,
            'reservoir_temperature_f': self.properties.reservoir_temperature_f,
            'molecular_weight': self.molecular_weight,
            'specific_gravity_gas': self.specific_gravity_gas,
            'specific_gravity_oil': self.specific_gravity_oil,
            'surface_tension_nt_m': self.surface_tension_nt_m,
            'compressibility_factor_z': self.compressibility_factor_z,
            'is_gasoline': self.is_gasoline,
            'is_crude_oil': self.is_crude_oil,
            'is_condensate': self.is_condensate,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FluidModel':
        """
        Create fluid model from dictionary.
        
        Parameters
        ----------
        data : dict
            Dictionary data.
        
        Returns
        -------
        FluidModel
            Fluid model instance.
        """
        fluid_type = FluidType(data.get('fluid_type', 'oil'))
        
        return cls(
            fluid_type=fluid_type,
            properties=FluidProperties(
                reservoir_pressure_psia=data.get('reservoir_pressure_psia', 14.7),
                reservoir_temperature_f=data.get('reservoir_temperature_f', 60),
                oil_density_lb_ft3=data.get('properties', {}).get('oil_density_lb_ft3', 0),
                gas_density_lb_ft3=data.get('properties', {}).get('gas_density_lb_ft3', 0),
                water_density_lb_ft3=data.get('properties', {}).get('water_density_lb_ft3', 62.4),
                oil_viscosity_cP=data.get('properties', {}).get('oil_viscosity_cP', 1.0),
                gas_viscosity_cP=data.get('properties', {}).get('gas_viscosity_cP', 0.02),
                formation_volume_factor_oil_res=data.get('properties', {}).get('formation_volume_factor_oil_res', 1.0),
                formation_volume_factor_gas_res=data.get('properties', {}).get('formation_volume_factor_gas_res', 1.0),
            ),
            molecular_weight=data.get('molecular_weight'),
            specific_gravity_gas=data.get('specific_gravity_gas', 1.0),
            specific_gravity_oil=data.get('specific_gravity_oil', 1.0),
            composition=data.get('composition'),
            surface_tension_nt_m=data.get('surface_tension_nt_m'),
            compressibility_factor_z=data.get('compressibility_factor_z'),
            is_gasoline=data.get('is_gasoline', False),
            is_crude_oil=data.get('is_crude_oil', False),
            is_condensate=data.get('is_condensate', False),
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FluidModel(type={self.fluid_type.value}, "
            f"P={self.properties.reservoir_pressure_psia:.1f}PSI, "
            f"T={self.properties.reservoir_temperature_f:.1f}°F)"
        )


class FluidModelFactory:
    """Factory for creating fluid models from different sources."""
    
    @staticmethod
    def create_regular_oil(reservoir_pressure_psia: float = 2000.0) -> FluidModel:
        """Create a regular crude oil fluid model."""
        return FluidModel(
            fluid_type=FluidType.OIL_GAS,
            properties=FluidProperties(reservoir_pressure_psia=reservoir_pressure_psia),
            specific_gravity_oil=0.85,
            is_crude_oil=True,
        )
    
    @staticmethod
    def create_gas(reservoir_pressure_psia: float = 2500.0) -> FluidModel:
        """Create a gas fluid model."""
        return FluidModel(
            fluid_type=FluidType.GAS,
            properties=FluidProperties(
                reservoir_pressure_psia=reservoir_pressure_psia,
                reservoir_temperature_f=180.0,
                formation_volume_factor_gas_res=1.0,
            ),
            compressibility_factor_z=0.9,
            molecular_weight=16.0,
            specific_gravity_gas=0.6,
        )
    
    @staticmethod
    def create_condensate(reservoir_pressure_psia: float = 3000.0) -> FluidModel:
        """Create a condensate fluid model."""
        return FluidModel(
            fluid_type=FluidType.GAS,
            properties=FluidProperties(reservoir_pressure_psia=reservoir_pressure_psia),
            is_condensate=True,
        )
    
    @staticmethod
    def create_wellstream(flow_rate_oil_gpd: float = 1000.0) -> FluidModel:
        """Create a wellstream (oil+gas) fluid model."""
        quality = min(flow_rate_oil_gpd / 100000.0, 0.95)
        return FluidModel(
            fluid_type=FluidType.OIL_GAS,
            properties=FluidProperties(
                reservoir_pressure_psia=2000.0,
                oil_viscosity_cP=1.5,
            ),
            is_crude_oil=True,
            specific_gravity_oil=0.8,
        )