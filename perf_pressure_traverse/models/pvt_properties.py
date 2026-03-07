"""Reservoir PVT properties model."""

from __future__ import annotations

from typing import Optional


class PVTProperties:
    """
    Reservoir PVT properties model.
    
    Attributes
    ----------
    reservoir_pressure_psia : float
        Reservoir pressure in psi.
    reservoir_temperature_f : float
        Reservoir temperature in °F.
    oil_density_lb_ft3 : float
        Oil density at reservoir conditions in lb/ft³.
    gas_density_lb_ft3 : float
        Gas density at reservoir conditions in lb/ft³.
    water_density_lb_ft3 : float
        Water (formation water) density in lb/ft³.
    oil_viscosity_cP : float
        Oil viscosity at reservoir conditions in centipoise.
    gas_viscosity_cP : float
        Gas viscosity at reservoir conditions in centipoise.
    formation_volume_factor_oil_res : float
        Oil formation volume factor at reservoir conditions.
    formation_volume_factor_gas_res : float
        Gas formation volume factor at reservoir conditions.
    """
    
    def __init__(
        self,
        reservoir_pressure_psia: float,
        reservoir_temperature_f: float,
        oil_density_lb_ft3: float = 0.0,
        gas_density_lb_ft3: float = 0.0,
        water_density_lb_ft3: float = 0.0,
        oil_viscosity_cP: float = 0.0,
        gas_viscosity_cP: float = 0.0,
        formation_volume_factor_oil_res: float = 1.0,
        formation_volume_factor_gas_res: float = 1.0,
    ) -> None:
        """
        Initialize PVT properties.
        
        Parameters
        ----------
        reservoir_pressure_psia : float
            Reservoir pressure in psi.
        reservoir_temperature_f : float
            Reservoir temperature in °F.
        oil_density_lb_ft3 : float, optional
            Oil density in lb/ft³.
        gas_density_lb_ft3 : float, optional
            Gas density in lb/ft³.
        water_density_lb_ft3 : float, optional
            Water density in lb/ft³.
        oil_viscosity_cP : float, optional
            Oil viscosity in cP.
        gas_viscosity_cP : float, optional
            Gas viscosity in cP.
        formation_volume_factor_oil_res : float, optional
            Oil formation volume factor at reservoir.
        formation_volume_factor_gas_res : float, optional
            Gas formation volume factor at reservoir.
        """
        self.reservoir_pressure_psia = reservoir_pressure_psia
        self.reservoir_temperature_f = reservoir_temperature_f
        self.oil_density_lb_ft3 = oil_density_lb_ft3
        self.gas_density_lb_ft3 = gas_density_lb_ft3
        self.water_density_lb_ft3 = water_density_lb_ft3
        self.oil_viscosity_cP = oil_viscosity_cP
        self.gas_viscosity_cP = gas_viscosity_cP
        self.formation_volume_factor_oil_res = formation_volume_factor_oil_res
        self.formation_volume_factor_gas_res = formation_volume_factor_gas_res
    
    def __repr__(self) -> str:
        """String representation of PVT properties."""
        return (
            f"PVTProperties(res_psia={self.reservoir_pressure_psia}, "
            f"res_temp={self.reservoir_temperature_f}, "
            f"oil_rho={self.oil_density_lb_ft3})"
        )
