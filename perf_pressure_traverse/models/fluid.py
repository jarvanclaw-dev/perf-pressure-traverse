"""Surface fluid property model."""

from __future__ import annotations


class FluidProperties:
    """
    Surface fluid property model.
    
    Attributes
    ----------
    surface_pressure_psia : float
        Surface pressure in psi.
    surface_temperature_f : float
        Surface temperature in °F.
    oil_gravity_sg : float
        Oil specific gravity at standard conditions (density in lb/ft³).
    gas_gravity_sg : float
        Gas specific gravity (air = 1.0).
    water_specific_gravity : float
        Water specific gravity.
    water_cut : float
        Water cut as a fraction (0 to 1).
    gas_oil_ratio : float
        Gas-oil ratio at surface conditions (scf/stb or scf/bbl).
    solution_oil_ratio : float
        Solution gas-oil ratio (Rs) from reservoir to surface.
    """
    
    def __init__(
        self,
        surface_pressure_psia: float,
        surface_temperature_f: float,
        oil_gravity_sg: float = 0.9,
        gas_gravity_sg: float = 0.65,
        water_specific_gravity: float = 1.0,
        water_cut: float = 0.0,
        gas_oil_ratio: float = 0.0,
        solution_oil_ratio: float = 0.0,
    ) -> None:
        """
        Initialize fluid properties.
        
        Parameters
        ----------
        surface_pressure_psia : float
            Surface pressure in psi.
        surface_temperature_f : float
            Surface temperature in °F.
        oil_gravity_sg : float, optional
            Oil specific gravity at standard conditions.
        gas_gravity_sg : float, optional
            Gas specific gravity (air = 1.0).
        water_specific_gravity : float, optional
            Water specific gravity.
        water_cut : float, optional
            Water cut as a fraction (0 to 1).
        gas_oil_ratio : float, optional
            Gas-oil ratio at surface conditions.
        solution_oil_ratio : float, optional
            Solution gas-oil ratio (Rs).
        """
        self.surface_pressure_psia = surface_pressure_psia
        self.surface_temperature_f = surface_temperature_f
        self.oil_gravity_sg = oil_gravity_sg
        self.gas_gravity_sg = gas_gravity_sg
        self.water_specific_gravity = water_specific_gravity
        self.water_cut = water_cut
        self.gas_oil_ratio = gas_oil_ratio
        self.solution_oil_ratio = solution_oil_ratio
    
    def __repr__(self) -> str:
        """String representation of fluid properties."""
        return (
            f"FluidProperties(p={self.surface_pressure_psia}, "
            f"T={self.surface_temperature_f}, "
            f"GOR={self.gas_oil_ratio}, "
            f"Rs={self.solution_oil_ratio})"
        )
