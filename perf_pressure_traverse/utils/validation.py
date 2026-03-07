"""Parameter validation utility class."""

from typing import Optional
from perf_pressure_traverse.models.fluid import FluidProperties
from perf_pressure_traverse.models.well import WellGeometry
from perf_pressure_traverse.models.pvt_properties import PVTProperties


class ParameterValidator:
    """Validation utility for pressure traverse parameters."""
    
    def __init__(self) -> None:
        """Initialize the validator."""
        self.errors = []
    
    def validate_inputs(
        self,
        fluid_properties: FluidProperties,
        well_geometry: WellGeometry,
        pvt_properties: Optional[PVTProperties] = None,
    ) -> bool:
        """
        Validate all input parameters.
        
        Parameters
        ----------
        fluid_properties : FluidProperties
            Fluid property model.
        well_geometry : WellGeometry
            Well geometry model.
        pvt_properties : PVTProperties, optional
            PVT property model.
        
        Returns
        -------
        bool
            True if all validations pass, False otherwise.
        """
        self.errors = []
        
        self.validate_fluid_properties(fluid_properties)
        self.validate_well_geometry(well_geometry)
        
        if pvt_properties is not None:
            self.validate_pvt_properties(pvt_properties)
        
        return not self.has_errors()
    
    def validate_fluid_properties(self, fluid_properties: FluidProperties) -> bool:
        """Validate fluid properties ranges."""
        valid = True
        
        # Check specific gravity in valid range
        if not (0.5 <= fluid_properties.oil_specific_gravity <= 1.0):
            valid = False
        
        if not (0.30 <= fluid_properties.gas_specific_gravity <= 0.80):
            valid = False
        
        # Check pressure and temperature
        if fluid_properties.surface_pressure_psia <= 14.7:
            valid = False
        
        if fluid_properties.surface_temperature_f <= 0:
            valid = False
        
        return valid
    
    def validate_well_geometry(self, well_geometry: WellGeometry) -> bool:
        """Validate well geometry parameters."""
        valid = True
        
        if well_geometry.borehole_diameter_ft <= 0:
            valid = False
        
        return valid
    
    def has_errors(self) -> bool:
        """Check if there are any validation errors."""
        return len(self.errors) > 0
    
    def get_errors(self) -> list:
        """Get a list of all validation errors."""
        return self.errors.copy()
