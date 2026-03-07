"""Domain models and data structures module."""

from perf_pressure_traverse.models.fluid import FluidProperties
from perf_pressure_traverse.models.well import WellGeometry
from perf_pressure_traverse.models.pvt_properties import PVTProperties
from perf_pressure_traverse.models.pressure import (PressurePoint, PressureProfile)
from perf_pressure_traverse.models.wellflowpath import (
    WellFlowPath,
    FlowPathSegment
)
from perf_pressure_traverse.models.fluidmodel import (
    FluidModel,
    FluidModelFactory,
    FluidType
)

__all__ = [
    # Core models
    'FluidProperties',
    'WellGeometry',
    'PVTProperties',
    
    # Pressure traverse models
    'PressurePoint',
    'PressureProfile',
    
    # Well flow path models
    'WellFlowPath',
    'FlowPathSegment',
    
    # Fluid model
    'FluidModel',
    'FluidModelFactory',
    'FluidType',
]