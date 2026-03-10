"""Pressure traverse solver implementation."""

from __future__ import annotations

import numpy as np
from typing import Optional

from perf_pressure_traverse.models.fluid import FluidProperties
from perf_pressure_traverse.models.well import WellGeometry
from perf_pressure_traverse.models.result import PressureTraverseResult
from perf_pressure_traverse.utils.validation import ParameterValidator
from perf_pressure_traverse.utils.diagnostics import SolverDiagnostics
from perf_pressure_traverse.math.z_factor import calculate_z_factor_aga_dc


class PressureTraverseSolver:
    """Core pressure traverse solver for gas and liquid wells."""
    
    def __init__(
        self,
        fluid_properties: FluidProperties,
        well_geometry: WellGeometry,
        pvt_properties: Optional = None,
        max_iterations: int = 100,
        pressure_tolerance_psi: float = 0.01,
        depth_step_ft: float = 10.0,
        convergence_tolerance: float = 0.01,
    ) -> None:
        """Initialize the pressure traverse solver."""
        self.fluid_properties = fluid_properties
        self.well_geometry = well_geometry
        self.pvt_properties = pvt_properties
        self.max_iterations = max_iterations
        self.pressure_tolerance = pressure_tolerance_psi
        self.depth_step = depth_step_ft
        self.convergence_tolerance = convergence_tolerance
        
        ParameterValidator.validate_inputs(fluid_properties, well_geometry, pvt_properties)
    
    def solve(self) -> PressureTraverseResult:
        """Perform pressure traverse calculation."""
        diagnostics = SolverDiagnostics()
        
        depth_steps = self._create_depth_steps()
        pressure = self.fluid_properties.surface_pressure_psia
        temperature = self.fluid_properties.surface_temperature_f
        flow_rate_gpm = 100.0
        gas_rate_mcfd = 500.0
        
        n_points = len(depth_steps)
        pressure_profile = np.zeros(n_points)
        temperature_profile = np.zeros(n_points)
        flow_regime_profile = np.zeros(n_points, dtype=int)
        liquid_holdup_profile = np.zeros(n_points)
        frictional_loss_profile = np.zeros(n_points)
        hydrostatic_loss_profile = np.zeros(n_points)
        acceleration_loss_profile = np.zeros(n_points)
        
        from perf_pressure_traverse.constants import GEOTHERMAL_GRADIENT
        
        for i, depth_ft in enumerate(depth_steps):
            temperature, _, _ = self._update_temperature(temperature, depth_ft)
            
            pressure, flow_regime, liquid_holdup = self._calculate_pressure_at_depth(
                depth_ft, pressure, temperature, flow_rate_gpm, gas_rate_mcfd
            )
            
            pressure_profile[i] = pressure
            temperature_profile[i] = temperature
            flow_regime_profile[i] = flow_regime
            liquid_holdup_profile[i] = liquid_holdup
            
            hydrostatic_loss, frictional_loss, accel_loss = self._compute_losses(depth_ft)
            hydrostatic_loss_profile[i] = hydrostatic_loss
            frictional_loss_profile[i] = frictional_loss
            acceleration_loss_profile[i] = accel_loss
            
            diagnostics.log_iteration(i + 1, depth_ft, pressure, pressure if i > 0 else None)
        
        bottomhole_pressure = pressure
        total_pressure_loss_ft = float(self.well_geometry.true_vertical_depth_ft)
        convergence_message = self._check_convergence(diagnostics)
        
        return PressureTraverseResult(
            surface_pressure=self.fluid_properties.surface_pressure_psia,
            bottomhole_pressure=bottomhole_pressure,
            pressure_profile=pressure_profile,
            temperature_profile=temperature_profile,
            flow_regime_profile=flow_regime_profile,
            liquid_holdup_profile=liquid_holdup_profile,
            frictional_loss_profile=frictional_loss_profile,
            hydrostatic_loss_profile=hydrostatic_loss_profile,
            acceleration_loss_profile=acceleration_loss_profile,
            total_pressure_loss_ft=total_pressure_loss_ft,
            solver_iterations=diagnostics.iterations,
            convergence_message=convergence_message,
            flow_rate_gpm=flow_rate_gpm,
            gas_rate_mcfd=gas_rate_mcfd,
            has_succeeded=convergence_message == "Converged",
            warnings=diagnostics.errors if hasattr(diagnostics, 'errors') else [],
        )
