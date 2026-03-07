"""Pressure traverse result model."""

from __future__ import annotations

from typing import Dict, List
import numpy as np


class PressureTraverseResult:
    """
    Result object for pressure traverse calculation.
    
    Attributes
    ----------
    surface_pressure : float
        Surface pressure in psi.
    bottomhole_pressure : float
        Bottomhole pressure in psi.
    pressure_profile : np.ndarray
        1D array of pressure values along the wellbore.
    temperature_profile : np.ndarray
        1D array of temperature values along the wellbore.
    flow_regime_profile : np.ndarray
        1D array of flow regime classifications.
    liquid_holdup_profile : np.ndarray
        1D array of liquid holdup percentages.
    frictional_loss_profile : np.ndarray
        1D array of frictional pressure loss at each point.
    hydrostatic_loss_profile : np.ndarray
        1D array of hydrostatic pressure loss at each point.
    acceleration_loss_profile : np.ndarray
        1D array of acceleration pressure loss at each point.
    total_pressure_loss_ft : float
        Total pressure loss from surface to bottom.
    solver_iterations : int
        Total number of Newton-Raphson iterations used.
    convergence_message : str
        Message indicating convergence status.
    flow_rate_gpm : float
        Liquid flow rate at surface in gallons per minute.
    gas_rate_mcfd : float
        Gas flow rate at surface in thousands of cubic feet per day.
    has_succeeded : bool
        Whether the calculation succeeded.
    warnings : list
        List of warning messages.
    """
    
    def __init__(
        self,
        surface_pressure: float,
        bottomhole_pressure: float,
        pressure_profile: np.ndarray,
        temperature_profile: np.ndarray,
        flow_regime_profile: np.ndarray,
        liquid_holdup_profile: np.ndarray,
        frictional_loss_profile: np.ndarray,
        hydrostatic_loss_profile: np.ndarray,
        acceleration_loss_profile: np.ndarray,
        total_pressure_loss_ft: float,
        solver_iterations: int,
        convergence_message: str,
        flow_rate_gpm: float = 0.0,
        gas_rate_mcfd: float = 0.0,
        has_succeeded: bool = True,
        warnings: Optional[List] = None,
    ) -> None:
        """
        Initialize result object.
        
        Parameters
        ----------
        surface_pressure : float
            Surface pressure in psi.
        bottomhole_pressure : float
            Bottomhole pressure in psi.
        pressure_profile : np.ndarray
            Pressure profile along the wellbore.
        temperature_profile : np.ndarray
            Temperature profile along the wellbore.
        flow_regime_profile : np.ndarray
            Flow regime classifications along the wellbore.
        liquid_holdup_profile : np.ndarray
            Liquid holdup percentages along the wellbore.
        frictional_loss_profile : np.ndarray
            Frictional loss profile along the wellbore.
        hydrostatic_loss_profile : np.ndarray
            Hydrostatic loss profile along the wellbore.
        acceleration_loss_profile : np.ndarray
            Acceleration loss profile along the wellbore.
        total_pressure_loss_ft : float
            Total pressure loss.
        solver_iterations : int
            Number of solver iterations.
        convergence_message : str
            Convergence status message.
        flow_rate_gpm : float, optional
            Surface liquid flow rate in GPM.
        gas_rate_mcfd : float, optional
            Surface gas flow rate in MCFD.
        has_succeeded : bool, optional
            Whether the calculation succeeded.
        warnings : list, optional
            List of warning messages.
        """
        self.surface_pressure = surface_pressure
        self.bottomhole_pressure = bottomhole_pressure
        self.pressure_profile = pressure_profile
        self.temperature_profile = temperature_profile
        self.flow_regime_profile = flow_regime_profile
        self.liquid_holdup_profile = liquid_holdup_profile
        self.frictional_loss_profile = frictional_loss_profile
        self.hydrostatic_loss_profile = hydrostatic_loss_profile
        self.acceleration_loss_profile = acceleration_loss_profile
        self.total_pressure_loss_ft = total_pressure_loss_ft
        self.solver_iterations = solver_iterations
        self.convergence_message = convergence_message
        self.flow_rate_gpm = flow_rate_gpm
        self.gas_rate_mcfd = gas_rate_mcfd
        self.has_succeeded = has_succeeded
        self.warnings = warnings or []
    
    def get_surface_pressure_loss(self) -> float:
        """Calculate surface pressure loss."""
        if len(self.pressure_profile) > 1:
            return self.surface_pressure - self.bottomhole_pressure
        return 0.0
    
    def get_average_pressure_gradient_psi_ft(self) -> float:
        """Calculate average pressure gradient."""
        if self.total_pressure_loss_ft > 0:
            return self.total_pressure_loss_ft / self.total_pressure_loss_ft
        return 0.0
    
    def get_pressure_loss_percentage(self) -> float:
        """Calculate pressure loss percentage of surface pressure."""
        if self.surface_pressure > 0:
            return (self.get_surface_pressure_loss() / self.surface_pressure) * 100.0
        return 0.0
    
    def get_average_velocity_ft_s(self) -> float:
        """Calculate average mixture velocity."""
        if self.flow_rate_gpm > 0:
            return (self.flow_rate_gpm / 448.83) / self.surface_pressure
        return 0.0
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary."""
        return {
            "surface_pressure": self.surface_pressure,
            "bottomhole_pressure": self.bottomhole_pressure,
            "total_pressure_loss_ft": self.total_pressure_loss_ft,
            "pressure_loss_percentage": self.get_pressure_loss_percentage(),
            "convergence_message": self.convergence_message,
            "solver_iterations": self.solver_iterations,
            "has_succeeded": self.has_succeeded,
            "flow_rate_gpm": self.flow_rate_gpm,
            "gas_rate_mcfd": self.gas_rate_mcfd,
        }
    
    def __repr__(self) -> str:
        """String representation of result."""
        return (
            f"PressureTraverseResult(P={self.surface_pressure:.2f}PSI/ "
            f"{self.bottomhole_pressure:.2f}PSI, "
            f"TVD={self.total_pressure_loss_ft:.2f}FT, "
            f"iterations={self.solver_iterations}, "
            f"converged={self.convergence_message == 'Converged'})"
        )
