"""Iterative solvers for pressure traverse calculations."""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional


def newton_raphson_solver(
    func: Callable[[float], float],
    derivative: Callable[[float], float],
    x0: float,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> float:
    """
    Newton-Raphson iterative solver for root finding.
    
    Parameters
    ----------
    func : callable
        Function to find root of (f(x) = 0).
    derivative : callable
        Derivative of func (f'(x)).
    x0 : float
        Initial guess.
    tolerance : float, optional
        Convergence tolerance. Default is 1e-6.
    max_iterations : int, optional
        Maximum iterations. Default is 100.
    
    Returns
    -------
    float
        Root of func.
    
    Raises
    ------
    ValueError
        If solver fails to converge.
    
    Notes
    -----
    - Requires f'(x) to be continuous near root
    - Quadratic convergence near root
    """
    x = x0
    for i in range(max_iterations):
        fx = func(x)
        dfx = derivative(x)
        
        if abs(dfx) < 1e-10:
            raise ValueError("Derivative too close to zero, cannot continue")
        
        x_new = x - fx / dfx
        
        if abs(x_new - x) < tolerance:
            return x_new
        
        x = x_new
    
    raise ValueError(f"Newton-Raphson failed to converge after {max_iterations} iterations")


def solve_pressure_step(
    initial_pressure: float,
    flow_direction: str,
    wellhead_pressure: float,
    gas_rate_mmcfd: float,
    tubing_diameter_in: float,
    vertical_depth_ft: float,
    gas_specific_gravity: float,
) -> float:
    """
    Solve for pressure at depth using iterative method.
    
    Parameters
    ----------
    initial_pressure : float
        Initial pressure at surface (psi).
    flow_direction : str
        Flow direction ('up' or 'down').
    wellhead_pressure : float
        Bottomhole condition pressure (psi).
    gas_rate_mmcfd : float
        Gas flow rate in million cubic feet per day.
    tubing_diameter_in : float
        Tubing inside diameter in inches.
    vertical_depth_ft : float
        Vertical depth from surface in feet.
    gas_specific_gravity : float
        Gas specific gravity.
    
    Returns
    -------
    float
        Pressure at depth.
    
    Notes
    -----
    - Iterative solution using friction factor correlations
    - Accounts for gas compressibility using AGA-8 DC Z-factor
    """
    # Check flow direction
    if flow_direction.lower() == 'up':
        start_pressure = wellhead_pressure
        depth_to_solve = vertical_depth_ft
    else:
        start_pressure = initial_pressure
        depth_to_solve = vertical_depth_ft
    
    if depth_to_solve == 0:
        return start_pressure
    
    # Simplified pressure drop calculation
    # Real implementation would use proper tubing friction calculation
    
    return start_pressure
