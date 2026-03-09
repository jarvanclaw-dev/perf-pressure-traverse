"""Flow module for multiphase pressure traverse calculations.

Implements:
- Flow regime identification (Beggs-Brill)
- Multiphase correlations (Beggs-Brill)
- Friction factor calculations
"""

from perf_pressure_traverse.flow.correlations import (
    BeggsBrillCorrelation,
    FlowRegime,
)
from perf_pressure_traverse.flow.regime import (
    FlowRegime as Regime,
    calculate_F_Lo,
    calculate_Fr_Lo,
    calculate_gas_Fr,
    calculate_inclination_factor,
    calculate_liquid_rate_at_inlet,
    calculate_liquid_superficial_velocity,
    calculate_gas_superficial_velocity,
    identify_regime_BeggsBrill,
    identify_regime_at_depth,
)

__all__ = [
    # Correlations
    'BeggsBrillCorrelation',
    'FlowRegime',
    
    # Regime identification
    'Regime',
    'FlowRegime',
    'calculate_F_Lo',
    'calculate_Fr_Lo',
    'calculate_gas_Fr',
    'calculate_inclination_factor',
    'calculate_liquid_rate_at_inlet',
    'calculate_liquid_superficial_velocity',
    'calculate_gas_superficial_velocity',
    'identify_regime_BeggsBrill',
    'identify_regime_at_depth',
]
