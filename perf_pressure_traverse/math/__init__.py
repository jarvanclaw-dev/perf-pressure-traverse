"""Mathematical models and correlations for reservoir engineering.

This package contains mathematical models for:
- Equation of State (EOS) calculations
- Vapor-Liquid Equilibrium (VLE) flash calculations
- Black oil PVT correlations (Vasquez-Beggs, Standing)
- Z-factor correlations

Modules:
- eos: Equation of State implementations (SRK, Peng-Robinson)
- z_factor: Gas compressibility factor correlations
- vle: Vapor-liquid equilibrium calculations
- black_oil_pvt: Black oil PVT property correlations
- iterative: Iterative numerical methods
"""
# Import submodules for compatibility
from __future__ import annotations

from perf_pressure_traverse.math.eos import (
    SRKEOS,
    PengRobinsonEOS,
    EquationOfState,
    NumericalError
)

from perf_pressure_traverse.math.vle import (
    VLEFlash,
    VLEFlashSystem
)

from perf_pressure_traverse.math.black_oil_pvt import (
    VasquezBeggsCorrelations,
    VasquezBeggsError,
    StandingCorrections,
    BlackOilPVTCalculator,
    PVTUnits,
    calculate_vasquez_beggs_pvt
)

__all__ = [
    # EOS module
    'SRKEOS',
    'PengRobinsonEOS',
    'EquationOfState',
    'NumericalError',
    
    # VLE module
    'VLEFlash',
    'VLEFlashSystem',
    
    # Black oil PVT module
    'VasquezBeggsCorrelations',
    'VasquezBeggsError',
    'StandingCorrections',
    'BlackOilPVTCalculator',
    'PVTUnits',
    'calculate_vasquez_beggs_pvt',
]

__version__ = '1.0.0'
