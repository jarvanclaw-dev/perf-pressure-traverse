"""Physical constants used in pressure traverse calculations."""

# Geothermal gradient (°F per 1000 ft)
GEOTHERMAL_GRADIENT = 0.017

# Standard conditions
STD_PRESSURE_PSIA = 14.7
STD_TEMPERATURE_F = 60.0

# Water properties
WATER_DENSITY_AT_WCF = 62.43  # lb/ft³ @ 60°F and water-free conditions

# API standard constants
API_19_1_FACTOR = 32.1740  # ft-lbf/lb (gravitational acceleration in ft/s²)

# Convergence criteria
DEFAULT_DAILY_STEP_FT = 10.0
MAX_ITERATIONS = 50
PRESSURE_TOLERANCE_PSI = 0.01
CONVERGENCE_REL_TOLERANCE = 0.01

__all__ = [
    "GEOTHERMAL_GRADIENT",
    "STD_PRESSURE_PSIA",
    "STD_TEMPERATURE_F",
    "WATER_DENSITY_AT_WCF",
    "API_19_1_FACTOR",
    "DEFAULT_DAILY_STEP_FT",
    "MAX_ITERATIONS",
    "PRESSURE_TOLERANCE_PSI",
    "CONVERGENCE_REL_TOLERANCE"
]
