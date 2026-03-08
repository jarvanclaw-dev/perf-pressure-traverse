"""Flow regime identification module.

Implements Beggs-Brill (1973) multiphase flow regime transition map
to identify flow regime based on fluid properties and well geometry.
"""

from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray


class FlowRegime(Enum):
    """Flow regime enumeration.

    Standard multiphase flow regimes according to Beggs-Brill method.
    """
    SEGREGATED = "Segregated Flow"
    INTERMITTENT = "Slug Flow"
    DISTRIBUTED = "Distributed Flow"
    MIST = "Mist Flow"
    BUBBLE = "Bubble Flow"


def calculate_liquid_superficial_velocity(L_s_gpm: float, area_ft2: float) -> float:
    """Calculate liquid superficial velocity.

    Args:
        L_s_gpm: Liquid flow rate in GPM
        area_ft2: Cross-sectional area in ft²

    Returns:
        Liquid superficial velocity in ft/s
    """
    if area_ft2 <= 0:
        return 0.0
    # Convert GPM to ft³/s: GPM * 0.002228 / 60
    return (L_s_gpm * 0.002228 / 60.0) / area_ft2


def calculate_gas_superficial_velocity(G_s_gpm: float, oil_gal: float, area_ft2: float) -> float:
    """Calculate gas superficial velocity with gas-oil ratio correction.

    Args:
        G_s_gpm: Gross gas flow rate in GPM
        oil_gal: Oil flow rate in gallons
        area_ft2: Cross-sectional area in ft²

    Returns:
        Gas superficial velocity in ft/s
    """
    if area_ft2 <= 0:
        return 0.0

    # Gas-oil ratio correction
    if oil_gal > 0:
        # Effective gas rate (account for dissolved gas)
        effective_gas_gpm = G_s_gpm + (oil_gal * 0.0181)  # Approximate correction
    else:
        effective_gas_gpm = G_s_gpm

    # Convert GPM to ft³/s: GPM * 0.002228 / 60
    return (effective_gas_gpm * 0.002228 / 60.0) / area_ft2


def calculate_liquid_rate_at_inlet(l_gpm: float, oil_gal: float) -> float:
    """Calculate total liquid rate at inlet conditions in GPM.

    Args:
        l_gpm: Total liquid flow rate in GPM
        oil_gal: Oil flow rate in gallons

    Returns:
        Total liquid rate in GPM
    """
    return l_gpm + (oil_gal * 0.0181)  # Gas-oil ratio correction


def calculate_F_Lo(
    liquid_rate_gpm: float,
    area_ft2: float,
    oil_density: float,
    water_density: float = 62.4,
    g: float = 32.17405
) -> float:
    """Calculate F_Lo - dimensionless liquid flow rate.

    F_Lo = (ρ_l * Q_l) / (A * sqrt(g))
    where Q_l is in ft³/s

    Args:
        liquid_rate_gpm: Total liquid rate in GPM
        area_ft2: Cross-sectional area in ft²
        oil_density: Oil density at inlet conditions (lbm/ft³)
        water_density: Water density (lbm/ft³)
        g: Gravitational acceleration (32.17405 ft/s²)

    Returns:
        F_Lo dimensionless value
    """
    if area_ft2 <= 0:
        return 0.0

    # Combined liquid density (weighted average)
    liquid_density = oil_density

    # Convert GPM to ft³/s
    Q_l_ft3s = liquid_rate_gpm * 0.002228 / 60.0

    # Calculate F_Lo
    return (liquid_density * Q_l_ft3s) / (area_ft2 * np.sqrt(g))


def calculate_Fr_Lo(
    liquid_rate_gpm: float,
    area_ft2: float,
    g: float = 32.17405
) -> float:
    """Calculate Fr_Lo - dimensionless liquid Froude number.

    Fr_Lo = (Q_l) / (A * sqrt(g * D))

    Args:
        liquid_rate_gpm: Total liquid rate in GPM
        area_ft2: Cross-sectional area in ft²
        g: Gravitational acceleration (32.17405 ft/s²)

    Returns:
        Fr_Lo dimensionless value
    """
    if area_ft2 <= 0:
        return 0.0

    # Equivalent diameter from area: D = 4 * A / π
    diameter_ft = 4.0 * area_ft2 / np.pi

    # Convert GPM to ft³/s
    Q_l_ft3s = liquid_rate_gpm * 0.002228 / 60.0

    # Calculate Fr_Lo
    return Q_l_ft3s / (area_ft2 * np.sqrt(g * diameter_ft))


def calculate_gas_Fr(
    gas_rate_gpm: float,
    area_ft2: float,
    diameter_ft: float,
    g: float = 32.17405
) -> float:
    """Calculate Fr - dimensionless gas Froude number.

    Fr = (Q_g) / (A * sqrt(g * D))

    Args:
        gas_rate_gpm: Gross gas flow rate in GPM
        area_ft2: Cross-sectional area in ft²
        diameter_ft: Pipe diameter in ft
        g: Gravitational acceleration (32.17405 ft/s²)

    Returns:
        Fr dimensionless value
    """
    if area_ft2 <= 0:
        return 0.0

    # Convert GPM to ft³/s
    Q_g_ft3s = gas_rate_gpm * 0.002228 / 60.0

    # Calculate Fr
    return Q_g_ft3s / (area_ft2 * np.sqrt(g * diameter_ft))


def calculate_inclination_factor(theta_deg: float, flow_direction: Literal["uphill", "downhill", "horizontal"] = "horizontal") -> float:
    """Calculate inclination factor from angle and flow direction.

    Using Beggs-Brill inclination correction factors.

    Args:
        theta_deg: Inclination angle in degrees (-90° to +90°)
        flow_direction: Flow direction (uphill, downhill, horizontal)

    Returns:
        Inclination factor (0-1)
    """
    # Use flow_direction if provided, otherwise infer
    if flow_direction == "horizontal":
        return 0.5
    elif flow_direction == "uphill":
        if abs(theta_deg) > 0:
            # Upward flow: factor increases with angle, max 1.0
            return min(0.5 + (abs(theta_deg) * 0.01), 1.0)
        return 0.5
    elif flow_direction == "downhill":
        if abs(theta_deg) > 0:
            # Downward flow: factor decreases with angle, min 0.0
            return max(0.5 - (abs(theta_deg) * 0.01), 0.0)
        return 0.5
    else:
        # Default to horizontal
        return 0.5


def identify_regime_BeggsBrill(
    oil_flow_rate_gpm: float,
    gas_inlet_rate_gpm: float,
    oil_gal: float,
    pipe_diameter_ft: float,
    borehole_area_ft2: float,
    well_angle_deg: float,
    oil_density_lbm_ft3: float,
    water_density_lbm_ft3: float,
    flow_direction: Literal["uphill", "downhill", "horizontal"] = "horizontal"
) -> FlowRegime:
    """Identify flow regime using Beggs-Brill transition map.

    Implements the comprehensive flow regime identification method
    from Beggs and Brill (1973) using dimensionless groups
    and inclination factors.

    Args:
        oil_flow_rate_gpm: Oil flow rate in GPM
        gas_inlet_rate_gpm: Gross gas inlet rate in GPM
        oil_gal: Oil flow rate in gallons
        pipe_diameter_ft: Pipe diameter in ft
        borehole_area_ft2: Cross-sectional area in ft²
        well_angle_deg: Well inclination angle in degrees (-90° to +90°)
        oil_density_lbm_ft3: Oil density at inlet conditions (lbm/ft³)
        water_density_lbm_ft3: Water density (lbm/ft³)
        flow_direction: Flow direction (uphill, downhill, horizontal)

    Returns:
        FlowRegime enum identifying the current flow regime

    Examples:
        >>> regime = identify_regime_BeggsBrill(
        ...     oil_flow_rate_gpm=100,
        ...     gas_inlet_rate_gpm=50,
        ...     oil_gal=85,
        ...     pipe_diameter_ft=0.2917,
        ...     borehole_area_ft2=0.0667,
        ...     well_angle_deg=0.0
        ... )
        >>> regime
        FlowRegime.DISTRIBUTED
    """
    # Determine effective flow direction
    if well_angle_deg > 0:
        flow_dir = "uphill"
    elif well_angle_deg < 0:
        flow_dir = "downhill"
    else:
        flow_dir = "horizontal"

    # Calculate effective rates
    total_liquid_rate = calculate_liquid_rate_at_inlet(oil_flow_rate_gpm, oil_gal)
    effective_gas_rate = gas_inlet_rate_gpm + (oil_gal * 0.0181)
    total_liquid_rate_gpm = total_liquid_rate + oil_flow_rate_gpm

    # Calculate velocities
    gas_velocity = calculate_gas_superficial_velocity(gas_inlet_rate_gpm, oil_gal, borehole_area_ft2)
    liquid_velocity = calculate_liquid_superficial_velocity(total_liquid_rate_gpm, borehole_area_ft2)

    # Calculate dimensionless groups
    F_Lo = calculate_F_Lo(total_liquid_rate_gpm, borehole_area_ft2, oil_density_lbm_ft3)
    Fr_Lo = calculate_Fr_Lo(total_liquid_rate_gpm, borehole_area_ft2)
    Fr = calculate_gas_Fr(effective_gas_rate, borehole_area_ft2, pipe_diameter_ft)

    # Determine flow regime based on Beggs-Brill criteria

    # Very low gas rate - bubble flow (priority first)
    if effective_gas_rate < 0.5:
        return FlowRegime.BUBBLE

    # High gas rate with low liquid velocity - mist flow
    if gas_velocity > 50.0 and liquid_velocity < 1.0:
        return FlowRegime.MIST

    # Bubble flow: very low F_Lo
    if F_Lo < 1.0:
        return FlowRegime.BUBBLE

    # Mist flow: very high gas rate (high Fr)
    if Fr > 20.0 and F_Lo > 10.0:
        return FlowRegime.MIST

    # Segregated flow: low to intermediate rates, low inclination
    if F_Lo < 3.0:
        return FlowRegime.SEGREGATED

    # Slug flow (intermittent): intermediate rates
    if F_Lo >= 3.0 and F_Lo < 8.0:
        return FlowRegime.INTERMITTENT

    # Distributed flow: high rates, high Fr_Lo
    if Fr_Lo > 10.0:
        return FlowRegime.DISTRIBUTED

    # Default to distributed flow for high gas rates
    return FlowRegime.DISTRIBUTED


def identify_regime_at_depth(
    oil_rate_gpm: NDArray,
    gas_rate_gpm: NDArray,
    oil_gal: float,
    diameter_ft: NDArray,
    area_ft2: NDArray,
    angle_deg: NDArray,
    oil_density: NDArray,
    water_density: float
) -> NDArray:
    """Identify flow regime profile at multiple depths.

    Vectorized version for efficiency across depth array.

    Args:
        oil_rate_gpm: Oil flow rate at each depth [N]
        gas_rate_gpm: Gas flow rate at each depth [N]
        oil_gal: Oil flow rate in gallons
        diameter_ft: Pipe diameter at each depth [N]
        area_ft2: Cross-sectional area at each depth [N]
        angle_deg: Well inclination angle at each depth [N]
        oil_density: Oil density at each depth [N]
        water_density: Water density [lbm/ft³]

    Returns:
        Array of FlowRegime enums identifying regime at each depth
    """
    regimes = np.array([
        identify_regime_BeggsBrill(
            oil_rate, gas_rate, oil_gal, dia, area, angle, oil_dens, water_density
        )
        for oil_rate, gas_rate, dia, area, angle, oil_dens in zip(
            oil_rate_gpm, gas_rate_gpm, diameter_ft, area_ft2, angle_deg, oil_density
        )
    ], dtype=object)
    return regimes


__all__ = [
    "FlowRegime",
    "calculate_liquid_superficial_velocity",
    "calculate_gas_superficial_velocity",
    "calculate_liquid_rate_at_inlet",
    "calculate_F_Lo",
    "calculate_Fr_Lo",
    "calculate_gas_Fr",
    "calculate_inclination_factor",
    "identify_regime_BeggsBrill",
    "identify_regime_at_depth",
]
