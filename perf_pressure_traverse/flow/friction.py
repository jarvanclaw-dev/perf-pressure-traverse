"""Friction factor calculations for multiphase flow.

Implements friction factor models for pressure drop calculations.
"""

from math import log10, exp


def moody_diagram_lookup(re: float, roughness: float) -> float:
    """Direct lookup from Moody diagram coefficients.

    Args:
        re: Reynolds number
        roughness: Pipe roughness in ft

    Returns:
        Fanning friction factor
    """
    if re == 0:
        return 0.01

    if re < 2000:
        # Laminar flow
        return 16.0 / re

    # Turbulent flow
    if roughness > 0:
        t = 5.74 * (100 * roughness / re) ** 0.9
        f = 0.0055 * (1 + (2000 * roughness / re) ** 0.9)

        if t < 1.0:
            return max(f, 0.01)
        return f

    return 0.01


def darcy_weisbach_friction_factor(re: float, relative_roughness: float) -> float:
    """Darcy-Weisbach friction factor.

    Args:
        re: Reynolds number
        relative_roughness: Surface roughness to diameter ratio

    Returns:
        Fanning friction factor
    """
    if re < 2000:
        return 16.0 / re

    # Swamee-Jain approximation
    from math import log
    try:
        t = relative_roughness
        f = 0.25 / (log10(t / 3.7 + 5.74 / re ** 0.9)) ** 2
        return max(f, 0.01)
    except:
        return 0.01


def api_friction_factor(roughness: float, reynolds_no: float) -> float:
    """API standard friction factor correlation."""
    if reynolds_no == 0:
        return 0.01

    if reynolds_no < 2000:
        return 16.0 / reynolds_no

    # Using a simplified API approximation
    if roughness > 0:
        relative_roughness = roughness
        term1 = 5.74 / (reynolds_no ** 0.9)
        term2 = 100 * relative_roughness
        f = 0.0055 * (1 + term2 ** 0.9)
        if term1 < 1.0:
            return max(f, 0.01)
        return f

    return 0.01


__all__ = [
    "moody_diagram_lookup",
    "darcy_weisbach_friction_factor",
    "api_friction_factor",
]
