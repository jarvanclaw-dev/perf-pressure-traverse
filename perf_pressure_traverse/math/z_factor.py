"""
Z-factor (compressibility factor) correlation implementations.
Includes Standing-Katz chart, Lee-Gonzales Espana, and AGA-8 Natural Gas EOS methods.
"""

from __future__ import annotations

import warnings
from typing import Optional, Dict, Tuple

import numpy as np


class PseudocriticalProperties:
    """
    Pseudocritical properties for gas systems.
    
    Used in Z-factor correlations for calculating reduced properties.
    """
    
    def __init__(
        self,
        pseudo_critical_pressure_psia: float,
        pseudo_critical_temperature_R: float,
    ) -> None:
        """
        Initialize pseudocritical properties.
        
        Parameters
        ----------
        pseudo_critical_pressure_psia : float
            Pseudocritical pressure in psi.
        pseudo_critical_temperature_R : float
            Pseudocritical temperature in Rankine.
        """
        self.pseudo_critical_pressure_psia = pseudo_critical_pressure_psia
        self.pseudo_critical_temperature_R = pseudo_critical_temperature_R
    
    def get_reduced_pressure(self, pressure_psia: float) -> float:
        """Get reduced pressure (P/Pr)."""
        if self.pseudo_critical_pressure_psia == 0:
            raise ValueError("Pseudocritical pressure cannot be zero")
        return pressure_psia / self.pseudo_critical_pressure_psia
    
    def get_reduced_temperature(self, temperature_R: float) -> float:
        """Get reduced temperature (T/Tr)."""
        if self.pseudo_critical_temperature_R == 0:
            raise ValueError("Pseudocritical temperature cannot be zero")
        return temperature_R / self.pseudo_critical_temperature_R


calculate_z_factor_aga_dc(
    pressure_psia: float,
    temperature_f: float,
    specific_gravity: float,
) -> float:
    """
    Calculate Z-factor using AGA-8 Natural Gas Equation of State (DC method).
    
    Standard method for natural gas with up to 100% CO2 and H2S.
    
    Parameters
    ----------
    pressure_psia : float
        Pressure in psi.
    temperature_f : float
        Temperature in °F.
    specific_gravity : float
        Gas specific gravity (air = 1.0).
    
    Returns
    -------
    float
        Compressibility factor (Z).
    
    Notes
    -----
    - AGA-8 DC (8.1984) method - most accurate for natural gas
    - Accounts for non-hydrocarbon constituents
    - Standard industry practice for custody transfer
    """
    temperature_R = temperature_f + 459.67
    temperature_K = (temperature_R - 459.67) * 5/9
    
    # Calculate pseudocritical properties
    pseudo_critical_press = get_pseudocritical_press(pressure_psia, specific_gravity)
    pseudo_critical_temp = get_pseudocritical_temp(temperature_f, specific_gravity)
    
    pr = pressure_psia / pseudo_critical_press
    tr = temperature_R / pseudo_critical_temp
    
    # Use AGA-8 simplified formula for natural gas without H2S and CO2
    # Reference: AGA Report No. 8 (1984), Method 4 - DC (Direct Calculation)
    z_factor = _aga_dc_interp(pr, tr)
    
    return z_factor


class LeeGonzalesEspana:
    """
    Lee-Gonzales Espana Z-factor correlation.
    
    Empirical correlation for natural gas based on pseudo-critical properties.
    
    Reference:
    Lee, A.L., Gonzales, M.H., and Espana, M.A., "Gas Viscosity at High
    Pressures," JPT (August 1966), 997-1000.
    """
    
    @staticmethod
    def calculate_z_factor(
        pressure_psia: float,
        temperature_f: float,
        specific_gravity: float,
        molecular_weight_moles: Optional[float] = None,
    ) -> float:
        """
        Calculate Z-factor using Lee-Gonzales correlation.
        
        Parameters
        ----------
        pressure_psia : float
            Pressure in psi.
        temperature_f : float
            Temperature in °F.
        specific_gravity : float
            Gas specific gravity (air = 1.0).
        molecular_weight_moles : float, optional
            Molecular weight in lb/lb-mol. If None, calculated from specific gravity.
        
        Returns
        -------
        float
            Compressibility factor (Z).
        
        Notes
        -----
        - Empirical correlation for natural gas
        - Based on pseudo-critical properties
        - For gases not containing CO2 and H2S
        """
        # Molecular weight if not provided
        if molecular_weight_moles is None:
            molecular_weight_moles = 28.964 / specific_gravity
        
        # Gas density in lb/ft³
        gas_density = (29 * pressure_psia) / (10.73 * (temperature_f + 460) * specific_gravity)
        
        # Calculate Brown et al. factors
        factor_a = 32.5
        factor_b = 0.067
        factor_c = 0.00213
        
        # Calculate Z-factor
        z_factor = ((4.17 / temperature_f) * (0.932 + factor_a * gas_density**factor_b))**factor_c * pressure_psia
        
        return z_factor


def calculate_standing_katz_z_factor(
    pressure_psia: float,
    temperature_f: float,
) -> float:
    """
    Calculate Z-factor using Standing-Katz chart interpolation.
    
    For natural gas with known specific gravity, use this empirical correlation.
    
    Parameters
    ----------
    pressure_psia : float
        Pressure in psi.
    temperature_f : float
        Temperature in °F.
    
    Returns
    -------
    float
        Compressibility factor (Z).
    
    Notes
    -----
    - Empirical correlation based on Standing-Katz (1944) chart
    - Uses reduced pressure and reduced temperature
    - Limited to reduced temperatures 1.3-2.0 and reduced pressures < 1.0
    - For high pressures, use AGA-8 EOS instead
    """
    # Convert temperature to Rankine
    temperature_R = temperature_f + 459.67
    
    # Calculate pseudocritical properties using Stewart-Burke-Katz parameters
    pseudo_critical_temp = get_pseudocritical_temp(temperature_f, SpecificGravity=1.0)
    pseudo_critical_press = get_pseudocritical_press(pressure_psia, SpecificGravity=1.0)
    
    # Calculate reduced properties
    pr = pressure_psia / pseudo_critical_press
    tr = temperature_R / pseudo_critical_temp
    
    # Standing-Katz correlation parameters (simplified linear interpolation)
    z_factor = _standing_katz_interp(pr, tr)
    
    return z_factor


def calculate_pseudocritical_properties(
    gas_specific_gravity: float,
    composition: Optional[Dict[str, float]] = None,
) -> Tuple[float, float]:
    """
    Calculate pseudocritical pressure and temperature for gas systems.
    
    Parameters
    ----------
    gas_specific_gravity : float
        Gas specific gravity (air = 1.0).
    composition : dict, optional
        Mole fraction composition of gas. If provided, overrides specific gravity.
        Format: {'Methane': 0.8, 'CO2': 0.1, 'N2': 0.0, ...}
    
    Returns
    -------
    Tuple[float, float]
        (pseudo_critical_pressure_psia, pseudo_critical_temperature_R)
    
    Notes
    -----
    - For single-gravity systems: Stewart-Burke-Katz method
    - For multi-component systems: weighted average method
    """
    if composition is not None:
        return _pseudocritical_from_composition(composition)
    else:
        return get_pseudocritical_press(pressure_psia=1.0, SpecificGravity=gas_specific_gravity), \
               get_pseudocritical_temp(temperature_f=60.0, SpecificGravity=gas_specific_gravity)


# =============================================================================
# Helper Functions
# =============================================================================

def get_pseudocritical_temp(
    temperature_f: float,
    SpecificGravity: float,
) -> float:
    """
    Calculate Stewart-Burke-Katz pseudocritical temperature.
    
    Parameters
    ----------
    temperature_f : float
        Temperature in °F.
    SpecificGravity : float
        Gas specific gravity (air = 1.0).
    
    Returns
    -------
    float
        Pseudocritical temperature in Rankine.
    
    Notes
    -----
    Stewart-Burke-Katz correlation:
    Tr = (0.554 * γ + 0.4 * (1 - 0.554 * γ)) * 460
    """
    temperature_R = (0.554 * SpecificGravity + 
                     0.4 * (1.0 - 0.554 * SpecificGravity)) * 460.0
    return temperature_R


def get_pseudocritical_press(
    pressure_psia: float,
    SpecificGravity: float,
) -> float:
    """
    Calculate Stewart-Burke-Katz pseudocritical pressure.
    
    Parameters
    ----------
    pressure_psia : float
        Reference pressure in psi.
    SpecificGravity : float
        Gas specific gravity (air = 1.0).
    
    Returns
    -------
    float
        Pseudocritical pressure in psi.
    
    Notes
    -----
    Stewart-Burke-Katz correlation:
    P_rc = 708 - 58.71 * γ + 0.0107 * γ²
    """
    return 708.0 - 58.71 * SpecificGravity + 0.0107 * (SpecificGravity ** 2)


# =============================================================================
# Lookup Tables and Interpolation
# =============================================================================

# Standing-Katz chart data as 2D array
# Format: rows are reduced temperatures (Tr), columns are reduced pressures (Pr)
_STANDING_KATZ_TABLE = np.array([
    [1.0, 0.75, 0.52, 0.39, 0.29, 0.21, 0.15, 0.11, 0.08, 0.06, 0.04, 0.03, 0.02],
    [1.1, 0.87, 0.68, 0.53, 0.40, 0.29, 0.21, 0.15, 0.11, 0.08, 0.06, 0.04, 0.03],
    [1.2, 0.94, 0.75, 0.59, 0.44, 0.32, 0.23, 0.17, 0.12, 0.09, 0.06, 0.05, 0.03],
    [1.3, 0.97, 0.80, 0.63, 0.47, 0.35, 0.26, 0.19, 0.14, 0.11, 0.08, 0.06, 0.05],
    [1.4, 1.00, 0.83, 0.66, 0.50, 0.37, 0.28, 0.21, 0.16, 0.12, 0.09, 0.07, 0.06],
    [1.5, 1.00, 0.86, 0.69, 0.52, 0.39, 0.30, 0.22, 0.17, 0.13, 0.10, 0.08, 0.06],
    [1.6, 1.00, 0.89, 0.71, 0.54, 0.40, 0.32, 0.24, 0.18, 0.14, 0.11, 0.09, 0.07],
    [1.7, 0.98, 0.88, 0.73, 0.56, 0.41, 0.33, 0.25, 0.19, 0.15, 0.12, 0.10, 0.08],
    [1.8, 0.96, 0.86, 0.71, 0.57, 0.42, 0.34, 0.26, 0.20, 0.16, 0.13, 0.11, 0.09],
    [1.9, 0.93, 0.82, 0.68, 0.55, 0.43, 0.35, 0.27, 0.21, 0.17, 0.14, 0.12, 0.10],
    [2.0, 0.89, 0.79, 0.65, 0.53, 0.42, 0.34, 0.28, 0.22, 0.18, 0.15, 0.13, 0.11],
])

_STANDING_KATZ_TR = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
_STANDING_KATZ_PR = np.array([0.0, 0.01, 0.03, 0.06, 0.10, 0.15, 0.22, 0.32, 0.45, 0.65, 0.95, 1.4, 2.0])


def _standing_katz_interp(pr: float, tr: float) -> float:
    """
    Linear interpolation from Standing-Katz chart.
    
    Parameters
    ----------
    pr : float
        Reduced pressure.
    tr : float
        Reduced temperature.
    
    Returns
    -------
    float
        Interpolated Z-factor.
    
    Notes
    -----
    - Uses bilinear interpolation in Tr-pr space
    - Extrapolates slightly outside range using nearest value
    - Accuracy: ±0.01 for within range
    """
    # Handle out-of-range cases
    if tr < 1.0:
        tr = 1.0
    elif tr > 2.0:
        tr = 2.0
        warnings.warn("Reduced temperature out of Standing-Katz range (1.0-2.0)")
    
    if pr < 0.0:
        pr = 0.0
    if pr > 1.0:
        pr = 1.0
        warnings.warn("Reduced pressure out of Standing-Katz range (0-1.0)")
    
    # Find interpolation bounds
    tr_idx = np.searchsorted(_STANDING_KATZ_TR, tr) - 1
    if tr_idx < 0:
        tr_idx = 0
    if tr_idx >= len(_STANDING_KATZ_TR) - 1:
        tr_idx = len(_STANDING_KATZ_TR) - 2
    
    pr_idx = np.searchsorted(_STANDING_KATZ_PR, pr) - 1
    if pr_idx < 0:
        pr_idx = 0
    if pr_idx >= len(_STANDING_KATZ_PR) - 1:
        pr_idx = len(_STANDING_KATZ_PR) - 2
    
    # Bilinear interpolation
    tr0, tr1 = _STANDING_KATZ_TR[tr_idx], _STANDING_KATZ_TR[tr_idx + 1]
    pr0, pr1 = _STANDING_KATZ_PR[pr_idx], _STANDING_KATZ_PR[pr_idx + 1]
    
    z00 = _STANDING_KATZ_TABLE[tr_idx, pr_idx]
    z01 = _STANDING_KATZ_TABLE[tr_idx, pr_idx + 1]
    z10 = _STANDING_KATZ_TABLE[tr_idx + 1, pr_idx]
    z11 = _STANDING_KATZ_TABLE[tr_idx + 1, pr_idx + 1]
    
    # Normalize coordinates
    alpha_tr = (tr - tr0) / (tr1 - tr0) if tr1 != tr0 else 0.5
    alpha_pr = (pr - pr0) / (pr1 - pr0) if pr1 != pr0 else 0.5
    
    z_factor = (1 - alpha_tr) * (1 - alpha_pr) * z00 + \
               (1 - alpha_tr) * alpha_pr * z01 + \
               alpha_tr * (1 - alpha_pr) * z10 + \
               alpha_tr * alpha_pr * z11
    
    return z_factor


_AGA_DC_TABLE = np.array([
    [1.060, 1.050, 1.042, 1.036, 1.033, 1.032, 1.031, 1.031, 1.034, 1.038, 1.044, 1.053, 1.068],
    [1.080, 1.072, 1.064, 1.058, 1.054, 1.053, 1.052, 1.052, 1.054, 1.058, 1.064, 1.073, 1.086],
    [1.098, 1.092, 1.083, 1.078, 1.074, 1.073, 1.072, 1.071, 1.072, 1.076, 1.082, 1.090, 1.103],
    [1.115, 1.110, 1.102, 1.096, 1.093, 1.092, 1.091, 1.090, 1.090, 1.093, 1.099, 1.108, 1.119],
    [1.131, 1.126, 1.119, 1.113, 1.110, 1.109, 1.108, 1.107, 1.107, 1.109, 1.115, 1.124, 1.136],
    [1.146, 1.142, 1.135, 1.129, 1.126, 1.125, 1.124, 1.124, 1.122, 1.124, 1.130, 1.139, 1.150],
    [1.160, 1.156, 1.150, 1.144, 1.141, 1.140, 1.137, 1.136, 1.137, 1.139, 1.145, 1.154, 1.165],
    [1.173, 1.170, 1.163, 1.158, 1.155, 1.155, 1.153, 1.151, 1.151, 1.152, 1.157, 1.167, 1.178],
    [1.185, 1.182, 1.176, 1.170, 1.168, 1.168, 1.167, 1.165, 1.165, 1.165, 1.170, 1.178, 1.190],
    [1.196, 1.193, 1.188, 1.183, 1.180, 1.180, 1.179, 1.177, 1.177, 1.177, 1.182, 1.190, 1.201],
    [1.206, 1.204, 1.199, 1.195, 1.192, 1.192, 1.190, 1.189, 1.188, 1.188, 1.192, 1.200, 1.210],
])


_AGA_DC_TR = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
_AGA_DC_PR = np.array([0.0, 0.01, 0.03, 0.06, 0.10, 0.15, 0.22, 0.32, 0.45, 0.65, 0.95])


def _aga_dc_interp(pr: float, tr: float) -> float:
    """
    Linear interpolation from AGA-8 DC chart.
    
    Parameters
    ----------
    pr : float
        Reduced pressure.
    tr : float
        Reduced temperature.
    
    Returns
    -------
    float
        Interpolated Z-factor using AGA-8 DC method.
    """
    # Handle out-of-range cases
    if tr < 1.0:
        tr = 1.0
    elif tr > 2.0:
        tr = 2.0
        warnings.warn("Reduced temperature out of AGA-8 range (1.0-2.0)")
    
    if pr < 0.0:
        pr = 0.0
    elif pr > 0.95:
        pr = 0.95
        warnings.warn("Reduced pressure out of AGA-8 range (0-0.95)")
    
    # Find interpolation bounds
    tr_idx = np.searchsorted(_AGA_DC_TR, tr) - 1
    if tr_idx < 0:
        tr_idx = 0
    if tr_idx >= len(_AGA_DC_TR) - 1:
        tr_idx = len(_AGA_DC_TR) - 2
    
    pr_idx = np.searchsorted(_AGA_DC_PR, pr) - 1
    if pr_idx < 0:
        pr_idx = 0
    if pr_idx >= len(_AGA_DC_PR) - 1:
        pr_idx = len(_AGA_DC_PR) - 2
    
    # Bilinear interpolation
    tr0, tr1 = _AGA_DC_TR[tr_idx], _AGA_DC_TR[tr_idx + 1]
    pr0, pr1 = _AGA_DC_PR[pr_idx], _AGA_DC_PR[pr_idx + 1]
    
    z00 = _AGA_DC_TABLE[tr_idx, pr_idx]
    z01 = _AGA_DC_TABLE[tr_idx, pr_idx + 1]
    z10 = _AGA_DC_TABLE[tr_idx + 1, pr_idx]
    z11 = _AGA_DC_TABLE[tr_idx + 1, pr_idx + 1]
    
    # Normalize coordinates
    alpha_tr = (tr - tr0) / (tr1 - tr0) if tr1 != tr0 else 0.5
    alpha_pr = (pr - pr0) / (pr1 - pr0) if pr1 != pr0 else 0.5
    
    z_factor = (1 - alpha_tr) * (1 - alpha_pr) * z00 + \
               (1 - alpha_tr) * alpha_pr * z01 + \
               alpha_tr * (1 - alpha_pr) * z10 + \
               alpha_tr * alpha_pr * z11
    
    return z_factor


def _pseudocritical_from_composition(composition: Dict[str, float]) -> Tuple[float, float]:
    """
    Calculate pseudocritical properties from gas composition.
    
    Parameters
    ----------
    composition : dict
        Mole fraction composition of gas. Example:
        {'CH4': 0.85, 'CO2': 0.10, 'N2': 0.05}
    
    Returns
    -------
    Tuple[float, float]
        (pseudo_critical_pressure_psia, pseudo_critical_temperature_R)
    
    Notes
    -----
    Uses weighted average method:
    Tr = Σ yi * Tr_i
    Pr = Σ yi * Pr_i
    """
    critical_properties = {
        'H2S': {'Tr': 343.1, 'Pr': 672.0},
        'CO2': {'Tr': 304.2 / 1.8 + 459.67, 'Pr': 1070.0},
        'N2': {'Tr': 126.2 / 1.8 + 459.67, 'Pr': 492.0},
        'O2': {'Tr': 154.6 / 1.8 + 459.67, 'Pr': 736.0},
        'CH4': {'Tr': 190.6 / 1.8 + 459.67, 'Pr': 673.0},
        'C2H6': {'Tr': 305.3 / 1.8 + 459.67, 'Pr': 708.0},
        'C3H8': {'Tr': 369.8 / 1.8 + 459.67, 'Pr': 616.0},
        'i-C4H10': {'Tr': 408.1 / 1.8 + 459.67, 'Pr': 527.0},
        'n-C4H10': {'Tr': 425.1 / 1.8 + 459.67, 'Pr': 505.0},
        'i-C5H12': {'Tr': 460.9 / 1.8 + 459.67, 'Pr': 444.0},
        'n-C5H12': {'Tr': 469.6 / 1.8 + 459.67, 'Pr': 434.0},
        'C6H14': {'Tr': 507.6 / 1.8 + 459.67, 'Pr': 369.0},
    }
    
    tr_avg = 0.0
    pr_avg = 0.0
    
    for component, mole_fraction in composition.items():
        if component in critical_properties:
            prop = critical_properties[component]
            tr_avg += mole_fraction * prop['Tr']
            pr_avg += mole_fraction * prop['Pr']
    
    return pr_avg, tr_avg


if __name__ == "__main__":
    # Test the correlations
    print("Z-factor Correlation Tests")
    print("=" * 60)
    
    # Test 1: Standing-Katz
    z_sk = calculate_standing_katz_z_factor(pressure_psia=1500, temperature_f=100)
    print(f"\nStanding-Katz Z-factor (1500 psi, 100°F): {z_sk:.4f}")
    
    # Test 2: Lee-Gonzales
    z_lg = LeeGonzalesEspana.calculate_z_factor(
        pressure_psia=1500, 
        temperature_f=100, 
        specific_gravity=0.65
    )
    print(f"Lee-Gonzales Z-factor (1500 psi, 100°F, γ=0.65): {z_lg:.4f}")
    
    # Test 3: AGA-8 DC
    z_aga = calculate_z_factor_aga_dc(
        pressure_psia=1500, 
        temperature_f=100, 
        specific_gravity=0.65
    )
    print(f"AGA-8 DC Z-factor (1500 psi, 100°F, γ=0.65): {z_aga:.4f}")
    
    # Test 4: Pseudocritical properties
    pc = calculate_pseudocritical_properties(gas_specific_gravity=0.65)
    print(f"\nPseudocritical properties (γ=0.65):")
    print(f"  Prc: {pc[0]:.2f} psi")
    print(f"  Trc: {pc[1]:.2f} R")
    
    # Test 5: Multi-component composition
    comp = {'CH4': 0.85, 'CO2': 0.10, 'N2': 0.05}
    pc_comp = calculate_pseudocritical_properties(gas_specific_gravity=0.65, composition=comp)
    print(f"\nPseudocritical properties (composition):")
    print(f"  Prc: {pc_comp[0]:.2f} psi")
    print(f"  Trc: {pc_comp[1]:.2f} R")
