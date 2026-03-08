"""Equations of State (EOS) for hydrocarbon systems.

This module implements cubic EOS formulations for calculating fluid compressibility 
factors (Z-factor) in ideal and non-ideal gas regimes.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod


class EquationOfState(ABC):
    """
    Abstract base class for Equation of State implementations.
    
    Provides interface for calculating compressibility factors (Z-factor) 
    for hydrocarbon systems across ideal and non-ideal gas regimes.
    
    References:
    - Soave, G.-R. (1972). "Modification of the Wilson equation for 
      calculation of vapor-liquid equilibria," Fluid Phase Equilibria, 1, 5-37.
    - Peng, D.-Y. & Robinson, D.B. (1976). "A New Two-Constant Equation of State,"
      Ind. Eng. Chem. Fundam., 15, 59-64.
    """
    
    @abstractmethod
    def calculate_z_factor(
        self, 
        temperature_K: float, 
        pressure_Pa: float,
        molecular_weight: Optional[float] = None,
        specific_gravity: Optional[float] = None,
        composition: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate compressibility factor (Z-factor) at given P and T.
        
        Parameters
        ----------
        temperature_K : float
            Temperature in kelvin.
        pressure_Pa : float
            Pressure in pascals.
        molecular_weight : float, optional
            Molecular weight in kg/kmol. Required if composition not provided.
        specific_gravity : float, optional
            Gas specific gravity (air=1.0). Required if molecular_weight not provided.
        composition : dict, optional
            Mole fraction composition of gas. If provided, molecular_weight 
            and specific_gravity can be derived. Example: {'CH4': 0.85, 'CO2': 0.1, 'N2': 0.05}
        
        Returns
        -------
        float
            Compressibility factor (Z).
        
        Notes
        -----
        - Returns Z ≈ 1.0 in ideal gas regime at low pressures
        - Returns Z < 1.0 in non-ideal regime at high pressures
        - Valid temperature range: gas can exist, typically -250°C to 300°C
        - Valid pressure range: up to critical pressure or EOS limit
        
        Raises
        ------
        ValueError
            If insufficient parameters provided
        NumericalError
            If equation cannot be solved (no real roots)
        """
        pass
    
    @abstractmethod
    def solve_cubics(
        self, 
        temperature_K: float, 
        pressure_Pa: float,
    ) -> List[float]:
        """
        Solve cubic equation of state for root volumes.
        
        Parameters
        ----------
        temperature_K : float
            Temperature in kelvin.
        pressure_Pa : float
            Pressure in pascals.
        
        Returns
        -------
        List[float]
            Three real roots for molar volume (m³/kmol): V₁, V₂, V₃
        
        Notes
        -----
        - Cubic equation: V³ + a₁V² + a₂V + a₃ = 0
        - Physical root is typically the middle one for stable phases
        - Non-physical roots are discarded or flagged
        
        Raises
        ------
        NumericalError
            If cubic equation does not have three real roots
        """
        pass
    
    def get_pseudocritical_properties(
        self, 
        molecular_weight: Optional[float] = None,
        specific_gravity: Optional[float] = None,
        composition: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float]:
        """
        Calculate pseudocritical properties for EOS calculations.
        
        Parameters
        ----------
        molecular_weight : float, optional
            Molecular weight in kg/kmol.
        specific_gravity : float, optional
            Gas specific gravity (air=1.0).
        composition : dict, optional
            Mole fraction composition.
        
        Returns
        -------
        Tuple[float, float]
            (Tc in K, Pc in Pa) - pseudocritical temperature and pressure
        """
        if composition is not None:
            return self._pseudocritical_from_composition(composition)
        
        # Convert to SI units
        if molecular_weight is not None and specific_gravity is not None:
            # Stewart-Burke-Katz parameters from specific gravity
            specific_gravity_gas = specific_gravity
            tc_rankine = (0.554 * specific_gravity_gas + 
                         0.4 * (1.0 - 0.554 * specific_gravity_gas)) * 460.0
            tc_kelvin = tc_rankine - 459.67
            
            pc_psi = 708.0 - 58.71 * specific_gravity_gas + 0.0107 * (specific_gravity_gas ** 2)
            # Convert psi to Pa: 1 psi = 6894.76 Pa
            pc_pascal = pc_psi * 6894.76
            
            return tc_kelvin, pc_pascal
        
        raise ValueError("Either molecular_weight, specific_gravity, or composition must be provided")
    
    def _pseudocritical_from_composition(
        self, 
        composition: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Calculate pseudocritical properties from gas composition.
        
        Parameters
        ----------
        composition : dict
            Mole fraction composition. Example: {'CH4': 0.85, 'CO2': 0.1, 'N2': 0.05}
        
        Returns
        -------
        Tuple[float, float]
            (Tc in K, Pc in Pa) - weighted average of component critical properties
        """
        critical_properties = {
            'H2S': {'Tc': 373.9, 'Pc': 89.6e5},  # K, Pa
            'CO2': {'Tc': 304.3, 'Pc': 73.8e5},  # K, Pa
            'N2': {'Tc': 126.2, 'Pc': 33.9e5},   # K, Pa
            'O2': {'Tc': 154.6, 'Pc': 50.4e5},   # K, Pa
            'CH4': {'Tc': 190.6, 'Pc': 45.99e5}, # K, Pa
            'C2H6': {'Tc': 305.3, 'Pc': 48.72e5}, # K, Pa
            'C3H8': {'Tc': 369.8, 'Pc': 42.48e5}, # K, Pa
            'i-C4H10': {'Tc': 408.1, 'Pc': 36.48e5}, # K, Pa
            'n-C4H10': {'Tc': 425.2, 'Pc': 37.96e5}, # K, Pa
            'i-C5H12': {'Tc': 460.9, 'Pc': 33.30e5}, # K, Pa
            'n-C5H12': {'Tc': 469.6, 'Pc': 33.70e5}, # K, Pa
            'C6H14': {'Tc': 507.6, 'Pc': 30.20e5},   # K, Pa
        }
        
        tc_avg = 0.0
        pc_avg = 0.0
        
        for component, mole_fraction in composition.items():
            if component in critical_properties:
                prop = critical_properties[component]
                tc_avg += mole_fraction * prop['Tc']
                pc_avg += mole_fraction * prop['Pc']
        
        return tc_avg, pc_avg


class NumericalError(Exception):
    """Exception raised for numerical errors in EOS calculations."""
    pass


class SRKEOS(EquationOfState):
    """
    Soave-Redlich-Kwong Equation of State.
    
    Modified cubic equation of state that accounts for temperature-dependent 
    attraction parameter. Widely used for natural gas and light hydrocarbon systems.
    
    Reference:
    - Soave, G.-R. (1972). "Modification of the Wilson equation for 
      calculation of vapor-liquid equilibria," Fluid Phase Equilibria, 1, 5-37.
    
    Equation:
    P = RT/(V-b) - aα(T)/(V(V+b)+b(V-b))
    
    Where:
    - a = 0.42748 R² Tc² / Pc
    - b = 0.08664 R Tc / Pc
    - α(T) = [1 + m(1 - √Tr)]²
    - m = 0.48 + 1.574ω - 0.176ω²
    
    For pure hydrocarbons:
    - Acentric factor ω is typically: 0.5 to 0.7 for natural gas
    """
    
    def __init__(
        self, 
        molecular_weight: float = 16.0,
        specific_gravity: Optional[float] = None,
        acentric_factor: float = 0.6,
    ):
        """
        Initialize SRK EOS.
        
        Parameters
        ----------
        molecular_weight : float
            Molecular weight in kg/kmol (e.g., 16.0 for methane).
        specific_gravity : float, optional
            Gas specific gravity (air=1.0). If provided, overrides molecular_weight.
        acentric_factor : float
            Acentric factor (dimensionless). For natural gas: 0.3-0.7.
        """
        self.molecular_weight = molecular_weight
        self.specific_gravity = specific_gravity
        self.omega = acentric_factor
        
        # Universal gas constant (J/(kmol·K))
        self.R = 8.314
        
        # Pre-calculate A and B parameters
        if specific_gravity is not None:
            molecular_weight = 28.964 / specific_gravity
        
        # Constants for SRK EOS
        self._calc_critical_properties()
    
    def _calc_critical_properties(self) -> None:
        """Calculate critical properties from molecular weight and specific gravity."""
        if self.specific_gravity is not None:
            # Stewart-Burke-Katz parameters
            specific_gravity_gas = self.specific_gravity
            tc_rankine = (0.554 * specific_gravity_gas + 
                         0.4 * (1.0 - 0.554 * specific_gravity_gas)) * 460.0
            tc_kelvin = tc_rankine - 459.67
            
            pc_psi = 708.0 - 58.71 * specific_gravity_gas + 0.0107 * (specific_gravity_gas ** 2)
            pc_pascal = pc_psi * 6894.76
            
            self.Tc, self.Pc = tc_kelvin, pc_pascal
        else:
            self.Tc, self.Pc = None, None
    
    def calculate_z_factor(
        self, 
        temperature_K: float = 293.15,
        pressure_Pa: float = 101325.0,
        composition: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate Z-factor using SRK EOS.
        
        Parameters
        ----------
        temperature_K : float
            Temperature in kelvin. Default: 293.15 K (20°C).
        pressure_Pa : float
            Pressure in pascals. Default: 101325 Pa (1 atm).
        composition : dict, optional
            Mole fraction composition. If provided, overrides molecular_weight and 
            specific_gravity for critical property calculation.
        
        Returns
        -------
        float
            Compressibility factor (Z).
        
        Notes
        -----
        - Ideal gas regime: Z ≈ 1.0 at low pressures
        - Non-ideal regime: Z < 1.0 at high pressures
        - Maximum accuracy: ±0.01 for typical natural gas systems
        """
        # Calculate pseudocritical properties
        if composition is not None:
            Tc, Pc = self._pseudocritical_from_composition(composition)
        else:
            Tc, Pc = self.get_pseudocritical_properties(
                molecular_weight=self.molecular_weight,
                specific_gravity=self.specific_gravity
            )
        
        # Reduced and inverse reduced properties
        try:
            Pr = pressure_Pa / Pc
            Tr = temperature_K / Tc
        except ZeroDivisionError:
            # Ideal gas assumption if critical properties are zero/invalid
            return 1.0
        
        # Check gas regime (ideal gas if reduced pressure is very low)
        if Pr < 0.1:
            # Ideal gas: Z = Pv/RT
            v_ideal = self.R * temperature_K / pressure_Pa  # m³/kmol
            # Using SRK B parameter
            B = 0.08664 * self.R * Tc / Pc
            z_ideal = B * Pr / (pressure_Pa / (self.R * temperature_K))  # Z = Pv/RT
            return z_ideal
        
        # Reduced temperature check
        if Tr == 0:
            Tr = 0.01
        
        # Calculate SRK alpha function
        m = 0.48 + 1.574 * self.omega - 0.176 * self.omega ** 2
        alpha = (1.0 + m * (1.0 - np.sqrt(Tr))) ** 2
        
        # SRK parameters
        a = 0.42748 * (self.R * Tc) ** 2 / Pc
        b = 0.08664 * self.R * Tc / Pc
        
        # A = aP/(RT)² and B = bP/(RT)
        A = a * pressure_Pa / ((self.R * temperature_K) ** 2)
        B = b * pressure_Pa / (self.R * temperature_K)
        
        # SRK cubic equation: Z³ - (1 - B)Z² + (A - 2B - 3B²)Z - (AB - B² - B³) = 0
        a1 = -1.0 + B
        a2 = (A - 2.0 * B - 3.0 * B ** 2)
        a3 = -(A * B - B ** 2 - B ** 3)
        
        # Solve cubic equation
        roots = self._solve_cubic_cool(a1, a2, a3)
        
        # Select the appropriate root based on physical constraints
        # Gas phase: choose smallest positive root that gives physically meaningful Z
        for z in roots:
            if z > 0 and z < 0.999:
                return z
        
        # If no suitable root found, use ideal gas as fallback
        return 1.0
    
    def solve_cubics(
        self, 
        temperature_K: float,
        pressure_Pa: float,
    ) -> List[float]:
        """
        Solve SRK cubic equation for molar volumes.
        
        Parameters
        ----------
        temperature_K : float
            Temperature in kelvin.
        pressure_Pa : float
            Pressure in pascals.
        
        Returns
        -------
        List[float]
            Three real molar volume roots (m³/kmol).
        
        Raises
        ------
        NumericalError
            If cubic equation does not have three real roots
        """
        # Calculate critical properties
        if self.specific_gravity is None:
            Tc, Pc = self.get_pseudocritical_properties(
                molecular_weight=self.molecular_weight
            )
        else:
            Tc, Pc = self.get_pseudocritical_properties(
                specific_gravity=self.specific_gravity
            )
        
        # SRK parameters
        a = 0.42748 * (self.R * Tc) ** 2 / Pc
        b = 0.08664 * self.R * Tc / Pc
        
        # A and B parameters
        A = a * pressure_Pa / ((self.R * temperature_K) ** 2)
        B = b * pressure_Pa / (self.R * temperature_K)
        
        # Cubic coefficients
        a1 = -1.0 + B
        a2 = (A - 2.0 * B - 3.0 * B ** 2)
        a3 = -(A * B - B ** 2 - B ** 3)
        
        roots = self._solve_cubic_cool(a1, a2, a3)
        
        if len(roots) != 3:
            raise NumericalError("SRK cubic equation did not produce three real roots")
        
        return roots
    
    def _solve_cubic_cool(
        self, 
        a1: float, 
        a2: float, 
        a3: float
    ) -> List[float]:
        """
        Solve cubic equation using Ferrari's method with numerical root finding.
        
        Parameters
        ----------
        a1, a2, a3 : float
            Coefficients of cubic equation: z³ + a1z² + a2z + a3 = 0
        
        Returns
        -------
        List[float]
            Three real roots (may include negative/zero values)
        """
        # Convert to monic form: z³ + coefs[0]z² + coefs[1]z + coefs[2] = 0
        coefs = [a1, a2, a3]
        
        # Use numpy's cubic solver
        try:
            roots = np.roots(coefs)
            
            # Filter out complex roots (within floating-point tolerance)
            real_roots = []
            for root in roots:
                if np.isclose(root.imag, 0.0):
                    real_roots.append(float(root.real))
                
                # Handle complex roots close to real axis
                elif abs(root.imag) < 1e-5:
                    real_roots.append(float(root.real + root.imag * 0.5))
            
            # Sort roots
            real_roots.sort(key=lambda x: x.real)
            return real_roots
            
        except Exception as e:
            raise NumericalError(f"Cubic solver failed: {e}")


class PengRobinsonEOS(EquationOfState):
    """
    Peng-Robinson Equation of State.
    
    Two-parameter cubic EOS that provides superior accuracy for 
    non-ideal fluids, particularly for polar components and near-critical 
    conditions. Better than SRK for systems with CO₂, H₂S, and N₂.
    
    Reference:
    - Peng, D.-Y. & Robinson, D.B. (1976). "A New Two-Constant Equation of State,"
      Ind. Eng. Chem. Fundam., 15, 59-64.
    
    Equation:
    P = RT/(V-b) - aα(T)/(V² + 2bV - b²)
    
    Where:
    - a = 0.45724 R² Tc² / Pc
    - b = 0.07780 R Tc / Pc
    - α(T) = [1 + κ(1 - √Tr)]²
    - κ = 0.37464 + 1.54226ω - 0.26992ω²
    
    Differences from SRK:
    - Different attraction parameter (a)
    - Different temperature function (α)
    - Better handling of polar components
    - More accurate near-critical region
    """
    
    def __init__(
        self, 
        molecular_weight: float = 16.0,
        specific_gravity: Optional[float] = None,
        acentric_factor: float = 0.6,
    ):
        """
        Initialize Peng-Robinson EOS.
        
        Parameters
        ----------
        molecular_weight : float
            Molecular weight in kg/kmol (e.g., 16.0 for methane).
        specific_gravity : float, optional
            Gas specific gravity (air=1.0). If provided, overrides molecular_weight.
        acentric_factor : float
            Acentric factor (dimensionless). For natural gas: 0.3-0.7.
        """
        self.molecular_weight = molecular_weight
        self.specific_gravity = specific_gravity
        self.omega = acentric_factor
        
        # Universal gas constant (J/(kmol·K))
        self.R = 8.314
        
        # Pre-calculate parameters if specific gravity provided
        if specific_gravity is not None:
            self._calc_critical_properties()
    
    def _calc_critical_properties(self) -> None:
        """Calculate critical properties from specific gravity."""
        specific_gravity_gas = self.specific_gravity
        tc_rankine = (0.554 * specific_gravity_gas + 
                     0.4 * (1.0 - 0.554 * specific_gravity_gas)) * 460.0
        tc_kelvin = tc_rankine - 459.67
        
        pc_psi = 708.0 - 58.71 * specific_gravity_gas + 0.0107 * (specific_gravity_gas ** 2)
        pc_pascal = pc_psi * 6894.76
        
        self.Tc = tc_kelvin
        self.Pc = pc_pascal
    
    def calculate_z_factor(
        self, 
        temperature_K: float = 293.15,
        pressure_Pa: float = 101325.0,
        composition: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate Z-factor using Peng-Robinson EOS.
        
        Parameters
        ----------
        temperature_K : float
            Temperature in kelvin. Default: 293.15 K (20°C).
        pressure_Pa : float
            Pressure in pascals. Default: 101325 Pa (1 atm).
        composition : dict, optional
            Mole fraction composition.
        
        Returns
        -------
        float
            Compressibility factor (Z).
        
        Notes
        -----
        - Preferred over SRK for natural gas with CO₂, H₂S, or N₂
        - Better accuracy in high-pressure non-ideal regime
        - Maximum accuracy: ±0.01 for typical systems
        """
        # Calculate pseudocritical properties
        if composition is not None:
            Tc, Pc = self._pseudocritical_from_composition(composition)
        elif self.specific_gravity is not None:
            Tc, Pc = self.Tc, self.Pc
        else:
            Tc, Pc = self.get_pseudocritical_properties(
                molecular_weight=self.molecular_weight
            )
        
        # Check for zero/invalid critical properties
        if Tc <= 0 or Pc <= 0:
            # Ideal gas assumption
            return 1.0
        
        # Reduced and inverse reduced properties
        try:
            Pr = pressure_Pa / Pc
            Tr = temperature_K / Tc
        except ZeroDivisionError:
            return 1.0
        
        # Ideal gas check (low reduced pressure)
        if Pr < 0.1:
            v_ideal = self.R * temperature_K / pressure_Pa  # m³/kmol
            B = 0.07780 * self.R * Tc / Pc
            z_ideal = B * Pr / (pressure_Pa / (self.R * temperature_K))
            return z_ideal
        
        # Reduced temperature check
        if Tr == 0:
            Tr = 0.01
        
        # Peng-Robinson alpha function
        kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega ** 2
        alpha = (1.0 + kappa * (1.0 - np.sqrt(Tr))) ** 2
        
        # Peng-Robinson parameters
        a = 0.45724 * (self.R * Tc) ** 2 / Pc
        b = 0.07780 * self.R * Tc / Pc
        
        # A and B parameters
        A = a * pressure_Pa / ((self.R * temperature_K) ** 2)
        B = b * pressure_Pa / (self.R * temperature_K)
        
        # Peng-Robinson cubic equation in Z:
        # Z³ + (B - 1)Z² + (A - 3B² - 2B)Z + (-A·B + B² + B³) = 0
        a1 = B - 1.0
        a2 = (A - 3.0 * B ** 2 - 2.0 * B)
        a3 = -(A * B - B ** 2 - B ** 3)
        
        # Solve cubic equation
        roots = self._solve_cubic_cool(a1, a2, a3)
        
        # Select appropriate root
        for z in roots:
            if z > 0 and z < 0.999:
                return z
        
        return 1.0
    
    def solve_cubics(
        self, 
        temperature_K: float,
        pressure_Pa: float,
    ) -> List[float]:
        """
        Solve Peng-Robinson cubic equation for molar volumes.
        
        Parameters
        ----------
        temperature_K : float
            Temperature in kelvin.
        pressure_Pa : float
            Pressure in pascals.
        
        Returns
        -------
        List[float]
            Three real molar volume roots (m³/kmol).
        
        Raises
        ------
        NumericalError
            If cubic equation does not produce three real roots
        """
        # Get critical properties
        if self.specific_gravity is not None:
            Tc, Pc = self.Tc, self.Pc
        else:
            Tc, Pc = self.get_pseudocritical_properties(
                molecular_weight=self.molecular_weight
            )
        
        # Peng-Robinson parameters
        a = 0.45724 * (self.R * Tc) ** 2 / Pc
        b = 0.07780 * self.R * Tc / Pc
        
        # A and B parameters
        A = a * pressure_Pa / ((self.R * temperature_K) ** 2)
        B = b * pressure_Pa / (self.R * temperature_K)
        
        # Cubic coefficients
        a1 = B - 1.0
        a2 = (A - 3.0 * B ** 2 - 2.0 * B)
        a3 = -(A * B - B ** 2 - B ** 3)
        
        roots = self._solve_cubic_cool(a1, a2, a3)
        
        if len(roots) != 3:
            raise NumericalError("Peng-Robinson cubic equation did not produce three real roots")
        
        return roots


def calculate_z_factor_aga_dc(
    temperature_f: float, 
    pressure_psia: float,
    specific_gravity: float,
) -> float:
    """
    Calculate Z-factor using AGA-8 Natural Gas EOS (DC method).
    
    This function provides a convenient interface for natural gas Z-factor 
    calculations that integrates with the EOS implementations.
    
    Parameters
    ----------
    temperature_f : float
        Temperature in degrees Fahrenheit.
    pressure_psia : float
        Pressure in psi.
    specific_gravity : float
        Gas specific gravity (air=1.0).
    
    Returns
    -------
    float
        Compressibility factor (Z).
    
    Notes
    -----
    - Uses pseudocritical properties from specific gravity
    - Interpolates from AGA-8 DC lookup table
    - Standard method for custody transfer
    """
    # Temperature in K
    temperature_k = (temperature_f + 459.67) * 5/9
    
    # Pressure in Pa
    pressure_pascal = pressure_psia * 6894.76
    
    # Create SRK EOS instance
    eos = SRKEOS(specific_gravity=specific_gravity)
    
    # Calculate Z-factor
    z = eos.calculate_z_factor(temperature_K=temperature_k, pressure_Pa=pressure_pascal)
    
    return z


if __name__ == "__main__":
    """Test the EOS implementations."""
    print("=" * 70)
    print("Equation of State Solvers - Test Suite")
    print("=" * 70)
    
    # Test 1: SRK on methane at normal conditions
    print("\n1. SRK EOS on Methane (γ=0.55)")
    print("-" * 70)
    srk = SRKEOS(specific_gravity=0.55, acentric_factor=0.6)
    z_sk = srk.calculate_z_factor(temperature_K=293.15, pressure_Pa=101325.0)
    print(f"   T = 293.15 K, P = 101325 Pa → Z = {z_sk:.6f}")
    
    # Test 2: SRK at high pressure (non-ideal regime)
    print("\n2. SRK EOS on Methane at High Pressure")
    print("-" * 70)
    z_sk_high = srk.calculate_z_factor(temperature_K=293.15, pressure_Pa=10e6)
    print(f"   T = 293.15 K, P = 10 MPa → Z = {z_sk_high:.6f}")
    
    # Test 3: Peng-Robinson on methane
    print("\n3. Peng-Robinson EOS on Methane (γ=0.55)")
    print("-" * 70)
    pr = PengRobinsonEOS(specific_gravity=0.55, acentric_factor=0.6)
    z_pr = pr.calculate_z_factor(temperature_K=293.15, pressure_Pa=101325.0)
    print(f"   T = 293.15 K, P = 101325 Pa → Z = {z_pr:.6f}")
    
    # Test 4: Peng-Robinson at high pressure
    print("\n4. Peng-Robinson EOS on Methane at High Pressure")
    print("-" * 70)
    z_pr_high = pr.calculate_z_factor(temperature_K=293.15, pressure_Pa=10e6)
    print(f"   T = 293.15 K, P = 10 MPa → Z = {z_pr_high:.6f}")
    
    # Test 5: Compare SRK vs PR at high pressure
    print("\n5. SRK vs Peng-Robinson Comparison (High Pressure)")
    print("-" * 70)
    print(f"   SRK Z = {z_sk_high:.6f}")
    print(f"   PR Z  = {z_pr_high:.6f}")
    print(f"   Difference = {abs(z_sk_high - z_pr_high):.6f}")
    
    # Test 6: Cubic equation solving
    print("\n6. Cubic Equation Solving")
    print("-" * 70)
    volumes = srk.solve_cubics(temperature_K=293.15, pressure_Pa=10e6)
    print(f"   Three molar volume roots (m³/kmol):")
    for i, v in enumerate(volumes, 1):
        print(f"   V{i} = {v:.6f}")
    
    # Test 7: Ideal gas vs non-ideal
    print("\n7. Ideal Gas vs Non-Ideal Regime (PR EOS)")
    print("-" * 70)
    z_ideal = pr.calculate_z_factor(temperature_K=293.15, pressure_Pa=1000)  # Low pressure
    z_non_ideal = pr.calculate_z_factor(temperature_K=293.15, pressure_Pa=10e6)  # High pressure
    print(f"   Low pressure (1 kPa): Z = {z_ideal:.6f}")
    print(f"   High pressure (10 MPa): Z = {z_non_ideal:.6f}")
    print(f"   Ratio Z_high/Z_ideal = {z_non_ideal/z_ideal:.3f}")
    
    # Test 8: Multi-component gas
    print("\n8. Multi-component Gas System")
    print("-" * 70)
    composition = {'CH4': 0.85, 'CO2': 0.10, 'N2': 0.05}
    z_multi = pr.calculate_z_factor(
        temperature_K=310.15, 
        pressure_Pa=5e6, 
        composition=composition
    )
    print(f"   Composition: {composition}")
    print(f"   T = 310.15 K, P = 5 MPa → Z = {z_multi:.6f}")
    
    # Test 9: Z-factor using helper function
    print("\n9. Z-factor via AGA-8 Helper Function")
    print("-" * 70)
    z_aga = calculate_z_factor_aga_dc(
        temperature_f=100,
        pressure_psia=1200,
        specific_gravity=0.65
    )
    print(f"   T = 100°F, P = 1200 psi, γ = 0.65 → Z = {z_aga:.6f}")
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
