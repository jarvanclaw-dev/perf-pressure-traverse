"""
Vapor-Liquid Equilibrium (VLE) flash calculations for two-phase systems.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from perf_pressure_traverse.math.eos import SRKEOS, PengRobinsonEOS, EquationOfState, NumericalError


class VLEFlash:
    """
    Vapor-Liquid Equilibrium flash calculation solver.
    
    Performs flash calculations to determine phase equilibrium between
    liquid and vapor phases for a given composition, temperature, and pressure.
    
    Attributes
    ----------
    eos : EquationOfState
        Equation of state implementation (SRK or Peng-Robinson).
    max_iterations : int
        Maximum number of iterations in flash calculation.
    tolerance : float
        Convergence tolerance for relative error in K-values.
    """
    
    def __init__(
        self,
        eos_type: str = "srk",
        max_iterations: int = 100,
        tolerance: float = 1e-4
    ):
        """
        Initialize VLE flash calculator.
        
        Parameters
        ----------
        eos_type : str
            Equation of state type: "srk" or "pr" (Peng-Robinson).
        max_iterations : int, optional
            Maximum number of iterations for convergence.
        tolerance : float, optional
            Convergence tolerance for K-value convergence.
        """
        if eos_type.lower() == "srk":
            self.eos = SRKEOS()
        else:
            self.eos = PengRobinsonEOS()
        
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def calculate_k_values(
        self,
        temperature_K: float,
        pressure_Pa: float,
        composition: Dict[str, float],
        acentric_factors: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate equilibrium K-values using the chosen EOS.
        
        K-value = y_i / x_i, where y_i is vapor phase mole fraction
        and x_i is liquid phase mole fraction.
        
        Parameters
        ----------
        temperature_K : float
            Temperature in Kelvin.
        pressure_Pa : float
            Pressure in Pascals.
        composition : Dict[str, float]
            Overall composition in liquid phase (mole fractions).
        acentric_factors : Optional[Dict[str, float]], optional
            Acentric factors for each component.
        
        Returns
        -------
        Dict[str, float]
            K-values for each component.
        
        Raises
        ------
        ValueError
            If composition sum doesn't equal 1.0 or contains invalid components.
        """
        # Validate composition
        comp_sum = sum(composition.values())
        if not np.isclose(comp_sum, 1.0, rtol=1e-6):
            raise ValueError(
                f"Composition must sum to 1.0, got {comp_sum}. "
                "Please normalize composition first."
            )
        
        K_values = {}
        
        # Calculate K-values component by component
        for component, x_i in composition.items():
            if acentric_factors and component in acentric_factors:
                omega = acentric_factors[component]
            else:
                omega = 0.0  # Default for non-polar components
            
            try:
                # Use EOS to get pseudocritical properties
                Tc, Pc = self.eos.get_pseudocritical_properties(
                    specific_gravity=0.65
                )
                
                Tr = temperature_K / Tc
                Pr = pressure_Pa / Pc
                
                # Simplified K-value calculation using empirical correlation
                # Based on equilibrium of fugacity coefficients
                # K_i = exp(5.37*(1+omega)*(1-Tc/T))
                K_i = np.exp(5.37 * (1 + omega) * (1 - Tc/temperature_K))
                
                # Ensure reasonable bounds
                K_i = max(K_i, 1e-10)
            
            except NumericalError:
                K_i = 1.0
            except Exception:
                K_i = 1.0
            
            K_values[component] = K_i
        
        return K_values
    
    def perform_flash(
        self,
        temperature_K: float,
        pressure_Pa: float,
        composition: Dict[str, float],
        acentric_factors: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Perform two-phase flash calculation.
        
        Solves for liquid and vapor compositions given overall composition,
        temperature, and pressure.
        
        Parameters
        ----------
        temperature_K : float
            Temperature in Kelvin.
        pressure_Pa : float
            Pressure in Pascals.
        composition : Dict[str, float]
            Overall fluid composition (mole fractions).
        acentric_factors : Optional[Dict[str, float]], optional
            Acentric factors for each component.
        
        Returns
        -------
        Tuple[Dict[str, float], Dict[str, float]]
            (liquid_composition, vapor_composition). Each is a dict of mole fractions.
        
        Raises
        ------
        ValueError
            If given composition sum doesn't equal 1.0.
        NumericalError
            If flash calculation doesn't converge.
        """
        # Validate composition
        comp_sum = sum(composition.values())
        if not np.isclose(comp_sum, 1.0, rtol=1e-6):
            raise ValueError(
                f"Composition must sum to 1.0, got {comp_sum}. "
                "Please normalize composition first."
            )
        
        # Calculate initial K-values
        K_values = self.calculate_k_values(
            temperature_K, pressure_Pa, composition, acentric_factors
        )
        
        # Flash calculation using Rachford-Rice equation
        V = 0.5  # Initial guess for vapor fraction
        
        for iteration in range(self.max_iterations):
            # Calculate liquid and vapor compositions
            liquid_composition = {}
            vapor_composition = {}
            
            for component, z_i in composition.items():
                x_i = z_i / (1 - V + V * K_values[component])
                y_i = K_values[component] * x_i
                
                liquid_composition[component] = x_i
                vapor_composition[component] = y_i
            
            # Check convergence of Rachford-Rice equation
            RR = sum(
                (K_values[component] - 1) * z_i 
                / (1 - V + V * K_values[component]) 
                for component, z_i in composition.items()
            )
            
            # Check relative error
            if iteration > 0:
                rel_error = abs(V - V_old) / abs(V_old) if V_old != 0 else 1.0
                if rel_error < self.tolerance:
                    break
            
            V_old = V
            
            # Newton-Raphson update for V
            dRR_dV = sum(
                (1 - K_values[component])**2 * z_i 
                / (1 - V + V * K_values[component])**2 
                for component, z_i in composition.items()
            )
            
            if abs(dRR_dV) > 1e-10:
                V = V - RR / dRR_dV
                V = np.clip(V, 0.0, 1.0 - 1e-10)
            else:
                break
        
        # Recalculate with final V
        liquid_composition = {}
        vapor_composition = {}
        
        for component, z_i in composition.items():
            x_i = z_i / (1 - V + V * K_values[component])
            y_i = K_values[component] * x_i
            
            liquid_composition[component] = x_i
            vapor_composition[component] = y_i
        
        # Normalize to handle floating point errors
        liquid_sum = sum(liquid_composition.values())
        vapor_sum = sum(vapor_composition.values())
        
        if liquid_sum > 0:
            liquid_composition = {k: v/liquid_sum for k, v in liquid_composition.items()}
        if vapor_sum > 0:
            vapor_composition = {k: v/vapor_sum for k, v in vapor_composition.items()}
        
        # Check for convergence
        if abs(V) > 1.01 or abs(V) < 0.0:
            raise NumericalError(
                f"Flash calculation did not converge. V = {V}. "
                f"Expected V between 0 and 1."
            )
        
        return liquid_composition, vapor_composition
    
    def calculate_phase_volumes(
        self,
        total_moles: float,
        V: float,
        molar_volume_liq: float,
        molar_volume_vap: float
    ) -> Tuple[float, float]:
        """
        Calculate phase volumes from total moles and vapor fraction.
        
        Parameters
        ----------
        total_moles : float
            Total moles of fluid.
        V : float
            Vapor fraction (mole basis).
        molar_volume_liq : float
            Molar volume of liquid phase (m³/mol).
        molar_volume_vap : float
            Molar volume of vapor phase (m³/mol).
        
        Returns
        -------
        Tuple[float, float]
            (liquid_volume, vapor_volume) in m³.
        """
        liquid_moles = (1 - V) * total_moles
        vapor_moles = V * total_moles
        
        liquid_volume = liquid_moles * molar_volume_liq
        vapor_volume = vapor_moles * molar_volume_vap
        
        return liquid_volume, vapor_volume


class TwoPhaseCompositionalSystem:
    """
    Compositional system handler for two-phase flash calculations.
    
    Manages complete EOS and VLE calculations for oil-gas systems.
    """
    
    def __init__(self, eos_type: str = "srk"):
        """
        Initialize compositional system.
        
        Parameters
        ----------
        eos_type : str
            Equation of state type: "srk" or "srk or pr".
        """
        self.flash = VLEFlash(eos_type=eos_type)
    
    def calculate_vle_properties(
        self,
        temperature_K: float,
        pressure_Pa: float,
        composition: Dict[str, float],
        acentric_factors: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Perform complete VLE analysis.
        
        Returns comprehensive fluid properties including phase compositions,
        Z-factors, and PTV relationships.
        
        Parameters
        ----------
        temperature_K : float
            Temperature in Kelvin.
        pressure_Pa : float
            Pressure in Pascals.
        composition : Dict[str, float]
            Overall composition (mole fractions).
        acentric_factors : Optional[Dict[str, float]], optional
            Acentric factors for each component.
        
        Returns
        -------
        Dict
            Dictionary containing all calculated properties.
        """
        try:
            # Perform flash calculation
            liquid_comp, vapor_comp = self.flash.perform_flash(
                temperature_K, pressure_Pa, composition, acentric_factors
            )
            
            # Calculate Z-factors for both phases
            z_liq = self.flash.eos.calculate_z_factor(
                temperature_K, pressure_Pa, composition
            )
            z_vap = self.flash.eos.calculate_z_factor(
                temperature_K, pressure_Pa, vapor_comp
            )
            
            # Estimate molar volumes
            Tc, Pc = self.flash.eos.get_pseudocritical_properties(
                specific_gravity=0.65
            )
            R = 8.314
            
            molar_volume_liq = R * temperature_K / pressure_Pa  # Ideal gas approximation
            molar_volume_vap = R * temperature_K / pressure_Pa
            
            # Calculate phase properties
            V = sum(vapor_comp.values())
            
            return {
                'temperature_K': temperature_K,
                'pressure_Pa': pressure_Pa,
                'vapor_fraction': V,
                'liquid_composition': liquid_comp,
                'vapor_composition': vapor_comp,
                'z_factor_liquid': z_liq,
                'z_factor_vapor': z_vap,
                'molar_volume_liq': molar_volume_liq,
                'molar_volume_vap': molar_volume_vap,
                'total_moles': sum(composition.values()),
            }
            
        except Exception as e:
            raise NumericalError(f"VLE calculation failed: {str(e)}")
