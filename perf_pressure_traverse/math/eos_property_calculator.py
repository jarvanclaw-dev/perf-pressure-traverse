"""
EOS Property Calculator for compositional systems.
Returns pressure-temperature-volume (PTV) relationships and phase compositions.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field

from perf_pressure_traverse.math.eos import SRKEOS, PengRobinsonEOS, EquationOfState, NumericalError, calculate_z_factor_aga_dc


@dataclass
class EOSPropertyResult:
    """
    Container for EOS property calculation results.
    
    Attributes
    ----------
    temperature_K : float
        Temperature in Kelvin.
    pressure_Pa : float
        Pressure in Pascals.
    volume_m3_mol : float
        Molar volume (m³/mol).
    z_factor : float
        Compressibility factor.
    liquid_composition : Dict[str, float]
        Mole fractions in liquid phase.
    vapor_composition : Dict[str, float]
        Mole fractions in vapor phase.
    phase_flag : str
        Phase flag: "liquid", "vapor", or "mix".
    total_moles : float
        Total moles of fluid.
    """
    temperature_K: float
    pressure_Pa: float
    volume_m3_mol: float
    z_factor: float
    liquid_composition: Dict[str, float] = field(default_factory=dict)
    vapor_composition: Dict[str, float] = field(default_factory=dict)
    phase_flag: str = "liquid"
    total_moles: float = 1.0
    
    def get_volumetric_flow_rate(
        self,
        pressure_psia: float,
        temperature_f: float,
        total_moles: Optional[float] = None
    ) -> Optional[float]:
        """
        Calculate volumetric flow rate at standard conditions.
        
        Parameters
        ----------
        pressure_psia : float
            Pressure in psia for flow rate.
        temperature_f : float
            Temperature in °F for flow rate.
        total_moles : Optional[float], optional
            Total moles (defaults to self.total_moles).
        
        Returns
        -------
        Optional[float]
            Volumetric flow rate at standard conditions, or None for error.
        """
        if pressure_psia is None or temperature_f is None:
            return None
        
        # Standard conditions: 60°F, 14.7 psia
        T_std_f = 60.0
        P_std_psia = 14.7
        
        # Convert temperature
        T_std_K = T_std_f + 459.67
        T_current_K = temperature_f + 459.67
        
        # Convert pressure
        P_std_Pa = P_std_psia * 6894.76
        P_current_Pa = pressure_psia * 6894.76
        
        # Use compressibility factor at standard conditions
        z_std = calculate_z_factor_aga_dc(T_std_f, P_std_psia, specific_gravity=0.65)
        
        # Calculate molar volume at standard conditions
        R = 8.314
        V_std_m3_mol = (z_std * R * T_std_K) / P_std_Pa
        
        # Calculate current molar volume
        V_current_m3_mol = (self.z_factor * R * T_current_K) / P_current_Pa
        
        # Standard volume per mole
        volume_per_mole_std = V_std_m3_mol / V_current_m3_mol * self.volume_m3_mol
        
        if total_moles is not None:
            total_volume_std = volume_per_mole_std * total_moles
        else:
            total_volume_std = volume_per_mole_std * self.total_moles
        
        return total_volume_std


class EOSPropertyCalculator:
    """
    EOS Property Calculator for compositional systems.
    
    Calculates comprehensive PTV relationships and phase compositions
    using Equation of State methods.
    """
    
    def __init__(
        self,
        eos_type: str = "srk",
        specific_gravity: Optional[float] = None,
        acentric_factor: float = 0.6
    ):
        """
        Initialize EOS Property Calculator.
        
        Parameters
        ----------
        eos_type : str
            Equation of state type: "srk" or "pr".
        specific_gravity : float, optional
            Fluid specific gravity.
        acentric_factor : float
            Fluid acentric factor.
        """
        if eos_type.lower() == "srk":
            # SRKEOS doesn't support acentric_factor in init
            self.eos = SRKEOS(specific_gravity=specific_gravity)
            self.omega = None
        else:
            from perf_pressure_traverse.math.eos import PengRobinsonEOS
            self.eos = PengRobinsonEOS(specific_gravity=specific_gravity, acentric_factor=acentric_factor)
            self.omega = acentric_factor if specific_gravity is not None else None
        
        self.specific_gravity = specific_gravity
    
    def calculate_property_at_conditions(
        self,
        temperature_K: float,
        pressure_Pa: float,
        composition: Optional[Dict[str, float]] = None
    ) -> EOSPropertyResult:
        """
        Calculate EOS properties at given conditions.
        
        Parameters
        ----------
        temperature_K : float
            Temperature in Kelvin.
        pressure_Pa : float
            Pressure in Pascals.
        composition : Optional[Dict[str, float]], optional
            Fluid composition (mole fractions).
        
        Returns
        -------
        EOSPropertyResult
            Calculation results containing PTV data.
        
        Raises
        ------
        NumericalError
            If calculation fails.
        """
        try:
            # Calculate Z-factor using base class equation of state method
            # which accepts specific_gravity or molecular_weight
            z_factor = self.eos.calculate_z_factor(
                temperature_K=temperature_K,
                pressure_Pa=pressure_Pa,
                specific_gravity=self.specific_gravity,
                composition=composition
            )
            
            # Calculate molar volume using ideal gas approximation
            R = 8.314
            Tc, Pc = self.eos.get_pseudocritical_properties(
                specific_gravity=self.specific_gravity
            )
            molar_volume = z_factor * R * temperature_K / pressure_Pa
            
            # Determine phase
            if z_factor > 0.95:
                phase_flag = "liquid" if molar_volume < 1.0 else "vapor"
            elif z_factor > 0.85:
                phase_flag = "mix"
            else:
                phase_flag = "vapor"
            
            return EOSPropertyResult(
                temperature_K=temperature_K,
                pressure_Pa=pressure_Pa,
                volume_m3_mol=molar_volume,
                z_factor=z_factor,
                phase_flag=phase_flag,
                total_moles=1.0
            )
            
        except Exception as e:
            raise NumericalError(f"Property calculation failed: {str(e)}")
    
    def calculate_ptv_relationship(
        self,
        temperature_K: float,
        pressure_range: List[float],
        step_size: float = 1e5,
        composition: Optional[Dict[str, float]] = None
    ) -> List[EOSPropertyResult]:
        """
        Calculate PTV relationship along a pressure traverse.
        
        Generates calculation results at multiple pressure points
        for a single temperature.
        
        Parameters
        ----------
        temperature_K : float
            Temperature in Kelvin.
        pressure_range : List[float]
            List of pressures in Pascals to evaluate.
        step_size : float, optional
            Sampling step if generating values.
        composition : Optional[Dict[str, float]], optional
            Fluid composition.
        
        Returns
        -------
        List[EOSPropertyResult]
            Generated results for all pressure points.
        
        Raises
        ------
        NumericalError
            If calculation fails at any point.
        """
        results = []
        
        if step_size is not None and len(pressure_range) == 1:
            # Generate pressure traverse
            pressures = np.arange(pressure_range[0], pressure_range[1] + step_size, step_size)
        else:
            pressures = pressure_range
        
        for pressure in pressures:
            try:
                result = self.calculate_property_at_conditions(
                    temperature_K, pressure, composition
                )
                results.append(result)
            except Exception as e:
                raise NumericalError(
                    f"PTV calculation failed at P={pressure}: {str(e)}"
                )
        
        return results
    
    def calculate_pvr_relationship(
        self,
        pressure_Pa: float,
        temperature_range: List[float],
        composition: Optional[Dict[str, float]] = None
    ) -> List[EOSPropertyResult]:
        """
        Calculate PVR relationship (Pressure-Volume-Temperature).
        
        Evaluates PTV properties across a temperature range at fixed pressure.
        
        Parameters
        ----------
        pressure_Pa : float
            Pressure in Pascals (fixed).
        temperature_range : List[float]
            List of temperatures in Kelvin to evaluate.
        composition : Optional[Dict[str, float]], optional
            Fluid composition.
        
        Returns
        -------
        List[EOSPropertyResult]
            Generated results for all temperature points.
        
        Raises
        ------
        NumericalError
            If calculation fails at any point.
        """
        results = []
        
        for temperature in temperature_range:
            try:
                result = self.calculate_property_at_conditions(
                    temperature, pressure_Pa, composition
                )
                results.append(result)
            except Exception as e:
                raise NumericalError(
                    f"PVR calculation failed at T={temperature}: {str(e)}"
                )
        
        return results
    
    def calculate_tvp_relationship(
        self,
        temperature_K: float,
        pressure_range: List[float],
        composition: Optional[Dict[str, float]] = None
    ) -> List[EOSPropertyResult]:
        """
        Calculate TVP relationship (Temperature-Volume-Pressure).
        
        Evaluates PTV properties across pressure and temperature ranges.
        
        Parameters
        ----------
        temperature_K : float
            Temperature in Kelvin.
        pressure_range : List[float]
            List of pressures in Pascals to evaluate.
        composition : Optional[Dict[str, float]], optional
            Fluid composition.
        
        Returns
        -------
        List[EOSPropertyResult]
            Generated results.
        
        Raises
        ------
        NumericalError
            If calculation fails at any point.
        """
        return self.calculate_ptv_relationship(
            temperature_K, pressure_range, composition=composition
        )
    
    def compare_eos_models(
        self,
        temperature_K: float,
        pressure_Pa: float,
        composition: Optional[Dict[str, float]] = None
    ) -> Dict[str, EOSPropertyResult]:
        """
        Compare SRK and Peng-Robinson EOS models.
        
        Parameters
        ----------
        temperature_K : float
            Temperature in Kelvin.
        pressure_Pa : float
            Pressure in Pascals.
        composition : Optional[Dict[str, float]], optional
            Fluid composition.
        
        Returns
        -------
        Dict[str, EOSPropertyResult]
            Dictionary with 'srk' and 'pr' results.
        """
        srk_calculator = EOSPropertyCalculator(
            eos_type="srk",
            specific_gravity=self.specific_gravity
        )
        srk_result = srk_calculator.calculate_property_at_conditions(
            temperature_K, pressure_Pa, composition
        )
        
        pr_calculator = EOSPropertyCalculator(
            eos_type="pr",
            specific_gravity=self.specific_gravity
        )
        pr_result = pr_calculator.calculate_property_at_conditions(
            temperature_K, pressure_Pa, composition
        )
        
        return {'srk': srk_result, 'pr': pr_result}
    
    def calculate_phase_composition(
        self,
        temperature_K: float,
        pressure_Pa: float,
        composition: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate phase compositions using VLE flash.
        
        Parameters
        ----------
        temperature_K : float
            Temperature in Kelvin.
        pressure_Pa : float
            Pressure in Pascals.
        composition : Dict[str, float]
            Overall composition.
        
        Returns
        -------
        Tuple[Dict[str, float], Dict[str, float]]
            (liquid_composition, vapor_composition).
        
        Raises
        ------
        NumericalError
            If VLE calculation fails.
        """
        from perf_pressure_traverse.math.vle import VLEFlash
        
        flash = VLEFlash(eos_type="srk")
        return flash.perform_flash(temperature_K, pressure_Pa, composition)
