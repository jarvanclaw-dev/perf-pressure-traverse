"""
Comprehensive parameter validation for pressure traverse calculations.

This module provides validation functions for input parameters used in pressure
traverse calculations, including pressures, temperatures, depths, flow rates,
and other domain model inputs.

The validators enforce physical constraints, numeric range checks, and
type safety to prevent invalid calculations.
"""


from typing import Union, Optional, List, Tuple
import numpy as np

from perf_pressure_traverse.utils.exceptions import (
    ValidationError,
    InputValidationError,
    PhysicalBoundsError,
    UnitConversionError,
    NumericalStabilityError,
    DimensionError,
    ConvergenceError,
    CorrelationError,
    PVTModelError,
)


class ParameterValidator:
    """
    Main validator class for pressure traverse parameters.
    
    This class provides a comprehensive validator with methods for each
    type of parameter validation, including fluent API support for chained
    validation calls.
    
    Example:
        >>> validator = ParameterValidator()
        >>> try:
        ...     validator.validate_pressure(pressure_psia=1000.0)
        ... except ValidationError as e:
        ...     print(e)
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize the validator.
        
        Parameters
        ----------
        strict_mode : bool, optional
            If True, raise exceptions immediately on invalid input.
            If False, collect all errors and report at the end.
            Default: True
        """
        self.strict_mode = strict_mode
        self.errors: List[ValidationError] = []
    
    def validate_all(
        self,
        fluid_properties: 'FluidProperties',
        well_geometry: 'WellGeometry',
        pvt_properties: Optional['PVTProperties'] = None,
        flow_rates: Optional[Tuple[float, float]] = None,
    ) -> bool:
        """
        Validate a complete set of pressure traverse inputs.
        
        Parameters
        ----------
        fluid_properties : FluidProperties
            Fluid property model.
        well_geometry : WellGeometry
            Well geometry model.
        pvt_properties : PVTProperties, optional
            PVT property model.
        flow_rates : Tuple[float, float], optional
            Gas and liquid flow rates (scf/d, bbl/d).
        
        Returns
        -------
        bool
            True if all validations pass, False otherwise.
        
        Raises
        ------
        ValidationError
            If strict_mode is True and any validation fails.
        """
        self.clear_errors()
        
        # Validate fluid properties
        self.validate_fluid_properties(fluid_properties, flow_rates)
        
        # Validate well geometry
        self.validate_well_geometry(well_geometry)
        
        # Validate PVT properties if provided
        if pvt_properties is not None:
            self.validate_pvt_properties(pvt_properties)
        
        # Validate surface conditions
        self.validate_surface_conditions(fluid_properties)
        
        # Validate flow rates if provided
        if flow_rates is not None:
            self.validate_flow_rates(*flow_rates)
        
        return not self.has_errors()
    
    # Fluid Properties Validation
    
    def validate_fluid_properties(
        self,
        fluid_properties: 'FluidProperties',
        flow_rates: Optional[Tuple[float, float]] = None,
    ) -> bool:
        """
        Validate fluid properties.
        
        Parameters
        ----------
        fluid_properties : FluidProperties
            Fluid property model.
        flow_rates : Tuple[float, float], optional
            Gas and liquid flow rates for gas-liquid ratio validation.
        
        Returns
        -------
        bool
            True if validation passes.
        
        Raises
        ------
        ValidationError
            If validation fails and strict_mode is True.
        """
        try:
            # Validate specific gravities
            self._validate_specific_gravity(
                fluid_properties.oil_specific_gravity,
                "oil_specific_gravity",
                min_value=0.5,
                max_value=1.1,
                default_msg="Oil specific gravity must be between 0.5 and 1.1"
            )
            
            self._validate_specific_gravity(
                fluid_properties.gas_specific_gravity,
                "gas_specific_gravity",
                min_value=0.30,
                max_value=0.80,
                default_msg="Gas specific gravity must be between 0.30 and 0.80"
            )
            
            self._validate_specific_gravity(
                fluid_properties.water_specific_gravity,
                "water_specific_gravity",
                min_value=0.90,
                max_value=1.08,
                default_msg="Water specific gravity must be between 0.90 and 1.08"
            )
            
            # Validate pressure
            self._validate_positive(
                fluid_properties.surface_pressure_psia,
                "surface_pressure_psia",
                min_value=14.7,
                default_msg="Surface pressure must be greater than atmospheric pressure"
            )
            
            # Validate temperature
            self._validate_positive(
                fluid_properties.surface_temperature_f,
                "surface_temperature_f",
                min_value=20,
                default_msg="Surface temperature must be above freezing point"
            )
            
            # Validate gas-oil ratios
            self._validate_positive(
                fluid_properties.gas_oil_ratio,
                "gas_oil_ratio",
                min_value=0,
                default_msg="Gas-oil ratio cannot be negative"
            )
            
            self._validate_positive(
                fluid_properties.solution_gas_ratio,
                "solution_gas_ratio",
                min_value=0,
                default_msg="Solution gas ratio cannot be negative"
            )
            
            self._validate_positive(
                fluid_properties.water_cut,
                "water_cut",
                min_value=0.0,
                max_value=1.0,
                default_msg="Water cut must be between 0 and 1"
            )
            
            # Validate gas-liquid ratio from flow rates if provided
            if flow_rates is not None and len(flow_rates) >= 2:
                gas_rate, liquid_rate = flow_rates
                if gas_rate > 0 and liquid_rate > 0:
                    gas_liquid_ratio = gas_rate / liquid_rate
                    self._validate_gor(gas_liquid_ratio)
            
            return True
            
        except ValidationError as e:
            self.errors.append(e)
            if self.strict_mode:
                raise
            return False
    
    def validate_pvt_properties(self, pvt_properties: 'PVTProperties') -> bool:
        """
        Validate PVT properties at reservoir conditions.
        
        Parameters
        ----------
        pvt_properties : PVTProperties
            PVT property model.
        
        Returns
        -------
        bool
            True if validation passes.
        
        Raises
        ------
        ValidationError
            If validation fails and strict_mode is True.
        """
        try:
            self._validate_positive(
                pvt_properties.oil_density_lbm_per_ft3,
                "oil_density_lbm_per_ft3",
                min_value=20,
                default_msg="Oil density must be positive"
            )
            
            self._validate_positive(
                pvt_properties.gas_density_lbm_per_ft3,
                "gas_density_lbm_per_ft3",
                min_value=0.01,
                default_msg="Gas density must be positive"
            )
            
            self._validate_positive(
                pvt_properties.water_density_lbm_per_ft3,
                "water_density_lbm_per_ft3",
                min_value=50,
                default_msg="Water density must be positive"
            )
            
            self._validate_positive(
                pvt_properties.gas_compressibility_factor,
                "gas_compressibility_factor",
                min_value=0.0,
                max_value=1.2,
                default_msg="Gas compressibility factor must be between 0 and 1.2"
            )
            
            return True
            
        except ValidationError as e:
            self.errors.append(e)
            if self.strict_mode:
                raise
            return False
    
    # Well Geometry Validation
    
    def validate_well_geometry(self, well_geometry: 'WellGeometry') -> bool:
        """
        Validate well geometry parameters.
        
        Parameters
        ----------
        well_geometry : WellGeometry
            Well geometry model.
        
        Returns
        -------
        bool
            True if validation passes.
        
        Raises
        ------
        ValidationError
            If validation fails and strict_mode is True.
        """
        try:
            # Validate borehole diameter
            self._validate_positive(
                well_geometry.borehole_diameter_ft,
                "borehole_diameter_ft",
                min_value=0.1,
                max_value=20.0,
                default_msg="Borehole diameter must be between 0.1 and 20 feet"
            )
            
            # Validate casing diameter
            self._validate_positive(
                well_geometry.casing_diameter_ft,
                "casing_diameter_ft",
                min_value=0.1,
                max_value=30.0,
                default_msg="Casing diameter must be positive"
            )
            
            # Validate deviation angles array if provided
            if hasattr(well_geometry, 'deviation_angles'):
                self._validate_deviation_angles(well_geometry.deviation_angles)
            
            return True
            
        except ValidationError as e:
            self.errors.append(e)
            if self.strict_mode:
                raise
            return False
    
    # Core Parameter Validation
    
    def validate_pressure(self, pressure: float, unit: str = "psi") -> bool:
        """
        Validate pressure value.
        
        Parameters
        ----------
        pressure : float
            Pressure value in specified unit.
        unit : str, optional
            Unit for pressure (psi, kPa, atm, MPa). Default: "psi".
        
        Returns
        -------
        bool
            True if validation passes.
        
        Raises
        ------
        ValidationError
            If validation fails and strict_mode is True.
        """
        try:
            # Convert to psi for validation
            pressure_psi = self._convert_pressure(pressure, unit)
            
            if pressure_psi <= 0:
                raise InputValidationError(
                    f"Pressure must be positive, got {pressure} {unit}",
                    parameter_name="pressure"
                )
            
            if pressure_psi < 14.7:
                raise InputValidationError(
                    "Pressure must be at least atmospheric pressure (14.7 psi)",
                    parameter_name="pressure"
                )
            
            return True
            
        except ValidationError as e:
            self.errors.append(e)
            if self.strict_mode:
                raise
            return False
    
    def validate_temperature(self, temperature: float, unit: str = "F") -> bool:
        """
        Validate temperature value.
        
        Parameters
        ----------
        temperature : float
            Temperature value in specified unit.
        unit : str, optional
            Unit for temperature (F, C, K). Default: "F".
        
        Returns
        -------
        bool
            True if validation passes.
        
        Raises
        ------
        ValidationError
            If validation fails and strict_mode is True.
        """
        try:
            # Convert to Fahrenheit for validation
            temp_f = self._convert_temperature(temperature, unit)
            
            if temp_f < 0:
                raise InputValidationError(
                    f"Temperature cannot be below 0°F, got {temperature} {unit}",
                    parameter_name="temperature"
                )
            
            if temp_f > 300:
                raise PhysicalBoundsError(
                    f"Temperature exceeds typical maximum of 300°F, got {temperature} {unit}",
                    parameter_name="temperature",
                    value=temperature
                )
            
            return True
            
        except ValidationError as e:
            self.errors.append(e)
            if self.strict_mode:
                raise
            return False
    
    def validate_depth(self, depth: Union[float, np.ndarray], unit: str = "ft") -> bool:
        """
        Validate depth value or array.
        
        Parameters
        ----------
        depth : float or ndarray
            Depth value(s) in specified unit.
        unit : str, optional
            Unit for depth (ft, m, km). Default: "ft".
        
        Returns
        -------
        bool
            True if validation passes.
        
        Raises
        ------
        ValidationError
            If validation fails and strict_mode is True.
        """
        try:
            # Convert to feet for validation (handle both scalar and array)
            if isinstance(depth, np.ndarray):
                depths_ft = np.array([self._convert_distance(d, unit) for d in depth])
                is_scalar = False
            else:
                depths_ft = np.array([self._convert_distance(depth, unit)])
                is_scalar = True
            
            # Validate each value
            min_valid = -1000  # Allow negative depths for geological depth calculations
            
            for i, d in enumerate(depths_ft):
                if d <= min_valid:
                    raise InputValidationError(
                        f"Depth at index {i} must be greater than {abs(min_valid)}, got {d} ft",
                        parameter_name="depth"
                    )
            
            # Check for NaN or inf
            if np.any(np.isnan(depths_ft)) or np.any(np.isinf(depths_ft)):
                raise NumericalStabilityError(
                    "Depth contains NaN or infinite values"
                )
            
            # Validate max depth for wellbore calculations
            max_depth_ft = float(np.max(depths_ft))  # Ensure scalar
            if max_depth_ft > 40000:
                message = f"Maximum depth {max_depth_ft:.2f} ft exceeds typical wellbore depth"
                if is_scalar:
                    raise PhysicalBoundsError(
                        message,
                        parameter_name="max_depth",
                        value=max_depth_ft
                    )
                else:
                    raise PhysicalBoundsError(message)
            
            return True
            
        except ValidationError as e:
            self.errors.append(e)
            if self.strict_mode:
                raise
            return False
    
    def validate_flow_rates(
        self,
        gas_rate: float,
        liquid_rate: float,
        units: str = "standard"
    ) -> bool:
        """
        Validate gas and liquid flow rates.
        
        Parameters
        ----------
        gas_rate : float
            Gas flow rate.
        liquid_rate : float
            Liquid flow rate.
        units : str, optional
            Units ("standard", "actual"). Default: "standard".
        
        Returns
        -------
        bool
            True if validation passes.
        
        Raises
        ------
        ValidationError
            If validation fails and strict_mode is True.
        """
        try:
            self._validate_positive(
                gas_rate,
                "gas_rate",
                min_value=0,
                default_msg="Gas flow rate cannot be negative"
            )
            
            self._validate_positive(
                liquid_rate,
                "liquid_rate",
                min_value=0,
                default_msg="Liquid flow rate cannot be negative"
            )
            
            if gas_rate == 0 and liquid_rate == 0:
                raise InputValidationError(
                    "At least one flow rate must be non-zero"
                )
            
            # Validate gas-oil ratio
            if liquid_rate > 0:
                gor = self._convert_rate(gas_rate, units)
                oil_rate = self._convert_rate(liquid_rate, units)
                self._validate_gor(oil_rate / gas_rate if gas_rate > 0 else 0)
            
            return True
            
        except ValidationError as e:
            self.errors.append(e)
            if self.strict_mode:
                raise
            return False
    
    def validate_surface_conditions(
        self,
        fluid_properties: 'FluidProperties'
    ) -> bool:
        """
        Validate surface pressure and temperature parameters.
        
        Parameters
        ----------
        fluid_properties : FluidProperties
            Fluid property model with surface conditions.
        
        Returns
        -------
        bool
            True if validation passes.
        
        Raises
        ------
        ValidationError
            If validation fails and strict_mode is True.
        """
        try:
            # Surface pressure check
            if not (fluid_properties.surface_pressure_psia >= 14.7):
                raise PhysicalBoundsError(
                    f"Surface pressure must be >= atmospheric pressure (14.7 psi), got {fluid_properties.surface_pressure_psia} psi"
                )
            
            # Surface temperature check
            if not (fluid_properties.surface_temperature_f >= 70 and fluid_properties.surface_temperature_f <= 120):
                raise InputValidationError(
                    f"Surface temperature should be realistic for field operations (70-120°F), got {fluid_properties.surface_temperature_f}°F"
                )
            
            return True
            
        except ValidationError as e:
            self.errors.append(e)
            if self.strict_mode:
                raise
            return False
    
    # Validation Result Helpers
    
    def has_errors(self) -> bool:
        """Return True if any validation errors exist."""
        return len(self.errors) > 0
    
    def get_errors(self) -> List[ValidationError]:
        """Get list of all validation errors."""
        return self.errors.copy()
    
    def get_error_summary(self) -> dict:
        """Get summary of all validation errors."""
        error_dict = {"total_errors": len(self.errors)}
        
        by_type = {}
        for error in self.errors:
            error_class = error.__class__.__name__
            if error_class not in by_type:
                by_type[error_class] = []
            by_type[error_class].append(str(error))
        
        error_dict["by_type"] = by_type
        return error_dict
    
    def clear_errors(self) -> None:
        """Clear all collected errors."""
        self.errors.clear()
    
    # Internal Validation Methods
    
    def _validate_positive(
        self,
        value: float,
        parameter_name: str,
        min_value: float = 0,
        max_value: Optional[float] = None,
        default_msg: str = "Value must be positive"
    ) -> bool:
        """
        Validate a positive value within optional range.
        
        Parameters
        ----------
        value : float
            Value to validate.
        parameter_name : str
            Name of the parameter being validated.
        min_value : float, optional
            Minimum allowed value. Default: 0.
        max_value : float, optional
            Maximum allowed value. If None, no upper bound.
        default_msg : str
            Default error message.
        
        Returns
        -------
        bool
            True if validation passes.
        
        Raises
        ------
        InputValidationError
            If value is not positive or out of range.
        """
        if value <= min_value:
            msg = f"{default_msg}, got {value}"
            if max_value is not None:
                msg = f"{default_msg} (valid range: {min_value} - {max_value}), got {value}"
            raise InputValidationError(msg, parameter_name=parameter_name)
        
        if max_value is not None and value > max_value:
            raise InputValidationError(
                f"Value exceeds maximum of {max_value}, got {value}",
                parameter_name=parameter_name
            )
        
        return True
    
    def _validate_specific_gravity(
        self,
        value: float,
        parameter_name: str,
        min_value: float,
        max_value: float,
        default_msg: str
    ) -> bool:
        """Validate specific gravity values."""
        if not (min_value <= value <= max_value):
            raise PhysicalBoundsError(
                f"{default_msg} (valid range: {min_value} - {max_value}), got {value}",
                parameter_name=parameter_name,
                value=value
            )
        return True
    
    def _validate_gor(self, gas_liquid_ratio: float) -> bool:
        """Validate gas-liquid ratio."""
        if gas_liquid_ratio > 10000:
            raise PhysicalBoundsError(
                f"Gas-liquid ratio is very high ({gas_liquid_ratio:.0f}), may be gas dominant well",
                parameter_name="gas_liquid_ratio",
                value=gas_liquid_ratio
            )
        return True
    
    def _validate_deviation_angles(self, angles: np.ndarray) -> bool:
        """Validate deviation angle array."""
        if len(angles) == 0:
            raise InputValidationError("Deviation angles array cannot be empty")
        
        if np.any(np.isnan(angles)) or np.any(np.isinf(angles)):
            raise NumericalStabilityError("Deviation angles contain NaN or infinite values")
        
        # Verify angles are in reasonable range (-180 to 180)
        invalid_angles = angles[np.abs(angles) > 180]
        if len(invalid_angles) > 0:
            raise PhysicalBoundsError(
                f"Deviation angles must be between -180 and 180 degrees, "
                f"found {invalid_angles[0]:.2f} degrees",
                parameter_name="deviation_angles"
            )
        return True
    
    # Unit Conversion Internal Methods
    
    def _convert_pressure(self, pressure: float, unit: str) -> float:
        """Convert pressure to psi."""
        unit = unit.lower()
        if unit == "psi":
            return pressure
        elif unit == "kpa":
            return pressure * 0.145038
        elif unit == "atm":
            return pressure * 14.6959
        elif unit == "mpa":
            return pressure * 145.038
        else:
            raise UnitConversionError(f"Unsupported pressure unit: {unit}")
    
    def _convert_temperature(self, temp: float, unit: str) -> float:
        """Convert temperature to Fahrenheit."""
        unit = unit.upper()
        if unit == "F":
            return temp
        elif unit == "C":
            return (temp * 9/5) + 32
        elif unit == "K":
            return (temp - 273.15) * 9/5 + 32
        else:
            raise UnitConversionError(f"Unsupported temperature unit: {unit}")
    
    def _convert_distance(self, distance: float, unit: str) -> float:
        """Convert distance to feet."""
        unit = unit.lower()
        if unit == "ft":
            return distance
        elif unit == "m" or unit == "meters":
            return distance * 3.28084
        elif unit == "km" or unit == "kilometers":
            return distance * 3280.84
        else:
            raise UnitConversionError(f"Unsupported distance unit: {unit}")
    
    def _convert_rate(self, rate: float, units: str) -> float:
        """Convert flow rate to appropriate units."""
        if units.lower() == "standard":
            return rate
        else:
            raise UnitConversionError(f"Unsupported rate units: {units}")
