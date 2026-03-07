"""
Custom exception hierarchy for perf-pressure-traverse library.

This module defines a comprehensive set of exceptions for parameter validation,
unit conversion errors, and other error conditions in pressure traverse calculations.

Exceptions are organized hierarchically with common base classes and specialized
exceptions for different error scenarios.
"""


class PressureTraverseError(Exception):
    """
    Base exception for all exceptions in the perf-pressure-traverse library.
    
    All custom exceptions in the library inherit from this class, providing a
    consistent error handling framework throughout the codebase.
    
    Example:
        >>> raise PressureTraverseError("Operation failed")
    """
    
    def __init__(self, message: str):
        """
        Initialize the exception with a descriptive message.
        
        Parameters
        ----------
        message : str
            Descriptive error message.
        """
        super().__init__(message)


class ValidationError(PressureTraverseError):
    """
    Raised when input parameters violate physical constraints or business rules.
    
    This exception is raised when user-supplied inputs fail validation checks,
    such as out-of-range values, invalid units, or inconsistent parameters.
    
    Example:
        >>> raise ValidationError("Surface pressure must be positive")
    """
    
    def __init__(self, message: str, error_code: str = "VALIDATION_ERROR"):
        """
        Initialize the validation error.
        
        Parameters
        ----------
        message : str
            Descriptive error message.
        error_code : str, optional
            Machine-readable error code for programmatic handling.
        """
        self.error_code = error_code
        super().__init__(message)
    
    def __str__(self):
        """Return error code and message in a standardized format."""
        return f"[{self.error_code}] {self.args[0]}"


class UnitConversionError(PressureTraverseError):
    """
    Raised when unit conversion fails or produces invalid results.
    
    This exception is raised when attempting to convert between incompatible
    units or when conversion results are outside valid physical ranges.
    
    Example:
        >>> raise UnitConversionError("Invalid flow rate to convert")
    """
    
    def __init__(self, message: str, error_code: str = "UNIT_CONVERSION_ERROR"):
        """
        Initialize the unit conversion error.
        
        Parameters
        ----------
        message : str
            Descriptive error message.
        error_code : str, optional
            Machine-readable error code.
        """
        self.error_code = error_code
        super().__init__(message)
    
    def __str__(self):
        """Return error code and message in a standardized format."""
        return f"[{self.error_code}] {self.args[0]}"


class InputValidationError(ValidationError):
    """
    Raised when input parameters fail basic validation checks.
    
    This is a specialized validation error for common input validation issues
    such as type mismatches, null values, or basic range violations.
    
    Example:
        >>> raise InputValidationError("Pressure value cannot be negative")
    """
    
    def __init__(self, message: str, parameter_name: str = None):
        """
        Initialize the input validation error.
        
        Parameters
        ----------
        message : str
            Descriptive error message.
        parameter_name : str, optional
            Name of the parameter that failed validation.
        """
        self.parameter_name = parameter_name
        full_message = message if parameter_name is None else f"{parameter_name}: {message}"
        super().__init__(full_message)
    
    def __str__(self):
        """Include parameter name in error string if available."""
        if self.parameter_name:
            return f"[INPUT_VALIDATION - {self.parameter_name}] {self.args[0]}"
        return super().__str__()


class PhysicalBoundsError(ValidationError):
    """
    Raised when input values violate physical bounds.
    
    This exception is raised when values are outside physically possible ranges
    such as temperature extremes or pressure limits.
    
    Example:
        >>> raise PhysicalBoundsError("Temperature exceeds maximum operating limit")
    """
    
    def __init__(self, message: str, parameter_name: str = None, value: float = None):
        """
        Initialize the physical bounds error.
        
        Parameters
        ----------
        message : str
            Descriptive error message.
        parameter_name : str, optional
            Name of the parameter that violated bounds.
        value : float, optional
            Value that caused the error.
        """
        self.parameter_name = parameter_name
        self.value = value
        full_message = ""
        
        if parameter_name is not None and value is not None:
            full_message = f"{parameter_name} = {value}: {message}"
        elif parameter_name is not None:
            full_message = f"{parameter_name}: {message}"
        else:
            full_message = message
        
        super().__init__(full_message)
    
    def __str__(self):
        """Include parameter and value in error string if available."""
        if self.parameter_name and self.value:
            return f"[PHYSICAL_BOUNDS - {self.parameter_name}={self.value}] {self.args[0]}"
        return super().__str__()


class ConvergenceError(PressureTraverseError):
    """
    Raised when iterative solver fails to converge.
    
    This exception is raised when Newton-Raphson or other iterative methods
    fail to reach convergence within the maximum iteration limit.
    
    Example:
        >>> raise ConvergenceError("Pressure solution failed to converge")
    """
    
    def __init__(self, message: str, max_iterations: int = 50, iterations_used: int = 0):
        """
        Initialize the convergence error.
        
        Parameters
        ----------
        message : str
            Descriptive error message.
        max_iterations : int, optional
            Maximum allowed iterations.
        iterations_used : int, optional
            Number of iterations actually performed.
        """
        self.max_iterations = max_iterations
        self.iterations_used = iterations_used
        super().__init__(message)
    
    def __str__(self):
        """Include iteration details in error string."""
        if self.max_iterations and self.iterations_used:
            return f"{self.args[0]} (used {self.iterations_used}/{self.max_iterations} iterations)"
        return super().__str__()


class NumericalStabilityError(PressureTraverseError):
    """
    Raised when calculations lead to numerical instability.
    
    This exception is raised when calculations produce NaN, inf, or other
    numerical artifacts.
    
    Example:
        >>> raise NumericalStabilityError("Division by zero occurred")
    """
    
    def __init__(self, message: str, error_code: str = "NUMERICAL_STABILITY_ERROR"):
        """
        Initialize the numerical stability error.
        
        Parameters
        ----------
        message : str
            Descriptive error message.
        error_code : str, optional
            Machine-readable error code.
        """
        self.error_code = error_code
        super().__init__(message)


class CorrelationError(PressureTraverseError):
    """
    Raised when requested correlation is not supported or not applicable.
    
    This exception is raised when attempting to use an unavailable correlation
    or when a correlation fails due to domain issues.
    
    Example:
        >>> raise CorrelationError("Correlation not supported: unknown_correlation")
    """
    
    def __init__(self, message: str, correlation_name: str = None):
        """
        Initialize the correlation error.
        
        Parameters
        ----------
        message : str
            Descriptive error message.
        correlation_name : str, optional
            Name of the correlation that caused the error.
        """
        self.correlation_name = correlation_name
        full_message = message if correlation_name is None else f"Correlation '{correlation_name}': {message}"
        super().__init__(full_message)


class PVTModelError(PressureTraverseError):
    """
    Raised when PVT model calculation fails.
    
    This exception is raised when fluid property calculations produce invalid
    results or fail due to physical constraints.
    
    Example:
        >>> raise PVTModelError("Failed to calculate formation volume factor")
    """
    
    def __init__(self, message: str, property_type: str = None):
        """
        Initialize the PVT model error.
        
        Parameters
        ----------
        message : str
            Descriptive error message.
        property_type : str, optional
            Type of PVT property that failed.
        """
        self.property_type = property_type
        full_message = message if property_type is None else f"{property_type}: {message}"
        super().__init__(full_message)


class DimensionError(ValidationError):
    """
    Raised when physical dimensions are invalid or non-physical.
    
    This exception is raised when wellbore dimensions or geometric parameters
    violate physical constraints.
    
    Example:
        >>> raise DimensionError("Well diameter cannot be zero")
    """
    
    def __init__(self, message: str, dimension_type: str = None):
        """
        Initialize the dimension error.
        
        Parameters
        ----------
        message : str
            Descriptive error message.
        dimension_type : str, optional
            Type of dimension that violated constraints.
        """
        self.dimension_type = dimension_type
        full_message = message if dimension_type is None else f"{dimension_type}: {message}"
        super().__init__(full_message)
