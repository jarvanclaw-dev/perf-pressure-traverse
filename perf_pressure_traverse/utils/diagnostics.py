"""Solver diagnostics utility class."""

from dataclasses import dataclass
from typing import List


@dataclass
class SolverError:
    """Represents a diagnostic error message."""
    message: str
    timestamp: float
    category: str = "error"


class SolverDiagnostics:
    """Diagnostic logger for pressure traverse solver."""
    
    def __init__(self) -> None:
        """Initialize the diagnostics logger."""
        self.iterations = 0
        self.errors: List[SolverError] = []
    
    def log_error(self, message: str) -> None:
        """Log an error message."""
        self.errors.append(SolverError(message, 0.0))
