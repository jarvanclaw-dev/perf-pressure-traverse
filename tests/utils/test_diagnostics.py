"""Tests for SolverDiagnostics utility class."""
import pytest

from perf_pressure_traverse.utils.diagnostics import SolverDiagnostics, SolverError


class TestSolverDiagnostics:
    """Test cases for SolverDiagnostics class."""
    
    def test_diagnostics_logger_init(self):
        """Test diagnostics logger initialization."""
        diagnostics = SolverDiagnostics()
        
        assert diagnostics.iterations == 0
        assert len(diagnostics.errors) == 0
    
    def test_diagnostics_logger_error(self):
        """Test error logging."""
        diagnostics = SolverDiagnostics()
        diagnostics.log_error("Test error message")
        
        assert len(diagnostics.errors) == 1
        assert diagnostics.errors[0].message == "Test error message"
        assert isinstance(diagnostics.errors[0].timestamp, float)
        assert diagnostics.errors[0].category == "error"
    
    def test_diagnostics_logger_warning(self):
        """Test warning logging."""
        diagnostics = SolverDiagnostics()
        diagnostics.log_warning("Test warning message")
        
        assert len(diagnostics.errors) == 1
        assert diagnostics.errors[0].category == "warning"
    
    def test_diagnostics_multiple_errors(self):
        """Test logging multiple errors."""
        diagnostics = SolverDiagnostics()
        
        for i in range(5):
            diagnostics.log_error(f"Error {i}")
        
        assert len(diagnostics.errors) == 5
    
    def test_diagnostics_reset(self):
        """Test diagnostics reset functionality."""
        diagnostics = SolverDiagnostics()
        
        # Log some errors
        diagnostics.log_error("Error 1")
        diagnostics.log_error("Error 2")
        
        assert len(diagnostics.errors) == 2
        
        # Reset
        diagnostics.reset()
        
        assert diagnostics.iterations == 0
        assert len(diagnostics.errors) == 0


class TestSolverError:
    """Test cases for SolverError dataclass."""
    
    def test_solver_error_init(self):
        """Test SolverError initialization."""
        error = SolverError(
            message="Test error",
            timestamp=1234567890.0
        )
        
        assert error.message == "Test error"
        assert error.timestamp == 1234567890.0
        assert error.category == "error"
    
    def test_solver_error_custom_category(self):
        """Test SolverError with custom category."""
        error = SolverError(
            message="Warning message",
            timestamp=1234567890.0,
            category="warning"
        )
        
        assert error.category == "warning"
    
    def test_solver_error_summary(self):
        """Test SolverError provides formatted summary."""
        error = SolverError(
            message="Connection timeout",
            timestamp=1234567890.0
        )
        
        summary = error.__str__()
        assert "Connection timeout" in summary
        assert "Error" in summary or "Timestamp" in summary
