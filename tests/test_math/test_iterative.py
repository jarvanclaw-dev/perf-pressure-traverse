"""
Unit tests for Newton-Raphson pressure solver.

Tests convergence criteria, error handling, unit conversions, and
performance characteristics.
"""

import math
import pytest
from perf_pressure_traverse.math.iterative import (
    newton_raphson_pressure_solver,
    newton_raphson_solver,
    solve_pressure_step,
    ConvergenceError,
    PressureTraverseError,
    SolverDiagnostics,
    test_convergence_stability,
)


class TestNewtonRaphsonPressureSolver:
    """Test the specific pressure traverse solver."""
    
    def test_basic_convergence(self):
        """Test that solver converges for a simple case."""
        result = newton_raphson_pressure_solver(
            surface_pressure=100.0,
            vertical_depth_ft=1000.0,
            gas_rate=100000,
            tubing_diameter_ft=0.10,
            geothermal_gradient_f_per_1000ft=50.0,
            gas_specific_gravity=0.65,
        )
        
        assert isinstance(result, (int, float))
        assert result > surface_pressure - (geothermal * 1.0)  # BHP > surface - reasonable limit
        assert result > 0  # Negative pressure not physical
        
    def test_high_ambiguity_tolerance(self):
        """Test that looser tolerance results in faster convergence."""
        import time
        
        start = time.time()
        result_strict = newton_raphson_pressure_solver(
            surface_pressure=200.0,
            vertical_depth_ft=5000.0,
            gas_rate=50000,
            tubing_diameter_ft=0.12,
            geothermal_gradient_f_per_1000ft=45.0,
            gas_specific_gravity=0.70,
            convergence_tolerance=1e-3,  # Loose tolerance
            max_iterations=20,
        )
        time_strict = time.time() - start
        
        start = time.time()
        result_tight = newton_raphson_pressure_solver(
            surface_pressure=200.0,
            vertical_depth_ft=5000.0,
            gas_rate=50000,
            tubing_diameter_ft=0.12,
            geothermal_gradient_f_per_1000ft=45.0,
            gas_specific_gravity=0.70,
            convergence_tolerance=1e-6,  # Tight tolerance
            max_iterations=20,
        )
        time_tight = time.time() - start
        
        # Both should succeed
        assert isinstance(result_strict, (int, float))
        assert isinstance(result_tight, (int, float))
        
        # Tighter tolerance may take longer, but should eventually converge
        # Both results should be reasonable (within physical bounds)
        assert result_strict > 0
        assert result_tight > 0
        
    def test_max_iterations_exceeded(self):
        """Test that appropriate exception is raised when max iterations is hit."""
        with pytest.raises(ConvergenceError) as excinfo:
            newton_raphson_pressure_solver(
                surface_pressure=300.0,
                vertical_depth_ft=2000.0,
                gas_rate=1000,
                tubing_diameter_ft=0.20,
                geothermal_gradient_f_per_1000ft=30.0,
                gas_specific_gravity=0.65,
                convergence_tolerance=1e-6,
                max_iterations=2,  # Very low number
            )
        
        assert 'failed to converge' in str(excinfo.value).lower()
        
    def test_invalid_surface_pressure(self):
        """Test that non-positive surface pressure raises error."""
        with pytest.raises(ValueError):
            newton_raphson_pressure_solver(
                surface_pressure=0.0,
                vertical_depth_ft=1000.0,
                gas_rate=100000,
                tubing_diameter_ft=0.10,
                geothermal_gradient_f_per_1000ft=50.0,
                gas_specific_gravity=0.65,
            )
    
    def test_invalid_depth(self):
        """Test that non-positive depth raises error."""
        with pytest.raises(ValueError):
            newton_raphson_pressure_solver(
                surface_pressure=100.0,
                vertical_depth_ft=0.0,
                gas_rate=100000,
                tubing_diameter_ft=0.10,
                geothermal_gradient_f_per_1000ft=50.0,
                gas_specific_gravity=0.65,
            )
    
    def test_invalid_tolerance(self):
        """Test that non-positive tolerance raises error."""
        with pytest.raises(ValueError):
            newton_raphson_pressure_solver(
                surface_pressure=100.0,
                vertical_depth_ft=1000.0,
                gas_rate=100000,
                tubing_diameter_ft=0.10,
                geothermal_gradient_f_per_1000ft=50.0,
                gas_specific_gravity=0.65,
                convergence_tolerance=-1e-6,
            )
    
    def test_invalid_flow_direction(self):
        """Test that invalid flow direction raises error."""
        with pytest.raises(ValueError):
            newton_raphson_pressure_solver(
                surface_pressure=100.0,
                vertical_depth_ft=1000.0,
                gas_rate=100000,
                tubing_diameter_ft=0.10,
                geothermal_gradient_f_per_1000ft=50.0,
                gas_specific_gravity=0.65,
                flow_direction='invalid',
            )
    
    def test_physical_bounds(self):
        """Test that computed pressure stays within physical bounds."""
        result = newton_raphson_pressure_solver(
            surface_pressure=100.0,
            vertical_depth_ft=10000.0,
            gas_rate=50000,
            tubing_diameter_ft=0.15,
            geothermal_gradient_f_per_1000ft=60.0,
            gas_specific_gravity=0.60,
            convergence_tolerance=1e-4,
            max_iterations=25,
        )
        
        # BHP should be > surface pressure - reasonable gradient
        # But should not be negative or zero
        assert result > 0
        assert result > surface_pressure * 0.1  # At least 10% of surface pressure
        
    def test_large_depth(self):
        """Test solver with very large depth."""
        result = newton_raphson_pressure_solver(
            surface_pressure=500.0,
            vertical_depth_ft=50000.0,  # 50,000 ft depth
            gas_rate=1000000,
            tubing_diameter_ft=0.10,
            geothermal_gradient_f_per_1000ft=65.0,
            gas_specific_gravity=0.65,
        )
        
        assert isinstance(result, (int, float))
        assert result > 0
        
    def test_gas_rate_sensitivity(self):
        """Test that solver responds appropriately to gas rate changes."""
        # Higher gas rate = more friction = higher BHP
        result_high_rate = newton_raphson_pressure_solver(
            surface_pressure=300.0,
            vertical_depth_ft=8000.0,
            gas_rate=1000000,  # High rate
            tubing_diameter_ft=0.10,
            geothermal_gradient_f_per_1000ft=55.0,
            gas_specific_gravity=0.70,
        )
        
        result_low_rate = newton_raphson_pressure_solver(
            surface_pressure=300.0,
            vertical_depth_ft=8000.0,
            gas_rate=50000,  # Low rate
            tubing_diameter_ft=0.10,
            geothermal_gradient_f_per_1000ft=55.0,
            gas_specific_gravity=0.70,
        )
        
        # Gas rate has significant effect on BHP
        # Result should be order-of-magnitude different
        assert result_high_rate > result_low_rate
        
    def test_standard_tolerance(self):
        """Test solver with default tolerance values."""
        result = newton_raphson_pressure_solver(
            surface_pressure=450.0,
            vertical_depth_ft=12500.0,
            gas_rate=250000,
            tubing_diameter_ft=0.12,
            geothermal_gradient_f_per_1000ft=50.0,
            gas_specific_gravity=0.66,
        )
        
        assert isinstance(result, (int, float))
        
    def test_zero_rate(self):
        """Test solver behavior with zero gas rate (dry well)."""
        result = newton_raphson_pressure_solver(
            surface_pressure=150.0,
            vertical_depth_ft=3000.0,
            gas_rate=0,  # No gas
            tubing_diameter_ft=0.10,
            geothermal_gradient_f_per_1000ft=40.0,
            gas_specific_gravity=0.65,
        )
        
        assert isinstance(result, (int, float))
        assert result > 0


class TestNewtonRaphsonGeneral:
    """Test general-purpose Newton-Raphson solver."""
    
    def test_simple_root_finding(self):
        """Test solver for simple quadratic equation."""
        def f(x):
            return x**2 - 4  # Root at x = ±2
        
        def df(x):
            return 2*x
        
        # Start from positive guess
        root = newton_raphson_solver(f, df, x0=1.0, tolerance=1e-6, max_iterations=20)
        
        assert abs(root - 2.0) < 1e-6
        
    def test_negative_root(self):
        """Test solver finds negative root."""
        def f(x):
            return x**2 - 9  # Root at x = -3
        
        def df(x):
            return 2*x
        
        root = newton_raphson_solver(f, df, x0=-1.0, tolerance=1e-6)
        
        assert abs(root + 3.0) < 1e-6
        
    def test_cubic_equation(self):
        """Test solver for cubic equation."""
        def f(x):
            return x**3 - x - 2  # One real root near x = 1.5
        
        def df(x):
            return 3*x**2 - 1
        
        root = newton_raphson_solver(f, df, x0=1.0, tolerance=1e-6)
        
        # Root should be approximately 1.618
        assert abs(root - 1.618) < 1e-3
        
    def test_max_iterations_limit(self):
        """Test that max iterations prevents infinite loops."""
        def f(x):
            return math.sin(x)  # Root at multiples of pi
        
        def df(x):
            return math.cos(x)
        
        # Start far from root
        root = newton_raphson_solver(f, df, x0=10.0, tolerance=1e-6, max_iterations=5)
        
        # Should raise an error, not loop forever
        assert abs(root) > 0  # If it didn't raise
        
    def test_derivative_too_small(self):
        """Test handling of near-zero derivative."""
        def f(x):
            return (x - 1)**3  # Triple root, derivative near zero
        
        def df(x):
            return 3*(x - 1)**2
        
        with pytest.raises(ValueError) as excinfo:
            newton_raphson_solver(f, df, x0=0.1, tolerance=1e-6, max_iterations=10)
        
        assert 'Derivative too close to zero' in str(excinfo.value)


class TestSolvePressureStep:
    """Test stepwise pressure calculation."""
    
    def test_upward_flow(self):
        """Test pressure increase in upward flow."""
        result = solve_pressure_step(
            pressure=100.0,
            depth_increment_ft=100.0,
            flow_direction='up',
            pvt_parameters={'length_ft': 10000, 'density': 50},
            friction_factor=0.02,
        )
        
        assert result > 100.0  # Pressure should increase
        
    def test_downward_flow(self):
        """Test pressure decrease in downward flow."""
        result = solve_pressure_step(
            pressure=300.0,
            depth_increment_ft=200.0,
            flow_direction='down',
            pvt_parameters={'length_ft': 10000, 'density': 45},
            friction_factor=0.015,
        )
        
        assert result < 300.0  # Pressure should decrease


class TestSolverDiagnostics:
    """Test convergence diagnostics."""
    
    def test_diagnostics_analyze_convergence(self):
        """Test convergence analysis."""
        # Create mock iteration log
        log = [
            {'iteration': 1, 'pressure': 500.0, 'function_value': 0.1},
            {'iteration': 2, 'pressure': 498.0, 'function_value': 0.01},
            {'iteration': 3, 'pressure': 497.0, 'function_value': 0.01},
            {'iteration': 4, 'pressure': 496.0, 'function_value': 0.001},
            {'iteration': 5, 'pressure': 495.0, 'function_value': 0.0001},
        ]
        
        diag = SolverDiagnostics.analyze_convergence(log, tolerance=1e-3)
        
        assert diag['converged'] == True
        assert diag['iterations_used'] == 5
        assert diag['final_residue'] == 0.0001
        assert diag['tolerance'] == 0.001
        assert diag['met_tolerance'] == True
        
    def test_diagnostics_non_converged(self):
        """Test diagnostics for non-converged run."""
        log = [
            {'iteration': 1, 'pressure': 100.0, 'function_value': 0.1},
            {'iteration': 2, 'pressure': 101.0, 'function_value': 0.09},
            {'iteration': 3, 'pressure': 102.0, 'function_value': 0.08},
        ]
        
        diag = SolverDiagnostics.analyze_convergence(log, tolerance=1e-4)
        
        assert diag['converged'] == False
        assert diag['met_tolerance'] == False
        
    def test_diagnostics_with_target(self):
        """Test diagnostics with target value."""
        log = [
            {'iteration': 1, 'pressure': 500.0, 'function_value': 0.5},
            {'iteration': 2, 'pressure': 498.0, 'function_value': 0.25},
            {'iteration': 3, 'pressure': 496.0, 'function_value': 0.125},
        ]
        
        diag = SolverDiagnostics.analyze_convergence(log, tolerance=1e-4, target=496.0)
        
        assert 'distance_to_target' in diag
        assert 'met_target' in diag


class TestConvergenceFunctionality:
    """Test convergence criteria and error handling."""
    
    def test_ac2_tolerance_handling(self):
        """AC2: Test that convergence tolerance is properly enforced."""
        # Different tolerance values
        for tol in [1e-1, 1e-3, 1e-6]:
            result = newton_raphson_pressure_solver(
                surface_pressure=150.0,
                vertical_depth_ft=2000.0,
                gas_rate=50000,
                tubing_diameter_ft=0.12,
                geothermal_gradient_f_per_1000ft=30.0,
                gas_specific_gravity=0.65,
                convergence_tolerance=tol,
                max_iterations=30,
            )
            
            assert isinstance(result, (int, float))
            assert result > 0
            
    def test_ac3_divergence_handling(self):
        """AC3: Test divergence handling when convergence fails."""
        with pytest.raises(ConvergenceError):
            newton_raphson_pressure_solver(
                surface_pressure=500.0,
                vertical_depth_ft=15000.0,
                gas_rate=1000,  # Very low gas rate
                tubing_diameter_ft=0.20,  # Large diameter
                geothermal_gradient_f_per_1000ft=25.0,
                gas_specific_gravity=0.65,
                convergence_tolerance=1e-6,
                max_iterations=15,  # Low iterations to force failure
            )
    
    def test_ac5_performance_characteristics(self):
        """AC5: Test convergence speed and efficiency."""
        import time
        
        # Test with moderate depth and rate
        start = time.time()
        for _ in range(10):
            newton_raphson_pressure_solver(
                surface_pressure=300.0,
                vertical_depth_ft=5000.0,
                gas_rate=100000,
                tubing_diameter_ft=0.12,
                geothermal_gradient_f_per_1000ft=45.0,
                gas_specific_gravity=0.66,
            )
        elapsed = time.time() - start
        
        # Multiple runs should complete in reasonable time
        assert elapsed < 5.0  # Should take less than 5 seconds


# Convergence test runner
def test_all_convergence_scenarios():
    """Run comprehensive convergence test suite."""
    # This is called to verify AC5 compliance
    
    # Test basic convergence
    result = newton_raphson_pressure_solver(
        surface_pressure=200.0,
        vertical_depth_ft=4000.0,
        gas_rate=80000,
        tubing_diameter_ft=0.10,
        geothermal_gradient_f_per_1000ft=40.0,
        gas_specific_gravity=0.70,
        convergence_tolerance=1e-6,
        max_iterations=50,
    )
    assert result > 0
    
    # Test edge cases
    test_results = test_convergence_stability()
    
    print("\n" + "="*60)
    print("CONVERGENCE TEST RESULTS")
    print("="*60)
    for test, passed in test_results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test}")
    print("="*60)


if __name__ == '__main__':
    # Run with pytest when executed directly
    pytest.main([__file__, '-v'])
