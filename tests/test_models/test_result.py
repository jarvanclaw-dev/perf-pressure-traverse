"""Tests for PressureTraverseResult model."""
import pytest
import numpy as np

from perf_pressure_traverse.models.result import PressureTraverseResult, CalculationStatus


class TestPressureTraverseResult:
    """Test cases for PressureTraverseResult class."""
    
    def test_init_successful_result(self):
        """Test initializing a successful calculation result."""
        result = PressureTraverseResult(
            well_name="Test Well",
            surface_pressure_psia=500.0,
            wellhead_pressure_psia=480.0,
            bottomhole_pressure_psia=4000.0,
            total_depth_ft=10000.0
        )
        
        assert result.status == CalculationStatus.SUCCESS
        assert result.well_name == "Test Well"
        assert result.surface_pressure_psia == 500.0
        assert result.bottomhole_pressure_psia == 4000.0
    
    def test_init_failed_result(self):
        """Test initializing a failed calculation result."""
        result = PressureTraverseResult(
            well_name="Test Well",
            status=CalculationStatus.FAILED,
            error_message="Div calculation failed"
        )
        
        assert result.status == CalculationStatus.FAILED
        assert result.error_message == "Div calculation failed"
    
    def test_result_properties(self):
        """Test result properties."""
        result = PressureTraverseResult(
            well_name="Oil Well",
            surface_pressure_psia=600.0,
            wellhead_pressure_psia=580.0,
            bottomhole_pressure_psia=3000.0,
            total_depth_ft=8000.0
        )
        
        assert result.pressure_drop = result.surface_pressure_psia - result.wellhead_pressure_psia == 20.0
        assert result.is_valid()
    
    def test_validate_result(self):
        """Test result validation."""
        result = PressureTraverseResult(
            well_name="Test",
            surface_pressure_psia=500.0,
            bottomhole_pressure_psia=4000.0
        )
        
        assert result.is_valid()
        
        # Invalid: pressure drop should be positive
        result.invalid = PressureTraverseResult(
            well_name="Test",
            surface_pressure_psia=600.0,
            wellhead_pressure_psia=700.0
        )
        
        assert not result.invalid.is_valid()
    
    def test_result_summary(self):
        """Test result summary generation."""
        result = PressureTraverseResult(
            well_name="Gas Well",
            surface_pressure_psia=300.0,
            wellhead_pressure_psia=350.0,
            bottomhole_pressure_psia=2000.0,
            total_depth_ft=15000.0
        )
        
        summary = result.summary()
        
        assert "Gas Well" in summary
        assert "3000 ft" in summary
        assert "20 psia" in summary  # The pressure rise is due to choke
        assert "2000 psia" in summary
    
    def test_multiple_results_comparison(self):
        """Test comparing multiple calculation results."""
        result1 = PressureTraverseResult(
            well_name="Well 1",
            surface_pressure_psia=500.0,
            bottomhole_pressure_psia=3800.0,
            total_depth_ft=10000.0
        )
        
        result2 = PressureTraverseResult(
            well_name="Well 2",
            surface_pressure_psia=500.0,
            bottomhole_pressure_psia=4100.0,
            total_depth_ft=10000.0
        )
        
        # Well 2 has higher BH pressure
        assert result2.bottomhole_pressure_psia > result1.bottomhole_pressure_psia


class TestCalculationStatus:
    """Test cases for CalculationStatus enum."""
    
    def test_calculation_status_values(self):
        """Test calculation status enum values."""
        assert CalculationStatus.SUCCESS.value == 0
        assert CalculationStatus.FAILED.value == 1
        assert CalculationStatus.WARNING.value == 2
        assert CalculationStatus.INPROGRESS.value == 3


class TestTraverseData:
    """Test cases for traverse data structure."""
    
    def test_traverse_data_array(self):
        """Test traverse data arrays."""
        data = PressureTraverseResult(
            well_name="Test",
            surface_pressure_psia=500.0,
            bottomhole_pressure_psia=3500.0,
            total_depth_ft=10000.0
        )
        
        # Simulated traverse points
        depths = np.linspace(0, 10000, 100)
        pressures = np.linspace(500, 3500, 100)
        
        data.traverse_depths = depths
        data.traverse_pressures = pressures
        
        assert len(data.traverse_depths) == 100
        assert len(data.traverse_pressures) == 100
        assert data.traverse_depths[0] == 0.0
        assert data.traverse_pressures[-1] == 3500.0
    
    def test_traverse_summary_stats(self):
        """Test summary statistics from traverse data."""
        data = PressureTraverseResult(
            well_name="Test",
            surface_pressure_psia=500.0,
            bottomhole_pressure_psia=3500.0
        )
        
        depths = np.linspace(0, 10000, 100)
        pressures = np.linspace(500, 3500, 100)
        
        data.traverse_depths = depths
        data.traverse_pressures = pressures
        
        stats = data.get_summary_stats()
        
        assert stats['max_depth'] == 10000.0
        assert stats['max_pressure'] == 3500.0
        assert stats['average'] in [2000, 2001]  # Round to reasonable precision
