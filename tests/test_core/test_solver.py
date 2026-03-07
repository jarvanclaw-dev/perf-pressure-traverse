"""Tests for PressureTraverseSolver class."""
import pytest

from perf_pressure_traverse.models.fluid import FluidProperties
from perf_pressure_traverse.models.well import WellGeometry, SectionGeometry
from perf_pressure_traverse.models.pvt_properties import PVTProperties
from perf_pressure_traverse.core.solver import PressureTraverseSolver
from perf_pressure_traverse.utils.validation import ParameterValidator
from perf_pressure_traverse.utils.diagnostics import SolverDiagnostics


class TestPressureTraverseSolver:
    """Test cases for PressureTraverseSolver class."""
    
    def test_init_solver(self):
        """Test solver initialization."""
        solver = PressureTraverseSolver()
        
        assert solver.solver_name == "Pressure Traverse Solver"
        assert solver.is_calculating is False
    
    def test_solver_with_fluid(self):
        """Test solver with fluid properties."""
        fluid = FluidProperties(
            oil_gravity=30.0,
            gas_gravity=0.65,
            specific_gravity=0.7,
            water_saturation=0.3
        )
        
        solver = PressureTraverseSolver(fluid)
        
        assert solver.fluid is not None
        assert solver.fluid.oil_gravity == 30.0
    
    def test_solver_with_well_geometry(self):
        """Test solver with well geometry."""
        sections = [
            SectionGeometry(
                start_depth_ft=0.0,
                end_depth_ft=5000.0,
                inclination_deg=0.0,
                azimuth_deg=0.0
            ),
            SectionGeometry(
                start_depth_ft=5000.0,
                end_depth_ft=10000.0,
                inclination_deg=45.0,
                azimuth_deg=90.0
            )
        ]
        
        well = WellGeometry(
            well_name="Test Well",
            sections=sections
        )
        
        solver = PressureTraverseSolver(fluid, well)
        
        assert solver.well is not None
        assert solver.well.well_name == "Test Well"
    
    def test_validate_parameters_no_errors(self):
        """Test parameter validation without errors."""
        fluid = FluidProperties(oil_gravity=30.0)
        well = WellGeometry(well_name="Test", sections=[])
        
        solver = PressureTraverseSolver(fluid, well)
        validation = solver.validate_parameters()
        
        assert validation.is_valid
        assert len(validation.errors) == 0
    
    def test_validate_parameters_with_errors(self):
        """Test parameter validation with invalid parameters."""
        # Fluid with no API gravity
        fluid = FluidProperties(oil_gravity=None)
        well = WellGeometry(well_name="Test", sections=[])
        
        solver = PressureTraverseSolver(fluid, well)
        validation = solver.validate_parameters()
        
        assert not validation.is_valid
        assert len(validation.errors) > 0
    
    def test_run_traverse_vertical_well(self):
        """Test pressure traverse calculation for vertical well."""
        fluid = FluidProperties(
            oil_gravity=30.0,
            gas_gravity=0.6
        )
        
        sections = [
            SectionGeometry(0.0, 5000.0, 0.0, 0.0)
        ]
        
        well = WellGeometry(
            well_name="Vertical Test",
            sections=sections
        )
        
        solver = PressureTraverseSolver(fluid, well)
        result = solver.run_traverse(surface_pressure=500.0)
        
        assert result.status.value == 0
        assert result.well_name == "Vertical Test"
        assert result.total_depth_ft == 5000.0
        assert not result.error_message
    
    def test_run_traverse_horizontal_section(self):
        """Test pressure traverse calculation with horizontal section."""
        fluid = FluidProperties(
            oil_gravity=35.0,
            gas_gravity=0.65
        )
        
        sections = [
            SectionGeometry(0.0, 3000.0, 0.0, 0.0),
            SectionGeometry(3000.0, 6000.0, 90.0, 0.0)  # Horizontal
        ]
        
        well = WellGeometry(
            well_name="Horizontal Test",
            sections=sections
        )
        
        solver = PressureTraverseSolver(fluid, well)
        result = solver.run_traverse(surface_pressure=500.0)
        
        assert result.status.value == 0
    
    def test_run_traverse_with_pvt(self):
        """Test pressure traverse calculation with PVT properties."""
        fluid = FluidProperties(oil_gravity=28.0)
        
        sections = [
            SectionGeometry(0.0, 3000.0, 0.0, 0.0)
        ]
        
        well = WellGeometry(
            well_name="Test",
            sections=sections
        )
        
        pvt = PVTProperties(
            pressure_psia=3000.0,
            absolute_temperature_rankine=600.0
        )
        
        solver = PressureTraverseSolver(fluid, well, pvt_properties=pvt)
        result = solver.run_traverse(surface_pressure=500.0)
        
        assert result.status.value == 0

    def test_run_traverse_diagnostic_logging(self):
        """Test that solver logs and tracks diagnostic information."""
        fluid = FluidProperties(oil_gravity=30.0)
        
        sections = [
            SectionGeometry(0.0, 1000.0, 0.0, 0.0)
        ]
        
        well = WellGeometry(
            well_name="Small Well",
            sections=sections
        )
        
        solver = PressureTraverseSolver(fluid, well)
        
        # Run traversal
        result = solver.run_traverse(surface_pressure=500.0)
        
        # Check diagnostics
        assert solver.diagnostics.iterations > 0
        assert len(solver.diagnostics.errors) == 0 or result.status == CalculationStatus.WARNING


class TestParameterValidation:
    """Test cases for parameter validation utilities."""
    
    @pytest.fixture
    def validator(self):
        """Create a ParameterValidator instance."""
        return ParameterValidator()
    
    def test_fluid_validation_valid(self, validator):
        """Test fluid validation with valid parameters."""
        fluid = FluidProperties(
            oil_gravity=30.0,
            gas_gravity=0.6,
            specific_gravity=0.7,
            water_saturation=0.3
        )
        
        errors = validator.validate_fluid(fluid)
        
        assert len(errors) == 0
    
    def test_fluid_validation_invalid_api_gravity(self, validator):
        """Test fluid validation with invalid API gravity."""
        fluid = FluidProperties(oil_gravity=-10.0)
        
        errors = validator.validate_fluid(fluid)
        
        assert len(errors) > 0
        assert "API gravity" in str(errors).lower()
    
    def test_well_validation_valid(self, validator):
        """Test well geometry validation with valid parameters."""
        sections = [
            SectionGeometry(0.0, 1000.0, 0.0, 0.0)
        ]
        
        well = WellGeometry(
            well_name="Test",
            sections=sections
        )
        
        errors = validator.validate_well(well)
        
        assert len(errors) == 0
    
    def test_well_validation_invalid_sections(self, validator):
        """Test well geometry validation with invalid sections."""
        sections = [
            SectionGeometry(1000.0, 0.0, 0.0, 0.0)  # Invalid: start > end
        ]
        
        well = WellGeometry(
            well_name="Test",
            sections=sections
        )
        
        errors = validator.validate_well(well)
        
        assert len(errors) > 0

class TestSolverDiagnostics:
    """Test cases for diagnostics utility."""
    
    def test_diagnostics_logger_init(self):
        """Test diagnostics logger initialization."""
        diagnostics = SolverDiagnostics()
        
        assert diagnostics.iterations == 0
        assert len(diagnostics.errors) == 0
    
    def test_diagnostics_logger_error(self):
        """Test error logging."""
        diagnostics = SolverDiagnostics()
        diagnostics.log_error("Test error")
        
        assert len(diagnostics.errors) == 1
        assert diagnostics.errors[0].message == "Test error"
