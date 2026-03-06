# Pressure Traverse Story Decomposition

## Foundation Stories

### Story 1: Project Setup and Package Structure
**Story ID:** 1
**Title:** Establish project structure and package skeleton
**Story Points:** 2
**Priority:** p0-critical
**Epic:** 4

**Description:**
Set up the project repository with complete directory structure, configuration files, and package initialization. Establish a clean, production-ready Python package structure following best practices.

**Acceptance Criteria:**
- AC1: Directory structure matches ARCHITECTURE.md specification
- AC2: `setup.py` properly configured with correct install requirements
- AC3: `pyproject.toml` configured for modern Python packaging with CI/CD hooks
- AC4: All core modules have `__init__.py` exposing intended public APIs
- AC5: GitHub repository properly initialized with README.md template
- AC6: Development environment working (install from source succeeds)

**Implementation Notes:**
1. Use black, pylint, and flake8 hooks for code quality
2. Include mypy for type checking
3. Use setuptools_scm for version management if available

**Dependencies:**
- None (foundation for all subsequent work)

---

### Story 2: Domain Models and Data Structures
**Story ID:** 2
**Title:** Implement domain model classes
**Story Points:** 3
**Priority:** p0-critical
**Epic:** 4

**Description:**
Create dataclasses for all domain models: FluidProperties, WellGeometry, PVTProperties, and PressureTraverseResult. Implement property methods for computed properties and basic validation in the data classes.

**Acceptance Criteria:**
- AC1: All domain classes implemented with dataclass decorators
- AC2: FluidProperties includes all PVT parameters with proper defaults
- AC3: WellGeometry correctly parses deviation data and provides lookup methods
- AC4: PressureTraverseResult includes all calculated profiles
- AC5: All @property methods implement computed values (densities, Z-factors)
- AC6: Dataclasses include input validation methods

**Implementation Notes:**
1. Use typing hints (NumPy array types, Optional, etc.)
2. Implement type checking with mypy stubs
3. Keep dataclass minimal - validation in separate utils module

**Dependencies:**
- Story 1 (completed)

---

### Story 3: Unit Conversion Utilities
**Story ID:** 1
**Title:** Implement unit conversion utilities
**Story Points:** 2
**Priority:** p2-medium
**Epic:** 4

**Description:**
Create comprehensive unit conversion functions for API units (ft, psi, °F, °R) and SI units (m, Pa, K). Ensure all conversions are consistent and well-tested.

**Acceptance Criteria:**
- AC1: `units.py` implemented with conversion functions for all core units
- AC2: Conversions use consistent formulas and are physically accurate
- AC3: Functions include docstrings with conversion details
- AC4: All conversions have corresponding unit tests (±0.01% accuracy)
- AC5: Consistent API - input validation on incompatible unit pairs

**Implementation Notes:**
1. Use numpy for array-compatible conversions
2. Include R ↔ Rankine conversions (R = F + 459.67)
3. Include standard gravitational acceleration constants

**Dependencies:**
- Story 1 (completed)

---

### Story 4: Parameter Validation Module
**Story ID:** 3
**Title:** Create parameter validation and error handling framework
**Story Points:** 5
**Priority:** p1-high
**Epic:** 4

**Description:**
Implement comprehensive validation utilities for fluid properties, well geometry, and calculation parameters. Define custom exceptions and validation rules to catch physical inconsistencies early.

**Acceptance Criteria:**
- AC1: `ParameterValidator` class implemented with validation methods
- AC2: Custom exception classes (InvalidParameterError, ConvergenceError, etc.) defined
- AC3: Fluid properties validation: specific gravities, gas/oil/water ratios
- AC4: Well geometry validation: diameters, deviation angles, depth measurements
- AC5: Physical bounds checks: positive pressures, realistic temperatures
- AC6: All validation functions have unit tests with expected exceptions

**Implementation Notes:**
1. Validation should fail fast with clear error messages
2. Use descriptive error messages with expected ranges
3. Include warnings for boundary values

**Dependencies:**
- Stories 1, 2 (completed)

---

### Story 5: Validation Test Module
**Story ID:** 2
**Title:** Create validation testing framework
**Story Points:** 3
**Priority:** p1-high
**Epic:** 4

**Description:**
Build framework for validating all parameter inputs with test-driven approach. Use pytest fixtures for common test parameters and parametrize validation tests.

**Acceptance Criteria:**
- AC1: pytest fixtures defined for fluid properties and well geometry
- AC2: Parametrized tests covering edge cases and invalid inputs
- AC3: Test coverage requirements defined: >90% for validation module
- AC4: Test suite fails gracefully on invalid configurations
- AC5: Validation test output includes clear failure diagnostics

**Implementation Notes:**
1. Use `pytest.mark.parametrize` extensively
2. Separate valid and invalid test cases
3. Document expected exception types for each test case

**Dependencies:**
- Story 2, 4 (completed)

---

## PVT Implementation Stories

### Story 6: Black Oil Property Correlations
**Story ID:** 8
**Title:** Implement black oil PVT property correlations (Vapor pressure, BPF, density)
**Story Points:** 8
**Priority:** p1-high
**Epic:** 4

**Description:**
Implement key black oil correlations: Standing vapor pressure, formation volume factors, solution gas ratio, and density calculations. Use standard correlations (Standing, Vasquez-Beggs, Glaso) per API RPI.

**Acceptance Criteria:**
- AC1: Vasquez-Beggs vapor pressure correlation implemented
- AC2: Standing liquid API gravity correction for BPF
- AC3: Vasquez-Beggs oil formation volume factor (FVF) correlation
- AC4: Vasquez-Beggs gas formation volume factor (FVF) correlation
- AC5: Gas density calculation at reservoir conditions
- AC6: Water properties (density at surface and reservoir conditions)
- AC7: All correlations validated against API test cases to ±1% accuracy

**Implementation Notes:**
1. Implement as class methods in `fluid.py`
2. Include correlation coefficient validation for API compatibility
3. Handle edge cases: low pressures, oil gravities at boundaries

**Dependencies:**
- Story 2 (completed)

---

### Story 7: Natural Gas Z-Factor Models
**Story ID:** 5
**Title:** Implement natural gas compressibility factor (Z-factor) correlations
**Story Points:** 5
**Priority:** p1-high
**Epic:** 4

**Description:**
Implement multiple Z-factor correlations (Standing-Katz, Lee-Gonzales, AGA-DC). Create reusable implementation for Z-factor tables and interpolation, supporting both natural gas and gas-condensate systems.

**Acceptance Criteria:**
- AC1: Standing-Katz Z-factor chart interpolation implemented
- AC2: Lee-Gonzales Espana correlation implemented
- AC3: AGA-8 Natural Gas Equation of State (DC method) implemented
- AC4: Pseudocritical properties calculator for gas systems
- AC5: Z-factor functions handle both single-gravity and multi-component systems
- AC6: All Z-factor implementations have unit tests with ±0.1% accuracy

**Implementation Notes:**
1. Standing-Katz uses pre-computed lookup table interpolation
2. Lee-Gonzales uses empirical formula with pseudo-critical properties
3. AGA-DC is equation-of-state based - more complex but most accurate

**Dependencies:**
- Story 6 (completed)

---

### Story 8: EOS Implementation (SRK and Peng-Robinson)
**Story ID:** 13
**Title:** Implement Equation of State (EOS) solvers for compositional systems
**Story Points:** 13
**Priority:** p2-medium
**Epic:** 4

**Description:**
Create abstract EOS base class and implement SRK and Peng-Robinson equations with cubic solving. This will support advanced gas-condensate systems requiring compositional calculations.

**Acceptance Criteria:**
- AC1: Abstract `EquationOfState` base class defined
- AC2: `SRKEOS` class implements full SRK equation solver (cubic roots)
- AC3: `PengRobinsonEOS` class implements full PR equation solver
- AC4: EOS solvers handle ideal gas regime and non-ideal gas regime
- AC5: Z-factor calculation returns appropriate real gas compressibility
- AC6: EOS implementations validated against known test cases
- AC7: Integration with Z-factor module for consistency

**Implementation Notes:**
1. Need robust cubic equation solver (Cardano's formula or numerical)
2. Must handle physically valid roots only (positive, realistic Z-factors)
3. Edge cases: near-critical point, very high pressures

**Dependencies:**
- Story 7 (completed)

---

## Flow Correlation Stories

### Story 9: Flow Regime Identification
**Story ID:** 3
**Title:** Implement flow regime identification logic
**Story Points:** 3
**Priority:** p1-high
**Epic:** 4

**Description:**
Create flow regime detection module that identifies flow regime types (Segregated, Intermittent, Distributed, Mist) based on fluid properties and well geometry. Implement Beggs-Brill flow regime transition map.

**Acceptance Criteria:**
- AC1: `FlowRegime` enum defined with all flow regime types
- AC2: Beggs-Brill flow regime transition map implemented
- AC3: Flow classification handles all pipe inclinations (0-90°)
- AC4: Helper functions for regime transition detection (F_Lo, Fr_Lo calculations)
- AC5: Flow regime identification has unit tests with expected outputs
- AC6: Edge cases covered (very low gas rate, high liquid rate)

**Implementation Notes:**
1. F_Lo = ρ_l * V_l / μ_w (superficial liquid velocity based)
2. Fr_Lo based on gravitational forces
3. Use standard Beggs-Brill transition criteria from API RPI

**Dependencies:**
- Story 2 (completed)

---

### Story 10: Friction Factor Calculations
**Story ID:** 5
**Title:** Implement friction factor models (Darcy-Weisbach, Moody, API)
**Story Points:** 5
**Priority:** p1-high
**Epic:** 4

**Description:**
Implement friction factor calculations for multiphase flow: Moody diagram lookup, API correlation, and Darcy-Weisbach equation integration. Support laminar, transitional, and turbulent regimes.

**Acceptance Criteria:**
- AC1: Moody diagram lookup function implemented with Reynolds number interpolation
- AC2: API standard friction factor correlation implemented
- AC3: Darcy-Weisbach friction factor calculation method
- AC4: Reynolds number calculation for single-phase and multiphase flow
- AC5: Laminar and transitional regime handling (Re < 2000)
- AC6: All friction factor functions validated against standard tables

**Implementation Notes:**
1. Moody coefficients must be hardcoded or loaded from database
2. Handle roughness input (pipe condition)
3. Friction factor returned should be Darcy friction factor (∂P/∂x vs V²)

**Dependencies:**
- Story 3 (completed)

---

### Story 11: Beggs-Brill Multiphase Flow Correlation
**Story ID:** 8
**Title:** Implement Beggs & Brill (1973) multiphase flow correlation
**Story Points:** 8
**Priority:** p1-high
**Epic:** 4

**Description:**
Fully implement Beggs-Brill (1973) correlation for pressure drop, liquid holdup, and friction factor in pipes. This is the industry standard for multiphase flow calculations.

**Acceptance Criteria:**
- AC1: Beggs-Brill pressure drop calculation for any pipe inclination
- AC2: Beggs-Brill liquid holdup correlation implementation
- AC3: Beggs-Brill friction factor correlation
- AC4: Complete function `calculate_pressure_drop()` returning pressure, holdup, friction
- AC5: Handles all flow regimes (Segregated, Intermittent, Distributed)
- AC6: Validates against API RPI test cases to ±1% accuracy
- AC7: Documentation and examples included

**Implementation Notes:**
1. Beggs-Brill uses flow regime and inclination-dependent models
2. Key parameters: F_Lo, Fr_Lo, F_g
3. Include correction factors C_1, C_2, C_3 based on inclination
4. Need correlation code from API RPI source

**Dependencies:**
- Stories 3, 4, 6, 8, 9, 10 (completed)

---

### Story 12: Additional Flow Correlations
**Story ID:** 5
**Title:** Implement Hagedorn-Brown and Gray correlations
**Story Points:** 5
**Priority:** p2-medium
**Epic:** 4

**Description:**
Implement Hagedorn-Brown (optimized for vertical flow) and Gray (optimized for gas wells) correlations for pressure traverse calculations.

**Acceptance Criteria:**
- AC1: Hagedorn-Brown correlation for vertical wells implemented
- AC2: Gray correlation for gas-liquid flow implemented
- AC3: Correlation selector/utility to choose based on application context
- AC4: Each correlation validated against API test cases
- AC5: Unit tests covering both correlation implementations

**Implementation Notes:**
1. Hagedorn-Brown requires lift gas property calculation
2. Gray correlation uses different holdup model for gas-rich systems
3. These serve as fallback or alternative predictions

**Dependencies:**
- Story 11 (completed)

---

### Story 13: Z-Factor Integration into Flow Solver
**Story ID:** 3
**Title:** Integrate Z-factor with flow correlation calculations
**Story Points:** 3
**Priority:** p1-high
**Epic:** 4

**Description:**
Connect Z-factor calculations with flow correlations and pressure solver. Ensure gas compressibility is correctly accounted for in pressure gradient calculations.

**Acceptance Criteria:**
- AC1: Z-factor lookup integrated into Beggs-Brill flow correlation
- AC2: Temperature-based Z-factor variation handled (gas temperature varies with depth)
- AC3: Gas density calculations use correct Z-factor and FVF
- AC4: Pressure gradient includes gas compressibility effects
- AC5: Single-phase limits work (gas-only, water-only, oil-only flows)

**Implementation Notes:**
1. Gas density: ρ_g = (P * SG_g * M) / (Z * R * T)
2. Gas FVF: B_g = 0.577 * T * Z / P (simplified API form)
3. Update Z-factor where gas phase exists in multiphase

**Dependencies:**
- Stories 3, 7, 11 (completed)

---

## Core Solver Stories

### Story 14: Newton-Raphson Iterative Solver
**Story ID:** 8
**Title:** Implement Newton-Raphson pressure solver
**Story Points:** 8
**Priority:** p0-critical
**Epic:** 4

**Description:**
Develop Newton-Raphson iterative solver for nonlinear pressure traverse equations. Implement convergence criteria, error handling, and diagnostics for iterative solver stability.

**Acceptance Criteria:**
- AC1: Generic Newton-Raphson solver implemented for any single-variable equation
- AC2: Pressure traverse pressure equation solver wrapper
- AC3: Convergence checks: absolute and relative criteria
- AC4: Maximum iteration limit (50 iterations) with diagnostics
- AC5: Convergence failure raises ConvergenceError with diagnostics
- AC6: Unit tests validating convergence for known functions
- AC7: Test with typical fluid properties and well conditions

**Implementation Notes:**
1. Pressure equation: P_{next} = P_current - f(P)/f'(P)
2. Need explicit derivative or finite difference approximation
3. Include damping factor for problematic cases

**Dependencies:**
- Story 4, 14 (will depend on itself)

---

### Story 15: Pressure Traverse Calculation Engine
**Story ID:** 13
**Title:** Implement main pressure traverse solver (surface-to-bottom and bottom-to-surface)
**Story Points:** 13
**Priority:** p0-critical
**Epic:** 4

**Description:**
Build the complete PressureTraverseSolver class with sweep algorithms and stepwise pressure traversals. Integrate all components (PVT, correlations, friction, solver). Implement surface-to-bottom and bottom-to-surface traverses.

**Acceptance Criteria:**
- AC1: `PressureTraverseSolver` class created with main calculation API
- AC2: Surface-to-bottom traverse calculation method (most common use case)
- AC3: Bottom-to-surface traverse calculation method for inflow testing
- AC4: Stepwise pressure calculation for each depth increment
- AC5: Coupled PVT and flow calculations at each depth point
- AC6: Integration with Newton-Raphson solver for pressure updates
- AC7: Result aggregation: collects all pressure profiles, regimes, loss components
- AC8: Full API RPI test suite validation (all test cases pass)
- AC9: Comprehensive documentation with usage examples

**Implementation Notes:**
1. Algorithm: Iterate depth by depth, solve pressure for next depth
2. Temperature model: linear geothermal gradient
3. Flow regime update each step
4. Performance optimization for >10,000 depth points
5. Return structured result with all profiles

**Dependencies:**
- Stories 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14 (all completed)

---

## Testing & Validation Stories

### Story 16: API RPI Test Suite Implementation
**Story ID:** 5
**Title:** Implement API Recommended Practice 14A (1976) test cases
**Story Points:** 5
**Priority:** p0-critical
**Epic:** 4

**Description:**
Create comprehensive test suite with all API RPI standard test cases. Implement test data structures and automated validation against known correct answers.

**Acceptance Criteria:**
- AC1: Test case data structures defined in `api_tests.py`
- AC2: All API RPI1976 test cases loaded (oil wells, gas wells, multiphase)
- AC3: Automated test runner for all test cases
- AC4: Validation: computed pressures match expected ±0.5% tolerance
- AC5: Test suite identifies failing cases for each correlation
- AC6: Test results report: which cases pass, which fail, relative errors

**Implementation Notes:**
1. Get API RPI test data from API reference documentation
2. Test cases cover range of conditions (shallow vs deep wells)
3. Track error metrics for debugging

**Dependencies:**
- Story 15 (completed)

---

### Story 17: Numerical Accuracy and Stability
**Story ID:** 5
**Title:** Verify numerical accuracy and handle edge cases
**Story Points:** 5
**Priority:** p1-high
**Epic:** 4

**Description:**
Comprehensive testing for numerical stability, edge cases, and accuracy. Test extreme conditions, transition zones, and convergence scenarios.

**Acceptance Criteria:**
- AC1: Edge case tests: 0° deviation (vertical), 90° deviation (horizontal)
- AC2: Low gas rate, high liquid rate extremes
- AC3: Very shallow wells (100 ft) and very deep wells (30,000 ft)
- AC4: Convergence stability tests (varying initial pressures)
- AC5: Error bounds reported for each API test case
- AC6: Numerical precision tests (±0.01 psi tolerance)

**Implementation Notes:**
1. Test with random noise in inputs to verify robustness
2. Compare against commercial software (if accessible)
3. Document any known limitation regions

**Dependencies:**
- Story 16 (completed)

---

## Documentation & Deployment Stories

### Story 18: API Documentation and Examples
**Story ID:** 8
**Title:** Generate comprehensive API documentation and usage examples
**Story Points:** 8
**Priority:** p2-medium
**Epic:** 4

**Description:**
Create Sphinx documentation with API reference, installation guide, and usage examples. Document all public APIs, parameters, return values, and error conditions.

**Acceptance Criteria:**
- AC1: Sphinx documentation project configured (conf.py, make files)
- AC2: NumPydoc style docstrings on all public functions and classes
- AC3: API reference pages generated for all modules
- AC4: Installation guide (pip install, Docker)
- AC5: Tutorial: end-to-end example from input to result
- AC6: Case studies: comparison of different correlations
- AC7: Troubleshooting guide for common errors

**Implementation Notes:**
1. Use numpydoc formatter for consistency
2. Include code snippets in docstrings
3. Use Sphinx autodoc to auto-generate from code

**Dependencies:**
- Story 15 (completed)

---

### Story 19: Docker and Environment Setup
**Story ID:** 8
**Title:** Create Docker container and deployment documentation
**Story Points:** 8
**Priority:** p1-high
**Epic:** 4

**Description:**
Create Dockerfile and docker-compose.yml for reproducible environment. Provide installation instructions for various platforms and deployment guidance.

**Acceptance Criteria:**
- AC1: Multi-stage Dockerfile for Python library
- AC2: Docker image successfully installs library
- AC3: Simple example Docker container demonstrates library usage
- AC4: Docker Compose for development environment (dev, test, prod)
- AC5: Deployment guide (cloud, local, containerized)
- AC6: CI/CD pipeline configuration (GitHub Actions)

**Implementation Notes:**
1. Use Python 3.10 base image
2. Alpine for smaller images, slim for better compatibility
3. Include test script in container

**Dependencies:**
- Story 15 (completed)

---

### Story 20: Continuous Verification and CI/CD
**Story ID:** 3
**Title:** Set up CI/CD pipeline with automated testing
**Story Points:** 3
**Priority:** p1-high
**Epic:** 4

**Description:**
Configure GitHub Actions for continuous integration with automated builds, testing, and documentation preview.

**Acceptance Criteria:**
- AC1: GitHub Actions workflow file created (.github/workflows/)
- AC2: Automated testing on every push and PR
- AC3: Test run on multiple Python versions
- AC4: Code coverage report generation
- AC5: Package build and upload to test PyPI
- AC6: Pre-commit hooks configured (black, flake8, mypy)

**Implementation Notes:**
1. Use matrix strategy for Python versions
2. Run complete test suite on all PRs
3. Coverage badge in README.md

**Dependencies:**
- Story 16-20 (development setup complete)

---

## Dependencies Overview

**Critical Path (for initial release):**
1 → 2 → 3 → 4 → 6 → 7 → 9 → 10 → 11 → 13 → 14 → 15 → 16

**Estimated Story Points for Initial Release:**
Total: 98 SP (Foundation: 18, PVT: 18, Flow: 39, Solver: 13, Testing: 8)

## Prioritized Sprint Order

**Sprint 1 - Foundation (Week 1):**
Story 1, Story 3, Story 4, Story 5

**Sprint 2 - PVT Core (Week 2-3):**
Story 6, Story 7, Story 8

**Sprint 3 - Flow Correlations (Week 3-4):**
Story 9, Story 10, Story 11, Story 12, Story 13

**Sprint 4 - Core Solver (Week 5-6):**
Story 14, Story 15

**Sprint 5 - Testing & Validation (Week 6-7):**
Story 16, Story 17

**Sprint 6 - Documentation & Deployment (Week 7-8):**
Story 18, Story 19, Story 20