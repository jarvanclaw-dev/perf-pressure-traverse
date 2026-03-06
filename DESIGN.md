# Pressure Traverse Calculation Library - Design Document

## Executive Summary

This document details the design for a production-ready Python library for calculating pressure traverses in gas and liquid wells. The library will implement industry-standard methods from API Recommended Practice 14A (RPI) including Beggs-Brill, Hagedorn-Brown, Gray flow correlations, and real gas EOS calculations.

## Research Findings

### Industry Standards

1. **API Recommended Practice 14A (RPI)** - The cornerstone of pressure traverse calculations
   - Stepwise depth increments (∆L)
   - Coupled multiphase flow modeling
   - Iterative pressure-temperature-flow solution

2. **Pressure Traverse Components**
   ```
   Total Pressure Drop = Frictional Loss + Hydrostatic Pressure + Acceleration
   - Frictional: Darcy-Weisbach equation with friction factor
   - Hydrostatic: ρ_avg * g * (P2-P1)
   - Acceleration: ρ_L * V_L² - ρ_L0 * V_L0² (minor effects)
   ```

3. **Flow Regimes**
   - Segregated (low gas velocity)
   - Intermittent (slug flow)
   - Distributed (high gas velocity, mist flow)
   - Determined by flow regime maps

### Key Correlations

1. **Beggs-Brill (1973)** - Handles all flow directions
   - Flow regime transition map
   - Liquid holdup correlation
   - Friction factor correlation
   - Works with any pipe inclination

2. **Hagedorn-Brown** - Optimized for vertical wells
   - Bubble flow to mist flow transitions
   - More accurate for deep wells

3. **Gray** - Designed for gas wells
   - Higher reliability for gas-condensate systems
   - Handles higher velocity ranges

4. **PVT Models**
   - Black oil correlations (standing, Glaso, Vasquez-Beggs)
   - Natural gas Z-factor (AGA-DC, Standing-Katz, Lee-Gonzales)
   - EOS: SRK, Peng-Robinson (for compositional systems)

### Numerical Considerations

1. **Iterative Solver**
   - Newton-Raphson method for pressure convergence
   - Convergence criteria: |ΔP| < 0.1 psi typically
   - Maximum iterations: 100

2. **Depth Discretization**
   - Variable depth increments (2-100 ft depending on required accuracy)
   - Automatic adjustment near flow regime transitions

3. **Error Handling**
   - Invalid parameters (negative rates, dimensions)
   - Convergence failures with diagnostics
   - Physics-based bounds validation

## Technology Stack Justification

1. **Python 3.10+**
   - Production-ready, extensive standard library
   - Industry standard for engineering/scientific computing
   - Bioinformatics and physics communities

2. **NumPy**
   - High-performance array operations
   - Essential for iterative calculations
   - Efficient dense matrix operations

3. **NumPy-Enhanced Libraries**
   - `pandas`: Data structures and time series for PVT tables
   - `scipy`: Optimization, special functions, ODE solvers
   - `numpy`: Core numerical computing

4. **Testing & Validation**
   - `pytest`: Test framework
   - `pytest-cov`: Code coverage
   - `numpy-testing`: NumPy-specific assertions
   - API-standard test suite validation

5. **Documentation**
   - `sphinx`: API reference generation
   - `numpydoc`: NumPy documentation style
   - `m2r` / `recommonmark`: Markdown to RST conversion
   - API docstring standards

## Design Principles

1. **Separation of Concerns**
   - Physical models (geometric calculations)
   - Flow correlations (mathematical relationships)
   - PVT models (fluid property evaluation)
   - Calculation engine (solver orchestration)
   - Data structures (well models, fluid properties)

2. **Extensibility**
   - Plugin architecture for new correlations
   - Strategy pattern for flow regime identification
   - Factory pattern for PVT model creation

3. **Computational Efficiency**
   - Vectorized operations where possible
   - Cached intermediate calculations
   - Early termination for converged solutions

4. **Reliability & Validity**
   - All calculations validated against API test cases
   - Physical bounds checking
   - Sensitivity analysis support
   - Clear error messages with diagnostics

## Module Organization

```
perf_pressure_traverse/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── pressure_traverse.py      # Main solver engine
│   └── units.py                  # Unit conversion utilities
├── models/
│   ├── __init__.py
│   ├── fluid.py                  # Fluid property models
│   ├── well.py                   # Wellbore geometry models
│   └── pvt.py                    # PVT calculation modules
├── flow/
│   ├── __init__.py
│   ├── correlations.py           # Flow correlation implementations
│   ├── regime.py                 # Flow regime identification
│   └── friction.py               # Friction factor calculations
├── math/
│   ├── __init__.py
│   ├── iterative.py              # Newton-Raphson solver
│   ├── z_factor.py               # Z-factor correlations
│   └── eos.py                    # Equation of state solvers
├── data/
│   ├── __init__.py
│   ├── api_tests.py              # API RPI test cases
│   └── lookup_tables.py          # Correlation coefficient caches
├── utils/
│   ├── __init__.py
│   ├── validation.py             # Parameter validation
│   └── diagnostics.py            # Error diagnostics
├── tests/
│   ├── test_pressure_traverse.py
│   ├── test_fluid_properties.py
│   ├── test_flow_correlations.py
│   └── test_eos.py
├── docs/
│   ├── api/
│   ├── examples/
│   └── conf.py
├── setup.py
├── requirements.txt
└── README.md
```

## Data Models

### Well Model
```python
@dataclass
class WellGeometry:
    """Wellbore geometry parameters"""
    borehole_diameter_ft: float  # Internal diameter
    casing_diameter_ft: float    # Outer casing diameter
    deviation_angles: NDArray    # Deviation vs. depth (degrees)
    depth_measurements: NDArray  # True vertical depth vs. measured depth
```

### Fluid Properties
```python
@dataclass
class FluidProperties:
    """Black oil PVT properties"""
    surface_pressure_psia: float
    surface_temperature_F: float
    oil_specific_gravity: float
    gas_specific_gravity: float
    water_specific_gravity: float
    gas_oil_ratio: float
    solution_oil_ratio: float
    water_cut: float
```

### Pressure Traverse Result
```python
@dataclass
class PressureTraverseResult:
    """Results of pressure traverse calculation"""
    surface_pressure: float
    bottomhole_pressure: float
    pressure_profile: NDArray    # [depth, pressure]
    temperature_profile: NDArray
    flow_regime_profile: NDArray
    liquid_holdup_profile: NDArray
    frictional_loss_profile: NDArray
    hydrostatic_loss_profile: NDArray
    convergence_message: str
```

## Implementation Strategy

### Phase 1: Core Foundation (Data Structures + Validation)
- Data models and domain classes
- Parameter validation
- Unit conversion utilities
- API documentation skeleton

### Phase 2: Fluid & PVT Models
- Black oil property correlations
- Gas Z-factor correlations
- EOS interfaces (SRK, Peng-Robinson)

### Phase 3: Flow Correlations
- Flow regime identification
- Beggs-Brill implementation
- Hagedorn-Brown implementation
- Gray implementation

### Phase 4: Pressure Traverse Solver
- Newton-Raphson iterative algorithm
- Depth discretization engine
- Result aggregation
- Error handling

### Phase 5: Testing & Validation
- API RPI test case validation
- Numerical accuracy verification
- Edge case testing
- Integration tests

### Phase 6: Documentation & Deployment
- API reference documentation
- Usage examples and tutorials
- Docker containerization
- Testing suite in CI/CD

## Performance Considerations

1. **Memory Efficiency**
   - Use memory views for large profiles
   - Lazy evaluation for intermediate results
   - Profile structure only stores necessary data

2. **Computational Speed**
   - Vectorized NumPy operations
   - Cached correlation coefficient tables
   - Early convergence termination

3. **Scalability**
   - Support for wells with 10,000+ depth points
   - Parallelizable where independent calculations exist
   - Optimize for common use cases (surface-to-bottom)

## Security & Reliability

1. **Input Validation**
   - Physical bounds checking for all parameters
   - NaN/inf detection
   - Unit consistency verification

2. **Error Handling**
   - Clear, actionable error messages
   - Convergence diagnostics
   - Fallback correlation suggestions

3. **Robustness**
   - No division by zero errors
   - NaN propagation prevention
   - Graceful degradation for out-of-range values

## API Design Philosophy

1. **Consistent Patterns**
   - `calculate_traverse()` for main solver
   - Separate `get_fluid_properties()` for PVT queries
   - Correlation-specific functions following naming conventions

2. **Documentation-First**
   - NumPy docstring standard
   - Parameter types clearly documented
   - Returns and raises explicitly declared
   - Example usage in docstrings

3. **Extensibility**
   - Configuration-based model selection
   - Plugin hook for custom correlations
   - Property setters for runtime adjustments

## Next Steps

1. Architecture document (ARCHITECTURE.md) - module breakdown, class designs
2. Story decomposition with acceptance criteria
3. Database entries for tracking implementation
4. GitHub issues for each story
5. Ready for Forge implementation