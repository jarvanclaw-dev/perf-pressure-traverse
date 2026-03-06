# Pressure Traverse Calculation Library - Architecture Document

## System Overview

The `perf-pressure-traverse` library calculates pressure traverses for gas and liquid wells using industry-standard methods from API Recommended Practice 14A (RPI). The architecture is designed for production readiness with modularity, extensibility, and numerical stability.

## Technology Stack

- **Language:** Python 3.10+
- **Core Libraries:** NumPy, SciPy, pandas
- **Testing:** pytest, pytest-cov, numpy-testing
- **Documentation:** Sphinx, numpydoc
- **Containerization:** Docker, docker-compose

## Directory Structure

```
perf_pressure_traverse/
├── perf_pressure_traverse/          # Top-level package
│   ├── __init__.py
│   ├── __version__.py            # Version management
│   ├── core/                    # Core calculation engine
│   │   ├── __init__.py
│   │   ├── pressure_traverse.py
│   │   └── units.py
│   ├── models/                  # Domain models and data structures
│   │   ├── __init__.py
│   │   ├── fluid.py
│   │   ├── well.py
│   │   └── pvt.py
│   ├── flow/                    # Flow correlation models
│   │   ├── __init__.py
│   │   ├── correlations.py
│   │   ├── regime.py
│   │   └── friction.py
│   ├── math/                    # Mathematical calculations
│   │   ├── __init__.py
│   │   ├── iterative.py
│   │   ├── z_factor.py
│   │   └── eos.py
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py
│   │   ├── validation.py
│   │   └── diagnostics.py
│   ├── api_rpi/                 # API RPI test cases
│   │   ├── __init__.py
│   │   └── test_cases.py
│   └── constants.py            # Physical constants
│
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── test_core/
│   ├── test_models/
│   ├── test_flow/
│   ├── test_math/
│   └── utils/
│
├── docs/                        # Documentation
│   ├── api/
│   ├── examples/
│   ├── conf.py
│   └── make.bat
│
├── docs/requirements.txt
├── setup.py
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── README.md
```

## Module Responsibilities

### Core Module (`core/`)

#### `pressure_traverse.py`
**Purpose:** Main pressure traverse calculation engine

**Classes:**
- `PressureTraverseSolver`: Primary solver class
  - `sweep_surface_to_bottom()`: Surface-to-bottom traverse
  - `sweep_bottom_to_surface()`: Bottom-to-surface traverse
  - `solve_step()`: Single step calculation with Newton-Raphson iteration
  - `convergence_check()`: Termination criteria

**Key Methods:**
```python
def calculate_traverse(
    surface_pressure: float,
    bottomhole_temperature_F: float,
    well_model: WellGeometry,
    fluid: FluidProperties,
    flow_correlation: str = "beggs_brill",
    depth_step_ft: float = 10.0
) -> PressureTraverseResult:
    """Calculate complete pressure traverse."""
```

**Data flow:**
1. Validate inputs
2. Initialize pressure and temperature arrays
3. Iterate through depth increments
4. For each segment:
   - Determine flow regime
   - Calculate PVT properties at current P, T
   - Determine liquid holdup
   - Calculate friction factor
   - Update pressure/temp using Newton-Raphson
   - Store results
5. Return aggregated result

#### `units.py`
**Purpose:** Unit conversion utilities

**Functions:**
- `psi_to_pascal()`
- `ft_to_meters()`
- `R_to_Rankine()`
- Consistent output in API units (ft, psi, °F)

### Models Module (`models/`)

#### `fluid.py`
**Purpose:** Black oil fluid property models

**Classes:**
- `FluidProperties`: Domain model
  - `from_surface_conditions()`: Create from basic surface data
  - `get_property_at_pt()`: Query property at any P,T

**Correlations:**
```python
def calculate_vapor_pressure(temp_F: float, specific_gravity_oil: float) -> float:
    """Standing vapor pressure correlation."""
    pass

def calculate_boiling_point_pressure(temp_F: float, specific_gravity_oil: float) -> float:
    """Calculate boiling point pressure using Glaso correlation."""
    pass

def calculate_formation_volume_factor(pressure_psia, temp_F, specific_gravity_oil, gas_oil_ratio: float) -> float:
    """Vasquez-Beggs FVF correlation."""
    pass

def calculate_oil_density(pressure_psia, temp_F, specific_gravity_oil, gas_oil_ratio: float, solution_gas_ratio: float) -> float:
    """Calculate oil density at reservoir conditions."""
    pass
```

#### `well.py`
**Purpose:** Wellbore geometry models

**Classes:**
- `WellGeometry`: Domain model
  - `from_wellhead_data()`: Parse wellhead data with deviation table
  - `interp_deviation()`: Get angle at any depth

**Properties:**
- `borehole_diameter_ft` orifice vs. annulus
- `casing_diameter_ft`
- `angle_vs_depth` (NDArray)
- `vertical_depth_vs_measured_depth` (NDArray)

#### `pvt.py`
**Purpose:** PVT calculation modules

**Z-Factor Models:**
```python
def standing_katz_z_factor(pressure_psia, temp_R) -> float:
    """Standing-Katz Z-factor chart interpolation."""
    pass

def aga_8_dc_gas_z_factor(**kwargs) -> float:
    """AGA-8 Natural Gas Equation of State (DC method)."""
    pass

def lee_gonzales_z_factor(**kwargs) -> float:
    """Lee-Gonzales Espana Z-factor correlation."""
    pass
```

**EOS Interfaces:**
```python
class EquationOfState(ABC):
    """Abstract base class for EOS implementations."""

    @abstractmethod
    def calculate_z_factor(self, components: List[Component], P: float, T: float) -> float:
        """Calculate compressibility factor."""
        pass
```

### Flow Module (`flow/`)

#### `correlations.py`
**Purpose:** Flow correlation implementations

**Correlation Selector:**
```python
class CorrelationSelector:
    def select_correlation(self, fluid: FluidProperties, well: WellGeometry) -> str:
        """Choose appropriate correlation based on context."""
        pass
```

**Beggs-Brill Implementation:**
```python
class BeggsBrillCorrelation:
    """Beggs-Brill (1973) multiphase flow correlation."""

    def get_friction_factor(self, reynolds_no: float, relative_roughness: float) -> float:
        """Fanning friction factor from Moody diagram."""
        pass

    def calculate_liquid_holdup(self, gas_liquid_ratio, liquid_velocity, inclination: float) -> float:
        """Calculate liquid holdup coefficient."""
        pass

    def calculate_pressure_drop(self, pressure, temp, flow_rates, geometry) -> dict:
        """Complete pressure drop calculation for a segment."""
```

**Other Correlations (to be implemented):**
- `HagedornBrownCorrelation`
- `GrayCorrelation`

#### `regime.py`
**Purpose:** Flow regime identification

**Flow Regime Types (Enum):**
```python
class FlowRegime(Enum):
    SEGREGATED = "Segregated Flow"
    INTERMITTENT = "Slug Flow"
    DISTRIBUTED = "Distributed Flow"
    MIST = "Mist Flow"
    BUBBLE = "Bubble Flow"
```

**Identification Methods:**
```python
def identify_regime_BeggsBrill(
    gas_liquid_ratio: float,
    liquid_velocity_ft_per_sec: float,
    gas_velocity: float,
    inclination_angle: float
) -> FlowRegime:
    """Beggs-Brill flow regime transition map."""
    pass

# Other regime determiners from other correlations
```

**Transition Maps:**
- Beggs-Brill: 584 experimental points
- Standard criteria: F_Lo, Fr_Lo, Fr, and inclination factors

#### `friction.py`
**Purpose:** Friction factor calculations

**Models:**
```python
def darcy_weisbach_friction_factor(reynolds_no: float, relative_roughness: float, flow_regime: FlowRegime) -> float:
    """Darcy-Weisbach friction factor for turbulent flow."""
    pass

def moody_diagram_lookup(re: float, roughness_ratio: float) -> float:
    """Direct lookup from Moody coefficients."""
    pass

def api_friction_factor(roughness: float, reynolds_no: float) -> float:
    """API standard friction factor correlation."""
    pass
```

### Math Module (`math/`)

#### `iterative.py`
**Purpose:** Numerical solver utilities

**Solvers:**
```python
def newton_raphson_solver(
    function: Callable[[float], float],
    derivative: Callable[[float], float],
    initial_guess: float,
    tolerance: float = 0.01,
    max_iterations: int = 50
) -> float:
    """Newton-Raphson numerical solution."""
    pass

def solve_pressure_step(
    pressure: float,
    depth_increment: float,
    function: Callable,
    gradient: Callable
) -> float:
    """Wrapper for unit-agnostic pressure solver."""
    pass
```

#### `z_factor.py`
**Purpose:** Z-factor correlation implementations

**Detailed Correlations:**
```python
def calculate_z_factor_aga_dc(**kwargs) -> float:
    """
    AGA-8 Natural Gas Equation of State (DC 8-1984)
    - Calculates Z-factor from composition, pressure, temperature
    """
    pass

def calculate_z_factor_standing_katz(pressure_psia, temp_R) -> float:
    """
    Standing-Katz Z-factor chart
    - Two-parameter interpolation (P/T)
    - For natural gas with known specific gravity
    """
    pass

def calculate_z_factor_lee_gonzales(**kwargs) -> float:
    """
    Lee-Gonzales Espana Z-factor correlation
    - Empirical correlation for natural gas
    - Based on pseudo-critical properties
    """
    pass
```

**Helper Functions:**
```python
def calculate_pseudocritical_properties(
    gas_specific_gravity: float,
    composition: Optional[Dict[str, float]] = None
) -> Tuple[float, float]:
    """Calculate critical pressure and temperature."""
    pass

def get_pseudocritical_temp(temp_F, specific_gravity_gas: float) -> float:
    """Stewart-Burke-Katz pseudocritical temperature."""
    pass

def get_pseudocritical_press(press_psia, specific_gravity_gas: float) -> float:
    """Stewart-Burke-Katz pseudocritical pressure."""
    pass
```

#### `eos.py`
**Purpose:** Equation of state solvers

**EOS Implementations:**
```python
class SRKEOS(EquationOfState):
    """Soave-Redlich-Kwong Equation of State."""

    def calculate_z_factor(self, temperature_K, pressure_Pa):
        pass

    def solve_cubics():
        pass

class PengRobinsonEOS(EquationOfState):
    """Peng-Robinson Equation of State."""

    def calculate_z_factor(self, temperature_K, pressure_Pa):
        pass
```

## Core Data Structures

### Domain Models

#### `models/fluid.py`
```python
@dataclass
class FluidProperties:
    """Black oil fluid properties."""
    surface_pressure_psia: float = 14.7
    surface_temperature_F: float = 60.0
    oil_specific_gravity: float = 0.85  # Specific gravity relative to water
    gas_specific_gravity: float = 0.65
    water_specific_gravity: float = 1.0
    gas_oil_ratio: float = 500.0
    solution_gas_ratio: float = 200.0
    water_cut: float = 0.3

    # Runtime properties (computed)
    @property
    def surface_oil_density_lbm_per_ft3(self) -> float:
        """Oil density at separator conditions."""
        pass

    @property
    def surface_gas_density_lbm_per_ft3(self) -> float:
        """Gas density at separator conditions."""
        pass

    def get_oil_fvf(self, pressure_psia: float) -> float:
        """Formation volume factor of oil."""
        pass

    def get_gas_fvf(self, pressure_psia: float, temp_R: float) -> float:
        """Formation volume factor of gas."""
        pass
```

#### `models/well.py`
```python
@dataclass
class WellGeometry:
    """Wellbore geometric model."""
    borehole_diameter_ft: float
    casing_diameter_ft: float
    deviation_angles: NDArray   # [m_idx, [angle_ft, angle_deg]]
    vertical_depth_vs_measured_depth: NDArray  # [m_idx, [measured_ft, vertical_ft]]

    @classmethod
    def from_wellhead_data(cls, deviation_table: NDArray) -> "WellGeometry":
        """Parse deviation table to create WellGeometry."""
        pass

    def get_deviation_at_depth(self, measured_depth_ft: float) -> float:
        """Get inclination angle at a specific depth."""
        pass

    def get_vertical_depth(self, measured_depth_ft: float) -> float:
        """Convert measured depth to vertical depth."""
        pass
```

#### `models/pvt.py`
```python
@dataclass
class PVTProperties:
    """PVT properties at reservoir conditions."""
    oil_density_lbm_per_ft3: float
    gas_density_lbm_per_ft3: float
    water_density_lbm_per_ft3: float
    gas_compressibility_factor: float
    oil_fvf: float
    gas_fvf: float
```

#### `core/pressure_traverse.py`
```python
@dataclass
class PressureTraverseResult:
    """Complete pressure traverse calculation result."""
    surface_pressure: float
    bottomhole_pressure: float
    pressure_profile: NDArray  # [depth_index, pressure_psia]
    temperature_profile: NDArray
    flow_regime_profile: NDArray
    liquid_holdup_profile: NDArray
    frictional_loss_profile: NDArray
    hydrostatic_loss_profile: NDArray
    convergence_message: str

    def to_dataframe(self) -> DataFrame:
        """Convert to pandas DataFrame for analysis."""
        pass
```

### Lookup Tables

#### `data/api_tests.py`
```python
class API_RPI_1976_TestCases:
    """
    API Recommended Practice 14A (1976) Test Cases

    Standard validation test cases with known correct answers:
    - Known surface and bottomhole conditions
    - Known pressure profiles
    - Known regimes and liquid holdup
    """
    @classmethod
    def get_test_case(cls, test_id: int) -> dict:
        """Extract specific test case."""
        pass

    @classmethod
    def run_validation(cls):
        """Run all test cases and verify accuracy."""
        pass
```

#### `data/lookup_tables.py`
```python
class ZFactorLookupTable:
    """
    Pre-computed Z-factor lookup data
    - Standing-Katz chart values
    - Coded as 2D array
    - Interpolation logic
    """
    _STANDING_KATZ_DATA: NDArray = ...

    @classmethod
    def interpolate(cls, pressure_psia, temp_R) -> float:
        """Linear interpolation from lookup table."""
        pass
```

## Error Handling Strategy

### Custom Exceptions

```python
class PressureTraverseError(Exception):
    """Base exception for pressure traverse errors."""
    pass

class InvalidParameterError(PressureTraverseError):
    """Raised when input parameters violate physical constraints."""
    pass

class ConvergenceError(PressureTraverseError):
    """Raised when Newton-Raphson iteration fails to converge."""
    pass

class NumericalStabilityError(PressureTraverseError):
    """Raised when calculations lead to numerical instability."""
    pass

class UnsupportedCorrelationError(PressureTraverseError):
    """Raised when requested correlation is not available."""
    pass
```

### Validation Layer

**`utils/validation.py`**
```python
class ParameterValidator:
    @staticmethod
    def validate_fluid_properties(fluid: FluidProperties):
        """Check all fluid properties are physical."""
        checks = [
            (fluid.surface_pressure_psia > 0, "Surface pressure must be positive"),
            (fluid.surface_temperature_F > -40, "Surface temperature too cold"),
            (0.5 <= fluid.oil_specific_gravity <= 1.1, "Oil specific gravity invalid"),
        ]
        for condition, msg in checks:
            if not condition:
                raise InvalidParameterError(msg)

    @staticmethod
    def validate_geometry(well: WellGeometry):
        """Check well geometry consistency."""
        pass

    @staticmethod
    def validate_correlation(correlation_name: str):
        """Check if correlation is supported."""
        supported = ["beggs_brill", "hagedorn_brown", "gray"]
        if correlation_name not in supported:
            raise UnsupportedCorrelationError(f"Correlation not supported: {correlation_name}")
```

### Diagnostics

**`utils/diagnostics.py`**
```python
class SolverDiagnostics:
    @staticmethod
    def convergence_report() -> dict:
        """Return convergence statistics."""
        return {
            "iterations_used": 23,
            "converged": True,
            "error_at_end": 0.35,
            "max_delta": 1.2
        }

    @staticmethod
    def regime_transitions():
        """Log flow regime transitions along depth."""
        return [10.1, 45.3, 89.7, 145.2, 198.5, 250.1]
```

## API Design

### Public API Patterns

```python
# Core solver
from perf_pressure_traverse import PressureTraverseSolver

solver = PressureTraverseSolver()
result = solver.calculate_traverse(surface_pressure=450.0, fluid=fluid, well=well)

# PVT queries
from perf_pressure_traverse.models.fluid import FluidProperties

f = FluidProperties.from_surface_conditions(
    surface_pressure=200.0,
    oil_temperature=100.0,
    ...
)
oil_fvf = f.get_oil_fvf(pressure=3000.0)

# Flow correlations
from perf_pressure_traverse.flow.beggs_brill import BeggsBrillCorrelation

correlation = BeggsBrillCorrelation()
friction_factor = correlation.get_friction_factor(re=5000, roughness=0.0001)

# Z-factor
from perf_pressure_traverse.math.z_factor import StandingKatz

z_factor = StandingKatz.calculate(pressure=3000, temp=520)
```

## Computational Workflow

### Pressure Traverse Algorithm

```
1. Input Validation
   ├─ Check fluid properties
   ├─ Check well geometry
   └─ Check convergence parameters

2. Initialization
   ├─ Set initial pressure (surface or BHP)
   ├─ Set initial temperature (surface or BHT)
   ├─ Create depth array from geometry
   └─ Allocate result arrays

3. Depth Loop (for each depth step)
   ├─ Get local deviation angle at current depth
   ├─ Calculate local geometry (area, equivalent diameter)
   ├─ Estimate PVT properties at current (P, T)
   │  ├─ Get reservoir temperature at depth (geothermal gradient)
   │  ├─ Call fluid property correlations
   │  └─ Call Z-factor correlation for gas
   ├─ Determine flow regime (regime.py)
   │  ├─ Check gas rate
   │  ├─ Check liquid velocity
   │  └─ Apply regime transition criteria
   ├─ Calculate flow correlation parameters
   │  ├─ Calculate Re
   │  ├─ Apply Beggs-Brill or other correlation
   │  └─ Compute friction factor
   ├─ Solve pressure for next depth step
   │  ├─ Apply Newton-Raphson to solve: P_next = F(P_current)
   │  ├─ Check convergence criteria
   │  ├─ Update P and T for next iteration
   └─ Store calculated segment results

4. Result Aggregation
   ├─ Collect all segment data
   ├─ Check for physical consistency
   ├─ Generate detailed diagnostics
   └─ Return structured result
```

### Numerical Considerations

1. **Convergence Criteria**
   - Absolute pressure change: |P_next - P_current| < 0.1 psi
   - Relative pressure change: < 0.01%
   - Maximum iterations: 50
   - Convergence failure = exception with diagnostics

2. **Depth Increment Size**
   - Default: 10 ft
   - Auto-adjust near regime transitions (5-20 ft)
   - User-configurable option

3. **Thermal Model**
   - Geothermal gradient: °F/1000 ft (standard values)
   - Heat transfer efficiency factor
   - Temperature at depth: T = T_surface + gradient * (depth-ft / 1000)

## Testing Strategy

### Unit Tests

**`tests/test_core/test_pressure_traverse.py`**
```python
def test_surface_to_bottom_traverse():
    """Standard API test case validation."""
    test_case = API_RPI_1976_TestCases.get_test_case(1)
    result = solver.calculate_traverse(
        surface_pressure=test_case["surface_pressure"],
        fluid=test_case["fluid"],
        well=test_case["well"]
    )
    assert abs(result.bottomhole_pressure - test_case["bhp"]) < 0.5 * test_case["tolerance"]
```

**`tests/test_models/test_fluid.py`**
```python
def test_vasquez_beggs_fvf():
    """Validate FVF correlation against known values."""
    assert approx_equal(fvf, expected_value)
```

**`tests/test_flow/test_correlations.py`**
```python
def test_beggs_brill_pressure_drop():
    """Compare Beggs-Brill against API test results."""
    ...
```

**`tests/test_math/test_iterative.py`**
```python
def test_newton_raphson_convergence():
    """Verify Newton-Raphson converges for known root."""
    root = newton_raphson_solver(shoulder, 0.1)
    assert abs(root - true_solution) < 1e-6
```

### Integration Tests

```python
def test_complete_workflow():
    """Full from start to finish end-to-end test."""
    ...
```

### Validation Tests

```python
def test_api_test_suite():
    """Run all API RPI test cases."""
    ...
```

## Deployment

### Docker Setup

**`Dockerfile`**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install --no-cache-dir -e .

CMD ["python", "-c", "from perf_pressure_traverse import PressureTraverseSolver; print('OK')"]
```

**`requirements.txt`**
```
numpy>=1.24.0
scipy>=1.11.0
pandas>=2.1.0
pytest>=7.4.0
pytest-cov>=4.1.0
sphinx>=7.2.0
numpydoc>=1.6.0
```

## Security & Quality

1. **Input Sanitization**
   - Type checking for all inputs
   - Range validation
   - Unit consistency checks

2. **Code Quality**
   - Pylint for static analysis
   - Black for formatting compliance
   - Flake8 for linting

3. **Test Coverage**
   - Minimum 90% code coverage
   - Critical paths 100%
   - Edge cases covered

## Next Steps

1. Create 15 focused stories with acceptance criteria
2. Assign story points (Fibonacci)
3. Define dependencies between stories
4. Forge implementation
5. Continuous integration and validation