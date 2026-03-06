# perf-pressure-traverse

Pressure traverse calculation library for gas and liquid wells. This library implements industry-standard pressure traverse calculations according to API Recommended Practice 14A (RPI).

## Features

- **Pressure Traverse Calculations**: Surface-to-bottom and bottom-to-surface traverses
- **Flow Correlations**: Beggs-Brill, Hagedorn-Brown, and Gray correlations
- **PVT Models**: Vasquez-Beggs, Standing-Katz, AGA-8 Z-factor correlations
- **Flow Regime Identification**: Automatic regime detection with transition maps
- **Numerical Solvers**: Newton-Raphson iteration for convergence
- **Unit Conversion**: Complete unit handling with psi, ft, and R conversions

## Installation

### From PyPI (Coming Soon)

```bash
pip install perf-pressure-traverse
```

### From Source

```bash
git clone https://github.com/jarvanclaw-dev/perf-pressure-traverse.git
cd perf-pressure-traverse
pip install -e .
```

### Docker

```bash
docker build -t perf-pressure-traverse:latest .
docker run -it perf-pressure-traverse:latest
```

## Quick Start

```python
from perf_pressure_traverse import PressureTraverseSolver
from perf_pressure_traverse.models.fluid import FluidProperties
from perf_pressure_traverse.models.well import WellGeometry

# Create fluid properties
fluid = FluidProperties.from_surface_conditions(
    surface_pressure=200.0,  # psi
    surface_temperature=100.0,  # °F
    oil_specific_gravity=0.85,
    gas_specific_gravity=0.65,
    gas_oil_ratio=500.0,
    solution_gas_ratio=200.0,
    water_cut=0.3
)

# Create well geometry (simplified for example)
well = WellGeometry.from_wellhead_data(deviation_table=None)

# Calculate traverse
solver = PressureTraverseSolver()
result = solver.calculate_traverse(
    surface_pressure=450.0,  # psi
    fluid=fluid,
    well=well
)

print(f"Bottomhole pressure: {result.bottomhole_pressure:.2f} psi")
```

## Documentation

- [API Reference](docs/api/)
- [Design Document](DESIGN.md)
- [Architecture](ARCHITECTURE.md)

## Testing

```bash
pytest tests/ -v
pytest tests/ -cov=perf_pressure_traverse --cov-report=html
pytest tests/ --mypy
```

## Project Status

This project is in active development. Currently implementing stories according to Jira.

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please read our contributing guidelines.

## Authors

- Jarvan Claw

## References

- API Recommended Practice 14A (1976)
- "Multiphase Flow in Pipes" - Beggs & Brill
- Standing, Kattan & Katz (1977) Z-factor correlations
