"""Unit tests for Fluid Model."""

import enum
import pytest
from unittest.mock import Mock

from perf_pressure_traverse.models.fluidmodel import (
    FluidModel,
    FluidModelFactory,
    FluidType
)
from perf_pressure_traverse.models.fluid import FluidProperties


class TestFluidType(enum.Enum):
    """Alias for FluidType from FluidModel."""
    ...


class TestFluidModel:
    """Tests for FluidModel class."""
