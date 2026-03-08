"""Flow correlation models module.

Placeholder for flow correlation implementations.
"""

from perf_pressure_traverse.flow.regime import FlowRegime, identify_regime_BeggsBrill, calculate_F_Lo


class CorrelationSelector:
    """Selector for flow correlation methods."""

    @staticmethod
    def select_correlation(fluid, well) -> str:
        """Choose appropriate correlation based on context.

        Args:
            fluid: FluidProperties instance
            well: WellGeometry instance

        Returns:
            Correlation name (e.g., 'beggs_brill')
        """
        return "beggs_brill"


__all__ = ["CorrelationSelector", "FlowRegime", "identify_regime_BeggsBrill", "calculate_F_Lo"]
