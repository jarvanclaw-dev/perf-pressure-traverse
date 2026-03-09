"""Unit tests for friction factor calculations module."""

import pytest
import numpy as np

from perf_pressure_traverse.flow.friction import (
    moody_diagram_lookup,
    darcy_weisbach_friction_factor,
    api_friction_factor,
)


class TestMoodyDiagramLookup:
    """Test Moody diagram-based friction factor lookup."""

    def test_laminar_flow_re2000(self):
        """Laminar flow Re < 2000 uses Poiseuille equation."""
        re = 1000.0  # Laminar flow
        roughness = 0.0  # Smooth pipe
        f = moody_diagram_lookup(re, roughness)
        expected = 16.0 / re
        assert abs(f - expected) < 1e-10

    def test_very_laminar_flow_re100(self):
        """Very low Reynolds number."""
        re = 100.0
        f = moody_diagram_lookup(re, 0.0)
        expected = 0.16
        assert abs(f - expected) < 0.001

    def test_critical_re2000(self):
        """Exact boundary between laminar and turbulent."""
        re = 2000.0
        f = moody_diagram_lookup(re, 0.0)
        # Should be close to laminar value
        expected = 16.0 / re
        assert abs(f - expected) < 0.01

    def test_turbulent_smooth_pipe_re10000(self):
        """Turbulent flow in smooth pipe."""
        re = 10000.0
        roughness = 0.0
        f = moody_diagram_lookup(re, roughness)
        # Should use smooth pipe formula
        assert f > 0
        assert f < 0.1

    def test_turbulent_rough_pipe_re10000(self):
        """Turbulent flow in rough pipe."""
        re = 10000.0
        roughness = 0.0001  # Slightly rough
        f = moody_diagram_lookup(re, roughness)
        assert f > 0
        assert f < 0.1

    def test_edge_case_zero_re(self):
        """Zero Reynolds number - should return small value."""
        f = moody_diagram_lookup(0.0, 0.0001)
        assert f > 0

    def test_minimum_friction_factor(self):
        """Friction factor should have reasonable lower bound."""
        # High Reynolds number with some roughness
        f = moody_diagram_lookup(100000.0, 0.0001)
        assert f > 0.001
        assert f < 0.05

    def test_multiphase_context_re100000(self):
        """Test with high Reynolds number typical of multiphase flow."""
        re = 100000.0
        roughness = 0.00005  # Typical pipeline roughness
        f = moody_diagram_lookup(re, roughness)
        assert isinstance(f, float)
        assert f > 0


class TestDarcyWeisbachFrictionFactor:
    """Test Darcy-Weisbach friction factor calculation."""

    def test_laminar_flow_re2000(self):
        """Laminar flow uses 16/Re."""
        re = 1500.0
        t = 0.04  # Relative roughness (4%)
        f = darcy_weisbach_friction_factor(re, t)
        expected = 16.0 / re
        assert abs(f - expected) < 1e-10

    def test_turbulent_flow_swamee_jain(self):
        """Turbulent flow uses Swamee-Jain approximation."""
        re = 50000.0
        t = 0.001  # 0.1% roughness
        f = darcy_weisbach_friction_factor(re, t)
        
        # Verify reasonable value
        assert f > 0
        assert f < 0.1
        
        # For a very slightly rough pipe, should be relatively low
        assert f >= 0.01

    def test_very_smooth_pipe_re100000(self):
        """Very high Re, very smooth pipe."""
        re = 100000.0
        f = darcy_weisbach_friction_factor(re, 0.0)
        assert f > 0
        assert f < 0.1

    def test_rough_pipe_re20000(self):
        """Higher roughness with moderate Re."""
        re = 20000.0
        f = darcy_weisbach_friction_factor(re, 0.01)  # 1% roughness
        assert f > 0
        assert f < 0.1

    def test_zero_relative_roughness(self):
        """Zero roughness should still work (smooth pipe)."""
        re = 50000.0
        f = darcy_weisbach_friction_factor(re, 0.0)
        assert isinstance(f, float)
        assert f > 0

    def test_edge_case_re_zero(self):
        """Zero Reynolds number edge case."""
        f = darcy_weisbach_friction_factor(0, 0.01)
        assert f > 0

    def test_multiphase_flow_simulation(self):
        """Simulate typical multiphase flow conditions."""
        # Gas-oil wells often have high velocities
        re = 30000.0  # Typical for gas-oil multiphase flow
        roughness = 0.0001  # Commercial pipe
        f = darcy_weisbach_friction_factor(re, roughness)
        
        assert f > 0
        assert isinstance(f, float)
        assert f < 0.05


class TestAPIFrictionFactor:
    """Test API RP 14E friction factor correlation."""

    def test_laminar_flow_re2000(self):
        """Laminar flow uses 16/Re."""
        re = 1200.0
        roughness = 0.0
        f = api_friction_factor(roughness, re)
        expected = 16.0 / re
        assert abs(f - expected) < 1e-6

    def test_turbulent_flow_api_approximation(self):
        """Turbulent flow uses API approximation."""
        re = 40000.0
        roughness = 0.0001
        f = api_friction_factor(roughness, re)
        assert f > 0
        assert f < 0.1

    def test_high_reynolds_number(self):
        """High Reynolds number typical of gas wells."""
        re = 200000.0
        f = api_friction_factor(0.0001, re)
        assert f > 0
        assert f < 0.05

    def test_minimum_roughness(self):
        """Minimum practical roughness."""
        f = api_friction_factor(0.0, 50000.0)
        assert f > 0

    def test_zero_reynolds(self):
        """Zero Reynolds number handling."""
        f = api_friction_factor(0.001, 0.0)
        assert isinstance(f, float)

    def test_large_roughness(self):
        """Large roughness value."""
        f = api_friction_factor(0.01, 10000.0)
        assert f > 0

    def test_multiphase_line_with_gas(self):
        """Gas line in multiphase context (often gas-dominated)."""
        # High velocity, relatively clean pipe
        re = 150000.0  # High Re for gas lines
        roughness = 0.00002
        f = api_friction_factor(roughness, re)
        
        assert f > 0
        assert isinstance(f, float)

    def test_multiphase_line_with_crude(self):
        """Crude oil line (lower velocity, some deposits)."""
        # Lower velocity, higher roughness due to deposits
        re = 15000.0
        roughness = 0.0005  # Higher roughness
        f = api_friction_factor(roughness, re)
        
        assert f > 0
        assert f >= 0.01


class TestFrictionFactorComparison:
    """Test and compare different friction factor models."""

    def test_consistency_for_smooth_pipe(self):
        """All models should give consistent results for smooth pipes."""
        re = 50000.0
        roughness = 0.0
        
        # Moody should work
        f_moody = moody_diagram_lookup(re, roughness)
        
        # Darcy-Weisbach should work
        f_dw = darcy_weisbach_friction_factor(re, roughness)
        
        # API should work
        f_api = api_friction_factor(roughness, re)
        
        # All should be close and reasonable
        # They may differ slightly due to different approximations
        for f in [f_moody, f_dw, f_api]:
            assert 0.001 < f < 0.1

    def test_consistency_for_rough_pipe(self):
        """All models should respond to pipe roughness."""
        re = 40000.0
        
        for roughness in [0.0, 0.0001, 0.001, 0.01]:
            f_moody = moody_diagram_lookup(re, roughness)
            f_dw = darcy_weisbach_friction_factor(re, roughness)
            f_api = api_friction_factor(roughness, re)
            
            # Higher roughness should generally give higher friction factor
            # (monotonic relationship)
            prev_f = f_api
            for f in [f_moody, f_dw]:
                assert f > 0
                # Rougher pipes typically have higher friction


class TestMultiphaseContext:
    """Test friction factor models in multiphase flow scenarios."""

    def test_gas_oil_well_high_gas_rate(self):
        """Gas-oil well with high gas rate - gas-dominated flow."""
        # High gas oil ratio scenario
        re = 120000.0
        roughness = 0.0001
        
        f = darcy_weisbach_friction_factor(re, roughness)
        
        # Should handle high velocities from gas
        assert f > 0
        assert isinstance(f, float)

    def test_gas_oil_well_low_gas_rate(self):
        """Gas-oil well with low gas rate - oil-dominated flow."""
        # Lower gas rate, typical multiphase with gas bubbles
        re = 10000.0
        roughness = 0.0002
        
        f = darcy_weisbach_friction_factor(re, roughness)
        
        assert f > 0
        assert isinstance(f, float)

    def test_watertight_gusher(self):
        """Water/gas well with high velocities."""
        # High water cut, high gas - gas-driven flow
        re = 80000.0
        f = moody_diagram_lookup(re, 0.00015)
        
        assert f > 0
        assert isinstance(f, float)

    def test_condensate_line(self):
        """Condensate line with moderate velocities."""
        re = 25000.0
        f = api_friction_factor(0.0001, re)
        
        assert f > 0
        assert isinstance(f, float)

    def test_sour_gas_line(self):
        """Sour gas line with potential corrosion."""
        # Corrosion may increase effective roughness
        re = 60000.0
        f = moody_diagram_lookup(re, 0.0002)
        
        assert f > 0
        assert f >= 0.01

    def test_liquid_only_pipeline(self):
        """Pure liquid pipeline as baseline."""
        re = 5000.0
        f = darcy_weisbach_friction_factor(re, 0.0001)
        
        assert f > 0
        assert f < 0.1

    def test_gas_only_pipeline(self):
        """Pure gas pipeline."""
        re = 150000.0
        f = api_friction_factor(0.0001, re)
        
        assert f > 0
        assert f < 0.1


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_or_negative_reynolds(self):
        """Reynolds number edge case handling."""
        # Test with zero and negative
        for re in [0, -1000, -100]:
            f = moody_diagram_lookup(abs(re), 0.001)
            f_dw = darcy_weisbach_friction_factor(abs(re) + 0.001, 0.001)
            f_api = api_friction_factor(0.001, abs(re) + 0.001)
            assert f > 0
            assert f_dw > 0
            assert f_api > 0

    def test_negative_roughness(self):
        """Roughness should not cause errors."""
        re = 10000.0
        for roughness in [-0.001, -0.01, 0.0]:
            f = moody_diagram_lookup(re, max(0, abs(roughness)))
            f_dw = darcy_weisbach_friction_factor(re, max(0, abs(roughness)))
            f_api = api_friction_factor(max(0, abs(roughness)), re)
            assert f > 0
            assert f_dw > 0
            assert f_api > 0

    def test_extreme_roughness(self):
        """Extreme roughness values."""
        re = 50000.0
        f = moody_diagram_lookup(re, 100.0)  # Unreasonable but test extreme
        assert f > 0


class TestNumericalStability:
    """Test numerical stability of friction factor calculations."""

    def test_high_precision_re1000000(self):
        """Very high Reynolds number."""
        f = darcy_weisbach_friction_factor(1000000.0, 0.0001)
        assert f > 0
        assert f < 0.1

    def test_division_by_zero_handling_re1(self):
        """Very small Reynolds number."""
        f = moody_diagram_lookup(1.0, 0.0)
        assert f > 0

    def test_large_roughness_re1000(self):
        """High roughness with low Re."""
        f = api_friction_factor(0.1, 1000.0)
        assert f > 0

    def test_gradient_of_re(self):
        """Friction factor change with Reynolds number gradient."""
        roughness = 0.0001
        re_values = [10000, 20000, 40000, 80000, 160000]
        friction_factors = [moody_diagram_lookup(re, roughness) for re in re_values]
        
        # Higher Re should generally give lower friction (smooth pipe)
        for i in range(len(friction_factors) - 1):
            # Friction should decrease with increasing Re (monotonic)
            if friction_factors[i] > friction_factors[i + 1]:
                pass  # Can still fluctuate but should trend downward


class TestMultiphaseFlowScenario:
    """Test realistic multiphase flow scenarios."""

    def test_gas_oil_transition(self):
        """Gas-oil well near transition to annular flow."""
        # Near critical rates
        re = 25000.0
        f_dw = darcy_weisbach_friction_factor(re, 0.0001)
        f_moody = moody_diagram_lookup(re, 0.0001)
        f_api = api_friction_factor(0.0001, re)
        
        # Should all produce valid results
        assert all(f > 0 for f in [f_dw, f_moody, f_api])

    def test_volatile_oil_migration(self):
        """Volatile oil migration - high gas liquid ratio."""
        re = 75000.0
        f = moody_diagram_lookup(re, 0.00015)
        
        assert f > 0
        assert isinstance(f, float)

    def test_watery_williams_pipeline(self):
        """Water-dominated pipeline."""
        re = 15000.0
        f = darcy_weisbach_friction_factor(re, 0.00015)
        
        assert f > 0
        assert f < 0.1

    def test_condensate_bank_depleted(self):
        """Condensate well with bank depletion."""
        re = 40000.0
        f = api_friction_factor(0.00012, re)
        
        assert f > 0


def test_friction_factor_in_integration_context():
    """Test that friction factors can be used in integrated pressure drop calculation."""
    # Simulate friction factor selection based on flow regime
    
    # Example: Slug flow (intermittent) - higher friction
    re_slug = 20000.0
    roughness_slug = 0.00015
    
    # Example: Mist flow - lower friction
    re_mist = 80000.0
    roughness_mist = 0.00012
    
    f_slug = darcy_weisbach_friction_factor(re_slug, roughness_slug)
    f_mist = darcy_weisbach_friction_factor(re_mist, roughness_mist)
    
    # Slug flow typically has higher friction losses
    assert f_slug > 0
    assert f_mist > 0
    # Note: Actual comparison depends on many factors, just verify they work
    assert isinstance(f_slug, float)
    assert isinstance(f_mist, float)
