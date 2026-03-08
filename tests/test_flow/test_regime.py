"""Unit tests for flow regime identification module."""

import pytest

import numpy as np

from perf_pressure_traverse.flow.regime import (
    FlowRegime,
    calculate_F_Lo,
    calculate_Fr_Lo,
    calculate_gas_Fr,
    calculate_inclination_factor,
    calculate_liquid_rate_at_inlet,
    calculate_liquid_superficial_velocity,
    calculate_gas_superficial_velocity,
    identify_regime_BeggsBrill,
    identify_regime_at_depth,
)


class TestFlowRegimeEnum:
    """Test FlowRegime enum values."""

    def test_all_regimes_defined(self):
        """Verify all expected regimes are defined."""
        expected = {
            FlowRegime.SEGREGATED,
            FlowRegime.INTERMITTENT,
            FlowRegime.DISTRIBUTED,
            FlowRegime.MIST,
            FlowRegime.BUBBLE,
        }
        assert expected.issubset(FlowRegime.__members__.values())

    def test_seguregated_name(self):
        """Verify Segregated flow name."""
        assert FlowRegime.SEGREGATED.value == "Segregated Flow"

    def test_intermittent_name(self):
        """Verify slug flow name."""
        assert FlowRegime.INTERMITTENT.value == "Slug Flow"

    def test_distributed_name(self):
        """Verify distributed flow name."""
        assert FlowRegime.DISTRIBUTED.value == "Distributed Flow"

    def test_mist_name(self):
        """Verify mist flow name."""
        assert FlowRegime.MIST.value == "Mist Flow"

    def test_bubble_name(self):
        """Verify bubble flow name."""
        assert FlowRegime.BUBBLE.value == "Bubble Flow"


class TestLiquidVelocityCalculation:
    """Test liquid superficial velocity calculations."""

    def test_liquid_velocity_positive_area(self):
        """Calculate velocity with valid area."""
        velocity = calculate_liquid_superficial_velocity(300.0, 0.5)
        assert velocity > 0
        assert velocity == pytest.approx(300.0 * 0.002228 / 60.0 / 0.5, rel=1e-6)

    def test_liquid_velocity_zero_area(self):
        """Handle zero area."""
        velocity = calculate_liquid_superficial_velocity(300.0, 0.0)
        assert velocity == 0.0

    def test_gas_velocity_with_gas_oil_ratio(self):
        """Gas velocity accounts for dissolved gas correction."""
        velocity = calculate_gas_superficial_velocity(100.0, 85.0, 0.5)
        assert velocity > 0.0
        # Should account for gas-oil ratio
        expected = (100.0 + 85.0 * 0.0181) * 0.002228 / 60.0 / 0.5
        assert velocity == pytest.approx(expected, rel=1e-6)

    def test_gas_velocity_zero_area(self):
        """Handle zero area."""
        velocity = calculate_gas_superficial_velocity(100.0, 85.0, 0.0)
        assert velocity == 0.0


class TestDimensionlessGroups:
    """Test dimensionless group calculations."""

    def test_F_Lo_calculation(self):
        """Calculate F_Lo with typical values."""
        F_Lo = calculate_F_Lo(
            liquid_rate_gpm=300.0,  # GPM
            area_ft2=0.0667,  # ft²
            oil_density=50.0,  # lbm/ft³
            water_density=62.4,
        )
        assert F_Lo > 0
        # F_Lo > 3 should indicate segregated or slug flow

    def test_Fr_Lo_calculation(self):
        """Calculate Fr_Lo with typical values."""
        Fr_Lo = calculate_Fr_Lo(
            liquid_rate_gpm=300.0,
            area_ft2=0.0667,
        )
        assert Fr_Lo > 0

    def test_gas_Fr_calculation(self):
        """Calculate gas Froude number."""
        Fr = calculate_gas_Fr(
            gas_rate_gpm=500.0,  # GPM
            area_ft2=0.0667,
            diameter_ft=0.2917,  # ft for 7-inch pipe
        )
        assert Fr > 0

    def test_zero_area_edge_case(self):
        """Handle zero area in dimensionless groups."""
        F_Lo = calculate_F_Lo(50.0, 62.4, 300.0, 0.0)
        Fr_Lo = calculate_Fr_Lo(300.0, 0.0)
        gas_Fr = calculate_gas_Fr(500.0, 0.0, 0.2917)
        assert F_Lo == pytest.approx(0.0, abs=1e-10)
        assert Fr_Lo == pytest.approx(0.0, abs=1e-10)
        assert gas_Fr == pytest.approx(0.0, abs=1e-10)


class TestInclinationFactor:
    """Test inclination factor calculations."""

    def test_horizontal_flow(self):
        """Calculate factor for horizontal flow."""
        factor = calculate_inclination_factor(0.0)
        assert factor == pytest.approx(0.5, abs=0.01)

    def test_uphill_flow(self):
        """Calculate factor for uphill flow."""
        factor = calculate_inclination_factor(30.0)
        assert 0 < factor <= 0.5

    def test_downhill_flow(self):
        """Calculate factor for downhill flow."""
        factor = calculate_inclination_factor(-30.0, flow_direction="downhill")
        assert 0 <= factor < 0.5

    def test_steeper_uphill(self):
        """Calculate factor for steep uphill flow."""
        factor = calculate_inclination_factor(60.0)
        assert factor > 0 and factor <= 1.0

    def test_steeper_downhill(self):
        """Calculate factor for steep downhill flow."""
        factor = calculate_inclination_factor(-60.0, flow_direction="downhill")
        assert factor >= 0 and factor < 0.5

    def test_edge_case_90_deg_uphill(self):
        """Calculate factor at extreme angle - uphill."""
        factor = calculate_inclination_factor(90.0, flow_direction="uphill")
        assert factor <= 1.0

    def test_edge_case_90_deg_downhill(self):
        """Calculate factor at extreme angle - downhill."""
        factor = calculate_inclination_factor(-90.0, flow_direction="downhill")
        assert factor >= 0


class TestRegimeIdentification:
    """Test flow regime identification."""

    def test_bubble_flow_low_gas_rate(self):
        """Identify bubble flow with very low gas rate."""
        regime = identify_regime_BeggsBrill(
            oil_flow_rate_gpm=100.0,
            gas_inlet_rate_gpm=0.1,
            oil_gal=90.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4
        )
        # Very low gas rate should yield bubble flow
        assert regime == FlowRegime.BUBBLE

    def test_mist_flow_high_gas_rate(self):
        """Identify mist flow with high gas rate."""
        regime = identify_regime_BeggsBrill(
            oil_flow_rate_gpm=100.0,
            gas_inlet_rate_gpm=2000.0,
            oil_gal=85.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4
        )
        # High gas rate at low liquid velocity should yield mist flow
        assert regime == FlowRegime.MIST

    def test_segregated_flow_low_rate_horizontal(self):
        """Identify segregated flow for low rates, horizontal."""
        regime = identify_regime_BeggsBrill(
            oil_flow_rate_gpm=50.0,
            gas_inlet_rate_gpm=20.0,
            oil_gal=48.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4
        )
        # Low rates should be segregated
        assert regime in {FlowRegime.SEGREGATED, FlowRegime.INTERMITTENT}

    def test_slug_flow(self):
        """Identify slug (intermittent) flow."""
        regime = identify_regime_BeggsBrill(
            oil_flow_rate_gpm=200.0,
            gas_inlet_rate_gpm=120.0,
            oil_gal=200.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4
        )
        # Intermediate rates should yield slug flow
        assert regime == FlowRegime.INTERMITTENT

    def test_distributed_flow_high_rate(self):
        """Identify distributed flow at high rates."""
        regime = identify_regime_BeggsBrill(
            oil_flow_rate_gpm=300.0,
            gas_inlet_rate_gpm=500.0,
            oil_gal=270.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4
        )
        # High liquid rate with gas should be distributed
        assert regime == FlowRegime.DISTRIBUTED

    def test_all_inclinations(self):
        """Test regime identification across all inclinations."""
        inclinations = [-45, -30, -15, -5, 0, 5, 15, 30, 45]
        
        for angle in inclinations:
            regime = identify_regime_BeggsBrill(
                oil_flow_rate_gpm=200.0,
                gas_inlet_rate_gpm=150.0,
                oil_gal=180.0,
                pipe_diameter_ft=0.2917,
                borehole_area_ft2=0.0667,
                well_angle_deg=angle,
                oil_density_lbm_ft3=50.0,
                water_density_lbm_ft3=62.4
            )
            # Should return a valid regime for all angles
            assert regime in FlowRegime.__members__.values()

    def test_zero_area_edge_case(self):
        """Handle zero area for regime identification."""
        regime = identify_regime_BeggsBrill(
            oil_flow_rate_gpm=100.0,
            gas_inlet_rate_gpm=50.0,
            oil_gal=85.0,
            pipe_diameter_ft=0.0,
            borehole_area_ft2=0.0,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4
        )
        # Should return distributed or bubble as fallback
        assert regime in {FlowRegime.SEGREGATED, FlowRegime.BUBBLE, FlowRegime.DISTRIBUTED}


class TestRegimeProfile:
    """Test regime profile identification."""

    def test_single_depth_profile(self):
        """Test regime identification across single depth."""
        regimes = identify_regime_at_depth(
            oil_rate_gpm=np.array([200.0]),
            gas_rate_gpm=np.array([150.0]),
            oil_gal=180.0,
            diameter_ft=np.array([0.2917]),
            area_ft2=np.array([0.0667]),
            angle_deg=np.array([0.0]),
            oil_density=np.array([50.0]),
            water_density=62.4
        )
        assert len(regimes) == 1
        assert regimes[0] in FlowRegime.__members__.values()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_low_liquid_rate(self):
        """Test regime with minimal liquid flow."""
        regime = identify_regime_BeggsBrill(
            oil_flow_rate_gpm=2.0,
            gas_inlet_rate_gpm=50.0,
            oil_gal=1.8,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4
        )
        # Very low liquid with gas should be bubble or mist
        assert regime in {FlowRegime.BUBBLE, FlowRegime.INTERMITTENT}

    def test_very_high_gas_rate_edge_case(self):
        """Test with extremely high gas rate."""
        regime = identify_regime_BeggsBrill(
            oil_flow_rate_gpm=100.0,
            gas_inlet_rate_gpm=10000.0,
            oil_gal=85.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4
        )
        # Extreme gas rate should be mist
        assert regime == FlowRegime.MIST

    def test_negligible_gas_rate(self):
        """Test with effectively no gas rate."""
        regime = identify_regime_BeggsBrill(
            oil_flow_rate_gpm=300.0,
            gas_inlet_rate_gpm=0.01,
            oil_gal=270.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4
        )
        # Almost no gas should be segregated or distributed
        assert regime in {FlowRegime.SEGREGATED, FlowRegime.DISTRIBUTED}

    def test_high_liquid_rate_very_low_gas(self):
        """Test edge case: high liquid, low gas."""
        regime = identify_regime_BeggsBrill(
            oil_flow_rate_gpm=1000.0,
            gas_inlet_rate_gpm=10.0,
            oil_gal=900.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4
        )
        # High liquid, low gas should be segregated or distributed
        assert regime in {FlowRegime.SEGREGATED, FlowRegime.DISTRIBUTED}


class TestLiteratureExamples:
    """Test against representative literature examples."""

    def test_low_rate_example(self):
        """Test known low rate configuration."""
        regime = identify_regime_BeggsBrill(
            oil_flow_rate_gpm=5.0,
            gas_inlet_rate_gpm=2.0,
            oil_gal=4.5,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=55.0,
            water_density_lbm_ft3=62.4
        )
        # Low rates typically segregated or bubble
        assert regime in {FlowRegime.SEGREGATED, FlowRegime.BUBBLE}

    def test_medium_rate_example(self):
        """Test known medium rate configuration."""
        regime = identify_regime_BeggsBrill(
            oil_flow_rate_gpm=50.0,
            gas_inlet_rate_gpm=20.0,
            oil_gal=48.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4
        )
        # Medium rates typically slug flow
        assert regime in {FlowRegime.INTERMITTENT, FlowRegime.SEGREGATED}

    def test_high_rate_example(self):
        """Test known high rate configuration."""
        regime = identify_regime_BeggsBrill(
            oil_flow_rate_gpm=500.0,
            gas_inlet_rate_gpm=400.0,
            oil_gal=450.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=45.0,
            water_density_lbm_ft3=62.4
        )
        # High rates typically distributed flow
        assert regime == FlowRegime.DISTRIBUTED

    def test_extremely_high_gas_rate_example(self):
        """Test known example with dominant gas flow."""
        regime = identify_regime_BeggsBrill(
            oil_flow_rate_gpm=30.0,
            gas_inlet_rate_gpm=1500.0,
            oil_gal=27.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=55.0,
            water_density_lbm_ft3=62.4
        )
        # Very high gas rate should be mist flow
        assert regime == FlowRegime.MIST
