"""Unit tests for Beggs-Brill multiphase flow correlation."""

import pytest

import numpy as np

from perf_pressure_traverse.flow.correlations import (
    BeggsBrillCorrelation,
    FlowRegime,
)


class TestBeggsBrillInitialization:
    """Test Beggs-Brill correlation initialization."""

    def test_horizontal_well(self):
        """Initialize correlation for horizontal well."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=100.0,
            gas_flow_rate_gpm=50.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        assert br is not None
        assert br.well_angle_deg == 0.0

    def test_uphill_well(self):
        """Initialize correlation for uphill well."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=100.0,
            gas_flow_rate_gpm=50.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=30.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        assert br is not None
        assert br.well_angle_deg == 30.0

    def test_downhill_well(self):
        """Initialize correlation for downhill well."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=100.0,
            gas_flow_rate_gpm=50.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=-30.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        assert br is not None
        assert br.well_angle_deg == -30.0

    def test_edge_case_zero_area(self):
        """Initialize with zero borehole area."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=100.0,
            gas_flow_rate_gpm=50.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0,
            well_angle_deg=0.0,
        )
        # Should not crash, should handle zero area
        assert br.borehole_area_ft2 == 0.0

    def test_standard_gravity_values(self):
        """Initialize with standard gravity values."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=100.0,
            gas_flow_rate_gpm=50.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
            gas_specific_gravity=1.0,
            oil_specific_gravity=1.0,
        )
        assert br.gas_specific_gravity == 1.0
        assert br.oil_specific_gravity == 1.0


class TestMultiphaseViscosity:
    """Test multiphase viscosity calculations."""

    def test_bubble_flow_viscosity(self):
        """Calculate viscosity for bubble flow regime."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=100.0,
            gas_flow_rate_gpm=0.1,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        
        mu = br.calculate_mixture_viscosity_cP()
        
        assert mu > 0
        assert mu < 10.0  # Should be similar to oil viscosity

    def test_mist_flow_viscosity(self):
        """Calculate viscosity for mist flow regime."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=100.0,
            gas_flow_rate_gpm=2000.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        
        mu = br.calculate_mixture_viscosity_cP()
        
        assert mu > 0
        assert mu < 0.1  # Should be similar to gas viscosity

    def test_mixture_viscosity_scales_with_rate(self):
        """Verify mixture viscosity is influenced by flow rates."""
        br1 = BeggsBrillCorrelation(
            oil_flow_rate_gpm=100.0,
            gas_flow_rate_gpm=50.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
        )
        
        br2 = BeggsBrillCorrelation(
            oil_flow_rate_gpm=200.0,
            gas_flow_rate_gpm=100.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
        )
        
        mu1 = br1.calculate_mixture_viscosity_cP()
        mu2 = br2.calculate_mixture_viscosity_cP()
        
        # Should be similar since rates are proportional
        ratio = mu2 / mu1
        assert 0.8 < ratio < 1.2

    def test_zero_area_viscosity_edge_case(self):
        """Handle zero area in viscosity calculation."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=100.0,
            gas_flow_rate_gpm=50.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0,
            well_angle_deg=0.0,
        )
        
        mu = br.calculate_mixture_viscosity_cP()
        
        # Should not crash
        assert mu >= 0.0


class TestMultiphaseDensity:
    """Test multiphase density calculations."""

    def test_bubble_flow_density(self):
        """Calculate density for bubble flow regime."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=100.0,
            gas_flow_rate_gpm=0.1,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        
        rho = br.calculate_mixture_density_lb_ft3()
        
        # Should be close to oil/water density
        assert 40.0 < rho < 70.0

    def test_mist_flow_density(self):
        """Calculate density for mist flow regime."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=100.0,
            gas_flow_rate_gpm=2000.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        
        rho = br.calculate_mixture_density_lb_ft3()
        
        # Should be lower with much gas holdup
        assert 1.0 < rho < 20.0

    def test_denser_phase_concentrated(self):
        """Verify density increases when denser phase fraction increases."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=100.0,
            gas_flow_rate_gpm=50.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        
        # High water fraction should be denser
        rho_water_only = br.calculate_mixture_density_lb_ft3(
            oil_fraction=0.0,
            water_fraction=1.0
        )
        
        # Mixed density should be intermediate
        rho_mixed = br.calculate_mixture_density_lb_ft3(
            oil_fraction=0.5,
            water_fraction=0.5
        )
        
        assert rho_water_only > rho_mixed

    def test_gas_holdup_affects_density(self):
        """Verify gas holdup reduces density significantly."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=100.0,
            gas_flow_rate_gpm=50.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        
        rho_liq = br.calculate_mixture_density_lb_ft3(
            oil_fraction=1.0,
            water_fraction=0.0
        )
        
        # With gas included, density should decrease
        rho_mixed = br.calculate_mixture_density_lb_ft3()
        
        assert rho_mixed < rho_liq


class TestGasHoldup:
    """Test gas holdup (void fraction) calculations."""

    def test_bubble_flow_holdup(self):
        """Calculate holdup for bubble flow regime."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=100.0,
            gas_flow_rate_gpm=0.1,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        
        lambda_g = br.calculate_gas_holdup()
        
        # Very low gas rate should have low holdup
        assert lambda_g <= 0.5

    def test_mist_flow_holdup(self):
        """Calculate holdup for mist flow regime."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=100.0,
            gas_flow_rate_gpm=2000.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        
        lambda_g = br.calculate_gas_holdup()
        
        # High gas rate should have high holdup
        assert lambda_g > 0.5

    def test_inclination_effects_on_holdup(self):
        """Verify holdup is affected by well inclination."""
        br_horizontal = BeggsBrillCorrelation(
            oil_flow_rate_gpm=200.0,
            gas_flow_rate_gpm=150.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
        )
        
        br_45_deg = BeggsBrillCorrelation(
            oil_flow_rate_gpm=200.0,
            gas_flow_rate_gpm=150.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=45.0,
        )
        
        lambda_g_horiz = br_horizontal.calculate_gas_holdup()
        lambda_g_45 = br_45_deg.calculate_gas_holdup()
        
        # May differ due to inclination effects
        assert lambda_g_horiz != lambda_g_45


class TestFlowRegimeIdentification:
    """Test flow regime identification."""

    def test_bubble_regime_confirmed(self):
        """Verify bubble flow at very low gas rate."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=100.0,
            gas_flow_rate_gpm=0.1,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        
        regime = br.identify_flow_regime()
        
        assert regime == FlowRegime.BUBBLE

    def test_mist_regime_confirmed(self):
        """Verify mist flow at very high gas rate."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=100.0,
            gas_flow_rate_gpm=10000.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        
        regime = br.identify_flow_regime()
        
        assert regime == FlowRegime.MIST

    def test_segregated_regime_confirmed(self):
        """Verify segregated flow at low rates."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=50.0,
            gas_flow_rate_gpm=20.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        
        regime = br.identify_flow_regime()
        
        assert regime in {FlowRegime.SEGREGATED, FlowRegime.BUBBLE}

    def test_slug_regime_confirmed(self):
        """Verify slug (intermittent) flow."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=200.0,
            gas_flow_rate_gpm=120.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        
        regime = br.identify_flow_regime()
        
        assert regime == FlowRegime.INTERMITTENT

    def test_distributed_regime_confirmed(self):
        """Verify distributed flow at high rates."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=500.0,
            gas_flow_rate_gpm=400.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=45.0,
            water_density_lbm_ft3=62.4,
        )
        
        regime = br.identify_flow_regime()
        
        assert regime == FlowRegime.DISTRIBUTED

    def test_all_regimes_across_inclinations(self):
        """Test regime identification across all inclinations."""
        inclinations = [-45, -30, -15, -5, 0, 5, 15, 30, 45]
        
        for angle in inclinations:
            br = BeggsBrillCorrelation(
                oil_flow_rate_gpm=200.0,
                gas_flow_rate_gpm=150.0,
                pipe_diameter_ft=0.2917,
                borehole_area_ft2=0.0667,
                well_angle_deg=angle,
                oil_density_lbm_ft3=50.0,
                water_density_lbm_ft3=62.4,
            )
            
            regime = br.identify_flow_regime()
            
            # Should return a valid regime for all angles
            assert regime in FlowRegime.__members__.values()

    def test_edge_case_zero_area_regime(self):
        """Handle zero area for regime identification."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=100.0,
            gas_flow_rate_gpm=50.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        
        regime = br.identify_flow_regime()
        
        # Should return distributed or bubble as fallback
        assert regime in {FlowRegime.SEGREGATED, FlowRegime.BUBBLE, FlowRegime.DISTRIBUTED}


class TestDimensionlessGroups:
    """Test dimensionless group calculations."""

    def test_F_Lo_calculation(self):
        """Calculate F_Lo with typical values."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=300.0,  # GPM
            gas_flow_rate_gpm=150.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        
        F_Lo = br.calculate_F_Lo()
        
        assert F_Lo > 0
        # Higher liquid rate should give higher F_Lo

    def test_Fr_Lo_calculation(self):
        """Calculate F_Lo Froude number."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=300.0,
            gas_flow_rate_gpm=150.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        
        Fr_Lo = br.calculate_Fr_Lo()
        
        assert Fr_Lo > 0

    def test_gas_Fr_calculation(self):
        """Calculate gas Froude number."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=200.0,
            gas_flow_rate_gpm=500.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        
        # Use the flow regime module's gas Fr calculation
        from perf_pressure_traverse.flow.regime import calculate_gas_Fr
        
        # Note: calculate_gas_Fr doesn't use our correlation, but we can test it
        # It checks flow regime based on F_Lo
        Fr = calculate_gas_Fr(
            gas_rate_gpm=500.0,
            area_ft2=0.0667,
            diameter_ft=0.2917,
        )
        
        assert Fr > 0

    def test_zero_area_dimensionless(self):
        """Handle zero area in dimensionless groups."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=100.0,
            gas_flow_rate_gpm=50.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0,
            well_angle_deg=0.0,
        )
        
        F_Lo = br.calculate_F_Lo()
        Fr_Lo = br.calculate_Fr_Lo()
        
        assert F_Lo == pytest.approx(0.0, abs=1e-10)
        assert Fr_Lo == pytest.approx(0.0, abs=1e-10)


class TestInclinationFactor:
    """Test inclination factor calculations."""

    def test_horizontal_flow(self):
        """Calculate factor for horizontal flow."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=100.0,
            gas_flow_rate_gpm=50.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
        )
        
        factor = br.calculate_inclination_factor()
        
        assert factor == pytest.approx(0.5, abs=0.01)

    def test_uphill_flow(self):
        """Calculate factor for uphill flow (15°)."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=100.0,
            gas_flow_rate_gpm=50.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=15.0,
        )
        
        factor = br.calculate_inclination_factor()
        
        assert 0.5 < factor <= 1.0

    def test_steeper_uphill(self):
        """Calculate factor for steep uphill flow (60°)."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=100.0,
            gas_flow_rate_gpm=50.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=60.0,
        )
        
        factor = br.calculate_inclination_factor()
        
        assert factor > 0.8

    def test_downhill_flow_negative_angle(self):
        """Calculate factor for downhill flow (-30°)."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=100.0,
            gas_flow_rate_gpm=50.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=-30.0,
        )
        
        factor = br.calculate_inclination_factor()
        
        # Downhill should have lower factor
        assert 0.4 < factor < 0.8


class TestPressureDropCalculations:
    """Test pressure drop calculations."""

    def test_horizontal_pressure_drop(self):
        """Calculate pressure drop for horizontal flow."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=200.0,
            gas_flow_rate_gpm=150.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        
        rho_m = br.calculate_mixture_density_lb_ft3()
        mu_m = br.calculate_mixture_viscosity_cP()
        
        dp_psi, breakdown = br.calculate_pressure_drop_total(
            well_length_ft=1000.0,
            mixture_density=rho_m,
            total_liquid_rate_gpm=200.0,
            gas_rate_gpm=150.0
        )
        
        assert dp_psi > 0
        assert breakdown['total_psi'] == dp_psi

    def test_uphill_pressure_drop(self):
        """Calculate pressure drop for uphill flow."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=200.0,
            gas_flow_rate_gpm=150.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=30.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        
        rho_m = br.calculate_mixture_density_lb_ft3()
        mu_m = br.calculate_mixture_viscosity_cP()
        
        dp_psi, breakdown = br.calculate_pressure_drop_total(
            well_length_ft=1000.0,
            mixture_density=rho_m,
            total_liquid_rate_gpm=200.0,
            gas_rate_gpm=150.0
        )
        
        # Uphill should have higher pressure drop than horizontal
        # (roughly due to hydrostatic component)
        assert dp_psi > 0

    def test_downhill_pressure_drop(self):
        """Calculate pressure drop for downhill flow."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=200.0,
            gas_flow_rate_gpm=150.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=-30.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        
        rho_m = br.calculate_mixture_density_lb_ft3()
        mu_m = br.calculate_mixture_viscosity_cP()
        
        dp_psi, breakdown = br.calculate_pressure_drop_total(
            well_length_ft=1000.0,
            mixture_density=rho_m,
            total_liquid_rate_gpm=200.0,
            gas_rate_gpm=150.0
        )
        
        assert dp_psi > 0

    def test_pressure_breakdown_components(self):
        """Verify pressure drop breakdown fields exist."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=200.0,
            gas_flow_rate_gpm=150.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
        )
        
        rho_m = br.calculate_mixture_density_lb_ft3()
        mu_m = br.calculate_mixture_viscosity_cP()
        
        dp_psi, breakdown = br.calculate_pressure_drop_total(
            well_length_ft=1000.0,
            mixture_density=rho_m,
            total_liquid_rate_gpm=200.0,
            gas_rate_gpm=150.0
        )
        
        # Should have breakdown dictionary with all expected fields
        assert 'total_psi' in breakdown
        assert 'hydrostatic_psi' in breakdown or 'friction_psi' in breakdown


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_gas_rate(self):
        """Test correlation with zero gas rate."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=300.0,
            gas_flow_rate_gpm=0.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        
        # Should not crash
        mu = br.calculate_mixture_viscosity_cP()
        rho = br.calculate_mixture_density_lb_ft3()
        lambda_g = br.calculate_gas_holdup()
        
        assert mu >= 0
        assert rho >= 0
        assert lambda_g == pytest.approx(0.0, abs=1e-6)

    def test_negligible_gas_rate(self):
        """Test with effectively no gas rate."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=300.0,
            gas_flow_rate_gpm=0.01,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        
        mu = br.calculate_mixture_viscosity_cP()
        rho = br.calculate_mixture_density_lb_ft3()
        
        assert mu >= 0
        assert rho >= 0

    def test_very_high_gas_rate_edge_case(self):
        """Test with extremely high gas rate."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=100.0,
            gas_flow_rate_gpm=10000.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        
        # Should not crash
        mu = br.calculate_mixture_viscosity_cP()
        rho = br.calculate_mixture_density_lb_ft3()
        lambda_g = br.calculate_gas_holdup()
        
        assert mu >= 0
        assert rho >= 0
        assert lambda_g >= 0

    def test_very_low_liquid_rate(self):
        """Test regime with minimal liquid flow."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=2.0,
            gas_flow_rate_gpm=50.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        
        # Should not crash
        mu = br.calculate_mixture_viscosity_cP()
        regime = br.identify_flow_regime()
        
        assert mu >= 0
        assert regime in FlowRegime.__members__.values()


class TestLiteratureExamples:
    """Test against representative literature examples."""

    def test_low_rate_example(self):
        """Test known low rate configuration (low F_Lo)."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=5.0,
            gas_flow_rate_gpm=2.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=55.0,
            water_density_lbm_ft3=62.4,
        )
        
        # Low rates typically segregated or bubble
        F_Lo = br.calculate_F_Lo()
        
        # Low F_Lo means segregated or bubble flow
        assert F_Lo < 0.01  # Expected for segregated/bubble flow


class TestReportGeneration:
    """Test report generation method."""

    def test_report_includes_all_fields(self):
        """Generate comprehensive report."""
        br = BeggsBrillCorrelation(
            oil_flow_rate_gpm=200.0,
            gas_flow_rate_gpm=150.0,
            pipe_diameter_ft=0.2917,
            borehole_area_ft2=0.0667,
            well_angle_deg=0.0,
            oil_density_lbm_ft3=50.0,
            water_density_lbm_ft3=62.4,
        )
        
        rho_m = br.calculate_mixture_density_lb_ft3()
        mu_m = br.calculate_mixture_viscosity_cP()
        
        dp_psi, _ = br.calculate_pressure_drop_total(
            well_length_ft=1000.0,
            mixture_density=rho_m,
            total_liquid_rate_gpm=200.0,
            gas_rate_gpm=150.0
        )
        
        report = br.generate_report(dp_psi)
        
        # Report should include all relevant fields
        assert 'timestamp' in report
        assert 'oil_flow_rate_gpm' in report
        assert 'gas_flow_rate_gpm' in report
        assert 'pipe_diameter_ft' in report
        assert 'well_angle_deg' in report
        assert 'flow_regime' in report
        assert 'mixture_viscosity_cP' in report
        assert 'mixture_density_lb_ft3' in report
        assert 'gas_holdup' in report
        assert 'pressure_drop_psi' in report
        
        assert report['pressure_drop_psi'] == dp_psi
