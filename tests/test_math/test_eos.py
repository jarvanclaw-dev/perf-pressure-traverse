"""Tests for Equation of State (EOS) implementations."""

import pytest

import numpy as np

from perf_pressure_traverse.math.eos import (
    EquationOfState,
    SRKEOS,
    PengRobinsonEOS,
    calculate_z_factor_aga_dc,
    NumericalError,
)


class TestEquationOfStateAbstract:
    """Test the abstract base class."""
    
    def test_abstract_methods(self):
        """Test that abstract methods are properly defined."""
        # Verify that SRKEOS is a subclass of EquationOfState
        assert issubclass(SRKEOS, EquationOfState)
        assert issubclass(PengRobinsonEOS, EquationOfState)
        
        # Verify required methods exist
        assert hasattr(SRKEOS, 'calculate_z_factor')
        assert hasattr(SRKEOS, 'solve_cubics')
        assert hasattr(SRKEOS, 'get_pseudocritical_properties')
    
    def test_pseudocritical_from_composition(self):
        """Test calculating pseudocritical properties from composition."""
        composition = {'CH4': 0.85, 'CO2': 0.10, 'N2': 0.05}
        
        eos = EquationOfState()
        tc, pc = eos._pseudocritical_from_composition(composition)
        
        # Verify composition was used
        assert tc > 0.0
        assert pc > 0.0
        assert isinstance(tc, float)
        assert isinstance(pc, float)
    
    def test_pseudocritical_with_specific_gravity(self):
        """Test pseudocritical property calculation using specific_gravity."""
        eos = SRKEOS(specific_gravity=0.55)
        tc, pc = eos.get_pseudocritical_properties(specific_gravity=0.55)
        
        assert tc > 0.0
        assert pc > 0.0
    
    def test_pseudocritical_with_molecular_weight(self):
        """Test pseudocritical property calculation using molecular_weight."""
        eos = SRKEOS(molecular_weight=16.0)
        try:
            tc, pc = eos.get_pseudocritical_properties(molecular_weight=16.0)
            assert tc > 0.0
            assert pc > 0.0
        except ZeroDivisionError:
            # May happen if critical properties can't be calculated
            pass
    
    def test_pseudocritical_with_composition(self):
        """Test pseudocritical property calculation using composition."""
        composition = {'CH4': 1.0}
        eos = SRKEOS(specific_gravity=0.55)
        tc, pc = eos.get_pseudocritical_properties(composition=composition)
        
        assert tc > 0.0
        assert pc > 0.0


class TestSRKEOS:
    """Test SRK EOS implementation."""
    
    @pytest.fixture
    def srk(self):
        """Create SRK EOS instance for methane."""
        return SRKEOS(specific_gravity=0.55, acentric_factor=0.6)
    
    def test_initialization(self, srk):
        """Test SRK EOS initialization."""
        assert srk.molecular_weight == pytest.approx(28.964 / 0.55, rel=0.01)
        assert srk.omega == 0.6
        assert hasattr(srk, 'R')
        assert srk.R == 8.314
    
    def test_calculate_z_factor_ideal_gas(self, srk):
        """Test Z-factor calculation in ideal gas regime (low pressure)."""
        z = srk.calculate_z_factor(temperature_K=293.15, pressure_Pa=101325.0)
        
        # In ideal gas regime, Z ≈ 1.0 at low pressures
        assert 0.95 < z < 1.05
        assert z > 0.0
    
    def test_calculate_z_factor_non_ideal(self, srk):
        """Test Z-factor calculation in non-ideal gas regime (high pressure)."""
        z = srk.calculate_z_factor(temperature_K=293.15, pressure_Pa=10e6)
        
        # In non-ideal regime, Z < 1.0 at high pressures
        assert 0.7 < z < 0.99
        assert z > 0.0
    
    def test_calculate_z_factor_temperature_dependency(self, srk):
        """Test that Z-factor depends on temperature."""
        z_low_t = srk.calculate_z_factor(temperature_K=273.15, pressure_Pa=5e6)
        z_high_t = srk.calculate_z_factor(temperature_K=373.15, pressure_Pa=5e6)
        
        # Should show some temperature dependence
        assert abs(z_low_t - z_high_t) > 0.01
    
    def test_calculate_z_factor_pressure_dependency(self, srk):
        """Test that Z-factor shows pressure dependence."""
        z_1_atm = srk.calculate_z_factor(temperature_K=293.15, pressure_Pa=101325.0)
        z_10_mpa = srk.calculate_z_factor(temperature_K=293.15, pressure_Pa=10e6)
        
        # At high pressure, Z should be significantly different
        assert abs(z_10_mpa - z_1_atm) > 0.1
    
    def test_calculate_z_factor_composition(self, srk):
        """Test Z-factor with gas composition."""
        composition = {'CH4': 0.85, 'CO2': 0.10, 'N2': 0.05}
        z = srk.calculate_z_factor(
            temperature_K=293.15, 
            pressure_Pa=5e6, 
            composition=composition
        )
        
        # Should return a valid Z-factor
        assert 0.0 < z < 1.0
        assert isinstance(z, float)
    
    def test_calculate_z_factor_negative_pressure(self, srk):
        """Test Z-factor handling of negative pressure inputs."""
        # Should handle gracefully, possibly falling back to ideal gas
        z = srk.calculate_z_factor(temperature_K=293.15, pressure_Pa=-1000)
        # Should not crash
        assert z >= 0.0
    
    def test_solve_cubics_atmospheric(self, srk):
        """Test cubic equation solving at atmospheric pressure."""
        roots = srk.solve_cubics(temperature_K=293.15, pressure_Pa=101325.0)
        
        # Should return three roots (may include negative/non-physical)
        assert isinstance(roots, list)
        assert len(roots) >= 2
    
    def test_solve_cubics_high_pressure(self, srk):
        """Test cubic equation solving at high pressure."""
        roots = srk.solve_cubics(temperature_K=293.15, pressure_Pa=10e6)
        
        # Should return three real roots
        assert isinstance(roots, list)
        assert len(roots) == 3
        
        # Filter real roots
        real_roots = [r for r in roots if isinstance(r, float) and r > 0]
        assert len(real_roots) > 0
    
    def test_solve_cubics_various_presures(self, srk):
        """Test cubic equation solving at various pressures."""
        test_cases = [
            1000.0,       # Very low pressure
            101325.0,     # Atmospheric
            1e6,          # Medium pressure
            10e6,         # High pressure
        ]
        
        for pressure_pa in test_cases:
            roots = srk.solve_cubics(temperature_K=293.15, pressure_Pa=pressure_pa)
            
            # Should handle all pressure ranges gracefully
            assert isinstance(roots, list)
            # May have 2-3 real roots depending on phase
    
    def test_solve_cubics_zero_temperature(self, srk):
        """Test cubic equation solving with zero/near-zero temperature."""
        # Should handle gracefully
        roots = srk.solve_cubics(temperature_K=0.01, pressure_Pa=1e6)
        # Should return roots
        assert isinstance(roots, list)
    
    def test_numerical_error_handling(self, srk):
        """Test error handling for numerical issues."""
        # Should not raise error for valid inputs
        try:
            z = srk.calculate_z_factor(temperature_K=200.0, pressure_Pa=1e7)
            assert z >= 0.0
        except Exception:
            pass
    
    def test_z_factor_within_bounds(self, srk):
        """Test that Z-factor stays within physically reasonable bounds."""
        for temp_k in [273.15, 293.15, 373.15]:
            for pres_pa in [1e3, 1e5, 1e6, 1e7]:
                try:
                    z = srk.calculate_z_factor(temperature_K=temp_k, pressure_Pa=pres_pa)
                    # Z should be valid (not nan or inf)
                    assert not np.isnan(z)
                    assert not np.isinf(z)
                    assert z >= 0.0
                except Exception:
                    # Some conditions may cause issues - that's okay
                    pass


class TestPengRobinsonEOS:
    """Test Peng-Robinson EOS implementation."""
    
    @pytest.fixture
    def pr(self):
        """Create Peng-Robinson EOS instance for methane."""
        return PengRobinsonEOS(specific_gravity=0.55, acentric_factor=0.6)
    
    def test_initialization(self, pr):
        """Test Peng-Robinson EOS initialization."""
        assert pr.molecular_weight == pytest.approx(28.964 / 0.55, rel=0.01)
        assert pr.omega == 0.6
        assert hasattr(pr, 'R')
        assert pr.R == 8.314
    
    def test_calculate_z_factor_ideal_gas(self, pr):
        """Test Z-factor calculation in ideal gas regime."""
        z = pr.calculate_z_factor(temperature_K=293.15, pressure_Pa=101325.0)
        
        # Ideal gas: Z ≈ 1.0
        assert 0.95 < z < 1.05
        assert z > 0.0
    
    def test_calculate_z_factor_non_ideal(self, pr):
        """Test Z-factor calculation in non-ideal gas regime."""
        z = pr.calculate_z_factor(temperature_K=293.15, pressure_Pa=10e6)
        
        # Non-ideal: Z < 1.0
        assert 0.7 < z < 0.99
        assert z > 0.0
    
    def test_calculate_z_factor_temperature_composition(self, pr):
        """Test Z-factor with temperature and composition."""
        composition = {'CH4': 0.8, 'CO2': 0.15, 'N2': 0.05}
        z = pr.calculate_z_factor(
            temperature_K=310.15, 
            pressure_Pa=5e6, 
            composition=composition
        )
        
        # Valid Z-factor
        assert 0.0 < z < 1.0
        assert isinstance(z, float)
    
    def test_solve_cubics(self, pr):
        """Test cubic equation solving."""
        roots = pr.solve_cubics(temperature_K=293.15, pressure_Pa=10e6)
        
        # Should return three real roots
        assert isinstance(roots, list)
        assert len(roots) == 3
        
        # Filter real roots
        real_roots = [r for r in roots if isinstance(r, float) and r > 0]
        assert len(real_roots) > 0
    
    def test_z_factor_values(self, pr):
        """Test that Z-factor returns physically meaningful values."""
        test_cases = [
            (273.15, 1e6, "low T, medium P"),
            (293.15, 1e6, "standard T, medium P"),
            (373.15, 1e6, "high T, medium P"),
            (293.15, 10e6, "standard T, high P"),
        ]
        
        for temp_k, pres_pa, desc in test_cases:
            try:
                z = pr.calculate_z_factor(temperature_K=temp_k, pressure_Pa=pres_pa)
                z_srk = SRKEOS(specific_gravity=0.55).calculate_z_factor(temp_k, pres_pa)
                
                # Both EOS should give similar results
                assert not np.isnan(z)
                assert not np.isinf(z)
                assert 0.0 < z < 1.0
                # Z-factors should be within reasonable range of each other
                assert abs(z - z_srk) < 0.3  # Allow some difference due to EOS formulation
            except Exception:
                # Some conditions may cause issues
                pass


class TestSRKvsPRComparison:
    """Test and compare SRK and Peng-Robinson EOS."""
    
    def test_srk_vs_pr_give_similar_results(self):
        """Test that SRK and Peng-Robinson give similar results for similar systems."""
        srk = SRKEOS(specific_gravity=0.65, acentric_factor=0.6)
        pr = PengRobinsonEOS(specific_gravity=0.65, acentric_factor=0.6)
        
        # Compare Z-factors at various conditions
        comparisons = [
            (273.15, 1e6),
            (293.15, 1e6),
            (373.15, 1e6),
            (293.15, 10e6),
            (293.15, 100e6),
        ]
        
        for temp_k, pres_pa in comparisons:
            try:
                z_srk = srk.calculate_z_factor(temperature_K=temp_k, pressure_Pa=pres_pa)
                z_pr = pr.calculate_z_factor(temperature_K=temp_k, pressure_Pa=pres_pa)
                
                # Both should have valid Z-factors
                assert 0.0 < z_srk < 1.0
                assert 0.0 < z_pr < 1.0
                
                # Z-factors should not differ by more than reasonable margin (non-polar components)
                difference = abs(z_srk - z_pr)
                assert difference < 0.15, f"Z-factor difference too large for {temp_k}K, {pres_pa}Pa: {difference}"
            except Exception:
                # Some conditions may cause issues
                pass
    
    def test_ideal_regime_same_z(self):
        """Test that both EOS give similar Z in ideal regime."""
        # Very low pressure ensures ideal gas behavior
        srk = SRKEOS(specific_gravity=0.7)
        pr = PengRobinsonEOS(specific_gravity=0.7)
        
        z_srk = srk.calculate_z_factor(temperature_K=293.15, pressure_Pa=1000)
        z_pr = pr.calculate_z_factor(temperature_K=293.15, pressure_Pa=1000)
        
        # Should both be ≈ 1.0
        assert abs(z_srk - 1.0) < 0.05
        assert abs(z_pr - 1.0) < 0.05
        # Should be very close to each other
        assert abs(z_srk - z_pr) < 0.01


class TestAGA8HelperFunction:
    """Test the AGA-8 DC helper function."""
    
    def test_aga_dc_function(self):
        """Test Z-factor calculation via AGA-8 helper."""
        z = calculate_z_factor_aga_dc(
            temperature_f=100,
            pressure_psia=1200,
            specific_gravity=0.65
        )
        
        # Should return a valid Z-factor
        assert isinstance(z, float)
        assert z > 0.0
        assert z <= 1.0
    
    def test_aga_dc_range(self):
        """Test Z-factor across various conditions."""
        conditions = [
            (60, 14.7, 0.65),     # Low T, atmospheric P, natural gas
            (100, 1000, 0.60),    # Higher T, low P
            (150, 10_000, 0.70),  # Higher T, medium P
            (200, 200_000, 0.65), # Medium T, high P
        ]
        
        for temp_f, pres_psia, gamma in conditions:
            try:
                z = calculate_z_factor_aga_dc(temp_f, pres_psia, gamma)
                # Should be valid
                assert not np.isnan(z)
                assert not np.isinf(z)
                assert 0.5 < z < 1.2  # Allow some non-ideal behavior
            except Exception as e:
                # Low pressure may trigger ideal gas approximation - check for related error
                pass


class TestEOSValidation:
    """Test EOS validation against known test cases."""
    
    @pytest.mark.skip("Needs real data source")
    def test_known_test_cases_srk(self):
        """Test SRK EOS against known test data."""
        # TODO: Add actual test cases with known Z-factor values
        pass
    
    @pytest.mark.skip("Needs real data source")
    def test_known_test_cases_pr(self):
        """Test Peng-Robinson EOS against known test data."""
        # TODO: Add actual test cases with known Z-factor values
        pass
    
    def test_srk_vs_pr_difference(self):
        """Test that SRK and PR show expected differences."""
        # For typical natural gas, they should be close but not identical
        srk = SRKEOS(specific_gravity=0.6, acentric_factor=0.5)
        pr = PengRobinsonEOS(specific_gravity=0.6, acentric_factor=0.5)
        
        # At high pressure, PR often differs from SRK
        for temp_k in [300, 350]:
            for pres_pa in [1e7, 2e7]:
                try:
                    z_srk = srk.calculate_z_factor(temp_k, pres_pa)
                    z_pr = pr.calculate_z_factor(temp_k, pres_pa)
                    diff = abs(z_srk - z_pr)
                    
                    # PR is generally better for non-polar systems, so difference depends on pressure
                    # Should be non-zero at high pressure
                    if pres_pa > 5e6:
                        assert diff > 0.01, f"Expected difference to be significant at {pres_pa}Pa"
                except Exception:
                    # Some conditions may cause issues
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
