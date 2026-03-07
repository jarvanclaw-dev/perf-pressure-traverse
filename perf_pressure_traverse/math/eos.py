"""Equations of State for hydrocarbon systems."""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional


class SRKEOS:
    """
    Soave-Redlich-Kwong Equation of State.
    
    Cubic equation of state for calculating fluid properties.
    """
    
    def __init__(self, molecular_weight: float, critical_temp_R: float, critical_press_PSI: float):
        """
        Initialize SRK EOS.
        
        Parameters
        ----------
        molecular_weight : float
            Molecular weight in lb/lb-mol.
        critical_temp_R : float
            Critical temperature in Rankine.
        critical_press_PSI : float
            Critical pressure in psi.
        """
        self.MW = molecular_weight
        self.Tc = critical_temp_R
        self.Pc = critical_press_PSI
    
    def calculate_fugacity(self, press_PSI: float, temp_R: float, volume_ft3: float) -> float:
        """Calculate fugacity using SRK EOS."""
        pass


class PengRobinsonEOS:
    """
    Peng-Robinson Equation of State.
    
    Cubic equation of state for calculating fluid properties.
    """
    
    def __init__(self, molecular_weight: float, critical_temp_R: float, critical_press_PSI: float):
        """
        Initialize Peng-Robinson EOS.
        
        Parameters
        ----------
        molecular_weight : float
            Molecular weight in lb/lb-mol.
        critical_temp_R : float
            Critical temperature in Rankine.
        critical_press_PSI : float
            Critical pressure in psi.
        """
        self.MW = molecular_weight
        self.Tc = critical_temp_R
        self.Pc = critical_press_PSI
    
    def calculate_fugacity(self, press_PSI: float, temp_R: float, volume_ft3: float) -> float:
        """Calculate fugacity using Peng-Robinson EOS."""
        pass
