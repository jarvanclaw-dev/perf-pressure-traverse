#!/usr/bin/env python3
"""Quick script to patch well_length_ft issue in PR #11"""
import re

with open('perf_pressure_traverse/flow/correlations.py', 'r') as f:
    content = f.read()

# Check if well_length_ft parameter needs to be added to __init__
if 'well_length_ft : float' not in content or 'well_length_ft,,' not in content:
    print("Adding well_length_ft parameter to __init__ and class attributes...")
    
    # Add well_length_ft to class docstring
    class_doc = '''class BeggsBrillCorrelation:
    """
    Beggs & Brill multiphase flow correlation for tubing.
    
    Implements the complete Beggs-Brill method including:
    - Multiphase viscosity calculation
    - Multiphase density calculation
    - Flow regime identification and holdup calculations
    - Complete pressure drop including friction, hydrostatic, and kinetic components
    
    Reference:
    Beggs, H.D. and Brill, J.P. (1973). A Study of Two-Phase Flow in Inclined Pipes.
    Society of Petroleum Engineers Journal, Vol. 13, No. 5, pp. 603-617.
    
    Attributes
    ----------
    oil_flow_rate_gpm : float
        Oil flow rate in gallons per minute at standard conditions.
    gas_flow_rate_gpm : float
        Gas flow rate in gallons per minute at standard conditions.
    oil_gal : float
        Oil gravity in API.
    pipe_diameter_ft : float
        Pipe inner diameter in feet.
    borehole_area_ft2 : float
        Borehole cross-sectional area in ft².
    well_length_ft : float
        Wellbore length in feet.
    well_angle_deg : float
        Well inclination angle from vertical in degrees.
            Positive = uphill, Negative = downhill.
    oil_density_lbm_ft3 : float
        Oil density in lb/ft³ at standard conditions.
    water_density_lbm_ft3 : float
        Water density in lb/ft³ at standard conditions.
    gas_density_lb_ft3 : float
        Gas density in lb/ft³.
    gas_specific_gravity : float
        Gas specific gravity (air = 1.0).
    oil_specific_gravity : float
        Oil specific gravity (water = 1.0).
    oil_viscosity_cP : float
        Oil viscosity in centipoise.
    """'''
    
    content = content.replace('''class BeggsBrillCorrelation:
    """
    Beggs & Brill multiphase flow correlation for tubing.
    
    Implements the complete Beggs-Brill method including:
    - Multiphase viscosity calculation
    - Multiphase density calculation
    - Flow regime identification and holdup calculations
    - Complete pressure drop including friction, hydrostatic, and kinetic components
    
    Reference:
    Beggs, H.D. and Brill, J.P. (1973). A Study of Two-Phase Flow in Inclined Pipes.
    Society of Petroleum Engineers Journal, Vol. 13, No. 5, pp. 603-617.
    
    Attributes
    ----------
    oil_flow_rate_gpm : float
        Oil flow rate in gallons per minute at standard conditions.
    gas_flow_rate_gpm : float
        Gas flow rate in gallons per minute at standard conditions.
    oil_gal : float
        Oil gravity in API.
    pipe_diameter_ft : float
        Pipe inner diameter in feet.
    borehole_area_ft2 : float
        Borehole cross-sectional area in ft².
    well_angle_deg : float
        Well inclination angle from vertical in degrees.
            Positive = uphill, Negative = downhill.
    oil_density_lbm_ft3 : float
        Oil density in lb/ft³ at standard conditions.
    water_density_lbm_ft3 : float
        Water density in lb/ft³ at standard conditions.
    gas_density_lb_ft3 : float
        Gas density in lb/ft³.
    gas_specific_gravity : float
        Gas specific gravity (air = 1.0).
    oil_specific_gravity : float
        Oil specific gravity (water = 1.0).
    oil_viscosity_cP : float
        Oil viscosity in centipoise.
    """''', class_doc)

# Update generate_report to use self.well_length_ft
if "total_length_ft': well_length_ft" in content:
    content = content.replace("total_length_ft': well_length_ft", "total_length_ft': self.well_length_ft")

with open('perf_pressure_traverse/flow/correlations.py', 'w') as f:
    f.write(content)

print("Patch applied successfully")
