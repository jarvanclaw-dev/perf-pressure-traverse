"""Wellbore geometry model."""

from __future__ import annotations

from typing import Optional, Union


class WellGeometry:
    """
    Wellbore geometry model.
    
    Attributes
    ----------
    borehole_diameter_ft : float
        Borehole (production tubing) inner diameter in feet.
    casing_diameter_ft : float
        Casing outer diameter in feet.
    true_vertical_depth_ft : float
        True vertical depth in feet.
    measured_depth_ft : float
        Measured depth in feet.
    is_vertical : bool
        Whether the well is vertical (deviation angle near 0°).
    deviation_angle_deg : Optional[float] or Optional[list]
        Measured inclination angle from vertical in degrees.
        Can be either a float or a list of deviation angles per unit depth.
    bearing_angle_deg : Optional[float]
        Measured bearing angle in degrees (azimuth).
    """
    
    def __init__(
        self,
        borehole_diameter_ft: float,
        casing_diameter_ft: float,
        true_vertical_depth_ft: float,
        measured_depth_ft: float,
        is_vertical: bool = True,
        deviation_angle_deg: Optional[Union[float, list]] = None,
        bearing_angle_deg: Optional[float] = None,
    ) -> None:
        """
        Initialize well geometry.
        
        Parameters
        ----------
        borehole_diameter_ft : float
            Borehole inner diameter in feet.
        casing_diameter_ft : float
            Casing outer diameter in feet.
        true_vertical_depth_ft : float
            True vertical depth in feet.
        measured_depth_ft : float
            Measured depth in feet.
        is_vertical : bool, optional
            Whether the well is vertical.
        deviation_angle_deg : float or list, optional
            Deviation angle from vertical.
        bearing_angle_deg : float, optional
            Bearing angle in degrees.
        """
        self.borehole_diameter_ft = borehole_diameter_ft
        self.casing_diameter_ft = casing_diameter_ft
        self.true_vertical_depth_ft = true_vertical_depth_ft
        self.measured_depth_ft = measured_depth_ft
        self.is_vertical = is_vertical
        self.deviation_angle_deg = deviation_angle_deg
        self.bearing_angle_deg = bearing_angle_deg
    
    @property
    def borehole_area_ft2(self) -> float:
        """Calculate borehole cross-sectional area in ft²."""
        r = self.borehole_diameter_ft / 2.0
        return 3.141592653589793 * r * r
    
    def __repr__(self) -> str:
        """String representation of well geometry."""
        return (
            f"WellGeometry(TVD={self.true_vertical_depth_ft}, "
            f"MD={self.measured_depth_ft}, "
            f"diam={self.borehole_diameter_ft * 12:.1f}in, "
            f"vertical={self.is_vertical})"
        )
