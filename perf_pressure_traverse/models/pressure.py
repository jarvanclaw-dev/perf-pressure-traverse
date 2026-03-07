"""Pressure point model representing measurement points along the wellbore."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
import numpy as np

from perf_pressure_traverse.models.well import WellGeometry


class PressurePoint:
    """
    Pressure point model representing a measurement point along the wellbore.
    
    Each point contains pressure, temperature, and other relevant data.
    Used to represent points in a pressure traverse profile.
    
    Attributes
    ----------
    measured_depth_ft : float
        Measured depth (MD) in feet from the wellhead.
    true_vertical_depth_ft : float
        True vertical depth (TVD) in feet from the wellhead.
        TVD is the vertical distance from the wellhead to the point.
    pressure_psi : float
        Pressure at this point in psi.
    temperature_f : float
        Temperature at this point in °F.
    liquid_holdup_fraction : float
        Liquid holdup fraction (0 to 1).
        Fraction of the pipe area occupied by liquid.
    mixture_velocity_ft_s : float
        Mixture velocity at this point in ft/s.
    quality : float
        Gas quality (x) at this point.
        Fraction of total flow that is gas.
    flow_regime : str
        Flow regime classification (Bubbly, Slug, Annular, Mist, etc.).
    time : datetime
        Date and time of measurement.
    iteration_number : int
        Iteration number for pressure traverse solver.
    residual : float
        Absolute residual at this point.
    """
    
    def __init__(
        self,
        measured_depth_ft: float,
        true_vertical_depth_ft: float,
        pressure_psi: float,
        temperature_f: float,
        liquid_holdup_fraction: float = 0.0,
        mixture_velocity_ft_s: float = 0.0,
        quality: float = 0.0,
        flow_regime: str = "Unknown",
        time: Optional[datetime] = None,
        iteration_number: int = 0,
        residual: float = 0.0,
    ) -> None:
        """
        Initialize a pressure point.
        
        Parameters
        ----------
        measured_depth_ft : float
            Measured depth in feet.
        true_vertical_depth_ft : float
            True vertical depth in feet.
        pressure_psi : float
            Pressure at the point in psi.
        temperature_f : float
            Temperature at the point in °F.
        liquid_holdup_fraction : float, optional
            Liquid holdup fraction.
        mixture_velocity_ft_s : float, optional
            Mixture velocity in ft/s.
        quality : float, optional
            Gas quality.
        flow_regime : str, optional
            Flow regime classification.
        time : datetime, optional
            Date and time of measurement.
        iteration_number : int, optional
            Iteration number.
        residual : float, optional
            Residual value.
        """
        self.measured_depth_ft = measured_depth_ft
        self.true_vertical_depth_ft = true_vertical_depth_ft
        self.pressure_psi = pressure_psi
        self.temperature_f = temperature_f
        self.liquid_holdup_fraction = liquid_holdup_fraction
        self.mixture_velocity_ft_s = mixture_velocity_ft_s
        self.quality = quality
        self.flow_regime = flow_regime
        self.time = time or datetime.now()
        self.iteration_number = iteration_number
        self.residual = residual
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PressurePoint':
        """
        Create a PressurePoint from a dictionary.
        
        Parameters
        ----------
        data : dict
            Dictionary containing point data.
            Expected keys: measured_depth_ft, true_vertical_depth_ft, 
                          pressure_psi, temperature_f, etc.
        
        Returns
        -------
        PressurePoint
            New PressurePoint instance.
        """
        return cls(
            measured_depth_ft=data.get('measured_depth_ft', 0),
            true_vertical_depth_ft=data.get('true_vertical_depth_ft', 0),
            pressure_psi=data.get('pressure_psi', 0),
            temperature_f=data.get('temperature_f', 0),
            liquid_holdup_fraction=data.get('liquid_holdup_fraction', 0),
            mixture_velocity_ft_s=data.get('mixture_velocity_ft_s', 0),
            quality=data.get('quality', 0),
            flow_regime=data.get('flow_regime', 'Unknown'),
            time=datetime.fromisoformat(data.get('time', '')) 
                 if data.get('time') else None,
            iteration_number=data.get('iteration_number', 0),
            residual=data.get('residual', 0),
        )
    
    def to_dict(self) -> dict:
        """
        Convert point data to dictionary.
        
        Returns
        -------
        dict
            Dictionary representation of the point.
        """
        return {
            'measured_depth_ft': self.measured_depth_ft,
            'true_vertical_depth_ft': self.true_vertical_depth_ft,
            'pressure_psi': self.pressure_psi,
            'temperature_f': self.temperature_f,
            'liquid_holdup_fraction': self.liquid_holdup_fraction,
            'mixture_velocity_ft_s': self.mixture_velocity_ft_s,
            'quality': self.quality,
            'flow_regime': self.flow_regime,
            'time': self.time.isoformat() if self.time else None,
            'iteration_number': self.iteration_number,
            'residual': self.residual,
        }
    
    def __repr__(self) -> str:
        """String representation of pressure point."""
        return (
            f"PressurePoint(MD={self.measured_depth_ft:.2f}FT, "
            f"TVD={self.true_vertical_depth_ft:.2f}FT, "
            f"P={self.pressure_psi:.2f}PSI, "
            f"T={self.temperature_f:.2f}°F, "
            f"holdup={self.liquid_holdup_fraction:.3f}, "
            f"regime={self.flow_regime})"
        )
    
    def __str__(self) -> str:
        """String representation for CSV export."""
        return (
            f"{self.measured_depth_ft:.6f}, "
            f"{self.true_vertical_depth_ft:.6f}, "
            f"{self.pressure_psi:.6f}, "
            f"{self.temperature_f:.6f}, "
            f"{self.liquid_holdup_fraction:.6f}, "
            f"{self.mixture_velocity_ft_s:.6f}, "
            f"{self.quality:.6f}, "
            f'{self.flow_regime}, '
            f"{self.time.strftime('%Y-%m-%d %H:%M:%S')}, "
            f"{self.iteration_number:.0f}, "
            f"{self.residual:.6f}"
        )


class PressureProfile:
    """
    Pressure profile containing all pressure points along the wellbore.
    
    Attributes
    ----------
    points : list[PressurePoint]
        List of pressure points from surface to bottom.
    well_geometry : WellGeometry
        Wellbore geometry information.
    """
    
    def __init__(self, well_geometry: WellGeometry) -> None:
        """
        Initialize a pressure profile.
        
        Parameters
        ----------
        well_geometry : WellGeometry
            Well geometry information.
        """
        self.points: list[PressurePoint] = []
        self.well_geometry = well_geometry
    
    def add_point(self, point: PressurePoint) -> None:
        """Add a pressure point to the profile."""
        self.points.append(point)
    
    def get_surface_point(self) -> Optional['PressurePoint']:
        """Get the surface pressure point (first point)."""
        return self.points[0] if self.points else None
    
    def get_bottomhole_point(self) -> Optional['PressurePoint']:
        """Get the bottomhole pressure point (last point)."""
        return self.points[-1] if self.points else None
    
    def get_pressure_array(self) -> np.ndarray:
        """Get Numpy array of pressures."""
        return np.array([p.pressure_psi for p in self.points])
    
    def get_temperature_array(self) -> np.ndarray:
        """Get Numpy array of temperatures."""
        return np.array([p.temperature_f for p in self.points])
    
    def get_depth_array(self) -> np.ndarray:
        """Get Numpy array of depths (MD or TVD)."""
        return np.array([p.measured_depth_ft for p in self.points])
    
    def to_dict(self) -> dict:
        """
        Convert profile to dictionary.
        
        Returns
        -------
        dict
            Dictionary representation containing points and geometry.
        """
        return {
            'well_geometry': self.well_geometry.to_dict(),
            'points': [p.to_dict() for p in self.points],
        }
    
    def export_to_csv(self, filename: str) -> None:
        """
        Export pressure profile to CSV file.
        
        Parameters
        ----------
        filename : str
            Output CSV filename.
        """
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'measured_depth_ft', 'true_vertical_depth_ft', 
                'pressure_psi', 'temperature_f',
                'liquid_holdup_fraction', 'mixture_velocity_ft_s',
                'quality', 'flow_regime', 'time',
                'iteration_number', 'residual'
            ])
            for point in self.points:
                writer.writerow([
                    point.measured_depth_ft,
                    point.true_vertical_depth_ft,
                    point.pressure_psi,
                    point.temperature_f,
                    point.liquid_holdup_fraction,
                    point.mixture_velocity_ft_s,
                    point.quality,
                    point.flow_regime,
                    point.time,
                    point.iteration_number,
                    point.residual,
                ])
    
    def __len__(self) -> int:
        """Return number of points in profile."""
        return len(self.points)
    
    def __getitem__(self, idx: int) -> PressurePoint:
        """Get point by index."""
        return self.points[idx]
    
    def __repr__(self) -> str:
        """String representation of pressure profile."""
        return (
            f"PressureProfile(points={len(self.points)}, "
            f"surface_psi={self.get_surface_point().pressure_psi if self.get_surface_point() else 0:.2f}, "
            f"bottomhole_psi={self.get_bottomhole_point().pressure_psi if self.get_bottomhole_point() else 0:.2f})"
        )