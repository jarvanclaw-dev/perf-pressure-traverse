"""Well flow path model representing wellbore segments and connectivity."""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

from perf_pressure_traverse.models.well import WellGeometry


@dataclass
class FlowPathSegment:
    """
    Represents a segment of the wellbore flow path.
    
    Attributes
    ----------
    start_depth_ft : float
        Starting depth of the segment (usually MD or TVD).
    end_depth_ft : float
        Ending depth of the segment.
    segment_length_ft : float
        Length of the segment.
    segment_type : str
        Type of segment (vertical, deviated, horizontal).
    deviation_angle_deg : float
        Deviation angle from vertical at segment.
    bearing_angle_deg : Optional[float]
        Bearing angle (azimuth) of this segment.
    has_perforations : bool
        Whether this segment contains perforations.
    perforation_intervals : List[Tuple[float, float]]
        List of perforation intervals in feet.
    is_wellhead : bool
        Whether this is the wellhead segment.
    is_bottomhole : bool
        Whether this is the bottomhole segment.
    """
    
    start_depth_ft: float
    end_depth_ft: float
    segment_length_ft: float
    segment_type: str = "vertical"
    deviation_angle_deg: float = 0.0
    bearing_angle_deg: Optional[float] = None
    has_perforations: bool = False
    perforation_intervals: List[Tuple[float, float]] = field(default_factory=list)
    is_wellhead: bool = False
    is_bottomhole: bool = False
    
    def __post_init__(self) -> None:
        """Calculate segment length after initialization."""
        if not hasattr(self, '_post_init_called'):
            self.segment_length_ft = self.end_depth_ft - self.start_depth_ft if self.end_depth_ft >= self.start_depth_ft else 0
            self._post_init_called = True
    
    def contains_depth(self, depth_ft: float) -> bool:
        """
        Check if a given depth falls within this segment.
        
        Parameters
        ----------
        depth_ft : float
            Depth to check.
        
        Returns
        -------
        bool
            True if depth is within segment.
        """
        return min(self.start_depth_ft, self.end_depth_ft) <= depth_ft <= max(self.start_depth_ft, self.end_depth_ft)
    
    def has_perforations_at_depth(self, depth_ft: float) -> bool:
        """
        Check if the segment has perforations at a specific depth.
        
        Parameters
        ----------
        depth_ft : float
            Depth to check.
        
        Returns
        -------
        bool
            True if perforations exist at this depth.
        """
        if not self.has_perforations:
            return False
        depth = min(self.start_depth_ft, self.end_depth_ft)
        for start, end in self.perforation_intervals:
            if start <= depth <= end:
                return True
        return False
    
    @classmethod
    def from_well_geometry(
        cls, 
        well: WellGeometry,
        use_tvd: bool = True,
        segment_length_ft: float = 50.0
    ) -> 'List[FlowPathSegment]':
        """
        Create flow path segments from well geometry.
        
        Parameters
        ----------
        well : WellGeometry
            Well geometry information.
        use_tvd : bool, optional
            Use true vertical depth if True, else measured depth.
        segment_length_ft : float, optional
            Target length of each segment.
        
        Returns
        -------
        List[FlowPathSegment]
            List of flow path segments.
        """
        total_depth = well.true_vertical_depth_ft if use_tvd else well.measured_depth_ft
        segments: List[FlowPathSegment] = []
        
        # Create wellhead segment
        if total_depth > 0:
            segments.append(cls(
                start_depth_ft=0.0,
                end_depth_ft=min(segment_length_ft, total_depth),
                segment_length_ft=0.0,  # Will be calculated
                segment_type="wellhead",
                is_wellhead=True,
            ))
        
        # Create deviated segments
        current_start = min(segment_length_ft, total_depth)
        current_depth = current_start
        
        if well.deviation_angle_deg:
            if isinstance(well.deviation_angle_deg, list):
                # Create segments based on deviation angle list
                for depth, angle in enumerate(well.deviation_angle_deg):
                    if current_depth < total_depth:
                        segment_end = min(current_depth + segment_length_ft, total_depth)
                        segments.append(cls(
                            start_depth_ft=current_depth,
                            end_depth_ft=segment_end,
                            segment_length_ft=0.0,
                            segment_type="deviated",
                            deviation_angle_deg=angle,
                            bearing_angle_deg=well.bearing_angle_deg,
                        ))
                        current_depth = segment_end
                    else:
                        break
            else:
                # Single constant deviation - create vertical segments
                while current_depth < total_depth:
                    segment_end = min(current_depth + segment_length_ft, total_depth)
                    segments.append(cls(
                        start_depth_ft=current_depth,
                        end_depth_ft=segment_end,
                        segment_length_ft=0.0,
                        segment_type="deviated",
                        deviation_angle_deg=well.deviation_angle_deg,
                        bearing_angle_deg=well.bearing_angle_deg,
                    ))
                    current_depth = segment_end
        
        # Create horizontal segment if needed
        if current_depth < total_depth:
            segments.append(cls(
                start_depth_ft=current_depth,
                end_depth_ft=total_depth,
                segment_length_ft=0.0,
                segment_type="horizontal",
                is_bottomhole=True,
            ))
        
        return segments


class WellFlowPath:
    """
    Well flow path model representing connectivity and segments.
    
    Attributes
    ----------
    segments : List[FlowPathSegment]
        List of flow path segments along the wellbore.
    well_geometry : WellGeometry
        Original well geometry.
    is_connected : bool
        Whether all segments form a continuous path.
    """
    
    def __init__(self, well_geometry: WellGeometry) -> None:
        """
        Initialize well flow path.
        
        Parameters
        ----------
        well_geometry : WellGeometry
            Well geometry information.
        """
        self.segments: List[FlowPathSegment] = []
        self.well_geometry = well_geometry
        self.is_connected = True
        
        # Auto-generate segments from well geometry
        self.segments = FlowPathSegment.from_well_geometry(
            well_geometry,
            use_tvd=True
        )
        
        # Set segment types based on topology
        self._set_segment_types()
    
    def _set_segment_types(self) -> None:
        """Set segment types based on depth positions."""
        total_depth = self.well_geometry.true_vertical_depth_ft
        
        for i, segment in enumerate(self.segments):
            if segment.is_wellhead:
                segment.segment_type = "wellhead"
            elif segment.is_bottomhole:
                segment.segment_type = "bottomhole"
            elif i == len(self.segments) - 1:
                segment.segment_type = "horizontal"
            elif abs(segment.deviation_angle_deg) > 10:
                segment.segment_type = "deviated"
            else:
                segment.segment_type = "vertical"
    
    def get_segment_at_depth(self, depth_ft: float) -> Optional[FlowPathSegment]:
        """
        Get segment containing a specific depth.
        
        Parameters
        ----------
        depth_ft : float
            Target depth.
        
        Returns
        -------
        FlowPathSegment or None
            Segment containing the depth, or None if not found.
        """
        for segment in self.segments:
            if segment.contains_depth(depth_ft):
                return segment
        return None
    
    def get_surface_segment(self) -> Optional[FlowPathSegment]:
        """Get the surface/segment at depth 0."""
        return self.get_segment_at_depth(0.0)
    
    def get_bottomhole_segment(self) -> Optional[FlowPathSegment]:
        """Get the bottomhole segment."""
        return self.get_segment_at_depth(
            self.well_geometry.true_vertical_depth_ft
        )
    
    def add_perforations(self, depth_range: List[Tuple[float, float]]) -> None:
        """
        Add perforation intervals to segments.
        
        Parameters
        ----------
        depth_range : list[tuple[float, float]]
            List of depth intervals [start, end] for perforations.
        """
        for start, end in depth_range:
            for segment in self.segments:
                if segment.contains_depth(start) and segment.contains_depth(end):
                    segment.has_perforations = True
                    segment.perforation_intervals.append((min(start, end), max(start, end)))
                    break
    
    def get_total_length_ft(self) -> float:
        """Calculate total length of all segments."""
        return sum(seg.segment_length_ft for seg in self.segments)
    
    def get_deviation_angle_at_depth(self, depth_ft: float) -> Optional[float]:
        """
        Get deviation angle at a specific depth.
        
        Parameters
        ----------
        depth_ft : float
            Target depth.
        
        Returns
        -------
        float or None
            Deviation angle in degrees, or None if unknown.
        """
        segment = self.get_segment_at_depth(depth_ft)
        return segment.deviation_angle_deg if segment else None
    
    def get_flow_regime_probability(self, flow_rate: float) -> Dict[str, float]:
        """
        Estimate flow regime probabilities at current conditions.
        
        Simplified implementation - would use detailed flow regime models.
        
        Parameters
        ----------
        flow_rate : float
            Current flow rate (GPM or similar).
        
        Returns
        -------
        dict
            Dictionary of flow regime probabilities.
        """
        # Simple heuristic probabilities based on flow rate
        if flow_rate < 1:
            return {"Bubbly": 0.6, "Slug": 0.3, "Annular": 0.1}
        elif flow_rate < 50:
            return {"Slug": 0.5, "Annular": 0.4, "Bubbly": 0.1}
        else:
            return {"Annular": 0.6, "Slug": 0.3, "Mist": 0.1}
    
    def to_dict(self) -> Dict:
        """
        Convert to dictionary.
        
        Returns
        -------
        dict
            Dictionary representation.
        """
        return {
            'well_depth_ft': self.well_geometry.true_vertical_depth_ft,
            'well_type': 'deviated' if self.well_geometry.deviation_angle_deg else 'vertical',
            'segment_count': len(self.segments),
            'total_length_ft': self.get_total_length_ft(),
            'segments': [
                {
                    'start_depth_ft': seg.start_depth_ft,
                    'end_depth_ft': seg.end_depth_ft,
                    'segment_type': seg.segment_type,
                    'deviation_angle_deg': seg.deviation_angle_deg,
                    'has_perforations': seg.has_perforations,
                }
                for seg in self.segments
            ],
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"WellFlowPath(segments={len(self.segments)}, "
            f"well_depth={self.well_geometry.true_vertical_depth_ft:.2f}FT, "
            f"connected={self.is_connected})"
        )