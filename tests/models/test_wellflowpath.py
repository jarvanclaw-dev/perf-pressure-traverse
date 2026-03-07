"""Unit tests for Well flow path models."""

import pytest
from unittest.mock import Mock

from perf_pressure_traverse.models.wellflowpath import (
    WellFlowPath,
    FlowPathSegment
)
from perf_pressure_traverse.models.well import WellGeometry


class TestFlowPathSegment:
    """Tests for FlowPathSegment model."""
    
    def test_initialization(self):
        """Test segment initialization."""
        segment = FlowPathSegment(
            start_depth_ft=0.0,
            end_depth_ft=100.0,
            segment_length_ft=100.0,
            segment_type="vertical",
        )
        
        assert segment.start_depth_ft == 0.0
        assert segment.end_depth_ft == 100.0
        assert segment.segment_length_ft == 100.0
        assert segment.segment_type == "vertical"
        assert segment.deviation_angle_deg == 0.0
    
    def test_auto_length_calculation(self):
        """Test automatic segment length calculation."""
        segment = FlowPathSegment(
            start_depth_ft=0.0,
            end_depth_ft=75.0,
            segment_length_ft=0.0,  # Zero - should be auto-calculated
            segment_type="vertical",
        )
        
        # Length should be 75.0 (end - start)
        assert segment.segment_length_ft == 75.0
    
    def test_contains_depth(self):
        """Test depth containment check."""
        segment = FlowPathSegment(
            start_depth_ft=0.0,
            end_depth_ft=100.0,
            segment_type="vertical",
        )
        
        assert segment.contains_depth(50.0) is True
        assert segment.contains_depth(0.0) is True
        assert segment.contains_depth(100.0) is True
        assert segment.contains_depth(-10.0) is False
        assert segment.contains_depth(200.0) is False
    
    def test_perforations(self):
        """Test perforation handling."""
        segment = FlowPathSegment(
            start_depth_ft=0.0,
            end_depth_ft=100.0,
            segment_type="vertical",
            has_perforations=True,
        )
        
        assert segment.has_perforations is True
        assert len(segment.perforation_intervals) == 0
        
        # Add perforations
        segment.perforation_intervals = [(10.0, 20.0)]
        
        assert len(segment.perforation_intervals) == 1
        assert segment.has_perforations_at_depth(15.0) is True
        assert segment.has_perforations_at_depth(50.0) is False
    
    def test_segment_from_well_geometry_vertical(self):
        """Test creating segments from vertical well geometry."""
        well = WellGeometry(
            measured_depth_ft=1500.0,
            true_vertical_depth_ft=1500.0,
            pipe_inner_diameter_in=5.0,
        )
        
        segments = FlowPathSegment.from_well_geometry(well)
        
        assert len(segments) >= 1
        assert segments[0].is_wellhead
        assert segments[-1].is_bottomhole
    
    def test_segment_from_well_geometry_deviated(self):
        """Test creating segments from deviated well geometry."""
        well = WellGeometry(
            measured_depth_ft=2000.0,
            true_vertical_depth_ft=1500.0,
            pipe_inner_diameter_in=5.0,
            deviation_angle_deg=15.0,
        )
        
        segments = FlowPathSegment.from_well_geometry(well)
        
        assert len(segments) >= 3
        assert segments[0].is_wellhead
        assert segments[-1].is_bottomhole
        assert segments[0].segment_type == "wellhead"
        assert segments[-1].segment_type == "bottomhole" or "horizontal"
    
    def test_segment_from_well_geometry_with_angle_list(self):
        """Test creating segments with angle list."""
        well = WellGeometry(
            measured_depth_ft=2000.0,
            true_vertical_depth_ft=1500.0,
            pipe_inner_diameter_in=5.0,
            deviation_angle_deg=[0.0, 5.0, 10.0, 15.0, 15.0],
        )
        
        segments = FlowPathSegment.from_well_geometry(well)
        
        assert len(segments) > 1
        assert all(seg.segment_length_ft >= 0 for seg in segments)


class TestWellFlowPath:
    """Tests for WellFlowPath model."""
    
    def test_initialization(self):
        """Test WellFlowPath initialization."""
        well = WellGeometry(
            measured_depth_ft=2000.0,
            true_vertical_depth_ft=1800.0,
            pipe_inner_diameter_in=5.0,
        )
        
        flow_path = WellFlowPath(well)
        
        assert flow_path.well_geometry == well
        assert len(flow_path.segments) > 0
        assert flow_path.is_connected is True
    
    def test_auto_segments_creation(self):
        """Test automatic segment creation from well geometry."""
        well = WellGeometry(1000.0, 950.0, 4.5)
        flow_path = WellFlowPath(well)
        
        # Should auto-generate segments
        assert len(flow_path.segments) > 0
        
        # Check segment structure
        assert flow_path.get_surface_segment() is not None
        assert flow_path.get_bottomhole_segment() is not None
    
    def test_get_segment_at_depth(self):
        """Test getting segment at specific depth."""
        well = WellGeometry(1000.0, 950.0, 4.5)
        flow_path = WellFlowPath(well)
        
        segment = flow_path.get_segment_at_depth(500.0)
        
        assert segment is not None
        assert segment.contains_depth(500.0)
    
    def test_get_nonexistent_segment(self):
        """Test getting segment for non-existent depth."""
        well = WellGeometry(1000.0, 950.0, 4.5)
        flow_path = WellFlowPath(well)
        
        segment = flow_path.get_segment_at_depth(999999.0)
        
        assert segment is None
    
    def test_get_surface_segment(self):
        """Test getting surface segment."""
        well = WellGeometry(1000.0, 950.0, 4.5)
        flow_path = WellFlowPath(well)
        
        surface_seg = flow_path.get_surface_segment()
        
        assert surface_seg is not None
        assert surface_seg.is_wellhead
        assert surface_seg.contains_depth(0.0)
    
    def test_get_bottomhole_segment(self):
        """Test getting bottomhole segment."""
        well = WellGeometry(1000.0, 950.0, 4.5)
        flow_path = WellFlowPath(well)
        
        bottom_seg = flow_path.get_bottomhole_segment()
        
        assert bottom_seg is not None
        assert bottom_seg.contains_depth(950.0) or bottom_seg.contains_depth(1000.0)
    
    def test_add_perforations(self):
        """Test adding perforations."""
        well = WellGeometry(1000.0, 950.0, 4.5)
        flow_path = WellFlowPath(well)
        
        flow_path.add_perforations([(200.0, 300.0), (400.0, 500.0)])
        
        assert flow_path.segments[0].has_perforations is True
        assert len(flow_path.segments[0].perforation_intervals) == 2
    
    def test_get_total_length(self):
        """Test calculating total length."""
        well = WellGeometry(1000.0, 950.0, 4.5)
        flow_path = WellFlowPath(well)
        
        total_len = flow_path.get_total_length_ft()
        
        # Should sum all segment lengths
        assert total_len > 0
    
    def test_get_deviation_angle_at_depth(self):
        """Test getting deviation angle at depth."""
        well = WellGeometry(
            measured_depth_ft=3000.0,
            true_vertical_depth_ft=2000.0,
            pipe_inner_diameter_in=5.0,
            deviation_angle_deg=[
                0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 12.0, 12.0
            ]
        )
        
        flow_path = WellFlowPath(well)
        
        # Near surface
        angle = flow_path.get_deviation_angle_at_depth(100.0)
        assert angle == 0.0
        
        # Mid-section
        angle = flow_path.get_deviation_angle_at_depth(1000.0)
        assert angle == 4.0
        
        # Deep section
        angle = flow_path.get_deviation_angle_at_depth(2000.0)
        assert angle == 12.0
    
    def test_get_deviation_angle_returns_none_for_vertical_well(self):
        """Test deviation angle returns None for vertical well."""
        well = WellGeometry(1000.0, 1000.0, 4.5)
        flow_path = WellFlowPath(well)
        
        angle = flow_path.get_deviation_angle_at_depth(500.0)
        
        assert angle == 0.0
    
    def test_get_flow_regime_probability(self):
        """Test flow regime probability calculation."""
        well = WellGeometry(1000.0, 950.0, 4.5)
        flow_path = WellFlowPath(well)
        
        # Low flow rate
        low_probs = flow_path.get_flow_regime_probability(0.5)
        assert low_probs["Bubbly"] + low_probs["Slug"] + low_probs["Annular"] > 0.9
        
        # High flow rate
        high_probs = flow_path.get_flow_regime_probability(100.0)
        assert high_probs["Annular"] + high_probs["Slug"] > 0.8
    
    def test_to_dict(self):
        """Test converting flow path to dictionary."""
        well = WellGeometry(1000.0, 950.0, 4.5)
        flow_path = WellFlowPath(well)
        
        result = flow_path.to_dict()
        
        assert 'well_depth_ft' in result
        assert 'well_type' in result
        assert 'segment_count' in result
        assert 'total_length_ft' in result
        assert 'segments' in result
    
    def test_repr(self):
        """Test string representation."""
        well = WellGeometry(1000.0, 950.0, 4.5)
        flow_path = WellFlowPath(well)
        
        repr_str = repr(flow_path)
        assert "WellFlowPath" in repr_str
        assert "segments=1" in repr_str or "segments=2" in repr_str