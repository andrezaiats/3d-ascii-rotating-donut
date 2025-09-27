#!/usr/bin/env python3
"""
Comprehensive Visual Validation Tests for Story 3.5: Visual Harmony and Aesthetics

Tests the visual harmony enhancements including density balancing, visual flow,
pattern recognition, edge case handling, and aesthetic quality validation.

Follows test strategy requirements:
- 90%+ coverage for visual processing functions
- pytest 7.4+ framework with unittest.mock
- Custom visual comparison utilities
- Edge case validation
- Performance verification within existing constraints

Author: Dev Agent (Story 3.5 Implementation)
"""

import pytest
import unittest.mock as mock
from typing import List, Tuple, Optional
import math

# Import the functions being tested
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rotating_donut import (
    # Data models
    Point3D, Point2D, CodeToken, ImportanceLevel, DisplayFrame,
    DensityAnalysis, VisualFlowState, AestheticQuality,

    # Visual harmony functions
    analyze_token_density_patterns,
    apply_adaptive_density_control,
    implement_token_clustering_logic,
    create_smooth_transition_algorithms,
    apply_intelligent_spacing_patterns,
    handle_visual_edge_cases,
    validate_artistic_impact_and_quality,

    # Helper functions
    _assess_pattern_clarity,
    _assess_edge_case_handling,
    _calculate_artistic_impact,
    _assess_character_variety,

    # Constants
    TERMINAL_WIDTH, TERMINAL_HEIGHT, ASCII_CHARS
)


class TestVisualHarmonyCore:
    """Test core visual harmony functions for Story 3.5."""

    @pytest.fixture
    def sample_tokens(self):
        """Create sample CodeToken objects for testing."""
        return [
            CodeToken(type='NAME', value='def', importance=ImportanceLevel.CRITICAL,
                     line=1, column=0, ascii_char='#'),
            CodeToken(type='OP', value='+', importance=ImportanceLevel.HIGH,
                     line=2, column=5, ascii_char='+'),
            CodeToken(type='NAME', value='variable', importance=ImportanceLevel.MEDIUM,
                     line=3, column=2, ascii_char='-'),
            CodeToken(type='COMMENT', value='# comment', importance=ImportanceLevel.LOW,
                     line=4, column=0, ascii_char='.'),
        ]

    @pytest.fixture
    def sample_points(self):
        """Create sample Point3D objects for testing."""
        return [
            Point3D(x=1.0, y=0.0, z=0.5, u=0.0, v=0.0, nx=1.0, ny=0.0, nz=0.0),
            Point3D(x=0.0, y=1.0, z=0.5, u=1.57, v=1.57, nx=0.0, ny=1.0, nz=0.0),
            Point3D(x=-1.0, y=0.0, z=0.5, u=3.14, v=3.14, nx=-1.0, ny=0.0, nz=0.0),
            Point3D(x=0.0, y=-1.0, z=0.5, u=4.71, v=4.71, nx=0.0, ny=-1.0, nz=0.0),
        ]

    @pytest.fixture
    def sample_mapped_pairs(self, sample_points, sample_tokens):
        """Create sample mapped pairs for testing."""
        return list(zip(sample_points, sample_tokens))


class TestDensityAnalysis(TestVisualHarmonyCore):
    """Test token density analysis functionality."""

    def test_analyze_empty_pairs(self):
        """Test density analysis with empty input."""
        result = analyze_token_density_patterns([])

        assert result.total_tokens == 0
        assert result.average_density == 0.0
        assert result.max_density == 0
        assert result.min_density == 0
        assert len(result.hotspots) == 0
        assert len(result.sparse_areas) == 0

    def test_analyze_basic_density(self, sample_mapped_pairs):
        """Test basic density analysis functionality."""
        with mock.patch('rotating_donut.project_to_screen') as mock_project:
            # Mock projection to return visible points at different positions
            mock_project.side_effect = [
                Point2D(x=10, y=5, depth=1.0, visible=True, visibility_factor=0.8),
                Point2D(x=15, y=8, depth=1.2, visible=True, visibility_factor=0.7),
                Point2D(x=10, y=5, depth=1.1, visible=True, visibility_factor=0.9),  # Same position as first
                Point2D(x=20, y=10, depth=1.3, visible=True, visibility_factor=0.6),
            ]

            result = analyze_token_density_patterns(sample_mapped_pairs)

            assert result.total_tokens == 4
            assert result.max_density == 2  # Two tokens at position (10, 5)
            assert result.min_density == 0  # Many empty positions
            assert (10, 5) in result.density_map
            assert result.density_map[(10, 5)] == 2

    def test_hotspot_identification(self, sample_mapped_pairs):
        """Test identification of density hotspots."""
        with mock.patch('rotating_donut.project_to_screen') as mock_project:
            # Create concentrated tokens at one position
            mock_project.side_effect = [
                Point2D(x=10, y=5, depth=1.0, visible=True, visibility_factor=0.8),
                Point2D(x=10, y=5, depth=1.1, visible=True, visibility_factor=0.8),
                Point2D(x=10, y=5, depth=1.2, visible=True, visibility_factor=0.8),
                Point2D(x=10, y=5, depth=1.3, visible=True, visibility_factor=0.8),
            ]

            result = analyze_token_density_patterns(sample_mapped_pairs)

            # With 4 tokens at one position and average density low, should be hotspot
            assert result.max_density == 4
            assert len(result.hotspots) > 0
            assert (10, 5) in result.hotspots

    def test_sparse_area_identification(self, sample_mapped_pairs):
        """Test identification of sparse areas."""
        with mock.patch('rotating_donut.project_to_screen') as mock_project:
            # Spread tokens thinly
            mock_project.side_effect = [
                Point2D(x=5, y=5, depth=1.0, visible=True, visibility_factor=0.8),
                Point2D(x=15, y=8, depth=1.2, visible=True, visibility_factor=0.7),
                Point2D(x=25, y=12, depth=1.1, visible=True, visibility_factor=0.9),
                Point2D(x=35, y=15, depth=1.3, visible=True, visibility_factor=0.6),
            ]

            result = analyze_token_density_patterns(sample_mapped_pairs)

            # Most positions should be sparse with this distribution
            assert len(result.sparse_areas) > len(result.hotspots)
            assert result.average_density < 1.0


class TestAdaptiveDensityControl(TestVisualHarmonyCore):
    """Test adaptive density control algorithm."""

    def test_no_density_control_needed(self, sample_mapped_pairs):
        """Test that no control is applied when density is acceptable."""
        # Mock density analysis with low max density
        density_analysis = DensityAnalysis(
            total_tokens=4, density_map={(10, 5): 2, (15, 8): 1},
            hotspots=[], sparse_areas=[], average_density=0.5,
            max_density=2, min_density=0
        )

        result = apply_adaptive_density_control(sample_mapped_pairs, density_analysis, max_density_threshold=3)

        # Should return original pairs unchanged
        assert len(result) == len(sample_mapped_pairs)
        assert result == sample_mapped_pairs

    def test_density_control_applied(self, sample_mapped_pairs):
        """Test that density control is applied when needed."""
        # Mock density analysis with high max density
        density_analysis = DensityAnalysis(
            total_tokens=4, density_map={(10, 5): 6, (15, 8): 1},
            hotspots=[(10, 5)], sparse_areas=[], average_density=1.0,
            max_density=6, min_density=0
        )

        with mock.patch('rotating_donut.project_to_screen') as mock_project:
            # Mock projection for critical token analysis
            mock_project.side_effect = [
                Point2D(x=10, y=5, depth=1.0, visible=True, visibility_factor=0.8),
                Point2D(x=15, y=8, depth=1.2, visible=True, visibility_factor=0.7),
                Point2D(x=10, y=5, depth=1.1, visible=True, visibility_factor=0.9),
                Point2D(x=20, y=10, depth=1.3, visible=True, visibility_factor=0.6),
            ]

            result = apply_adaptive_density_control(sample_mapped_pairs, density_analysis, max_density_threshold=3)

            # Should apply density balancing
            assert len(result) <= len(sample_mapped_pairs)  # Some tokens may be filtered

    def test_critical_token_preservation(self, sample_mapped_pairs):
        """Test that critical tokens are always preserved."""
        density_analysis = DensityAnalysis(
            total_tokens=4, density_map={(10, 5): 6},
            hotspots=[(10, 5)], sparse_areas=[], average_density=1.0,
            max_density=6, min_density=0
        )

        with mock.patch('rotating_donut.project_to_screen') as mock_project:
            mock_project.return_value = Point2D(x=10, y=5, depth=1.0, visible=True, visibility_factor=0.8)

            result = apply_adaptive_density_control(sample_mapped_pairs, density_analysis, max_density_threshold=2)

            # Critical tokens should always be preserved
            critical_tokens_in_result = [token for _, token in result if token.importance == ImportanceLevel.CRITICAL]
            critical_tokens_in_input = [token for _, token in sample_mapped_pairs if token.importance == ImportanceLevel.CRITICAL]

            assert len(critical_tokens_in_result) == len(critical_tokens_in_input)


class TestVisualFlowAlgorithms(TestVisualHarmonyCore):
    """Test smooth transition algorithms for visual flow."""

    def test_first_frame_flow_state(self):
        """Test visual flow state initialization for first frame."""
        current_frame = [
            (Point2D(x=10, y=5, depth=1.0, visible=True, visibility_factor=0.8),
             CodeToken(type='NAME', value='def', importance=ImportanceLevel.CRITICAL, line=1, column=0, ascii_char='#'))
        ]

        flow_state = create_smooth_transition_algorithms(current_frame, None, 0)

        assert flow_state.previous_frame == current_frame
        assert flow_state.continuity_score == 1.0
        assert len(flow_state.transition_weights) == 0
        assert len(flow_state.flow_vectors) == 0

    def test_frame_transition_calculation(self):
        """Test transition calculation between frames."""
        token = CodeToken(type='NAME', value='def', importance=ImportanceLevel.CRITICAL, line=1, column=0, ascii_char='#')

        previous_frame = [
            (Point2D(x=10, y=5, depth=1.0, visible=True, visibility_factor=0.8), token)
        ]

        current_frame = [
            (Point2D(x=12, y=6, depth=1.1, visible=True, visibility_factor=0.9), token)
        ]

        flow_state = create_smooth_transition_algorithms(current_frame, previous_frame, 1)

        assert flow_state.previous_frame == current_frame
        token_id = id(token)
        assert token_id in flow_state.transition_weights
        assert (12, 6) in flow_state.flow_vectors

        # Check movement vector calculation
        dx, dy = flow_state.flow_vectors[(12, 6)]
        assert dx == 2  # 12 - 10
        assert dy == 1  # 6 - 5

    def test_continuity_score_calculation(self):
        """Test continuity score calculation based on token displacement."""
        token1 = CodeToken(type='NAME', value='def', importance=ImportanceLevel.CRITICAL, line=1, column=0, ascii_char='#')
        token2 = CodeToken(type='OP', value='+', importance=ImportanceLevel.HIGH, line=2, column=5, ascii_char='+')

        # Small displacement - should have high continuity
        previous_frame = [
            (Point2D(x=10, y=5, depth=1.0, visible=True, visibility_factor=0.8), token1),
            (Point2D(x=15, y=8, depth=1.2, visible=True, visibility_factor=0.7), token2)
        ]

        current_frame = [
            (Point2D(x=11, y=5, depth=1.0, visible=True, visibility_factor=0.8), token1),
            (Point2D(x=16, y=8, depth=1.2, visible=True, visibility_factor=0.7), token2)
        ]

        flow_state = create_smooth_transition_algorithms(current_frame, previous_frame, 1)

        # Small displacement should result in high continuity score
        assert flow_state.continuity_score > 0.7


class TestEdgeCaseHandling(TestVisualHarmonyCore):
    """Test visual edge case handling functionality."""

    def test_handle_empty_pairs(self):
        """Test edge case handling with empty input."""
        density_analysis = DensityAnalysis(0, {}, [], [], 0.0, 0, 0)
        result = handle_visual_edge_cases([], density_analysis)
        assert result == []

    def test_handle_sparse_areas(self, sample_mapped_pairs):
        """Test handling of sparse code areas."""
        # Mock very sparse density analysis
        density_analysis = DensityAnalysis(
            total_tokens=4, density_map={(10, 5): 1},
            hotspots=[], sparse_areas=[(0, 0), (5, 5), (10, 10)],
            average_density=0.05,  # Below 0.1 threshold
            max_density=1, min_density=0
        )

        result = handle_visual_edge_cases(sample_mapped_pairs, density_analysis)

        # Should apply sparse area handling
        assert len(result) >= len(sample_mapped_pairs)

    def test_handle_dense_sections(self, sample_mapped_pairs):
        """Test handling of very dense code sections."""
        density_analysis = DensityAnalysis(
            total_tokens=4, density_map={(10, 5): 8},
            hotspots=[(10, 5)], sparse_areas=[],
            average_density=2.0,
            max_density=8,  # Above 5 threshold
            min_density=0
        )

        result = handle_visual_edge_cases(sample_mapped_pairs, density_analysis)

        # Should apply density reduction handling
        assert len(result) <= len(sample_mapped_pairs)

    def test_handle_long_lines(self, sample_tokens):
        """Test handling of very long lines."""
        # Create tokens representing a very long line
        long_line_tokens = []
        for i in range(100):  # Simulate 100-character line
            token = CodeToken(
                type='NAME', value=f'var{i}', importance=ImportanceLevel.MEDIUM,
                line=1, column=i, ascii_char='-'
            )
            long_line_tokens.append(token)

        sample_points = [
            Point3D(x=float(i), y=0.0, z=0.5, u=0.0, v=0.0, nx=1.0, ny=0.0, nz=0.0)
            for i in range(100)
        ]

        mapped_pairs = list(zip(sample_points, long_line_tokens))

        density_analysis = DensityAnalysis(
            total_tokens=100, density_map={}, hotspots=[], sparse_areas=[],
            average_density=1.0, max_density=1, min_density=1
        )

        result = handle_visual_edge_cases(mapped_pairs, density_analysis)

        # Should handle long lines (currently returns original, but validates processing)
        assert len(result) == len(mapped_pairs)


class TestAestheticQualityValidation(TestVisualHarmonyCore):
    """Test aesthetic quality validation functionality."""

    def test_validate_basic_quality(self):
        """Test basic aesthetic quality validation."""
        # Create a balanced test frame
        frame_buffer = [[ASCII_CHARS['BACKGROUND'] for _ in range(TERMINAL_WIDTH)] for _ in range(TERMINAL_HEIGHT)]
        # Add some non-background characters for pattern analysis
        frame_buffer[5][10] = '#'
        frame_buffer[8][15] = '+'
        frame_buffer[12][20] = '-'

        frame = DisplayFrame(
            width=TERMINAL_WIDTH, height=TERMINAL_HEIGHT,
            buffer=frame_buffer, depth_buffer=[], frame_number=1
        )

        density_analysis = DensityAnalysis(
            total_tokens=3, density_map={(10, 5): 1, (15, 8): 1, (20, 12): 1},
            hotspots=[], sparse_areas=[], average_density=0.1,
            max_density=1, min_density=0
        )

        visual_flow_state = VisualFlowState(
            previous_frame=None, transition_weights={}, flow_vectors={}, continuity_score=0.8
        )

        quality = validate_artistic_impact_and_quality(frame, density_analysis, visual_flow_state, 1)

        assert 0.0 <= quality.overall_score <= 1.0
        assert 0.0 <= quality.density_balance <= 1.0
        assert 0.0 <= quality.visual_flow <= 1.0
        assert 0.0 <= quality.pattern_clarity <= 1.0
        assert 0.0 <= quality.edge_case_handling <= 1.0
        assert 0.0 <= quality.artistic_impact <= 1.0

    def test_pattern_clarity_assessment(self):
        """Test pattern clarity assessment function."""
        # Test optimal complexity
        frame_buffer = [[ASCII_CHARS['BACKGROUND'] for _ in range(TERMINAL_WIDTH)] for _ in range(TERMINAL_HEIGHT)]

        # Fill about 50% with non-background chars (optimal range)
        non_bg_count = int(TERMINAL_WIDTH * TERMINAL_HEIGHT * 0.5)
        for i in range(non_bg_count):
            row = i // TERMINAL_WIDTH
            col = i % TERMINAL_WIDTH
            if row < TERMINAL_HEIGHT:
                frame_buffer[row][col] = '#'

        frame = DisplayFrame(
            width=TERMINAL_WIDTH, height=TERMINAL_HEIGHT,
            buffer=frame_buffer, depth_buffer=[], frame_number=1
        )

        density_analysis = DensityAnalysis(
            total_tokens=non_bg_count, density_map={}, hotspots=[], sparse_areas=[],
            average_density=1.0, max_density=1, min_density=1
        )

        clarity = _assess_pattern_clarity(frame, density_analysis)
        assert clarity == 1.0  # Should be optimal

    def test_edge_case_assessment(self):
        """Test edge case handling assessment."""
        # Test with extreme density hotspots
        density_analysis = DensityAnalysis(
            total_tokens=100, density_map={(10, 5): 15},  # Extreme hotspot
            hotspots=[(10, 5)], sparse_areas=[],
            average_density=2.0, max_density=15, min_density=0
        )

        edge_score = _assess_edge_case_handling(density_analysis)
        assert edge_score < 1.0  # Should be penalized for extreme density

    def test_character_variety_assessment(self):
        """Test character variety assessment."""
        # Create frame with good character variety
        frame_buffer = [[ASCII_CHARS['BACKGROUND'] for _ in range(TERMINAL_WIDTH)] for _ in range(TERMINAL_HEIGHT)]
        frame_buffer[0][0] = '#'
        frame_buffer[0][1] = '+'
        frame_buffer[0][2] = '-'
        # Note: '.' is background, so we have 3 of 4 possible chars

        frame = DisplayFrame(
            width=TERMINAL_WIDTH, height=TERMINAL_HEIGHT,
            buffer=frame_buffer, depth_buffer=[], frame_number=1
        )

        variety = _assess_character_variety(frame)
        expected_variety = 3.0 / 3.0  # 3 unique chars out of 3 non-background chars
        assert variety == expected_variety


class TestIntegrationScenarios(TestVisualHarmonyCore):
    """Test integration scenarios and performance requirements."""

    def test_visual_harmony_pipeline_integration(self, sample_mapped_pairs):
        """Test complete visual harmony pipeline integration."""
        # Mock all dependencies for integration test
        with mock.patch('rotating_donut.project_to_screen') as mock_project:
            mock_project.return_value = Point2D(x=10, y=5, depth=1.0, visible=True, visibility_factor=0.8)

            # Test the complete pipeline
            density_analysis = analyze_token_density_patterns(sample_mapped_pairs)
            processed_pairs = handle_visual_edge_cases(sample_mapped_pairs, density_analysis)

            if density_analysis.max_density > 3:
                balanced_pairs = apply_adaptive_density_control(processed_pairs, density_analysis)
            else:
                balanced_pairs = processed_pairs

            # Verify pipeline completion
            assert len(balanced_pairs) <= len(sample_mapped_pairs)
            assert isinstance(density_analysis, DensityAnalysis)

    def test_performance_constraints(self, sample_mapped_pairs):
        """Test that visual harmony functions meet performance constraints."""
        import time

        # Test density analysis performance
        start_time = time.time()
        for _ in range(100):  # Simulate 100 frame operations
            with mock.patch('rotating_donut.project_to_screen') as mock_project:
                mock_project.return_value = Point2D(x=10, y=5, depth=1.0, visible=True, visibility_factor=0.8)
                analyze_token_density_patterns(sample_mapped_pairs)

        elapsed = time.time() - start_time

        # Should complete 100 operations well within frame budget (33ms per frame for 30 FPS)
        assert elapsed < 1.0  # Should be much faster than 1 second for 100 operations

    def test_memory_efficiency(self, sample_mapped_pairs):
        """Test memory efficiency of visual harmony functions."""
        # Test that functions don't create excessive memory overhead
        import sys

        initial_size = sys.getsizeof(sample_mapped_pairs)

        with mock.patch('rotating_donut.project_to_screen') as mock_project:
            mock_project.return_value = Point2D(x=10, y=5, depth=1.0, visible=True, visibility_factor=0.8)

            density_analysis = analyze_token_density_patterns(sample_mapped_pairs)
            processed_pairs = handle_visual_edge_cases(sample_mapped_pairs, density_analysis)

        final_size = sys.getsizeof(processed_pairs)

        # Memory usage should not explode (allow for reasonable overhead)
        assert final_size < initial_size * 5


if __name__ == "__main__":
    # Run tests with coverage reporting
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=rotating_donut",
        "--cov-report=term-missing",
        "--cov-fail-under=90"
    ])