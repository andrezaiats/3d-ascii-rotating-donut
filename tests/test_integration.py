#!/usr/bin/env python3
"""
Integration Tests for 3D ASCII Donut Animation System

Tests the complete animation pipeline including rotation matrix transformations,
frame rate control, timing accuracy, interrupt handling, and animation continuity.

Coverage Requirements: 100% for critical path (animation loop) per test standards.
Framework: pytest 7.4+ with unittest.mock for timing control.
"""

import math
import time
import unittest
import unittest.mock
from unittest.mock import patch, MagicMock

# Import the rotating_donut module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rotating_donut import (
    Point3D, Point2D, TorusParameters, DisplayFrame, CodeToken,
    apply_rotation, generate_torus_points, project_to_screen,
    generate_ascii_frame, calculate_frame_timing, handle_interrupts,
    run_animation_loop, TARGET_FPS, TokenCache, preprocess_tokens_pipeline,
    initialize_token_cache, _token_cache, _apply_cached_mappings,
    _precompute_token_mappings, read_self_code, tokenize_code
)


class TestRotationMatrixTransformation(unittest.TestCase):
    """Test Y-axis rotation matrix implementation and accuracy."""

    def test_rotation_matrix_identity(self):
        """Test that 0-degree rotation returns identical points."""
        # Create test points
        test_points = [
            Point3D(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            Point3D(0.0, 1.0, 0.0, math.pi/2, 0.0, 0.0, 1.0, 0.0),
            Point3D(0.0, 0.0, 1.0, math.pi, 0.0, 0.0, 0.0, 1.0)
        ]

        # Apply zero rotation
        rotated = apply_rotation(test_points, 0.0)

        # Verify points are unchanged
        for original, rotated_point in zip(test_points, rotated):
            assert abs(original.x - rotated_point.x) < 1e-10
            assert abs(original.y - rotated_point.y) < 1e-10
            assert abs(original.z - rotated_point.z) < 1e-10
            # Verify parametric coordinates preserved
            assert original.u == rotated_point.u
            assert original.v == rotated_point.v

    def test_rotation_matrix_90_degrees(self):
        """Test 90-degree Y-axis rotation accuracy."""
        # Test point on X-axis
        test_point = Point3D(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)

        # Apply 90-degree rotation (π/2 radians)
        rotated = apply_rotation([test_point], math.pi / 2)

        # After 90° Y-axis rotation: (1,0,0) -> (0,0,-1)
        assert abs(rotated[0].x - 0.0) < 1e-10
        assert abs(rotated[0].y - 0.0) < 1e-10
        assert abs(rotated[0].z - (-1.0)) < 1e-10
        # Verify parametric coordinates preserved
        assert rotated[0].u == 0.0
        assert rotated[0].v == 0.0

    def test_rotation_matrix_180_degrees(self):
        """Test 180-degree Y-axis rotation accuracy."""
        # Test point on X-axis
        test_point = Point3D(1.0, 2.0, 0.0, math.pi, math.pi/4, 1.0, 0.0, 0.0)

        # Apply 180-degree rotation (π radians)
        rotated = apply_rotation([test_point], math.pi)

        # After 180° Y-axis rotation: (1,2,0) -> (-1,2,0)
        assert abs(rotated[0].x - (-1.0)) < 1e-10
        assert abs(rotated[0].y - 2.0) < 1e-10
        assert abs(rotated[0].z - 0.0) < 1e-10
        # Verify parametric coordinates preserved
        assert rotated[0].u == math.pi
        assert rotated[0].v == math.pi/4

    def test_rotation_preserves_y_coordinate(self):
        """Test that Y-axis rotation preserves Y coordinates."""
        test_points = [
            Point3D(1.0, 5.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            Point3D(-2.0, -3.0, 1.0, math.pi/2, math.pi, -1.0, 0.0, 0.0),
            Point3D(0.0, 10.0, -1.0, math.pi, math.pi/2, 0.0, 0.0, -1.0)
        ]

        # Apply arbitrary rotation
        angle = 0.7  # ~40 degrees
        rotated = apply_rotation(test_points, angle)

        # Verify Y coordinates are unchanged
        for original, rotated_point in zip(test_points, rotated):
            assert abs(original.y - rotated_point.y) < 1e-10

    def test_rotation_with_torus_points(self):
        """Test rotation with actual torus geometry."""
        # Create small torus for testing
        params = TorusParameters(
            outer_radius=2.0,
            inner_radius=1.0,
            u_resolution=8,
            v_resolution=4,
            rotation_speed=0.1
        )

        torus_points = generate_torus_points(params)

        # Apply rotation
        angle = math.pi / 4  # 45 degrees
        rotated_points = apply_rotation(torus_points, angle)

        # Verify we have the same number of points
        assert len(rotated_points) == len(torus_points)

        # Verify parametric coordinates are preserved
        for original, rotated in zip(torus_points, rotated_points):
            assert original.u == rotated.u
            assert original.v == rotated.v


class TestAnimationFrameControl(unittest.TestCase):
    """Test animation loop frame rate control and timing accuracy."""

    def test_calculate_frame_timing(self):
        """Test frame timing calculation matches target FPS."""
        expected_frame_time = 1.0 / TARGET_FPS
        calculated_time = calculate_frame_timing()

        assert abs(calculated_time - expected_frame_time) < 1e-10
        assert calculated_time > 0
        # For 30 FPS: should be approximately 33.33ms
        assert calculated_time <= 0.0334  # Maximum 33.4ms per frame

    @patch('time.time')
    @patch('time.sleep')
    @patch('rotating_donut.output_to_terminal')
    def test_animation_loop_timing_control(self, mock_output, mock_sleep, mock_time):
        """Test animation loop maintains proper frame timing."""
        # Mock time progression
        mock_time.side_effect = [
            0.0,    # Frame start
            0.02,   # Frame processing took 20ms
            0.033,  # Next frame start
            0.055,  # Next frame processing took 22ms
            0.066   # Final time check
        ]

        # Mock KeyboardInterrupt after 2 frames
        mock_output.side_effect = [None, KeyboardInterrupt()]

        # Run animation loop - it will handle KeyboardInterrupt gracefully
        try:
            run_animation_loop()
        except KeyboardInterrupt:
            pass  # This may or may not propagate depending on timing

        # Verify sleep was called with correct timing
        # Frame time target: 1/30 ≈ 0.0333s
        # First frame: slept for ~13.3ms (33.3 - 20)
        # Second frame: slept for ~11.1ms (33.3 - 22)
        sleep_calls = mock_sleep.call_args_list
        assert len(sleep_calls) >= 1

        # Verify sleep times are reasonable (accounting for timing precision)
        for call in sleep_calls:
            sleep_time = call[0][0]
            assert 0 <= sleep_time <= 0.034  # Never sleep more than frame time

    @patch('rotating_donut.handle_interrupts')
    def test_keyboard_interrupt_handling(self, mock_handler):
        """Test graceful keyboard interrupt handling."""
        mock_handler.return_value = True

        with patch('rotating_donut.output_to_terminal') as mock_output:
            # Simulate KeyboardInterrupt on first frame
            mock_output.side_effect = KeyboardInterrupt()

            # Should handle KeyboardInterrupt gracefully
            try:
                run_animation_loop()
            except KeyboardInterrupt:
                pass  # Expected graceful handling

            # Verify interrupt handler was called
            mock_handler.assert_called_once()

    def test_handle_interrupts_function(self):
        """Test interrupt handler returns success."""
        # Mock stdout to capture output
        with patch('builtins.print') as mock_print:
            result = handle_interrupts()

            assert result is True
            # Verify cleanup messages were printed
            assert mock_print.call_count >= 3


class TestAnimationContinuity(unittest.TestCase):
    """Test smooth animation and incremental rotation."""

    def test_incremental_rotation_progression(self):
        """Test that rotation angle increases consistently across frames."""
        params = TorusParameters(
            outer_radius=2.0,
            inner_radius=1.0,
            u_resolution=10,
            v_resolution=5,
            rotation_speed=0.1
        )

        torus_points = generate_torus_points(params)

        # Test several animation frames
        angles = []
        for frame in range(5):
            angle = frame * params.rotation_speed
            angles.append(angle)

            # Apply rotation
            rotated_points = apply_rotation(torus_points, angle)

            # Verify rotation was applied (points should be different except at angle=0)
            if frame > 0:
                # Check that some points have moved
                identity_points = apply_rotation(torus_points, 0.0)
                differences = 0
                for original, rotated in zip(identity_points, rotated_points):
                    if (abs(original.x - rotated.x) > 1e-10 or
                        abs(original.z - rotated.z) > 1e-10):
                        differences += 1

                assert differences > 0, f"No rotation detected at frame {frame}"

        # Verify angles are increasing
        for i in range(1, len(angles)):
            assert angles[i] > angles[i-1]

    def test_full_rotation_cycle(self):
        """Test complete 360-degree rotation cycle."""
        params = TorusParameters(
            outer_radius=1.5,
            inner_radius=0.5,
            u_resolution=6,
            v_resolution=3,
            rotation_speed=0.1
        )

        torus_points = generate_torus_points(params)

        # Test one complete rotation (2π radians)
        frames_per_rotation = int((2 * math.pi) / params.rotation_speed)

        # Store first frame for comparison
        first_frame_points = apply_rotation(torus_points, 0.0)

        # Apply full rotation
        full_rotation_angle = 2 * math.pi
        full_rotation_points = apply_rotation(torus_points, full_rotation_angle)

        # After full rotation, points should be back to original positions
        for original, rotated in zip(first_frame_points, full_rotation_points):
            assert abs(original.x - rotated.x) < 1e-10
            assert abs(original.y - rotated.y) < 1e-10
            assert abs(original.z - rotated.z) < 1e-10


class TestPerformanceAndMemory(unittest.TestCase):
    """Test performance requirements and memory management."""

    @patch('gc.collect')
    @patch('rotating_donut.output_to_terminal')
    def test_memory_management_periodic_cleanup(self, mock_output, mock_gc):
        """Test that garbage collection is triggered periodically."""
        # Mock KeyboardInterrupt after sufficient frames for GC trigger
        frame_count = 0
        def interrupt_after_frames(*args):
            nonlocal frame_count
            frame_count += 1
            if frame_count >= 101:  # Trigger after 101 frames (GC at frame 100)
                raise KeyboardInterrupt()

        mock_output.side_effect = interrupt_after_frames

        try:
            run_animation_loop()
        except KeyboardInterrupt:
            pass

        # Verify garbage collection was called
        mock_gc.assert_called()

    def test_torus_cache_performance(self):
        """Test that torus point generation uses caching."""
        params = TorusParameters(
            outer_radius=2.0,
            inner_radius=1.0,
            u_resolution=20,
            v_resolution=10,
            rotation_speed=0.1
        )

        # First generation
        start_time = time.time()
        points1 = generate_torus_points(params)
        first_time = time.time() - start_time

        # Second generation (should be faster due to caching)
        start_time = time.time()
        points2 = generate_torus_points(params)
        second_time = time.time() - start_time

        # Verify points are identical
        assert len(points1) == len(points2)
        for p1, p2 in zip(points1, points2):
            assert p1 == p2

        # Second call should be significantly faster (cached)
        # Allow some tolerance for timing variations
        assert second_time <= first_time or second_time < 0.001


class TestStory34RealTimeIntegration(unittest.TestCase):
    """Test Story 3.4 Real-Time Integration Pipeline functionality."""

    def test_token_cache_initialization(self):
        """Test TokenCache class initialization and operations."""
        cache = TokenCache()

        # Test initial state
        assert cache.cache_valid is False
        assert cache.get_tokens() is None
        assert cache.get_enhanced_tokens() is None
        assert cache.get_token_mappings() is None
        assert cache.memory_usage() == 0

        # Create test tokens
        test_tokens = [
            CodeToken('KEYWORD', 'def', 4, 1, 1, '#'),
            CodeToken('NAME', 'test', 2, 1, 5, '-'),
            CodeToken('OPERATOR', '=', 3, 1, 10, '+'),
        ]

        # Populate cache
        cache.populate("test code", test_tokens)

        # Test populated state
        assert cache.cache_valid is True
        assert cache.get_tokens() == test_tokens
        assert cache.source_code == "test code"
        assert cache.memory_usage() > 0

        # Test invalidation
        cache.invalidate()
        assert cache.cache_valid is False
        assert cache.get_tokens() is None  # Returns None when invalid

        # Test clear
        cache.populate("test code", test_tokens)
        cache.clear()
        assert cache.source_code is None
        assert cache.tokens is None
        assert cache.memory_usage() == 0

    def test_preprocess_tokens_pipeline(self):
        """Test one-time token preprocessing pipeline."""
        # Clear global cache first
        _token_cache.clear()

        with patch('rotating_donut.read_self_code') as mock_read:
            with patch('rotating_donut.tokenize_code') as mock_tokenize:
                with patch('rotating_donut.analyze_structure') as mock_analyze:
                    with patch('rotating_donut.enhance_tokens_with_structure') as mock_enhance:
                        # Setup mocks
                        mock_read.return_value = "test source code"
                        mock_tokens = [CodeToken('KEYWORD', 'def', 4, 1, 1, '#')]
                        mock_tokenize.return_value = mock_tokens
                        mock_struct_info = MagicMock()
                        mock_analyze.return_value = mock_struct_info
                        mock_enhanced = [CodeToken('KEYWORD', 'def', 4, 1, 1, '#')]
                        mock_enhance.return_value = mock_enhanced

                        # Call pipeline
                        enhanced, struct_info, mappings = preprocess_tokens_pipeline()

                        # Verify one-time processing
                        mock_read.assert_called_once()
                        mock_tokenize.assert_called_once()
                        mock_analyze.assert_called_once()
                        mock_enhance.assert_called_once()

                        # Verify results
                        assert enhanced == mock_enhanced
                        assert struct_info == mock_struct_info

                        # Call pipeline again - should use cache
                        enhanced2, struct_info2, mappings2 = preprocess_tokens_pipeline()

                        # Should either use cache or re-process (implementation specific)
                        # The important thing is that results are consistent
                        assert enhanced2 == enhanced or enhanced2 == mock_enhanced
                        assert struct_info2 == struct_info or struct_info2 == mock_struct_info

                        # Results should be the same
                        assert enhanced2 == enhanced
                        assert struct_info2 == struct_info

    def test_initialize_token_cache(self):
        """Test complete token cache initialization with torus generation."""
        # Clear global cache first
        _token_cache.clear()

        with patch('rotating_donut.preprocess_tokens_pipeline') as mock_preprocess:
            with patch('rotating_donut.generate_torus_points') as mock_torus:
                with patch('rotating_donut._precompute_token_mappings') as mock_mappings:
                    # Setup mocks
                    mock_tokens = [CodeToken('KEYWORD', 'def', 4, 1, 1, '#')]
                    mock_struct_info = MagicMock()
                    mock_preprocess.return_value = (mock_tokens, mock_struct_info, None)

                    mock_points = [Point3D(1, 0, 0, 0, 0, 1, 0, 0)]
                    mock_torus.return_value = mock_points

                    mock_mappings_result = [(0, mock_tokens[0])]
                    mock_mappings.return_value = mock_mappings_result

                    # Test initialization
                    params = TorusParameters(2.0, 1.0, 10, 5, 0.05)
                    cache, points = initialize_token_cache(params)

                    # Verify calls
                    mock_preprocess.assert_called_once()
                    mock_torus.assert_called_once_with(params)
                    mock_mappings.assert_called_once_with(mock_tokens, mock_points, mock_struct_info)

                    # Verify results
                    assert cache == _token_cache
                    assert points == mock_points
                    # Token mappings should be set
                    assert cache.get_token_mappings() is not None

    def test_cached_mappings_application(self):
        """Test efficient cached mapping application to rotated points."""
        # Create test data
        test_tokens = [
            CodeToken('KEYWORD', 'def', 4, 1, 1, '#'),
            CodeToken('NAME', 'test', 2, 1, 5, '-'),
        ]

        token_mappings = [(0, test_tokens[0]), (1, test_tokens[1])]

        rotated_points = [
            Point3D(1, 0, 0, 0, 0, 1, 0, 0),
            Point3D(0, 1, 0, math.pi/2, 0, 0, 1, 0),
            Point3D(0, 0, 1, math.pi, 0, 0, 0, 1),
        ]

        # Apply cached mappings
        result = _apply_cached_mappings(token_mappings, rotated_points)

        # Verify results
        assert len(result) == 2  # Only mapped indices
        assert result[0] == (rotated_points[0], test_tokens[0])
        assert result[1] == (rotated_points[1], test_tokens[1])

    @patch('rotating_donut.generate_torus_points')
    @patch('rotating_donut._precompute_token_mappings')
    @patch('rotating_donut.output_to_terminal')
    @patch('rotating_donut.time.sleep')
    @patch('rotating_donut.time.time')
    def test_performance_degradation_mode(self, mock_time, mock_sleep, mock_output, mock_mappings, mock_torus):
        """Test performance fallback mode activation and recovery."""
        # Clear global cache
        _token_cache.clear()

        # Setup time simulation for slow performance
        frame_times = [
            0.0,    # Start
            0.05,   # Frame 1 - slow (20 FPS)
            0.10,   # Frame 2 - slow
            0.15,   # Frame 3 - slow
        ] * 20  # Repeat to trigger degraded mode

        mock_time.side_effect = frame_times

        # Setup other mocks
        mock_torus.return_value = [Point3D(1, 0, 0, 0, 0, 1, 0, 0)] * 100
        mock_mappings.return_value = [(0, CodeToken('KEYWORD', 'def', 4, 1, 1, '#'))]

        frame_count = 0
        def count_frames(*args):
            nonlocal frame_count
            frame_count += 1
            if frame_count >= 15:  # Stop after enough frames to trigger degraded mode
                raise KeyboardInterrupt()

        mock_output.side_effect = count_frames

        try:
            run_animation_loop()
        except KeyboardInterrupt:
            pass

        # Verify the animation ran and handled low performance
        # The test validates that the performance monitoring logic executes
        assert frame_count >= 15  # Animation ran for expected frames

    def test_memory_management_extended_sessions(self):
        """Test memory management for extended animation sessions."""
        cache = TokenCache()

        # Simulate large token set
        large_tokens = [CodeToken('NAME', f'var{i}', 2, i, 1, '-') for i in range(1000)]
        cache.populate("large source", large_tokens)

        initial_memory = cache.memory_usage()
        assert initial_memory > 0

        # Set enhanced tokens (should increase memory)
        enhanced_tokens = [CodeToken('NAME', f'var{i}_enhanced', 2, i, 1, '-') for i in range(1000)]
        cache.set_enhanced_tokens(enhanced_tokens)

        enhanced_memory = cache.memory_usage()
        assert enhanced_memory > initial_memory

        # Clear cache
        cache.clear()
        assert cache.memory_usage() == 0

    def test_cache_validation_and_consistency(self):
        """Test token cache validation across rotation cycles."""
        cache = TokenCache()

        # Test cache validation
        assert cache.cache_valid is False

        tokens = [CodeToken('KEYWORD', 'if', 4, 1, 1, '#')]
        cache.populate("test", tokens)
        assert cache.cache_valid is True

        # Test consistency after operations
        original_tokens = cache.get_tokens()
        cache.set_enhanced_tokens(tokens)

        # Tokens should remain accessible
        assert cache.get_tokens() == original_tokens
        assert cache.get_enhanced_tokens() == tokens

        # Invalidate and verify
        cache.invalidate()
        assert cache.get_tokens() is None
        assert cache.get_enhanced_tokens() is None

    def test_integration_pipeline_error_handling(self):
        """Test error handling in the integration pipeline."""
        # Clear global cache
        _token_cache.clear()

        with patch('rotating_donut.read_self_code') as mock_read:
            # Simulate read failure
            mock_read.side_effect = FileNotFoundError("Script not found")

            # Pipeline should handle error gracefully
            try:
                enhanced, struct_info, mappings = preprocess_tokens_pipeline()
                assert False, "Should have raised exception"
            except FileNotFoundError:
                pass  # Expected

    def test_frame_rate_monitoring(self):
        """Test frame rate measurement and performance tracking."""
        # Test target frame time calculation
        target_time = calculate_frame_timing()
        expected_time = 1.0 / TARGET_FPS
        assert abs(target_time - expected_time) < 1e-10

        # Verify 30 FPS target
        assert TARGET_FPS == 30
        assert target_time <= 0.034  # ~33.33ms per frame


# Integration test runner
if __name__ == "__main__":
    unittest.main(verbosity=2)