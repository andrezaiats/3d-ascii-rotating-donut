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
    Point3D, Point2D, TorusParameters, DisplayFrame,
    apply_rotation, generate_torus_points, project_to_screen,
    generate_ascii_frame, calculate_frame_timing, handle_interrupts,
    run_animation_loop, TARGET_FPS
)


class TestRotationMatrixTransformation(unittest.TestCase):
    """Test Y-axis rotation matrix implementation and accuracy."""

    def test_rotation_matrix_identity(self):
        """Test that 0-degree rotation returns identical points."""
        # Create test points
        test_points = [
            Point3D(1.0, 0.0, 0.0, 0.0, 0.0),
            Point3D(0.0, 1.0, 0.0, math.pi/2, 0.0),
            Point3D(0.0, 0.0, 1.0, math.pi, 0.0)
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
        test_point = Point3D(1.0, 0.0, 0.0, 0.0, 0.0)

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
        test_point = Point3D(1.0, 2.0, 0.0, math.pi, math.pi/4)

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
            Point3D(1.0, 5.0, 2.0, 0.0, 0.0),
            Point3D(-2.0, -3.0, 1.0, math.pi/2, math.pi),
            Point3D(0.0, 10.0, -1.0, math.pi, math.pi/2)
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


# Integration test runner
if __name__ == "__main__":
    unittest.main(verbosity=2)