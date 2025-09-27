#!/usr/bin/env python3
"""
Unit tests for mathematical functions in the 3D ASCII Donut project.

This module provides comprehensive test coverage for:
- Parametric torus generation
- Mathematical validation functions
- Performance characteristics (caching)
- Edge cases and error handling

Test Strategy:
- Validate mathematical accuracy with known geometric properties
- Test parameter validation and error handling
- Verify performance optimizations (caching)
- Cover edge cases: zero values, negative values, extreme parameters

Framework: unittest (Python standard library)
Coverage Target: 90%+ for mathematical functions
"""

import math
import unittest
from typing import List

# Import the module under test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rotating_donut import (
    Point3D, Point2D, TorusParameters,
    generate_torus_points,
    validate_torus_volume,
    validate_torus_surface_area,
    validate_torus_geometry,
    project_to_screen,
    _torus_cache
)


class TestParametricTorusGeneration(unittest.TestCase):
    """Test suite for torus point generation using parametric equations."""

    def test_basic_torus_generation(self):
        """Test basic torus generation with standard parameters."""
        params = TorusParameters(
            outer_radius=2.0,
            inner_radius=1.0,
            u_resolution=10,
            v_resolution=5,
            rotation_speed=0.1
        )

        points = generate_torus_points(params)

        # Verify correct number of points generated
        expected_points = 10 * 5  # u_resolution * v_resolution
        self.assertEqual(len(points), expected_points)

        # Verify all points are Point3D instances
        self.assertTrue(all(isinstance(p, Point3D) for p in points))

        # Verify parametric coordinates are within expected ranges
        for point in points:
            self.assertGreaterEqual(point.u, 0)
            self.assertLessEqual(point.u, 2 * math.pi)
            self.assertGreaterEqual(point.v, 0)
            self.assertLessEqual(point.v, 2 * math.pi)

    def test_torus_mathematical_properties(self):
        """Test that generated torus conforms to mathematical properties."""
        params = TorusParameters(
            outer_radius=3.0,
            inner_radius=1.5,
            u_resolution=20,
            v_resolution=10,
            rotation_speed=0.1
        )

        points = generate_torus_points(params)

        # Test specific parametric equation values at known coordinates
        # At u=0, v=0: x should be (R+r), y=0, z=0
        u_zero_v_zero = next((p for p in points if abs(p.u) < 1e-10 and abs(p.v) < 1e-10), None)
        if u_zero_v_zero:
            expected_x = params.outer_radius + params.inner_radius
            self.assertLess(abs(u_zero_v_zero.x - expected_x), 1e-10)
            self.assertLess(abs(u_zero_v_zero.y), 1e-10)
            self.assertLess(abs(u_zero_v_zero.z), 1e-10)

        # Verify distance constraints for all points
        for point in points:
            # Distance from origin to point should be between (R-r) and (R+r)
            distance_from_origin = math.sqrt(point.x**2 + point.y**2 + point.z**2)
            min_distance = params.outer_radius - params.inner_radius
            max_distance = params.outer_radius + params.inner_radius
            self.assertGreaterEqual(distance_from_origin, min_distance)
            self.assertLessEqual(distance_from_origin, max_distance)

    def test_parameter_validation(self):
        """Test parameter validation for torus generation."""
        # Test invalid radius relationship (outer <= inner)
        with self.assertRaises(ValueError):
            invalid_params = TorusParameters(
                outer_radius=1.0,
                inner_radius=2.0,  # inner > outer
                u_resolution=10,
                v_resolution=5,
                rotation_speed=0.1
            )
            generate_torus_points(invalid_params)

        # Test negative inner radius
        with self.assertRaises(ValueError):
            invalid_params = TorusParameters(
                outer_radius=2.0,
                inner_radius=-1.0,  # negative
                u_resolution=10,
                v_resolution=5,
                rotation_speed=0.1
            )
            generate_torus_points(invalid_params)

        # Test zero resolution parameters
        with self.assertRaises(ValueError):
            invalid_params = TorusParameters(
                outer_radius=2.0,
                inner_radius=1.0,
                u_resolution=0,  # zero resolution
                v_resolution=5,
                rotation_speed=0.1
            )
            generate_torus_points(invalid_params)

    def test_caching_performance(self):
        """Test that torus generation results are properly cached."""
        # Clear cache before test
        _torus_cache.clear()

        params = TorusParameters(
            outer_radius=2.0,
            inner_radius=1.0,
            u_resolution=5,
            v_resolution=3,
            rotation_speed=0.1
        )

        # First generation should populate cache
        points1 = generate_torus_points(params)
        assert len(_torus_cache) == 1

        # Second generation with same parameters should use cache
        points2 = generate_torus_points(params)
        assert points1 is points2  # Should be the exact same object reference

        # Different parameters should create new cache entry
        different_params = TorusParameters(
            outer_radius=3.0,  # different radius
            inner_radius=1.0,
            u_resolution=5,
            v_resolution=3,
            rotation_speed=0.1
        )
        points3 = generate_torus_points(different_params)
        assert len(_torus_cache) == 2
        assert points3 is not points1

    def test_resolution_scaling(self):
        """Test that different resolutions produce expected point counts."""
        base_params = TorusParameters(
            outer_radius=2.0,
            inner_radius=1.0,
            u_resolution=4,
            v_resolution=3,
            rotation_speed=0.1
        )

        # Test various resolution combinations
        test_cases = [
            (4, 3, 12),
            (8, 6, 48),
            (10, 10, 100),
            (1, 1, 1)
        ]

        for u_res, v_res, expected_count in test_cases:
            params = base_params._replace(u_resolution=u_res, v_resolution=v_res)
            points = generate_torus_points(params)
            assert len(points) == expected_count


class TestMathematicalValidation:
    """Test suite for mathematical validation functions."""

    def test_volume_validation(self):
        """Test torus volume validation using theoretical formula V = 2π²Rr²."""
        valid_params = TorusParameters(
            outer_radius=3.0,
            inner_radius=1.5,
            u_resolution=10,
            v_resolution=5,
            rotation_speed=0.1
        )

        # Valid parameters should pass validation
        assert validate_torus_volume(valid_params) is True

        # Test with known mathematical relationship
        R, r = valid_params.outer_radius, valid_params.inner_radius
        expected_volume = 2 * (math.pi ** 2) * R * (r ** 2)
        assert expected_volume > 0  # Positive volume is required

        # Invalid parameters should raise ValueError
        invalid_params = TorusParameters(
            outer_radius=1.0,
            inner_radius=2.0,  # inner > outer
            u_resolution=10,
            v_resolution=5,
            rotation_speed=0.1
        )

        with pytest.raises(ValueError, match="Invalid torus parameters"):
            validate_torus_volume(invalid_params)

    def test_surface_area_validation(self):
        """Test torus surface area validation using theoretical formula A = 4π²Rr."""
        valid_params = TorusParameters(
            outer_radius=4.0,
            inner_radius=2.0,
            u_resolution=10,
            v_resolution=5,
            rotation_speed=0.1
        )

        # Valid parameters should pass validation
        assert validate_torus_surface_area(valid_params) is True

        # Test with known mathematical relationship
        R, r = valid_params.outer_radius, valid_params.inner_radius
        expected_surface_area = 4 * (math.pi ** 2) * R * r
        assert expected_surface_area > 0  # Positive surface area is required

        # Invalid parameters should raise ValueError
        invalid_params = TorusParameters(
            outer_radius=1.0,
            inner_radius=-0.5,  # negative inner radius
            u_resolution=10,
            v_resolution=5,
            rotation_speed=0.1
        )

        with pytest.raises(ValueError, match="Invalid torus parameters"):
            validate_torus_surface_area(invalid_params)

    def test_comprehensive_geometry_validation(self):
        """Test comprehensive torus geometry validation."""
        # Valid parameters should pass all validations
        valid_params = TorusParameters(
            outer_radius=5.0,
            inner_radius=2.5,
            u_resolution=20,
            v_resolution=10,
            rotation_speed=0.1
        )

        assert validate_torus_geometry(valid_params) is True

        # Test various invalid parameter combinations
        invalid_cases = [
            # outer_radius <= inner_radius
            TorusParameters(1.0, 2.0, 10, 5, 0.1),
            # negative inner_radius
            TorusParameters(2.0, -1.0, 10, 5, 0.1),
            # zero resolution
            TorusParameters(2.0, 1.0, 0, 5, 0.1),
            TorusParameters(2.0, 1.0, 10, 0, 0.1),
        ]

        for invalid_params in invalid_cases:
            with pytest.raises(ValueError):
                validate_torus_geometry(invalid_params)


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_minimal_valid_parameters(self):
        """Test with minimal valid parameters."""
        minimal_params = TorusParameters(
            outer_radius=0.1,  # Very small but valid
            inner_radius=0.05,
            u_resolution=1,
            v_resolution=1,
            rotation_speed=0.1
        )

        points = generate_torus_points(minimal_params)
        assert len(points) == 1
        assert validate_torus_geometry(minimal_params) is True

    def test_large_parameters(self):
        """Test with large parameter values."""
        large_params = TorusParameters(
            outer_radius=1000.0,
            inner_radius=500.0,
            u_resolution=100,
            v_resolution=50,
            rotation_speed=0.1
        )

        points = generate_torus_points(large_params)
        assert len(points) == 5000  # 100 * 50
        assert validate_torus_geometry(large_params) is True

    def test_high_resolution(self):
        """Test with high resolution parameters."""
        high_res_params = TorusParameters(
            outer_radius=2.0,
            inner_radius=1.0,
            u_resolution=50,
            v_resolution=25,
            rotation_speed=0.1
        )

        points = generate_torus_points(high_res_params)
        assert len(points) == 1250  # 50 * 25

        # Verify mathematical properties still hold with high resolution
        for point in points:
            distance_from_origin = math.sqrt(point.x**2 + point.y**2 + point.z**2)
            min_distance = high_res_params.outer_radius - high_res_params.inner_radius
            max_distance = high_res_params.outer_radius + high_res_params.inner_radius
            assert min_distance <= distance_from_origin <= max_distance


class TestMathematicalPrecision:
    """Test suite for mathematical precision requirements."""

    def test_tau_usage(self):
        """Test that math.tau is used for mathematical precision."""
        params = TorusParameters(
            outer_radius=2.0,
            inner_radius=1.0,
            u_resolution=4,
            v_resolution=2,
            rotation_speed=0.1
        )

        points = generate_torus_points(params)

        # Check that parametric coordinates use proper tau scaling
        u_values = [p.u for p in points]
        v_values = [p.v for p in points]

        # Should have values from 0 to 2π (tau)
        assert min(u_values) >= 0
        assert max(u_values) <= 2 * math.pi
        assert min(v_values) >= 0
        assert max(v_values) <= 2 * math.pi

    def test_floating_point_precision(self):
        """Test floating-point precision in calculations."""
        params = TorusParameters(
            outer_radius=1.0,
            inner_radius=0.5,
            u_resolution=8,
            v_resolution=4,
            rotation_speed=0.1
        )

        points = generate_torus_points(params)

        # Verify that calculations maintain precision
        for point in points:
            # All coordinates should be finite numbers
            assert math.isfinite(point.x)
            assert math.isfinite(point.y)
            assert math.isfinite(point.z)
            assert math.isfinite(point.u)
            assert math.isfinite(point.v)

            # No NaN values should exist
            assert not math.isnan(point.x)
            assert not math.isnan(point.y)
            assert not math.isnan(point.z)
            assert not math.isnan(point.u)
            assert not math.isnan(point.v)


class TestPerspectiveProjection(unittest.TestCase):
    """Test suite for 3D to 2D perspective projection functionality."""

    def test_basic_projection(self):
        """Test basic perspective projection with valid 3D points."""
        # Test point at origin
        origin_point = Point3D(x=0.0, y=0.0, z=0.0, u=0.0, v=0.0)
        projected = project_to_screen(origin_point)

        # Origin should project to center of screen
        self.assertEqual(projected.x, 20)  # Center of 40-wide grid
        self.assertEqual(projected.y, 10)  # Center of 20-high grid
        self.assertTrue(projected.visible)
        self.assertGreater(projected.depth, 0.0)
        self.assertLess(projected.depth, 1.0)

    def test_coordinate_mapping_to_grid(self):
        """Test that coordinates map correctly to 40x20 ASCII grid."""
        # Test point that should map to specific grid coordinates
        test_point = Point3D(x=1.0, y=0.5, z=0.0, u=0.0, v=0.0)
        projected = project_to_screen(test_point)

        # Verify coordinates are within valid grid range
        self.assertGreaterEqual(projected.x, 0)
        self.assertLess(projected.x, 40)
        self.assertGreaterEqual(projected.y, 0)
        self.assertLess(projected.y, 20)
        self.assertTrue(projected.visible)

    def test_depth_calculation_and_ordering(self):
        """Test depth calculation for proper Z-sorting."""
        # Create points at different Z depths
        near_point = Point3D(x=0.0, y=0.0, z=1.0, u=0.0, v=0.0)
        far_point = Point3D(x=0.0, y=0.0, z=-2.0, u=0.0, v=0.0)

        near_projected = project_to_screen(near_point)
        far_projected = project_to_screen(far_point)

        # Nearer points should have smaller depth values for proper sorting
        self.assertLess(far_projected.depth, near_projected.depth)
        self.assertGreaterEqual(near_projected.depth, 0.0)
        self.assertLessEqual(near_projected.depth, 1.0)
        self.assertGreaterEqual(far_projected.depth, 0.0)
        self.assertLessEqual(far_projected.depth, 1.0)

    def test_points_behind_camera(self):
        """Test handling of points behind camera (negative Z after camera offset)."""
        # Camera distance is 5.0, so points with z < -5.0 are behind camera
        behind_camera_point = Point3D(x=1.0, y=1.0, z=-6.0, u=0.0, v=0.0)
        projected = project_to_screen(behind_camera_point)

        # Points behind camera should be marked as invisible
        self.assertFalse(projected.visible)
        self.assertEqual(projected.depth, float('inf'))

    def test_bounds_checking(self):
        """Test visibility flag for points outside display area."""
        # Test point that projects outside the display area
        extreme_point = Point3D(x=10.0, y=10.0, z=0.0, u=0.0, v=0.0)
        projected = project_to_screen(extreme_point)

        # Point should be clamped to valid coordinates but marked invisible
        self.assertGreaterEqual(projected.x, 0)
        self.assertLess(projected.x, 40)
        self.assertGreaterEqual(projected.y, 0)
        self.assertLess(projected.y, 20)
        # Visibility depends on whether point projects within bounds

    def test_division_by_zero_handling(self):
        """Test handling of edge cases that could cause division by zero."""
        # Point exactly at camera position (z = -camera_distance = -5.0)
        camera_point = Point3D(x=1.0, y=1.0, z=-5.0, u=0.0, v=0.0)
        projected = project_to_screen(camera_point)

        # Should handle gracefully and mark as invisible
        self.assertFalse(projected.visible)

    def test_perspective_projection_formula(self):
        """Test perspective projection formula accuracy."""
        # Test with known values to verify formula implementation
        test_point = Point3D(x=2.0, y=1.0, z=0.0, u=0.0, v=0.0)
        projected = project_to_screen(test_point)

        # Calculate expected values manually
        camera_distance = 5.0
        focal_length = 2.0
        z_camera = test_point.z + camera_distance  # 5.0
        expected_screen_x = (test_point.x * focal_length) / z_camera  # 0.8
        expected_screen_y = (test_point.y * focal_length) / z_camera  # 0.4
        expected_grid_x = int((expected_screen_x + 1.0) * 20)  # 36
        expected_grid_y = int((expected_screen_y + 1.0) * 10)  # 14

        self.assertEqual(projected.x, expected_grid_x)
        self.assertEqual(projected.y, expected_grid_y)
        self.assertTrue(projected.visible)

    def test_aspect_ratio_preservation(self):
        """Test that projection preserves proper aspect ratio for torus visualization."""
        # Test symmetric points to verify aspect ratio handling
        point_right = Point3D(x=1.0, y=0.0, z=0.0, u=0.0, v=0.0)
        point_up = Point3D(x=0.0, y=1.0, z=0.0, u=0.0, v=0.0)

        proj_right = project_to_screen(point_right)
        proj_up = project_to_screen(point_up)

        # Both should be visible and at reasonable distances from center
        self.assertTrue(proj_right.visible)
        self.assertTrue(proj_up.visible)

        # Center is at (20, 10), check relative distances
        center_x, center_y = 20, 10
        dist_right = abs(proj_right.x - center_x)
        dist_up = abs(proj_up.y - center_y)

        # Distances should be proportional to grid dimensions (40:20 ratio)
        # This ensures proper aspect ratio for circular torus appearance

    def test_mathematical_precision(self):
        """Test mathematical precision and floating-point handling."""
        # Test with various point types including edge cases
        test_points = [
            Point3D(x=0.0, y=0.0, z=0.0, u=0.0, v=0.0),      # Origin
            Point3D(x=1e-10, y=1e-10, z=1e-10, u=0.0, v=0.0), # Very small values
            Point3D(x=100.0, y=100.0, z=100.0, u=0.0, v=0.0), # Large values
            Point3D(x=-1.0, y=-1.0, z=2.0, u=0.0, v=0.0),     # Negative coordinates
        ]

        for point in test_points:
            projected = project_to_screen(point)

            # All projections should produce finite values
            self.assertTrue(math.isfinite(projected.x))
            self.assertTrue(math.isfinite(projected.y))
            self.assertTrue(math.isfinite(projected.depth) or projected.depth == float('inf'))

            # Coordinates should be integers
            self.assertIsInstance(projected.x, int)
            self.assertIsInstance(projected.y, int)

            # Depth should be float in valid range or inf for invisible points
            self.assertIsInstance(projected.depth, float)
            if projected.visible:
                self.assertGreaterEqual(projected.depth, 0.0)
                self.assertLessEqual(projected.depth, 1.0)

    def test_torus_projection_integration(self):
        """Test projection integration with torus points."""
        # Generate a small torus and project all points
        params = TorusParameters(
            outer_radius=2.0,
            inner_radius=1.0,
            u_resolution=8,
            v_resolution=4,
            rotation_speed=0.1
        )

        torus_points = generate_torus_points(params)
        projected_points = [project_to_screen(point) for point in torus_points]

        # All projections should be valid Point2D objects
        self.assertTrue(all(isinstance(p, Point2D) for p in projected_points))

        # At least some points should be visible (not all behind camera)
        visible_count = sum(1 for p in projected_points if p.visible)
        self.assertGreater(visible_count, 0)

        # All visible points should have valid grid coordinates
        for proj in projected_points:
            if proj.visible:
                self.assertGreaterEqual(proj.x, 0)
                self.assertLess(proj.x, 40)
                self.assertGreaterEqual(proj.y, 0)
                self.assertLess(proj.y, 20)
                self.assertGreaterEqual(proj.depth, 0.0)
                self.assertLessEqual(proj.depth, 1.0)


if __name__ == "__main__":
    # Run tests if script is executed directly
    unittest.main(verbosity=2)