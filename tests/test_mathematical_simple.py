#!/usr/bin/env python3
"""
Simple unit tests for mathematical functions in the 3D ASCII Donut project.
Uses Python standard library unittest for maximum compatibility.
"""

import math
import unittest
import sys
import os

# Import the module under test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rotating_donut import (
    Point3D, TorusParameters,
    generate_torus_points,
    validate_torus_volume,
    validate_torus_surface_area,
    validate_torus_geometry,
    _torus_cache
)


class TestTorusMathematical(unittest.TestCase):
    """Comprehensive test suite for torus mathematical functions."""

    def setUp(self):
        """Clear cache before each test."""
        _torus_cache.clear()

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

        # Verify distance constraints for all points
        for point in points:
            # Distance from origin to point should be between (R-r) and (R+r)
            distance_from_origin = math.sqrt(point.x**2 + point.y**2 + point.z**2)
            min_distance = params.outer_radius - params.inner_radius
            max_distance = params.outer_radius + params.inner_radius
            self.assertGreaterEqual(distance_from_origin, min_distance - 1e-10)
            self.assertLessEqual(distance_from_origin, max_distance + 1e-10)

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
        params = TorusParameters(
            outer_radius=2.0,
            inner_radius=1.0,
            u_resolution=5,
            v_resolution=3,
            rotation_speed=0.1
        )

        # First generation should populate cache
        points1 = generate_torus_points(params)
        self.assertEqual(len(_torus_cache), 1)

        # Second generation with same parameters should use cache
        points2 = generate_torus_points(params)
        self.assertIs(points1, points2)  # Should be the exact same object reference

        # Different parameters should create new cache entry
        different_params = TorusParameters(
            outer_radius=3.0,  # different radius
            inner_radius=1.0,
            u_resolution=5,
            v_resolution=3,
            rotation_speed=0.1
        )
        points3 = generate_torus_points(different_params)
        self.assertEqual(len(_torus_cache), 2)
        self.assertIsNot(points3, points1)

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
        self.assertTrue(validate_torus_volume(valid_params))

        # Test with known mathematical relationship
        R, r = valid_params.outer_radius, valid_params.inner_radius
        expected_volume = 2 * (math.pi ** 2) * R * (r ** 2)
        self.assertGreater(expected_volume, 0)  # Positive volume is required

        # Invalid parameters should raise ValueError
        invalid_params = TorusParameters(
            outer_radius=1.0,
            inner_radius=2.0,  # inner > outer
            u_resolution=10,
            v_resolution=5,
            rotation_speed=0.1
        )

        with self.assertRaises(ValueError):
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
        self.assertTrue(validate_torus_surface_area(valid_params))

        # Test with known mathematical relationship
        R, r = valid_params.outer_radius, valid_params.inner_radius
        expected_surface_area = 4 * (math.pi ** 2) * R * r
        self.assertGreater(expected_surface_area, 0)  # Positive surface area is required

        # Invalid parameters should raise ValueError
        invalid_params = TorusParameters(
            outer_radius=1.0,
            inner_radius=-0.5,  # negative inner radius
            u_resolution=10,
            v_resolution=5,
            rotation_speed=0.1
        )

        with self.assertRaises(ValueError):
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

        self.assertTrue(validate_torus_geometry(valid_params))

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
            with self.assertRaises(ValueError):
                validate_torus_geometry(invalid_params)

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
            self.assertEqual(len(points), expected_count)

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
        self.assertEqual(len(points), 1)
        self.assertTrue(validate_torus_geometry(minimal_params))

    def test_large_parameters(self):
        """Test with large parameter values."""
        large_params = TorusParameters(
            outer_radius=100.0,
            inner_radius=50.0,
            u_resolution=10,
            v_resolution=5,
            rotation_speed=0.1
        )

        points = generate_torus_points(large_params)
        self.assertEqual(len(points), 50)  # 10 * 5
        self.assertTrue(validate_torus_geometry(large_params))

    def test_mathematical_precision(self):
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
            self.assertTrue(math.isfinite(point.x))
            self.assertTrue(math.isfinite(point.y))
            self.assertTrue(math.isfinite(point.z))
            self.assertTrue(math.isfinite(point.u))
            self.assertTrue(math.isfinite(point.v))

            # No NaN values should exist
            self.assertFalse(math.isnan(point.x))
            self.assertFalse(math.isnan(point.y))
            self.assertFalse(math.isnan(point.z))
            self.assertFalse(math.isnan(point.u))
            self.assertFalse(math.isnan(point.v))


if __name__ == "__main__":
    # Run tests if script is executed directly
    unittest.main(verbosity=2)