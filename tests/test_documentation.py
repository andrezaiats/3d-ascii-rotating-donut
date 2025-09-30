#!/usr/bin/env python3
"""
Test suite for documentation validation.

This module validates that:
1. All documented examples execute correctly
2. Parameter modification examples produce expected results
3. Mathematical formulas in documentation match implementation
4. Docstring code examples work as specified
"""

import unittest
import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rotating_donut as rd


class TestTorusParameterExamples(unittest.TestCase):
    """Test that TorusParameters example configurations work as documented."""

    def test_standard_donut_configuration(self):
        """Test standard donut configuration from TorusParameters docstring."""
        params = rd.TorusParameters(
            outer_radius=2.0,
            inner_radius=1.0,
            u_resolution=50,
            v_resolution=30,
            rotation_speed=0.02
        )
        points = rd.generate_torus_points(params)

        self.assertEqual(len(points), 50 * 30)
        self.assertTrue(all(isinstance(p, rd.Point3D) for p in points))
        self.assertGreater(params.outer_radius, params.inner_radius)

    def test_thin_ring_configuration(self):
        """Test thin ring configuration from TorusParameters docstring."""
        params = rd.TorusParameters(
            outer_radius=3.0,
            inner_radius=0.5,
            u_resolution=60,
            v_resolution=20,
            rotation_speed=0.03
        )
        points = rd.generate_torus_points(params)

        self.assertEqual(len(points), 60 * 20)
        self.assertGreater(params.outer_radius, params.inner_radius)
        self.assertAlmostEqual(params.outer_radius / params.inner_radius, 6.0)

    def test_fat_donut_configuration(self):
        """Test fat donut configuration from TorusParameters docstring."""
        params = rd.TorusParameters(
            outer_radius=1.5,
            inner_radius=1.2,
            u_resolution=40,
            v_resolution=40,
            rotation_speed=0.01
        )
        points = rd.generate_torus_points(params)

        self.assertEqual(len(points), 40 * 40)
        self.assertGreater(params.outer_radius, params.inner_radius)
        self.assertLess(params.outer_radius - params.inner_radius, 0.5)

    def test_high_detail_configuration(self):
        """Test high detail configuration from TorusParameters docstring."""
        params = rd.TorusParameters(
            outer_radius=2.0,
            inner_radius=1.0,
            u_resolution=100,
            v_resolution=60,
            rotation_speed=0.02
        )
        points = rd.generate_torus_points(params)

        self.assertEqual(len(points), 100 * 60)

    def test_performance_mode_configuration(self):
        """Test performance mode configuration from TorusParameters docstring."""
        params = rd.TorusParameters(
            outer_radius=2.0,
            inner_radius=1.0,
            u_resolution=30,
            v_resolution=20,
            rotation_speed=0.02
        )
        points = rd.generate_torus_points(params)

        self.assertEqual(len(points), 30 * 20)


class TestMathematicalFormulas(unittest.TestCase):
    """Validate that mathematical formulas in documentation match implementation."""

    def test_torus_parametric_equations(self):
        """Validate torus parametric equations from README and module docstring."""
        R = 2.0  # outer_radius
        r = 1.0  # inner_radius
        u = 0.0
        v = 0.0

        # Calculate using documented formulas
        x_expected = (R + r * math.cos(v)) * math.cos(u)
        y_expected = (R + r * math.cos(v)) * math.sin(u)
        z_expected = r * math.sin(v)

        # Generate using actual implementation
        params = rd.TorusParameters(
            outer_radius=R,
            inner_radius=r,
            u_resolution=1,
            v_resolution=1,
            rotation_speed=0.01
        )
        points = rd.generate_torus_points(params)

        # First point should match formula at u=0, v=0
        self.assertAlmostEqual(points[0].x, x_expected, places=10)
        self.assertAlmostEqual(points[0].y, y_expected, places=10)
        self.assertAlmostEqual(points[0].z, z_expected, places=10)

    def test_torus_surface_area_formula(self):
        """Validate surface area formula A = 4π²Rr from README."""
        R = 2.0
        r = 1.0
        params = rd.TorusParameters(
            outer_radius=R,
            inner_radius=r,
            u_resolution=50,
            v_resolution=30,
            rotation_speed=0.01
        )

        theoretical_area = 4 * (math.pi ** 2) * R * r

        self.assertTrue(rd.validate_torus_surface_area(params))
        self.assertGreater(theoretical_area, 0)

    def test_torus_volume_formula(self):
        """Validate volume formula V = 2π²Rr² from README."""
        R = 2.0
        r = 1.0
        params = rd.TorusParameters(
            outer_radius=R,
            inner_radius=r,
            u_resolution=50,
            v_resolution=30,
            rotation_speed=0.01
        )

        theoretical_volume = 2 * (math.pi ** 2) * R * (r ** 2)

        self.assertTrue(rd.validate_torus_volume(params))
        self.assertGreater(theoretical_volume, 0)

    def test_rotation_matrix_y_axis(self):
        """Validate Y-axis rotation matrix from README and module docstring."""
        angle = math.pi / 4  # 45 degrees
        point = rd.Point3D(x=1.0, y=2.0, z=0.0, u=0.0, v=0.0, nx=0.0, ny=0.0, nz=1.0)

        # Apply rotation using documented formula
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        x_expected = point.x * cos_angle + point.z * sin_angle
        y_expected = point.y
        z_expected = -point.x * sin_angle + point.z * cos_angle

        # Apply actual implementation
        rotated_points = rd.apply_rotation([point], angle)
        rotated = rotated_points[0]

        self.assertAlmostEqual(rotated.x, x_expected, places=10)
        self.assertAlmostEqual(rotated.y, y_expected, places=10)
        self.assertAlmostEqual(rotated.z, z_expected, places=10)


class TestImportanceLevelClassification(unittest.TestCase):
    """Test that ImportanceLevel classification examples work as documented."""

    def test_keyword_classification_example(self):
        """Test 'def' keyword classification from ImportanceLevel docstring."""
        token = rd.CodeToken(
            type='KEYWORD',  # Use custom type from tokenize_code mapping
            value='def',
            importance=0,
            line=1,
            column=0,
            ascii_char=''
        )
        importance = rd.classify_importance(token)

        self.assertEqual(importance, rd.ImportanceLevel.CRITICAL)

    def test_operator_classification_example(self):
        """Test operator classification from ImportanceLevel docstring."""
        token = rd.CodeToken(
            type='OPERATOR',  # Use custom type from tokenize_code mapping
            value='*',
            importance=0,
            line=1,
            column=0,
            ascii_char=''
        )
        importance = rd.classify_importance(token)

        self.assertEqual(importance, rd.ImportanceLevel.HIGH)

    def test_identifier_classification_example(self):
        """Test identifier classification from ImportanceLevel docstring."""
        token = rd.CodeToken(
            type='IDENTIFIER',  # Use custom type from tokenize_code mapping
            value='calculate_area',
            importance=0,
            line=1,
            column=0,
            ascii_char=''
        )
        importance = rd.classify_importance(token)

        self.assertEqual(importance, rd.ImportanceLevel.MEDIUM)

    def test_comment_classification_example(self):
        """Test comment classification from ImportanceLevel docstring."""
        token = rd.CodeToken(
            type='COMMENT',
            value='# Calculate the area',
            importance=0,
            line=1,
            column=0,
            ascii_char=''
        )
        importance = rd.classify_importance(token)

        self.assertEqual(importance, rd.ImportanceLevel.LOW)


class TestASCIICharacterMapping(unittest.TestCase):
    """Test ASCII character mapping examples from documentation."""

    def test_default_character_set(self):
        """Test default ASCII_CHARS configuration."""
        self.assertEqual(rd.ASCII_CHARS['HIGH'], '#')
        self.assertEqual(rd.ASCII_CHARS['MEDIUM'], '+')
        self.assertEqual(rd.ASCII_CHARS['LOW'], '-')
        self.assertEqual(rd.ASCII_CHARS['BACKGROUND'], '.')

    def test_character_set_contains_basic_ascii(self):
        """Verify all characters are basic ASCII as documented."""
        for char in rd.ASCII_CHARS.values():
            self.assertGreaterEqual(ord(char), 32)
            self.assertLessEqual(ord(char), 126)


class TestREADMEExamples(unittest.TestCase):
    """Test examples from README.md for accuracy."""

    def test_readme_torus_calculation_example(self):
        """Test example calculation from README Mathematical Background section."""
        R = 2.0
        r = 1.0
        u = 0.0
        v = 0.0

        # Calculate using documented formulas
        x = (R + r * math.cos(v)) * math.cos(u)
        y = (R + r * math.cos(v)) * math.sin(u)
        z = r * math.sin(v)

        # Should match README example: Point(3.0, 0.0, 0.0)
        self.assertAlmostEqual(x, 3.0, places=10)
        self.assertAlmostEqual(y, 0.0, places=10)
        self.assertAlmostEqual(z, 0.0, places=10)

    def test_readme_rotation_matrix_formula(self):
        """Test rotation matrix formula from README."""
        theta = math.pi / 2  # 90 degrees
        x, y, z = 1.0, 0.0, 0.0

        # Apply documented Y-axis rotation formula
        x_new = x * math.cos(theta) + z * math.sin(theta)
        y_new = y
        z_new = -x * math.sin(theta) + z * math.cos(theta)

        # Point should rotate from (1,0,0) to (0,0,-1)
        self.assertAlmostEqual(x_new, 0.0, places=10)
        self.assertAlmostEqual(y_new, 0.0, places=10)
        self.assertAlmostEqual(z_new, -1.0, places=10)


class TestDocumentationConstraints(unittest.TestCase):
    """Test documented constraints and requirements."""

    def test_torus_constraint_outer_greater_than_inner(self):
        """Test constraint: outer_radius > inner_radius > 0."""
        # Valid parameters
        valid_params = rd.TorusParameters(
            outer_radius=2.0,
            inner_radius=1.0,
            u_resolution=30,
            v_resolution=20,
            rotation_speed=0.01
        )
        self.assertTrue(rd.validate_torus_geometry(valid_params))

        # Invalid parameters (inner >= outer)
        invalid_params = rd.TorusParameters(
            outer_radius=1.0,
            inner_radius=2.0,
            u_resolution=30,
            v_resolution=20,
            rotation_speed=0.01
        )

        with self.assertRaises(ValueError) as context:
            rd.generate_torus_points(invalid_params)

        self.assertIn("outer_radius > inner_radius", str(context.exception))

    def test_resolution_constraint_positive(self):
        """Test constraint: u_resolution > 0, v_resolution > 0."""
        invalid_params = rd.TorusParameters(
            outer_radius=2.0,
            inner_radius=1.0,
            u_resolution=0,
            v_resolution=20,
            rotation_speed=0.01
        )

        with self.assertRaises(ValueError) as context:
            rd.generate_torus_points(invalid_params)

        self.assertIn("u_resolution > 0", str(context.exception))


class TestErrorMessageFormat(unittest.TestCase):
    """Test that error messages follow documented format."""

    def test_error_message_contains_solution(self):
        """Verify error messages include 'Solution:' as documented."""
        invalid_params = rd.TorusParameters(
            outer_radius=1.0,
            inner_radius=2.0,
            u_resolution=30,
            v_resolution=20,
            rotation_speed=0.01
        )

        with self.assertRaises(ValueError) as context:
            rd.generate_torus_points(invalid_params)

        error_message = str(context.exception)
        self.assertIn("Solution:", error_message)


class TestPlatformDetection(unittest.TestCase):
    """Test platform detection as documented."""

    def test_platform_detection_returns_valid_info(self):
        """Test that detect_platform returns expected PlatformInfo structure."""
        platform_info = rd.detect_platform()

        self.assertIsInstance(platform_info, rd.PlatformInfo)
        self.assertIn(platform_info.platform, ['win32', 'darwin', 'linux', 'linux2'])
        self.assertIsInstance(platform_info.is_windows, bool)
        self.assertIsInstance(platform_info.is_macos, bool)
        self.assertIsInstance(platform_info.is_linux, bool)
        self.assertIsInstance(platform_info.python_version, tuple)
        self.assertEqual(len(platform_info.python_version), 3)


class TestCachingOptimizations(unittest.TestCase):
    """Test caching optimizations as documented."""

    def test_geometry_caching_prevents_regeneration(self):
        """Verify torus geometry is cached as documented."""
        params = rd.TorusParameters(
            outer_radius=2.0,
            inner_radius=1.0,
            u_resolution=30,
            v_resolution=20,
            rotation_speed=0.01
        )

        # Generate twice
        points1 = rd.generate_torus_points(params)
        points2 = rd.generate_torus_points(params)

        # Should return identical cached results
        self.assertEqual(len(points1), len(points2))
        self.assertAlmostEqual(points1[0].x, points2[0].x, places=10)
        self.assertAlmostEqual(points1[0].y, points2[0].y, places=10)
        self.assertAlmostEqual(points1[0].z, points2[0].z, places=10)


class TestDocstringCompleteness(unittest.TestCase):
    """Test that key functions have complete docstrings."""

    def test_generate_torus_points_has_docstring(self):
        """Verify generate_torus_points has comprehensive docstring."""
        docstring = rd.generate_torus_points.__doc__

        self.assertIsNotNone(docstring, f"generate_torus_points docstring is None. Function exists: {hasattr(rd, 'generate_torus_points')}")
        self.assertGreater(len(docstring), 100, "Docstring should be comprehensive")
        self.assertTrue('Args:' in docstring or 'Parameters:' in docstring or 'param' in docstring.lower())
        self.assertIn('Returns:', docstring)
        self.assertIn('parametric', docstring.lower())

    def test_apply_rotation_has_docstring(self):
        """Verify apply_rotation has comprehensive docstring."""
        docstring = rd.apply_rotation.__doc__

        self.assertIsNotNone(docstring, f"apply_rotation docstring is None. Function exists: {hasattr(rd, 'apply_rotation')}")
        self.assertGreater(len(docstring), 50, "Docstring should be comprehensive")
        self.assertIn('rotation', docstring.lower())
        self.assertIn('matrix', docstring.lower())

    def test_module_docstring_exists(self):
        """Verify module has comprehensive docstring."""
        module_docstring = rd.__doc__

        self.assertIsNotNone(module_docstring)
        self.assertGreater(len(module_docstring), 500)
        self.assertIn('torus', module_docstring.lower())
        self.assertIn('parametric', module_docstring.lower())


if __name__ == '__main__':
    unittest.main()