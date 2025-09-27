#!/usr/bin/env python3
"""
Basic structure tests for rotating_donut.py

Tests the fundamental structure and imports of the main file to ensure
it meets the acceptance criteria for Story 1.1.
"""

import sys
import os
import unittest
from unittest.mock import patch, mock_open

# Add parent directory to path for importing rotating_donut
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rotating_donut


class TestBasicStructure(unittest.TestCase):
    """Test basic file structure and imports."""

    def test_imports_available(self):
        """Test that all required modules are imported."""
        # Check that standard library modules are available
        self.assertTrue(hasattr(rotating_donut, 'math'))
        self.assertTrue(hasattr(rotating_donut, 'sys'))
        self.assertTrue(hasattr(rotating_donut, 'time'))
        self.assertTrue(hasattr(rotating_donut, 'tokenize'))

    def test_data_models_defined(self):
        """Test that all required data models are defined."""
        self.assertTrue(hasattr(rotating_donut, 'Point3D'))
        self.assertTrue(hasattr(rotating_donut, 'Point2D'))
        self.assertTrue(hasattr(rotating_donut, 'CodeToken'))
        self.assertTrue(hasattr(rotating_donut, 'TorusParameters'))
        self.assertTrue(hasattr(rotating_donut, 'DisplayFrame'))

    def test_mathematical_functions_defined(self):
        """Test that mathematical engine functions are defined."""
        self.assertTrue(hasattr(rotating_donut, 'generate_torus_points'))
        self.assertTrue(hasattr(rotating_donut, 'apply_rotation'))
        self.assertTrue(hasattr(rotating_donut, 'project_to_screen'))

    def test_parsing_functions_defined(self):
        """Test that parsing engine functions are defined."""
        self.assertTrue(hasattr(rotating_donut, 'read_self_code'))
        self.assertTrue(hasattr(rotating_donut, 'tokenize_code'))
        self.assertTrue(hasattr(rotating_donut, 'classify_importance'))

    def test_rendering_functions_defined(self):
        """Test that rendering engine functions are defined."""
        self.assertTrue(hasattr(rotating_donut, 'map_tokens_to_surface'))
        self.assertTrue(hasattr(rotating_donut, 'generate_ascii_frame'))
        self.assertTrue(hasattr(rotating_donut, 'output_to_terminal'))

    def test_animation_functions_defined(self):
        """Test that animation controller functions are defined."""
        self.assertTrue(hasattr(rotating_donut, 'run_animation_loop'))
        self.assertTrue(hasattr(rotating_donut, 'main'))

    def test_torus_parameter_validation(self):
        """Test torus parameter validation in generate_torus_points."""
        # Invalid parameters should raise ValueError
        invalid_params = rotating_donut.TorusParameters(
            outer_radius=1.0,
            inner_radius=2.0,  # Invalid: inner > outer
            u_resolution=50,
            v_resolution=25,
            rotation_speed=0.1
        )

        with self.assertRaises(ValueError) as context:
            rotating_donut.generate_torus_points(invalid_params)

        self.assertIn("Solution:", str(context.exception))

    @patch('builtins.open', mock_open(read_data="# Test code"))
    def test_read_self_code(self):
        """Test read_self_code function with mocked file."""
        code = rotating_donut.read_self_code()
        self.assertEqual(code, "# Test code")

    def test_classify_importance_default(self):
        """Test classify_importance returns LOW for unknown tokens."""
        importance = rotating_donut.classify_importance("UNKNOWN", "unknown_value")
        self.assertEqual(importance, "LOW")

    def test_data_model_creation(self):
        """Test that data models can be instantiated correctly."""
        # Test Point3D
        point_3d = rotating_donut.Point3D(1.0, 2.0, 3.0, 0.5, 1.5)
        self.assertEqual(point_3d.x, 1.0)
        self.assertEqual(point_3d.u, 0.5)

        # Test TorusParameters
        params = rotating_donut.TorusParameters(2.0, 1.0, 50, 25, 0.1)
        self.assertEqual(params.outer_radius, 2.0)
        self.assertEqual(params.inner_radius, 1.0)

        # Test CodeToken
        token = rotating_donut.CodeToken("NAME", "test", "HIGH", 1, 5, "#")
        self.assertEqual(token.type, "NAME")
        self.assertEqual(token.importance, "HIGH")


if __name__ == '__main__':
    unittest.main()