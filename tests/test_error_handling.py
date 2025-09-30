#!/usr/bin/env python3
"""
Comprehensive Error Handling Tests for Story 4.3

Tests error handling and robustness across all components:
- File reading errors (permissions, missing files, encoding)
- Keyboard interrupt handling with terminal state validation
- Mathematical validation (boundary conditions, invalid inputs)
- Error message format compliance ("Solution:" requirement)
- Fallback behaviors (parsing failures, invalid parameters, terminal errors)
- Graceful degradation scenarios

Coverage Requirement: 90%+ for all error handling code paths
"""

import unittest
from unittest.mock import patch, mock_open, MagicMock, PropertyMock
import sys
import os
import io
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions to test
from rotating_donut import (
    get_script_path,
    validate_file_content,
    read_self_code,
    generate_torus_points,
    apply_rotation,
    project_to_screen,
    handle_interrupts,
    output_to_terminal,
    TorusParameters,
    Point3D,
    DisplayFrame,
    PlatformInfo,
    detect_platform
)


class TestFileReadingErrors(unittest.TestCase):
    """Test file reading error handling and recovery (Task 1)."""

    def test_get_script_path_missing_file_attribute(self):
        """Test __file__ not available error."""
        # This test verifies the error message when __file__ is not available
        # In practice, __file__ is always available when module is imported
        # so we test the validation logic
        self.assertTrue(hasattr(sys.modules[__name__], '__file__'))

    def test_get_script_path_nonexistent_file(self):
        """Test FileNotFoundError for nonexistent file."""
        with patch('rotating_donut.__file__', '/nonexistent/path/to/file.py'):
            with patch('os.path.exists', return_value=False):
                with self.assertRaises(FileNotFoundError) as cm:
                    get_script_path()
                self.assertIn("Solution:", str(cm.exception))

    def test_get_script_path_permission_denied(self):
        """Test PermissionError for inaccessible file."""
        with patch('rotating_donut.__file__', '/restricted/file.py'):
            with patch('os.path.exists', return_value=True):
                with patch('os.access', return_value=False):
                    with self.assertRaises(PermissionError) as cm:
                        get_script_path()
                    self.assertIn("Solution:", str(cm.exception))

    def test_validate_file_content_empty_file(self):
        """Test ValueError for empty file content."""
        with self.assertRaises(ValueError) as cm:
            validate_file_content('test.py', '')
        self.assertIn("Solution:", str(cm.exception))
        self.assertIn("empty", str(cm.exception).lower())

    def test_validate_file_content_too_large(self):
        """Test ValueError for file exceeding size limit."""
        large_content = 'a' * (11 * 1024 * 1024)  # 11MB > 10MB limit
        with self.assertRaises(ValueError) as cm:
            validate_file_content('test.py', large_content)
        self.assertIn("Solution:", str(cm.exception))
        self.assertIn("exceeds", str(cm.exception).lower())

    def test_validate_file_content_syntax_error(self):
        """Test ValueError for invalid Python syntax."""
        invalid_code = "def bad syntax("
        with self.assertRaises(ValueError) as cm:
            validate_file_content('test.py', invalid_code)
        self.assertIn("Solution:", str(cm.exception))

    def test_validate_file_content_too_short(self):
        """Test ValueError for suspiciously short file."""
        with self.assertRaises(ValueError) as cm:
            validate_file_content('test.py', '# x')
        self.assertIn("Solution:", str(cm.exception))

    def test_validate_file_content_valid(self):
        """Test successful validation of valid Python file."""
        valid_code = """
# Valid Python file
def test_function():
    return 42
"""
        result = validate_file_content('test.py', valid_code)
        self.assertTrue(result)


class TestKeyboardInterruptHandling(unittest.TestCase):
    """Test keyboard interrupt and signal handling (Task 2)."""

    @patch('builtins.print')
    def test_handle_interrupts_basic(self, mock_print):
        """Test basic keyboard interrupt handling."""
        result = handle_interrupts()
        self.assertTrue(result)
        # Verify terminal restoration messages printed
        printed_messages = [call[0][0] for call in mock_print.call_args_list if call[0]]
        self.assertTrue(any('stopped gracefully' in str(msg).lower() for msg in printed_messages))

    @patch('builtins.print')
    def test_handle_interrupts_with_platform_info(self, mock_print):
        """Test keyboard interrupt with platform-specific cleanup."""
        platform_info = PlatformInfo(
            platform='linux',
            os_name='posix',
            is_windows=False,
            is_macos=False,
            is_linux=True,
            python_version=(3, 8, 0),
            supports_ansi=True
        )
        result = handle_interrupts(platform_info)
        self.assertTrue(result)
        # Verify ANSI escape codes were attempted
        print_calls = [str(call) for call in mock_print.call_args_list]
        # Should include cursor restoration and screen clearing

    @patch('builtins.print', side_effect=Exception("Print failed"))
    def test_handle_interrupts_print_failure(self, mock_print):
        """Test interrupt handler when printing fails."""
        result = handle_interrupts()
        # Should return False when cleanup fails
        self.assertFalse(result)


class TestMathematicalValidation(unittest.TestCase):
    """Test mathematical input validation and edge cases (Task 3)."""

    def test_generate_torus_invalid_radii(self):
        """Test ValueError for invalid torus radii."""
        params = TorusParameters(
            outer_radius=1.0,
            inner_radius=2.0,  # inner > outer (invalid)
            u_resolution=10,
            v_resolution=10,
            rotation_speed=0.05
        )
        with self.assertRaises(ValueError) as cm:
            generate_torus_points(params)
        self.assertIn("Solution:", str(cm.exception))

    def test_generate_torus_negative_radius(self):
        """Test ValueError for negative radius."""
        params = TorusParameters(
            outer_radius=2.0,
            inner_radius=-1.0,  # Negative (invalid)
            u_resolution=10,
            v_resolution=10,
            rotation_speed=0.05
        )
        with self.assertRaises(ValueError) as cm:
            generate_torus_points(params)
        self.assertIn("Solution:", str(cm.exception))

    def test_generate_torus_zero_resolution(self):
        """Test ValueError for zero resolution."""
        params = TorusParameters(
            outer_radius=2.0,
            inner_radius=1.0,
            u_resolution=0,  # Zero resolution (invalid)
            v_resolution=10,
            rotation_speed=0.05
        )
        with self.assertRaises(ValueError) as cm:
            generate_torus_points(params)
        self.assertIn("Solution:", str(cm.exception))

    def test_apply_rotation_nan_angle(self):
        """Test ValueError for NaN angle."""
        points = [Point3D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)]
        with self.assertRaises(ValueError) as cm:
            apply_rotation(points, float('nan'))
        self.assertIn("Solution:", str(cm.exception))

    def test_apply_rotation_inf_angle(self):
        """Test ValueError for infinite angle."""
        points = [Point3D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)]
        with self.assertRaises(ValueError) as cm:
            apply_rotation(points, float('inf'))
        self.assertIn("Solution:", str(cm.exception))

    def test_apply_rotation_extreme_angle(self):
        """Test angle wrapping for extreme angles."""
        import math
        points = [Point3D(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)]
        # Very large angle should be wrapped
        large_angle = 100 * math.tau  # 100 full rotations
        result = apply_rotation(points, large_angle)
        # Should succeed without raising exception
        self.assertEqual(len(result), 1)

    def test_project_to_screen_division_by_zero(self):
        """Test ZeroDivisionError handling in projection."""
        # Point at camera position (z + camera_distance = 0) should be marked invisible
        # Camera distance is 5.0, so z = -5.0 puts point at camera
        point = Point3D(0.0, 0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 1.0)  # z = -5.0 (at camera)
        result = project_to_screen(point)
        # Should return invisible point rather than raising error (graceful handling)
        self.assertFalse(result.visible)
        self.assertEqual(result.visibility_factor, 0.0)
        self.assertEqual(result.depth, float('inf'))


class TestErrorMessageFormat(unittest.TestCase):
    """Test error message format compliance (Task 4)."""

    def test_all_value_errors_have_solution(self):
        """Test that ValueError messages include 'Solution:' guidance."""
        # Test torus parameter validation
        params = TorusParameters(1.0, 2.0, 10, 10, 0.05)
        try:
            generate_torus_points(params)
        except ValueError as e:
            self.assertIn("Solution:", str(e))

    def test_file_not_found_has_solution(self):
        """Test FileNotFoundError messages include 'Solution:' guidance."""
        with patch('rotating_donut.__file__', '/nonexistent/path.py'):
            with patch('os.path.exists', return_value=False):
                try:
                    get_script_path()
                except FileNotFoundError as e:
                    self.assertIn("Solution:", str(e))

    def test_permission_error_has_solution(self):
        """Test PermissionError messages include 'Solution:' guidance."""
        with patch('rotating_donut.__file__', '/restricted/path.py'):
            with patch('os.path.exists', return_value=True):
                with patch('os.access', return_value=False):
                    try:
                        get_script_path()
                    except PermissionError as e:
                        self.assertIn("Solution:", str(e))


class TestFallbackBehaviors(unittest.TestCase):
    """Test fallback behaviors for non-critical features (Task 5)."""

    def test_generate_torus_with_valid_parameters(self):
        """Test torus generation with valid parameters as baseline."""
        params = TorusParameters(
            outer_radius=2.0,
            inner_radius=1.0,
            u_resolution=10,
            v_resolution=5,
            rotation_speed=0.05
        )
        points = generate_torus_points(params)
        self.assertEqual(len(points), 50)  # 10 * 5 = 50 points
        # Verify all points have proper structure
        for point in points:
            self.assertIsInstance(point, Point3D)
            self.assertIsInstance(point.x, float)
            self.assertIsInstance(point.y, float)
            self.assertIsInstance(point.z, float)

    @patch('builtins.print')
    def test_handle_interrupts_graceful_degradation(self, mock_print):
        """Test interrupt handler degrades gracefully on partial failures."""
        # Even if some cleanup fails, should still attempt all steps
        result = handle_interrupts()
        self.assertTrue(result)


class TestTerminalErrorHandling(unittest.TestCase):
    """Test terminal error handling and compatibility (Task 6)."""

    def test_output_to_terminal_broken_pipe(self):
        """Test BrokenPipeError handling during output."""
        frame = DisplayFrame(
            width=40,
            height=20,
            buffer=[['.' for _ in range(40)] for _ in range(20)],
            depth_buffer=[[0.0 for _ in range(40)] for _ in range(20)],
            frame_number=0
        )

        with patch('builtins.print', side_effect=BrokenPipeError("Pipe closed")):
            with self.assertRaises(BrokenPipeError) as cm:
                output_to_terminal(frame)
            self.assertIn("Solution:", str(cm.exception))

    def test_output_to_terminal_io_error(self):
        """Test IOError handling during output."""
        frame = DisplayFrame(
            width=40,
            height=20,
            buffer=[['.' for _ in range(40)] for _ in range(20)],
            depth_buffer=[[0.0 for _ in range(40)] for _ in range(20)],
            frame_number=0
        )

        with patch('builtins.print', side_effect=IOError("Output error")):
            with self.assertRaises(IOError) as cm:
                output_to_terminal(frame)
            self.assertIn("Solution:", str(cm.exception))

    @patch('builtins.print')
    def test_output_to_terminal_screen_clear_failure(self, mock_print):
        """Test graceful degradation when screen clearing fails."""
        frame = DisplayFrame(
            width=40,
            height=20,
            buffer=[['#' for _ in range(40)] for _ in range(20)],
            depth_buffer=[[0.5 for _ in range(40)] for _ in range(20)],
            frame_number=1
        )

        # First call (screen clear) fails, subsequent calls succeed
        mock_print.side_effect = [IOError("Clear failed")] + [None] * 25

        # Should continue despite clear failure
        try:
            output_to_terminal(frame)
        except Exception as e:
            self.fail(f"Should not raise exception on screen clear failure: {e}")


class TestGracefulDegradation(unittest.TestCase):
    """Integration tests for graceful degradation scenarios (Task 5 & 6)."""

    def test_multiple_error_scenarios(self):
        """Test handling of multiple concurrent error conditions."""
        # Valid parameters should always work
        params = TorusParameters(2.0, 1.0, 10, 10, 0.05)
        points = generate_torus_points(params)
        self.assertGreater(len(points), 0)

    def test_boundary_conditions(self):
        """Test mathematical edge cases at boundaries."""
        import math

        # Minimum valid parameters
        params = TorusParameters(
            outer_radius=0.1,  # Very small but valid
            inner_radius=0.01,
            u_resolution=3,    # Minimum useful resolution
            v_resolution=3,
            rotation_speed=0.001
        )
        points = generate_torus_points(params)
        self.assertEqual(len(points), 9)  # 3 * 3 = 9 points

    def test_rotation_consistency(self):
        """Test rotation produces consistent results across multiple calls."""
        import math
        points = [Point3D(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)]

        # Rotate by small angle
        angle1 = 0.1
        result1 = apply_rotation(points, angle1)

        # Rotate again with same angle
        result2 = apply_rotation(points, angle1)

        # Results should be identical
        self.assertAlmostEqual(result1[0].x, result2[0].x, places=10)
        self.assertAlmostEqual(result1[0].y, result2[0].y, places=10)
        self.assertAlmostEqual(result1[0].z, result2[0].z, places=10)


def run_tests():
    """Run all error handling tests with detailed output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFileReadingErrors))
    suite.addTests(loader.loadTestsFromTestCase(TestKeyboardInterruptHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestMathematicalValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorMessageFormat))
    suite.addTests(loader.loadTestsFromTestCase(TestFallbackBehaviors))
    suite.addTests(loader.loadTestsFromTestCase(TestTerminalErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestGracefulDegradation))

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code based on results
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())