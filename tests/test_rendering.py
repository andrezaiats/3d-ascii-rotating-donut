#!/usr/bin/env python3
"""
Unit tests for ASCII rendering engine functionality.

Tests cover core rendering engine requirements:
- 40x20 character buffer creation
- Depth sorting (painter's algorithm)
- ASCII character mapping (., -, +, #)
- Screen clearing and frame output
- Terminal compatibility

Test Framework: unittest (Python standard library)
"""

import unittest
import sys
import io
from unittest.mock import patch
from typing import List

# Import the module under test
sys.path.insert(0, '..')
from rotating_donut import (
    generate_ascii_frame,
    output_to_terminal,
    Point2D,
    DisplayFrame,
    TERMINAL_WIDTH,
    TERMINAL_HEIGHT,
    ASCII_CHARS
)


class TestRenderingEngine(unittest.TestCase):
    """Test suite for ASCII rendering engine core functionality."""

    def test_buffer_dimensions_40x20(self):
        """Test AC1: Create 40x20 character buffer for frame rendering."""
        points = []
        frame = generate_ascii_frame(points, frame_number=1)

        # Verify 40x20 dimensions
        self.assertEqual(frame.width, 40)
        self.assertEqual(frame.height, 20)
        self.assertEqual(len(frame.buffer), 20)
        self.assertEqual(len(frame.buffer[0]), 40)
        self.assertEqual(len(frame.depth_buffer), 20)
        self.assertEqual(len(frame.depth_buffer[0]), 40)

        # Verify frame number tracking
        self.assertEqual(frame.frame_number, 1)

    def test_depth_sorting_painters_algorithm(self):
        """Test AC2: Implement depth sorting to render closest points last."""
        # Create overlapping points with different depths
        points = [
            Point2D(x=10, y=10, depth=0.1, visible=True),  # Closest (should win)
            Point2D(x=10, y=10, depth=0.9, visible=True),  # Farthest
            Point2D(x=10, y=10, depth=0.5, visible=True),  # Middle
        ]

        frame = generate_ascii_frame(points)

        # Closest point should be rendered (painter's algorithm)
        self.assertEqual(frame.buffer[10][10], '#')  # depth 0.1 maps to '#'
        self.assertEqual(frame.depth_buffer[10][10], 0.1)

    def test_ascii_character_mapping(self):
        """Test AC3: Use basic ASCII characters (., -, +, #) for depth/brightness levels."""
        test_cases = [
            (0.0, '#'),   # depth < 0.25 -> '#' (closest)
            (0.2, '#'),   # depth < 0.25 -> '#'
            (0.3, '+'),   # 0.25 <= depth < 0.5 -> '+'
            (0.4, '+'),   # 0.25 <= depth < 0.5 -> '+'
            (0.6, '-'),   # 0.5 <= depth < 0.75 -> '-'
            (0.7, '-'),   # 0.5 <= depth < 0.75 -> '-'
            (0.8, '.'),   # depth >= 0.75 -> '.'
            (0.9, '.'),   # depth >= 0.75 -> '.'
        ]

        for depth, expected_char in test_cases:
            with self.subTest(depth=depth, expected=expected_char):
                points = [Point2D(x=5, y=5, depth=depth, visible=True)]
                frame = generate_ascii_frame(points)
                self.assertEqual(frame.buffer[5][5], expected_char)

    def test_terminal_safe_characters_only(self):
        """Test AC3: Only use terminal-safe ASCII characters (., -, +, #)."""
        safe_chars = {'.', '-', '+', '#'}

        # Test with various depths across the valid range
        for depth in [0.0, 0.1, 0.3, 0.6, 0.8, 1.0]:
            points = [Point2D(x=5, y=5, depth=depth, visible=True)]
            frame = generate_ascii_frame(points)
            char = frame.buffer[5][5]
            self.assertIn(char, safe_chars, f"Character '{char}' not terminal-safe")

    @patch('builtins.print')
    def test_screen_clearing_ansi_codes(self, mock_print):
        """Test AC4: Clear screen and render new frame for smooth animation."""
        frame = DisplayFrame(
            width=40,
            height=20,
            buffer=[['.' for _ in range(40)] for _ in range(20)],
            depth_buffer=[[float('inf') for _ in range(40)] for _ in range(20)],
            frame_number=1
        )

        output_to_terminal(frame)

        # Verify screen clearing ANSI code is printed
        calls = mock_print.call_args_list
        self.assertTrue(len(calls) > 0)

        # First call should contain screen clearing codes
        first_call_args, first_call_kwargs = calls[0]
        output = ''.join(str(arg) for arg in first_call_args)
        self.assertIn('\033[2J\033[H', output)  # Clear screen + home cursor

    @patch('builtins.print')
    def test_frame_output_with_flush(self, mock_print):
        """Test AC4: Frame output uses flush=True for smooth animation."""
        frame = DisplayFrame(
            width=40,
            height=20,
            buffer=[['.' for _ in range(40)] for _ in range(20)],
            depth_buffer=[[float('inf') for _ in range(40)] for _ in range(20)],
            frame_number=1
        )

        output_to_terminal(frame)

        # Verify all print calls use flush=True
        for call in mock_print.call_args_list:
            args, kwargs = call
            self.assertTrue(kwargs.get('flush', False), "Print should use flush=True")

    def test_terminal_compatibility_cross_platform(self):
        """Test AC5: Handle terminal compatibility for character output."""
        # Create frame with test pattern
        buffer = [['.' for _ in range(40)] for _ in range(20)]
        buffer[10][20] = '#'  # Test character

        frame = DisplayFrame(
            width=40,
            height=20,
            buffer=buffer,
            depth_buffer=[[float('inf') for _ in range(40)] for _ in range(20)],
            frame_number=1
        )

        # Should not raise exception (basic compatibility test)
        try:
            with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
                output_to_terminal(frame)
                output = mock_stdout.getvalue()
                # Verify output contains expected content
                self.assertIn('#', output)
                self.assertTrue(len(output) > 0)
        except Exception as e:
            self.fail(f"Terminal output failed: {e}")

    def test_invisible_points_filtered(self):
        """Test that invisible points are properly filtered out."""
        points = [
            Point2D(x=10, y=10, depth=0.1, visible=False),  # Should be ignored
            Point2D(x=15, y=15, depth=0.2, visible=True),   # Should be rendered
        ]

        frame = generate_ascii_frame(points)

        # Invisible point should not affect buffer
        self.assertEqual(frame.buffer[10][10], '.')  # Background
        self.assertEqual(frame.depth_buffer[10][10], float('inf'))

        # Visible point should be rendered
        self.assertEqual(frame.buffer[15][15], '#')  # depth 0.2 -> '#'
        self.assertEqual(frame.depth_buffer[15][15], 0.2)

    def test_bounds_checking_safety(self):
        """Test that out-of-bounds coordinates are handled safely."""
        points = [
            Point2D(x=-1, y=10, depth=0.1, visible=True),   # x out of bounds
            Point2D(x=40, y=10, depth=0.1, visible=True),   # x out of bounds
            Point2D(x=10, y=-1, depth=0.1, visible=True),   # y out of bounds
            Point2D(x=10, y=20, depth=0.1, visible=True),   # y out of bounds
            Point2D(x=10, y=10, depth=0.1, visible=True),   # Valid point
        ]

        # Should not crash with out-of-bounds coordinates
        try:
            frame = generate_ascii_frame(points)
            # Only valid point should be rendered
            self.assertEqual(frame.buffer[10][10], '#')
        except Exception as e:
            self.fail(f"Bounds checking failed: {e}")

    def test_buffer_initialization_cleared(self):
        """Test that buffers are properly initialized and cleared."""
        frame = generate_ascii_frame([])

        # All buffer positions should be background
        background_count = 0
        for row in frame.buffer:
            for char in row:
                if char == '.':
                    background_count += 1

        self.assertEqual(background_count, 800)  # 40 * 20 = 800

        # All depth buffer positions should be infinity
        for row in frame.depth_buffer:
            for depth in row:
                self.assertEqual(depth, float('inf'))

    def test_frame_number_tracking(self):
        """Test frame number tracking for debugging purposes."""
        points = [Point2D(x=5, y=5, depth=0.5, visible=True)]

        frame1 = generate_ascii_frame(points, frame_number=42)
        frame2 = generate_ascii_frame(points, frame_number=100)

        self.assertEqual(frame1.frame_number, 42)
        self.assertEqual(frame2.frame_number, 100)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete rendering pipeline."""

    def test_complete_rendering_pipeline(self):
        """Test complete pipeline from Point2D list to terminal output."""
        # Create a simple cross pattern
        points = []

        # Horizontal line (depth 0.3)
        for i in range(10, 30):
            points.append(Point2D(x=i, y=10, depth=0.3, visible=True))

        # Vertical line (depth 0.2 - closer)
        for j in range(5, 15):
            points.append(Point2D(x=20, y=j, depth=0.2, visible=True))

        frame = generate_ascii_frame(points, frame_number=1)

        # Verify cross pattern is rendered correctly
        # Horizontal line should be '+' (depth 0.3)
        for i in range(10, 30):
            if i != 20:  # Skip intersection
                self.assertEqual(frame.buffer[10][i], '+')

        # Vertical line should be '#' (depth 0.2 - closer)
        for j in range(5, 15):
            self.assertEqual(frame.buffer[j][20], '#')

        # Intersection should show closest point ('#')
        self.assertEqual(frame.buffer[10][20], '#')

        # Test terminal output doesn't crash
        with patch('sys.stdout', new_callable=io.StringIO):
            output_to_terminal(frame)


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)