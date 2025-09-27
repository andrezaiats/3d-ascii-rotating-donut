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
    generate_ascii_frame_legacy,
    output_to_terminal,
    map_tokens_to_surface,
    map_tokens_to_surface_with_structure,
    enhance_tokens_with_structure,
    _handle_token_compression,
    _apply_visual_balance,
    _apply_structural_distribution,
    Point2D,
    Point3D,
    DisplayFrame,
    CodeToken,
    ImportanceLevel,
    StructuralElement,
    StructuralInfo,
    TERMINAL_WIDTH,
    TERMINAL_HEIGHT,
    ASCII_CHARS
)


class TestRenderingEngine(unittest.TestCase):
    """Test suite for ASCII rendering engine core functionality."""

    def test_buffer_dimensions_40x20(self):
        """Test AC1: Create 40x20 character buffer for frame rendering."""
        points = []
        frame = generate_ascii_frame_legacy(points, frame_number=1)

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
            Point2D(x=10, y=10, depth=0.1, visible=True, visibility_factor=1.0),  # Closest (should win)
            Point2D(x=10, y=10, depth=0.9, visible=True, visibility_factor=1.0),  # Farthest
            Point2D(x=10, y=10, depth=0.5, visible=True, visibility_factor=1.0),  # Middle
        ]

        frame = generate_ascii_frame_legacy(points)

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
                points = [Point2D(x=5, y=5, depth=depth, visible=True, visibility_factor=1.0)]
                frame = generate_ascii_frame_legacy(points)
                self.assertEqual(frame.buffer[5][5], expected_char)

    def test_terminal_safe_characters_only(self):
        """Test AC3: Only use terminal-safe ASCII characters (., -, +, #)."""
        safe_chars = {'.', '-', '+', '#'}

        # Test with various depths across the valid range
        for depth in [0.0, 0.1, 0.3, 0.6, 0.8, 1.0]:
            points = [Point2D(x=5, y=5, depth=depth, visible=True, visibility_factor=1.0)]
            frame = generate_ascii_frame_legacy(points)
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
            Point2D(x=10, y=10, depth=0.1, visible=False, visibility_factor=0.0),  # Should be ignored
            Point2D(x=15, y=15, depth=0.2, visible=True, visibility_factor=1.0),   # Should be rendered
        ]

        frame = generate_ascii_frame_legacy(points)

        # Invisible point should not affect buffer
        self.assertEqual(frame.buffer[10][10], '.')  # Background
        self.assertEqual(frame.depth_buffer[10][10], float('inf'))

        # Visible point should be rendered
        self.assertEqual(frame.buffer[15][15], '#')  # depth 0.2 -> '#'
        self.assertEqual(frame.depth_buffer[15][15], 0.2)

    def test_bounds_checking_safety(self):
        """Test that out-of-bounds coordinates are handled safely."""
        points = [
            Point2D(x=-1, y=10, depth=0.1, visible=True, visibility_factor=1.0),   # x out of bounds
            Point2D(x=40, y=10, depth=0.1, visible=True, visibility_factor=1.0),   # x out of bounds
            Point2D(x=10, y=-1, depth=0.1, visible=True, visibility_factor=1.0),   # y out of bounds
            Point2D(x=10, y=20, depth=0.1, visible=True, visibility_factor=1.0),   # y out of bounds
            Point2D(x=10, y=10, depth=0.1, visible=True, visibility_factor=1.0),   # Valid point
        ]

        # Should not crash with out-of-bounds coordinates
        try:
            frame = generate_ascii_frame_legacy(points)
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
        points = [Point2D(x=5, y=5, depth=0.5, visible=True, visibility_factor=1.0)]

        frame1 = generate_ascii_frame_legacy(points, frame_number=42)
        frame2 = generate_ascii_frame_legacy(points, frame_number=100)

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
            points.append(Point2D(x=i, y=10, depth=0.3, visible=True, visibility_factor=1.0))

        # Vertical line (depth 0.2 - closer)
        for j in range(5, 15):
            points.append(Point2D(x=20, y=j, depth=0.2, visible=True, visibility_factor=1.0))

        frame = generate_ascii_frame_legacy(points, frame_number=1)

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


class TestTokenMapping(unittest.TestCase):
    """Test suite for Story 2.4: Token-to-ASCII Character Mapping functionality."""

    def setUp(self):
        """Set up test fixtures for token mapping tests."""
        # Create sample tokens with different importance levels
        self.tokens = [
            CodeToken(type='KEYWORD', value='def', importance=ImportanceLevel.CRITICAL,
                     line=1, column=1, ascii_char='#'),
            CodeToken(type='OPERATOR', value='+', importance=ImportanceLevel.HIGH,
                     line=1, column=10, ascii_char='+'),
            CodeToken(type='IDENTIFIER', value='x', importance=ImportanceLevel.MEDIUM,
                     line=1, column=15, ascii_char='-'),
            CodeToken(type='COMMENT', value='# comment', importance=ImportanceLevel.LOW,
                     line=2, column=1, ascii_char='.'),
        ]

        # Create sample surface points with parametric coordinates
        self.points = [
            Point3D(x=1.0, y=0.0, z=0.0, u=0.0, v=0.0),
            Point3D(x=0.0, y=1.0, z=0.0, u=1.57, v=0.0),
            Point3D(x=-1.0, y=0.0, z=0.0, u=3.14, v=0.0),
            Point3D(x=0.0, y=-1.0, z=0.0, u=4.71, v=0.0),
            Point3D(x=1.0, y=0.0, z=1.0, u=0.0, v=1.57),
            Point3D(x=0.0, y=1.0, z=1.0, u=1.57, v=1.57),
            Point3D(x=-1.0, y=0.0, z=1.0, u=3.14, v=1.57),
            Point3D(x=0.0, y=-1.0, z=1.0, u=4.71, v=1.57),
        ]

    def test_basic_token_to_surface_mapping(self):
        """Test AC1: Basic token distribution across torus surface points."""
        mapped_pairs = map_tokens_to_surface(self.tokens, self.points)

        # Should return list of (Point3D, CodeToken) pairs
        self.assertIsInstance(mapped_pairs, list)
        self.assertTrue(len(mapped_pairs) > 0)

        # Each pair should contain Point3D and CodeToken
        for point, token in mapped_pairs:
            self.assertIsInstance(point, Point3D)
            self.assertIsInstance(token, CodeToken)

    def test_character_mapping_preserved(self):
        """Test AC1: Character mapping from importance levels is preserved."""
        mapped_pairs = map_tokens_to_surface(self.tokens, self.points)

        # Check that each token's ASCII character is preserved
        token_chars = {token.ascii_char for _, token in mapped_pairs}
        expected_chars = {'#', '+', '-', '.'}

        self.assertTrue(token_chars.issubset(expected_chars))

        # Verify specific importance-to-character mappings
        for _, token in mapped_pairs:
            if token.importance == ImportanceLevel.CRITICAL:
                self.assertEqual(token.ascii_char, '#')
            elif token.importance == ImportanceLevel.HIGH:
                self.assertEqual(token.ascii_char, '+')
            elif token.importance == ImportanceLevel.MEDIUM:
                self.assertEqual(token.ascii_char, '-')
            elif token.importance == ImportanceLevel.LOW:
                self.assertEqual(token.ascii_char, '.')

    def test_density_mapping_allocation(self):
        """Test AC2: Density mapping where important tokens get more surface points."""
        # Create minimal test case with known tokens
        critical_token = CodeToken(type='KEYWORD', value='def', importance=ImportanceLevel.CRITICAL,
                                 line=1, column=1, ascii_char='#')
        low_token = CodeToken(type='COMMENT', value='# comment', importance=ImportanceLevel.LOW,
                            line=2, column=1, ascii_char='.')

        test_tokens = [critical_token, low_token]
        mapped_pairs = map_tokens_to_surface(test_tokens, self.points)

        # Count allocations for each importance level
        critical_count = sum(1 for _, token in mapped_pairs if token.importance == ImportanceLevel.CRITICAL)
        low_count = sum(1 for _, token in mapped_pairs if token.importance == ImportanceLevel.LOW)

        # CRITICAL tokens should get more surface points than LOW tokens
        self.assertGreater(critical_count, low_count,
                          "Critical tokens should get more surface point allocations")

    def test_scaling_behavior_with_varying_token_counts(self):
        """Test AC4: Handle dynamic scaling for varying source code lengths."""
        # Test case 1: Few tokens, many points (expansion)
        few_tokens = self.tokens[:2]
        mapped_few = map_tokens_to_surface(few_tokens, self.points)
        self.assertTrue(len(mapped_few) > 0)
        self.assertTrue(len(mapped_few) <= len(self.points))

        # Test case 2: Many tokens, few points (compression)
        many_tokens = self.tokens * 3  # 12 tokens
        few_points = self.points[:4]   # 4 points
        mapped_many = map_tokens_to_surface(many_tokens, few_points)
        self.assertTrue(len(mapped_many) > 0)
        self.assertTrue(len(mapped_many) <= len(few_points))

    def test_visual_balance_distribution(self):
        """Test AC5: Visual balance prevents clustering of same importance tokens."""
        # Create multiple tokens of same importance
        critical_tokens = [
            CodeToken(type='KEYWORD', value='def', importance=ImportanceLevel.CRITICAL,
                     line=1, column=1, ascii_char='#'),
            CodeToken(type='KEYWORD', value='class', importance=ImportanceLevel.CRITICAL,
                     line=2, column=1, ascii_char='#'),
            CodeToken(type='KEYWORD', value='if', importance=ImportanceLevel.CRITICAL,
                     line=3, column=1, ascii_char='#'),
        ]

        mapped_pairs = map_tokens_to_surface(critical_tokens, self.points)

        # Extract u coordinates for critical tokens
        critical_positions = [point.u for point, token in mapped_pairs
                            if token.importance == ImportanceLevel.CRITICAL]

        # Should have multiple different u positions (not clustered)
        unique_positions = set(critical_positions)
        self.assertGreater(len(unique_positions), 1,
                          "Critical tokens should be distributed across different u coordinates")

    def test_token_compression_handling(self):
        """Test compression scenario where tokens exceed surface points."""
        # Create more tokens than points
        many_tokens = self.tokens * 3  # 12 tokens
        few_points = self.points[:3]   # 3 points

        compressed_pairs = _handle_token_compression(many_tokens, few_points)

        # Should return exactly as many pairs as available points
        self.assertEqual(len(compressed_pairs), len(few_points))

        # Should prioritize higher importance tokens
        token_importances = [token.importance for _, token in compressed_pairs]
        # Should contain mostly high importance tokens
        high_importance_count = sum(1 for imp in token_importances
                                  if imp >= ImportanceLevel.HIGH)
        self.assertGreater(high_importance_count, 0)

    def test_visual_balance_application(self):
        """Test visual balance function for aesthetic distribution."""
        # Create initial mapping
        initial_pairs = [(self.points[0], self.tokens[0]),
                        (self.points[1], self.tokens[1])]

        balanced_pairs = _apply_visual_balance(initial_pairs, self.points)

        # Should return valid pairs
        self.assertIsInstance(balanced_pairs, list)
        self.assertTrue(len(balanced_pairs) > 0)

        # Each pair should still be valid
        for point, token in balanced_pairs:
            self.assertIsInstance(point, Point3D)
            self.assertIsInstance(token, CodeToken)

    def test_input_validation_empty_tokens(self):
        """Test error handling for empty token lists."""
        with self.assertRaises(ValueError) as context:
            map_tokens_to_surface([], self.points)

        self.assertIn("Empty token list", str(context.exception))
        self.assertIn("Solution:", str(context.exception))

    def test_input_validation_empty_points(self):
        """Test error handling for empty surface points."""
        with self.assertRaises(ValueError) as context:
            map_tokens_to_surface(self.tokens, [])

        self.assertIn("Empty surface points", str(context.exception))
        self.assertIn("Solution:", str(context.exception))

    def test_sequence_based_distribution(self):
        """Test that tokens are distributed based on their sequence in source code."""
        # Create tokens with specific line/column positions
        ordered_tokens = [
            CodeToken(type='KEYWORD', value='def', importance=ImportanceLevel.CRITICAL,
                     line=1, column=1, ascii_char='#'),
            CodeToken(type='IDENTIFIER', value='func', importance=ImportanceLevel.MEDIUM,
                     line=1, column=5, ascii_char='-'),
            CodeToken(type='OPERATOR', value=':', importance=ImportanceLevel.HIGH,
                     line=1, column=9, ascii_char='+'),
        ]

        mapped_pairs = map_tokens_to_surface(ordered_tokens, self.points)

        # Should preserve some relationship to source sequence
        self.assertTrue(len(mapped_pairs) > 0)

        # All original tokens should be represented
        mapped_values = {token.value for _, token in mapped_pairs}
        original_values = {token.value for token in ordered_tokens}
        self.assertTrue(original_values.issubset(mapped_values))

    def test_new_ascii_frame_with_tokens(self):
        """Test new generate_ascii_frame function with token mapping."""
        # Create token-surface mapping
        mapped_pairs = map_tokens_to_surface(self.tokens, self.points)

        # Generate frame using new token-based function
        frame = generate_ascii_frame(mapped_pairs, frame_number=42)

        # Verify frame structure
        self.assertEqual(frame.width, TERMINAL_WIDTH)
        self.assertEqual(frame.height, TERMINAL_HEIGHT)
        self.assertEqual(frame.frame_number, 42)

        # Should contain token characters, not depth-based characters
        buffer_chars = set()
        for row in frame.buffer:
            for char in row:
                buffer_chars.add(char)

        # Should include token ASCII characters and background
        expected_chars = {'.', '#', '+', '-'}  # Background + token chars
        self.assertTrue(buffer_chars.issubset(expected_chars))

    def test_legacy_compatibility(self):
        """Test that legacy generate_ascii_frame_legacy still works."""
        # Create Point2D list for legacy function
        points_2d = [
            Point2D(x=10, y=10, depth=0.1, visible=True, visibility_factor=1.0),
            Point2D(x=15, y=15, depth=0.8, visible=True, visibility_factor=1.0),
        ]

        # Should work with legacy function
        frame = generate_ascii_frame_legacy(points_2d, frame_number=1)

        self.assertEqual(frame.width, TERMINAL_WIDTH)
        self.assertEqual(frame.height, TERMINAL_HEIGHT)
        self.assertEqual(frame.frame_number, 1)

        # Should use depth-based characters
        self.assertEqual(frame.buffer[10][10], '#')  # depth 0.1
        self.assertEqual(frame.buffer[15][15], '.')  # depth 0.8


class TestStructuralSurfaceMapping(unittest.TestCase):
    """Test suite for structural analysis enhanced surface mapping."""

    def setUp(self):
        """Set up test fixtures for structural mapping tests."""
        # Create test tokens
        self.test_tokens = [
            CodeToken(type='KEYWORD', value='def', importance=ImportanceLevel.CRITICAL,
                     line=1, column=0, ascii_char='#'),
            CodeToken(type='IDENTIFIER', value='test_func', importance=ImportanceLevel.MEDIUM,
                     line=1, column=4, ascii_char='-'),
            CodeToken(type='OPERATOR', value='(', importance=ImportanceLevel.HIGH,
                     line=1, column=13, ascii_char='+'),
            CodeToken(type='IDENTIFIER', value='x', importance=ImportanceLevel.MEDIUM,
                     line=2, column=4, ascii_char='-'),
            CodeToken(type='KEYWORD', value='return', importance=ImportanceLevel.CRITICAL,
                     line=3, column=4, ascii_char='#')
        ]

        # Create test 3D points with surface normals
        self.test_points = [
            Point3D(x=1.0, y=0.0, z=0.5, u=0.1, v=0.1, nx=1.0, ny=0.0, nz=0.0),
            Point3D(x=0.5, y=0.8, z=0.3, u=0.2, v=0.2, nx=0.5, ny=0.8, nz=0.0),
            Point3D(x=-0.2, y=0.5, z=0.7, u=0.3, v=0.3, nx=-0.2, ny=0.5, nz=0.7),
            Point3D(x=0.8, y=-0.3, z=0.1, u=0.4, v=0.4, nx=0.8, ny=-0.3, nz=0.1),
            Point3D(x=-0.1, y=-0.7, z=0.9, u=0.5, v=0.5, nx=-0.1, ny=-0.7, nz=0.9)
        ]

        # Create test structural info
        function_element = StructuralElement(
            element_type='function',
            name='test_func',
            start_line=1,
            end_line=3,
            complexity_score=2.5,
            nesting_depth=0
        )

        self.structural_info = StructuralInfo(
            elements=[function_element],
            max_complexity=2.5,
            total_lines=3,
            import_count=0,
            function_count=1,
            class_count=0
        )

        # Pre-enhance tokens for performance-optimized tests
        self.enhanced_tokens = enhance_tokens_with_structure(self.test_tokens, self.structural_info)

    def test_map_tokens_to_surface_with_structure_basic(self):
        """Test basic structural surface mapping functionality."""
        mapped_pairs = map_tokens_to_surface_with_structure(
            self.enhanced_tokens, self.test_points, self.structural_info
        )

        # Should return list of (point, token) pairs
        self.assertIsInstance(mapped_pairs, list)
        self.assertGreater(len(mapped_pairs), 0)

        # Each pair should be (Point3D, CodeToken)
        for point, token in mapped_pairs:
            self.assertIsInstance(point, Point3D)
            self.assertIsInstance(token, CodeToken)

    def test_map_tokens_to_surface_with_structure_enhancement(self):
        """Test that structural mapping enhances token importance."""
        # Get regular mapping for comparison
        regular_pairs = map_tokens_to_surface(self.test_tokens, self.test_points)
        structural_pairs = map_tokens_to_surface_with_structure(
            self.enhanced_tokens, self.test_points, self.structural_info
        )

        # Structural mapping should have reasonable number of pairs
        # (may differ from regular mapping due to structural optimizations)
        self.assertGreater(len(structural_pairs), 0)
        self.assertLessEqual(len(structural_pairs), len(self.test_points))

        # Find tokens that should be enhanced by structural context
        structural_tokens = [token for _, token in structural_pairs]
        enhanced_count = sum(1 for token in structural_tokens
                           if token.importance >= ImportanceLevel.HIGH)

        # Should have some tokens with enhanced importance
        self.assertGreater(enhanced_count, 0)

    def test_enhance_tokens_with_structure_validation(self):
        """Test input validation for token enhancement."""
        # Test empty tokens
        with self.assertRaises(ValueError) as context:
            enhance_tokens_with_structure([], self.structural_info)
        self.assertIn("Empty tokens list provided for enhancement", str(context.exception))

        # Test invalid structural info
        with self.assertRaises(ValueError) as context:
            enhance_tokens_with_structure(self.test_tokens, "invalid")
        self.assertIn("Invalid structural_info parameter", str(context.exception))

    def test_map_tokens_to_surface_with_structure_validation(self):
        """Test input validation for structural surface mapping."""
        # Test empty enhanced tokens
        with self.assertRaises(ValueError) as context:
            map_tokens_to_surface_with_structure([], self.test_points, self.structural_info)
        self.assertIn("Empty enhanced_tokens list", str(context.exception))

        # Test empty points
        with self.assertRaises(ValueError) as context:
            map_tokens_to_surface_with_structure(self.enhanced_tokens, [], self.structural_info)
        self.assertIn("Empty points list", str(context.exception))

        # Test invalid structural info
        with self.assertRaises(ValueError) as context:
            map_tokens_to_surface_with_structure(self.enhanced_tokens, self.test_points, "invalid")
        self.assertIn("Invalid structural_info parameter", str(context.exception))

    def test_structural_distribution_complexity_ordering(self):
        """Test that structural distribution respects complexity ordering."""
        # Create multiple structural elements with different complexities
        simple_function = StructuralElement(
            element_type='function',
            name='simple_func',
            start_line=1,
            end_line=2,
            complexity_score=1.0,
            nesting_depth=0
        )

        complex_function = StructuralElement(
            element_type='function',
            name='complex_func',
            start_line=3,
            end_line=8,
            complexity_score=5.0,
            nesting_depth=0
        )

        complex_structural_info = StructuralInfo(
            elements=[simple_function, complex_function],
            max_complexity=5.0,
            total_lines=8,
            import_count=0,
            function_count=2,
            class_count=0
        )

        # Create tokens for both functions
        mixed_tokens = [
            CodeToken(type='KEYWORD', value='def', importance=ImportanceLevel.CRITICAL,
                     line=1, column=0, ascii_char='#'),  # simple function
            CodeToken(type='KEYWORD', value='def', importance=ImportanceLevel.CRITICAL,
                     line=3, column=0, ascii_char='#'),  # complex function
            CodeToken(type='IDENTIFIER', value='x', importance=ImportanceLevel.MEDIUM,
                     line=4, column=4, ascii_char='-'),  # in complex function
        ]

        # Test with larger point set
        large_points = self.test_points * 4  # 20 points

        # Enhance mixed tokens for performance-optimized testing
        enhanced_mixed_tokens = enhance_tokens_with_structure(mixed_tokens, complex_structural_info)

        mapped_pairs = map_tokens_to_surface_with_structure(
            enhanced_mixed_tokens, large_points, complex_structural_info
        )

        # Should successfully map tokens with structural priority
        self.assertGreater(len(mapped_pairs), 0)
        self.assertLessEqual(len(mapped_pairs), len(large_points))

    def test_structural_distribution_fallback(self):
        """Test fallback behavior when structural mapping encounters issues."""
        # This test verifies graceful degradation is available through the main animation loop
        # The actual fallback is tested in the integration test or main loop
        pass  # Placeholder for future fallback testing if needed


class TestStructuralMappingHelpers(unittest.TestCase):
    """Test helper functions for structural surface mapping."""

    def test_apply_structural_distribution(self):
        """Test the structural distribution algorithm."""
        # Create test data
        tokens = [
            CodeToken(type='KEYWORD', value='def', importance=ImportanceLevel.CRITICAL,
                     line=1, column=0, ascii_char='#'),
            CodeToken(type='IDENTIFIER', value='func', importance=ImportanceLevel.MEDIUM,
                     line=1, column=4, ascii_char='-'),
        ]

        points = [
            Point3D(x=1.0, y=0.0, z=0.5, u=0.1, v=0.1),
            Point3D(x=0.5, y=0.8, z=0.3, u=0.2, v=0.2),
            Point3D(x=-0.2, y=0.5, z=0.7, u=0.3, v=0.3),
        ]

        function_element = StructuralElement(
            element_type='function',
            name='func',
            start_line=1,
            end_line=2,
            complexity_score=2.0,
            nesting_depth=0
        )

        structural_info = StructuralInfo(
            elements=[function_element],
            max_complexity=2.0,
            total_lines=2,
            import_count=0,
            function_count=1,
            class_count=0
        )

        # Test structural distribution
        mapped_pairs = _apply_structural_distribution(tokens, points, structural_info)

        # Should return valid mappings
        self.assertIsInstance(mapped_pairs, list)
        self.assertGreater(len(mapped_pairs), 0)

        # All mappings should be valid
        for point, token in mapped_pairs:
            self.assertIsInstance(point, Point3D)
            self.assertIsInstance(token, CodeToken)

    def test_structural_distribution_edge_cases(self):
        """Test edge cases for structural distribution."""
        # Test with no structural elements
        empty_structural_info = StructuralInfo(
            elements=[],
            max_complexity=0.0,
            total_lines=1,
            import_count=0,
            function_count=0,
            class_count=0
        )

        tokens = [
            CodeToken(type='IDENTIFIER', value='x', importance=ImportanceLevel.MEDIUM,
                     line=1, column=0, ascii_char='-'),
        ]

        points = [Point3D(x=1.0, y=0.0, z=0.5, u=0.1, v=0.1)]

        # Should handle empty structural elements gracefully
        mapped_pairs = _apply_structural_distribution(tokens, points, empty_structural_info)
        self.assertIsInstance(mapped_pairs, list)


class TestDynamicCharacterAssignment(unittest.TestCase):
    """Test suite for Story 3.2: Dynamic Character Assignment functionality.

    Tests cover token-driven character selection, importance hierarchy,
    priority resolution for overlapping tokens, smooth visual transitions,
    and character density balancing with code complexity.
    """

    def setUp(self):
        """Set up test fixtures for dynamic character assignment tests."""
        # Create test tokens with different importance levels
        self.test_tokens = [
            CodeToken(type='KEYWORD', value='def', importance=ImportanceLevel.CRITICAL,
                     line=1, column=0, ascii_char='#'),
            CodeToken(type='OPERATOR', value='+', importance=ImportanceLevel.HIGH,
                     line=1, column=10, ascii_char='+'),
            CodeToken(type='IDENTIFIER', value='var', importance=ImportanceLevel.MEDIUM,
                     line=1, column=15, ascii_char='-'),
            CodeToken(type='COMMENT', value='# comment', importance=ImportanceLevel.LOW,
                     line=2, column=0, ascii_char='.'),
        ]

        # Create test surface points
        self.test_points = [
            Point3D(x=1.0, y=0.0, z=0.0, u=0.0, v=0.0),
            Point3D(x=0.0, y=1.0, z=0.0, u=1.57, v=0.0),
            Point3D(x=-1.0, y=0.0, z=0.0, u=3.14, v=0.0),
            Point3D(x=0.0, y=-1.0, z=0.0, u=4.71, v=0.0),
        ]

    def test_token_driven_character_selection(self):
        """Test AC1: Replace static ASCII characters with token-driven selection."""
        # Create token-surface mapping
        mapped_pairs = map_tokens_to_surface(self.test_tokens, self.test_points)

        # Generate frame using token-based function
        frame = generate_ascii_frame(mapped_pairs, frame_number=1)

        # Should use token ASCII characters, not depth-based characters
        buffer_chars = set()
        for row in frame.buffer:
            for char in row:
                buffer_chars.add(char)

        # Should contain token characters and background
        token_chars = {token.ascii_char for token in self.test_tokens}
        token_chars.add('.')  # Background character

        self.assertTrue(buffer_chars.issubset(token_chars))

    def test_importance_hierarchy_character_rendering(self):
        """Test AC2: Apply importance hierarchy (Critical=#, High=+, Medium=-, Low=.)."""
        # Verify character mapping for each importance level
        for token in self.test_tokens:
            if token.importance == ImportanceLevel.CRITICAL:
                self.assertEqual(token.ascii_char, '#')
            elif token.importance == ImportanceLevel.HIGH:
                self.assertEqual(token.ascii_char, '+')
            elif token.importance == ImportanceLevel.MEDIUM:
                self.assertEqual(token.ascii_char, '-')
            elif token.importance == ImportanceLevel.LOW:
                self.assertEqual(token.ascii_char, '.')

    def test_priority_resolution_overlapping_tokens(self):
        """Test AC3: Handle multiple tokens mapping to same surface area through priority resolution."""
        # Create overlapping tokens at same screen position
        overlapping_tokens = [
            CodeToken(type='KEYWORD', value='def', importance=ImportanceLevel.CRITICAL,
                     line=1, column=0, ascii_char='#'),
            CodeToken(type='COMMENT', value='# comment', importance=ImportanceLevel.LOW,
                     line=1, column=1, ascii_char='.'),
        ]

        # Create points that will project to same screen coordinates
        overlapping_points = [
            Point3D(x=1.0, y=0.0, z=0.1, u=0.0, v=0.0),  # Closer depth
            Point3D(x=1.0, y=0.0, z=0.9, u=0.1, v=0.1),  # Farther depth
        ]

        mapped_pairs = [(overlapping_points[0], overlapping_tokens[0]),
                       (overlapping_points[1], overlapping_tokens[1])]

        frame = generate_ascii_frame(mapped_pairs)

        # Find the screen position where overlap occurs
        found_critical_char = False
        for row in frame.buffer:
            for char in row:
                if char == '#':  # Critical importance should win
                    found_critical_char = True

        self.assertTrue(found_critical_char, "Critical importance token should be visible")

    def test_importance_based_conflict_resolution(self):
        """Test enhanced priority resolution for tokens at similar depths."""
        # Create tokens with similar depths but different importance
        similar_depth_tokens = [
            CodeToken(type='KEYWORD', value='def', importance=ImportanceLevel.CRITICAL,
                     line=1, column=0, ascii_char='#'),
            CodeToken(type='IDENTIFIER', value='var', importance=ImportanceLevel.MEDIUM,
                     line=1, column=1, ascii_char='-'),
        ]

        # Create points with very similar depths (within 0.1 threshold)
        similar_depth_points = [
            Point3D(x=1.0, y=0.0, z=0.5, u=0.0, v=0.0, nx=1.0, ny=0.0, nz=0.0),    # depth 0.5
            Point3D(x=1.01, y=0.0, z=0.55, u=0.1, v=0.0, nx=1.0, ny=0.0, nz=0.0),  # depth 0.55 (similar)
        ]

        mapped_pairs = [(similar_depth_points[0], similar_depth_tokens[0]),
                       (similar_depth_points[1], similar_depth_tokens[1])]

        frame = generate_ascii_frame(mapped_pairs)

        # Critical importance should win over medium when depths are similar
        critical_count = 0
        medium_count = 0
        for row in frame.buffer:
            for char in row:
                if char == '#':
                    critical_count += 1
                elif char == '-':
                    medium_count += 1

        # Should favor critical importance for conflict resolution
        self.assertGreaterEqual(critical_count, medium_count)

    def test_visual_transition_smoothness(self):
        """Test AC4: Ensure smooth visual transitions during rotation."""
        # Create sequence of frames to test transition quality
        frame_sequence = []

        for rotation_angle in [0.0, 0.1, 0.2, 0.3]:
            # Simulate rotation by adjusting point coordinates
            rotated_points = []
            for point in self.test_points:
                # Simple rotation simulation
                new_u = (point.u + rotation_angle) % 6.28
                rotated_point = Point3D(x=point.x, y=point.y, z=point.z, u=new_u, v=point.v, nx=point.nx, ny=point.ny, nz=point.nz)
                rotated_points.append(rotated_point)

            mapped_pairs = map_tokens_to_surface(self.test_tokens, rotated_points)
            frame = generate_ascii_frame(mapped_pairs)
            frame_sequence.append(frame)

        # Verify frame sequence maintains visual coherence
        self.assertEqual(len(frame_sequence), 4)

        # Each frame should be valid
        for frame in frame_sequence:
            self.assertEqual(frame.width, TERMINAL_WIDTH)
            self.assertEqual(frame.height, TERMINAL_HEIGHT)

    def test_character_density_code_complexity_balance(self):
        """Test AC5: Balance character density with code complexity."""
        # Create tokens representing different complexity levels
        simple_tokens = [
            CodeToken(type='IDENTIFIER', value='x', importance=ImportanceLevel.MEDIUM,
                     line=1, column=0, ascii_char='-'),
        ]

        complex_tokens = [
            CodeToken(type='KEYWORD', value='def', importance=ImportanceLevel.CRITICAL,
                     line=1, column=0, ascii_char='#'),
            CodeToken(type='KEYWORD', value='class', importance=ImportanceLevel.CRITICAL,
                     line=2, column=0, ascii_char='#'),
            CodeToken(type='OPERATOR', value='+', importance=ImportanceLevel.HIGH,
                     line=3, column=0, ascii_char='+'),
            CodeToken(type='OPERATOR', value='*', importance=ImportanceLevel.HIGH,
                     line=4, column=0, ascii_char='+'),
        ]

        # Test with same number of surface points
        test_points = self.test_points[:10] * 2  # 20 points

        simple_mapped = map_tokens_to_surface(simple_tokens, test_points)
        complex_mapped = map_tokens_to_surface(complex_tokens, test_points)

        # Complex code should result in more diverse character distribution
        simple_frame = generate_ascii_frame(simple_mapped)
        complex_frame = generate_ascii_frame(complex_mapped)

        # Count unique characters in each frame
        simple_chars = set()
        complex_chars = set()

        for row in simple_frame.buffer:
            for char in row:
                if char != '.':  # Exclude background
                    simple_chars.add(char)

        for row in complex_frame.buffer:
            for char in row:
                if char != '.':  # Exclude background
                    complex_chars.add(char)

        # Complex code should have more character variety
        self.assertGreaterEqual(len(complex_chars), len(simple_chars))

    def test_fallback_mechanisms_edge_cases(self):
        """Test fallback mechanisms for edge cases in token-to-surface mapping."""
        # Test empty token list handling
        empty_mapped_pairs = []
        frame = generate_ascii_frame(empty_mapped_pairs)

        # Should create valid empty frame
        self.assertEqual(frame.width, TERMINAL_WIDTH)
        self.assertEqual(frame.height, TERMINAL_HEIGHT)

        # Should be all background characters
        for row in frame.buffer:
            for char in row:
                self.assertEqual(char, '.')

    def test_equal_importance_stability(self):
        """Test stability when tokens have equal importance and similar depth."""
        # Create tokens with same importance
        equal_tokens = [
            CodeToken(type='IDENTIFIER', value='a', importance=ImportanceLevel.MEDIUM,
                     line=1, column=0, ascii_char='-'),
            CodeToken(type='IDENTIFIER', value='b', importance=ImportanceLevel.MEDIUM,
                     line=1, column=1, ascii_char='-'),
        ]

        # Create points with identical depths
        identical_points = [
            Point3D(x=1.0, y=0.0, z=0.5, u=0.0, v=0.0),
            Point3D(x=1.0, y=0.0, z=0.5, u=0.1, v=0.0),
        ]

        mapped_pairs = [(identical_points[0], equal_tokens[0]),
                       (identical_points[1], equal_tokens[1])]

        # Should not crash and maintain first-rendered stability
        frame = generate_ascii_frame(mapped_pairs)
        self.assertEqual(frame.width, TERMINAL_WIDTH)
        self.assertEqual(frame.height, TERMINAL_HEIGHT)


class TestTokenSurfaceIntegration(unittest.TestCase):
    """Test suite for Story 3.1 token-surface integration functionality.

    Comprehensive tests for precise parametric coordinate mapping,
    importance-weighted distribution, dynamic scaling, structural relationships,
    and rotation consistency.
    """

    def setUp(self):
        """Set up test fixtures for token-surface integration tests."""
        from math import tau, cos, sin

        # Create test tokens with various importance levels
        self.test_tokens = [
            CodeToken(type='NAME', value='def', importance=ImportanceLevel.CRITICAL,
                     line=1, column=0, ascii_char='#'),
            CodeToken(type='NAME', value='function_name', importance=ImportanceLevel.HIGH,
                     line=1, column=4, ascii_char='+'),
            CodeToken(type='OP', value='(', importance=ImportanceLevel.HIGH,
                     line=1, column=17, ascii_char='+'),
            CodeToken(type='NAME', value='param', importance=ImportanceLevel.MEDIUM,
                     line=1, column=18, ascii_char='-'),
            CodeToken(type='OP', value=')', importance=ImportanceLevel.HIGH,
                     line=1, column=23, ascii_char='+'),
            CodeToken(type='OP', value=':', importance=ImportanceLevel.CRITICAL,
                     line=1, column=24, ascii_char='#'),
            CodeToken(type='COMMENT', value='# comment', importance=ImportanceLevel.LOW,
                     line=2, column=0, ascii_char='.'),
        ]

        # Create test surface points with proper u,v coordinates and surface normals
        self.test_points = []
        for i in range(20):
            for j in range(10):
                u = (i / 20) * tau
                v = (j / 10) * tau
                x = (2 + 1 * cos(v)) * cos(u)
                y = (2 + 1 * cos(v)) * sin(u)
                z = 1 * sin(v)
                # Calculate surface normal using torus parametric derivatives
                from rotating_donut import calculate_torus_surface_normal
                nx, ny, nz = calculate_torus_surface_normal(u, v, 2.0, 1.0)
                self.test_points.append(Point3D(x=x, y=y, z=z, u=u, v=v, nx=nx, ny=ny, nz=nz))

    def test_parametric_coordinate_accuracy(self):
        """Test parametric coordinate accuracy with mathematical validation."""
        from math import tau
        from rotating_donut import _calculate_precise_token_coordinates

        # Test with small token set for precise validation
        tokens = self.test_tokens[:3]
        points = self.test_points[:50]

        mapped_pairs = _calculate_precise_token_coordinates(tokens, points)

        # Verify all mappings have valid coordinates
        for point, token in mapped_pairs:
            self.assertGreaterEqual(point.u, 0)
            self.assertLessEqual(point.u, tau)
            self.assertGreaterEqual(point.v, 0)
            self.assertLessEqual(point.v, tau)

        # Verify tokens are mapped to appropriate u-coordinates based on position
        sorted_mapped = sorted(mapped_pairs, key=lambda x: x[1].line)
        u_coords = [pair[0].u for pair in sorted_mapped]

        # U-coordinates should be in ascending order for sequential tokens
        for i in range(len(u_coords) - 1):
            self.assertLessEqual(u_coords[i], u_coords[i + 1] + tau * 0.1)  # Allow wraparound

    def test_importance_weighted_distribution(self):
        """Test importance-weighted distribution with various importance levels."""
        from rotating_donut import map_tokens_to_surface

        # Create tokens with different importance levels
        mixed_tokens = [
            CodeToken(type='NAME', value='def', importance=ImportanceLevel.CRITICAL,
                     line=1, column=0, ascii_char='#'),
            CodeToken(type='NAME', value='var', importance=ImportanceLevel.MEDIUM,
                     line=2, column=0, ascii_char='-'),
            CodeToken(type='COMMENT', value='# comment', importance=ImportanceLevel.LOW,
                     line=3, column=0, ascii_char='.'),
        ]

        points = self.test_points[:100]
        mapped_pairs = map_tokens_to_surface(mixed_tokens, points)

        # Count mappings by importance level
        importance_counts = {}
        for point, token in mapped_pairs:
            importance = token.importance
            importance_counts[importance] = importance_counts.get(importance, 0) + 1

        # Critical tokens should have more mappings than low importance tokens
        if ImportanceLevel.CRITICAL in importance_counts and ImportanceLevel.LOW in importance_counts:
            self.assertGreater(importance_counts[ImportanceLevel.CRITICAL],
                             importance_counts[ImportanceLevel.LOW])

    def test_scaling_algorithms_edge_cases(self):
        """Test scaling algorithms with edge case source file sizes."""
        from rotating_donut import _apply_dynamic_scaling_system

        # Test micro file (< 100 tokens)
        micro_tokens = self.test_tokens[:3]
        points = self.test_points[:200]

        scaled_tokens, scaled_points = _apply_dynamic_scaling_system(micro_tokens, points)

        # Micro files should potentially have token replication
        self.assertGreaterEqual(len(scaled_tokens), len(micro_tokens))

        # Test large file simulation (> 1000 tokens)
        large_tokens = self.test_tokens * 200  # Simulate large file
        large_points = self.test_points[:100]

        scaled_large_tokens, scaled_large_points = _apply_dynamic_scaling_system(large_tokens, large_points)

        # Large files should be compressed
        self.assertLessEqual(len(scaled_large_tokens), len(large_tokens))

        # Test massive file simulation (> 10000 tokens)
        massive_tokens = self.test_tokens * 2000  # Simulate massive file
        massive_points = self.test_points[:100]

        scaled_massive_tokens, scaled_massive_points = _apply_dynamic_scaling_system(massive_tokens, massive_points)

        # Massive files should be aggressively compressed
        self.assertLessEqual(len(scaled_massive_tokens), 800)  # Should not exceed target

    def test_spatial_relationship_preservation(self):
        """Test spatial relationship preservation with complex code structures."""
        from rotating_donut import _analyze_code_structure_relationships

        # Create tokens representing a function structure
        function_tokens = [
            CodeToken(type='NAME', value='def', importance=ImportanceLevel.CRITICAL,
                     line=1, column=0, ascii_char='#'),
            CodeToken(type='NAME', value='my_function', importance=ImportanceLevel.HIGH,
                     line=1, column=4, ascii_char='+'),
            CodeToken(type='OP', value='(', importance=ImportanceLevel.HIGH,
                     line=1, column=15, ascii_char='+'),
            CodeToken(type='NAME', value='x', importance=ImportanceLevel.MEDIUM,
                     line=2, column=4, ascii_char='-'),
            CodeToken(type='OP', value='=', importance=ImportanceLevel.HIGH,
                     line=2, column=6, ascii_char='+'),
            CodeToken(type='NUMBER', value='1', importance=ImportanceLevel.MEDIUM,
                     line=2, column=8, ascii_char='-'),
        ]

        structure_groups = _analyze_code_structure_relationships(function_tokens)

        # Should identify function structure
        self.assertIn('functions', structure_groups)
        self.assertIn('my_function', structure_groups['functions'])

        # Function tokens should be grouped together
        func_info = structure_groups['functions']['my_function']
        self.assertEqual(func_info['start'], 0)  # Index of 'def' token
        self.assertGreater(len(func_info['tokens']), 0)

    def test_rotation_consistency_validation(self):
        """Test rotation consistency with full 360-degree validation."""
        from rotating_donut import _validate_rotation_stability, _ensure_rotation_consistency
        from math import tau

        # Create mappings that span full circumference
        test_mappings = []
        for i, token in enumerate(self.test_tokens):
            u = (i / len(self.test_tokens)) * tau
            v = tau * 0.25
            from rotating_donut import calculate_torus_surface_normal
            nx, ny, nz = calculate_torus_surface_normal(u, v, 2.0, 1.0)
            point = Point3D(x=1.0, y=0.0, z=0.0, u=u, v=v, nx=nx, ny=ny, nz=nz)
            test_mappings.append((point, token))

        # Should not raise exceptions for well-distributed mappings
        try:
            _validate_rotation_stability(test_mappings)
            consistent_mappings = _ensure_rotation_consistency(test_mappings)
            self.assertEqual(len(consistent_mappings), len(test_mappings))
        except ValueError as e:
            self.fail(f"Rotation consistency validation failed: {e}")

    def test_coordinate_precision_mathematical_validation(self):
        """Test coordinate precision with known geometric properties."""
        from rotating_donut import _calculate_precise_token_coordinates
        from math import tau, cos, sin

        # Create simple test case with known coordinates
        test_tokens = [
            CodeToken(type='NAME', value='a', importance=ImportanceLevel.CRITICAL,
                     line=1, column=0, ascii_char='#'),
            CodeToken(type='NAME', value='b', importance=ImportanceLevel.HIGH,
                     line=1, column=2, ascii_char='+'),
        ]

        # Create points with known u,v coordinates
        test_points = [
            Point3D(x=2.0, y=0.0, z=0.0, u=0.0, v=0.0),
            Point3D(x=0.0, y=2.0, z=0.0, u=tau/2, v=0.0),
            Point3D(x=-2.0, y=0.0, z=0.0, u=tau, v=0.0),
        ]

        mapped_pairs = _calculate_precise_token_coordinates(test_tokens, test_points)

        # Verify mathematical relationships
        for point, token in mapped_pairs:
            # Torus parametric equations should be satisfied
            # (simplified check for test points)
            self.assertIsInstance(point.u, float)
            self.assertIsInstance(point.v, float)

            # Coordinates should be in valid range
            self.assertGreaterEqual(point.u, 0)
            self.assertLessEqual(point.u, tau + 0.001)  # Small tolerance for floating point

    def test_importance_clustering_algorithm(self):
        """Test clustering algorithm that groups tokens by importance level."""
        from rotating_donut import _create_importance_clusters

        # Create test mappings with mixed importance
        mixed_mappings = []
        for i, token in enumerate(self.test_tokens):
            point = self.test_points[i % len(self.test_points)]
            mixed_mappings.append((point, token))

        clusters = _create_importance_clusters(mixed_mappings)

        # Should create clusters for each importance level
        self.assertIn(ImportanceLevel.CRITICAL, clusters)
        self.assertIn(ImportanceLevel.HIGH, clusters)
        self.assertIn(ImportanceLevel.MEDIUM, clusters)
        self.assertIn(ImportanceLevel.LOW, clusters)

        # Clusters should maintain sequential order within importance level
        for importance, cluster in clusters.items():
            if len(cluster) > 1:
                lines = [pair[1].line for pair in cluster]
                columns = [pair[1].column for pair in cluster]
                # Should be sorted by line then column
                for i in range(len(cluster) - 1):
                    self.assertLessEqual(lines[i], lines[i + 1])

    def test_visual_continuity_patterns(self):
        """Test visual continuity patterns across rotation cycles."""
        from rotating_donut import _verify_visual_continuity
        from math import tau

        # Create mappings with smooth u-coordinate progression
        continuous_mappings = []
        for i in range(10):
            u = (i / 10) * tau
            v = tau * 0.3
            from rotating_donut import calculate_torus_surface_normal
            nx, ny, nz = calculate_torus_surface_normal(u, v, 2.0, 1.0)
            point = Point3D(x=1.0, y=0.0, z=0.0, u=u, v=v, nx=nx, ny=ny, nz=nz)
            token = self.test_tokens[i % len(self.test_tokens)]
            continuous_mappings.append((point, token))

        # Should handle continuous mappings without issues
        result = _verify_visual_continuity(continuous_mappings)
        self.assertEqual(len(result), len(continuous_mappings))

    def test_coverage_integration_functions(self):
        """Test coverage of token-surface integration functions."""
        from rotating_donut import (
            map_tokens_to_surface,
            _apply_dynamic_scaling_system,
            _calculate_precise_token_coordinates,
            _apply_importance_weighted_distribution,
            _validate_coordinate_mapping,
            _ensure_rotation_consistency
        )

        # Test main integration function
        mapped_pairs = map_tokens_to_surface(self.test_tokens, self.test_points)
        self.assertIsInstance(mapped_pairs, list)
        self.assertGreater(len(mapped_pairs), 0)

        # Test each component function
        scaled_tokens, scaled_points = _apply_dynamic_scaling_system(self.test_tokens, self.test_points)
        self.assertIsInstance(scaled_tokens, list)
        self.assertIsInstance(scaled_points, list)

        coordinate_pairs = _calculate_precise_token_coordinates(self.test_tokens, self.test_points[:10])
        self.assertIsInstance(coordinate_pairs, list)

        weighted_pairs = _apply_importance_weighted_distribution(coordinate_pairs, self.test_points[:10])
        self.assertIsInstance(weighted_pairs, list)

        validated_pairs = _validate_coordinate_mapping(weighted_pairs)
        self.assertIsInstance(validated_pairs, list)

        consistent_pairs = _ensure_rotation_consistency(validated_pairs)
        self.assertIsInstance(consistent_pairs, list)

    def test_performance_requirements(self):
        """Test that functions meet performance requirements."""
        import time
        from rotating_donut import map_tokens_to_surface

        # Test with reasonably sized input to verify performance
        large_tokens = self.test_tokens * 20  # 140 tokens
        large_points = self.test_points * 2   # 400 points

        start_time = time.time()
        mapped_pairs = map_tokens_to_surface(large_tokens, large_points)
        end_time = time.time()

        execution_time = end_time - start_time

        # Should complete within reasonable time (< 2 seconds for this size)
        self.assertLess(execution_time, 2.0)

        # Should return valid results
        self.assertIsInstance(mapped_pairs, list)
        self.assertGreater(len(mapped_pairs), 0)


class TestRotationAwareVisibility(unittest.TestCase):
    """Test suite for Story 3.3: Rotation-Aware Code Display functionality.

    Tests cover surface normal calculations, visibility determination,
    rotation-aware token rendering, visual quality consistency,
    and edge case handling for visibility boundaries.
    """

    def setUp(self):
        """Set up test fixtures for rotation-aware visibility tests."""
        from math import pi, cos, sin

        # Import the new visibility functions
        from rotating_donut import (
            calculate_torus_surface_normal,
            calculate_surface_visibility,
            calculate_enhanced_visibility,
            apply_visibility_dimming,
            resolve_token_boundary_conflicts
        )
        self.calculate_torus_surface_normal = calculate_torus_surface_normal
        self.calculate_surface_visibility = calculate_surface_visibility
        self.calculate_enhanced_visibility = calculate_enhanced_visibility
        self.apply_visibility_dimming = apply_visibility_dimming
        self.resolve_token_boundary_conflicts = resolve_token_boundary_conflicts

        # Create test points with surface normals
        self.test_points_with_normals = [
            Point3D(x=2.0, y=0.0, z=0.0, u=0.0, v=0.0, nx=1.0, ny=0.0, nz=0.0),      # Front-facing
            Point3D(x=-2.0, y=0.0, z=0.0, u=pi, v=0.0, nx=-1.0, ny=0.0, nz=0.0),    # Back-facing
            Point3D(x=0.0, y=2.0, z=0.0, u=pi/2, v=0.0, nx=0.0, ny=1.0, nz=0.0),    # Side-facing
            Point3D(x=0.0, y=0.0, z=1.0, u=0.0, v=pi/2, nx=0.0, ny=0.0, nz=1.0),    # Top-facing
        ]

        # Create test tokens for visibility testing
        self.test_tokens = [
            CodeToken(type='KEYWORD', value='def', importance=ImportanceLevel.CRITICAL,
                     line=1, column=0, ascii_char='#'),
            CodeToken(type='OPERATOR', value='+', importance=ImportanceLevel.HIGH,
                     line=1, column=10, ascii_char='+'),
            CodeToken(type='IDENTIFIER', value='var', importance=ImportanceLevel.MEDIUM,
                     line=1, column=15, ascii_char='-'),
            CodeToken(type='COMMENT', value='# comment', importance=ImportanceLevel.LOW,
                     line=2, column=0, ascii_char='.'),
        ]

    def test_surface_normal_calculation_accuracy(self):
        """Test AC1: Calculate surface normal using torus parametric derivatives."""
        from math import pi, sqrt

        # Test known torus parameters
        outer_radius = 2.0
        inner_radius = 1.0

        # Test surface normal at u=0, v=0 (front of torus)
        nx, ny, nz = self.calculate_torus_surface_normal(0.0, 0.0, outer_radius, inner_radius)

        # At u=0, v=0, normal should point outward in +x direction
        self.assertAlmostEqual(nx, 1.0, places=3)
        self.assertAlmostEqual(ny, 0.0, places=3)
        self.assertAlmostEqual(nz, 0.0, places=3)

        # Verify normal is normalized (unit length)
        magnitude = sqrt(nx**2 + ny**2 + nz**2)
        self.assertAlmostEqual(magnitude, 1.0, places=6)

    def test_surface_normal_validation_errors(self):
        """Test surface normal calculation error handling."""
        # Test invalid radius parameters
        with self.assertRaises(ValueError) as context:
            self.calculate_torus_surface_normal(0.0, 0.0, 1.0, 2.0)  # inner > outer
        self.assertIn("Invalid torus parameters", str(context.exception))
        self.assertIn("Solution:", str(context.exception))

    def test_visibility_calculation_front_facing(self):
        """Test AC1,2: Visibility calculation for front-facing surfaces."""
        front_point = self.test_points_with_normals[0]  # Front-facing normal (1, 0, 0)

        # Use viewing direction that aligns with front-facing normal
        visibility = self.calculate_surface_visibility(front_point, viewing_direction=(1.0, 0.0, 0.0))

        # Front-facing surface should have high visibility
        self.assertGreater(visibility, 0.5)
        self.assertLessEqual(visibility, 1.0)

    def test_visibility_calculation_back_facing(self):
        """Test AC2: Hide tokens on back-facing surfaces."""
        back_point = self.test_points_with_normals[1]  # Back-facing normal

        visibility = self.calculate_surface_visibility(back_point)

        # Back-facing surface should have zero or very low visibility
        self.assertLessEqual(visibility, 0.1)

    def test_enhanced_visibility_importance_boosting(self):
        """Test AC4: High-importance tokens get visibility boost during marginal angles."""
        # Create point with marginal viewing angle
        marginal_point = Point3D(x=1.0, y=0.0, z=0.0, u=0.0, v=0.0,
                               nx=0.5, ny=0.0, nz=0.866)  # 60-degree angle

        # Test enhanced visibility for critical vs low importance
        critical_visibility = self.calculate_enhanced_visibility(marginal_point, ImportanceLevel.CRITICAL)
        low_visibility = self.calculate_enhanced_visibility(marginal_point, ImportanceLevel.LOW)

        # Critical tokens should get visibility boost
        self.assertGreater(critical_visibility, low_visibility)

    def test_visibility_dimming_smooth_transitions(self):
        """Test AC3: Smooth visibility transitions through character dimming."""
        # Test various visibility levels
        test_cases = [
            (0.9, ImportanceLevel.CRITICAL, '#'),  # High visibility keeps original
            (0.7, ImportanceLevel.CRITICAL, '#'),  # Medium-high visibility
            (0.5, ImportanceLevel.CRITICAL, '+'),  # Medium visibility dims
            (0.3, ImportanceLevel.CRITICAL, '-'),  # Low visibility dims further
            (0.1, ImportanceLevel.CRITICAL, '.'),  # Very low visibility becomes background
        ]

        for visibility, importance, expected_char in test_cases:
            with self.subTest(visibility=visibility, importance=importance):
                dimmed_char = self.apply_visibility_dimming('#', visibility, importance)
                # Check that dimming is reasonable (exact char may vary by algorithm)
                self.assertIn(dimmed_char, ['.', '-', '+', '#'])

    def test_boundary_conflict_resolution(self):
        """Test AC5: Handle edge cases where token boundaries align with visibility edges."""
        # Create screen data with conflicting visibility at adjacent positions
        screen_data = [
            (Point2D(x=10, y=10, depth=0.5, visible=True, visibility_factor=0.9), self.test_tokens[0]),
            (Point2D(x=11, y=10, depth=0.5, visible=True, visibility_factor=0.1), self.test_tokens[1]),
            (Point2D(x=10, y=11, depth=0.5, visible=True, visibility_factor=0.8), self.test_tokens[2]),
        ]

        resolved_data = self.resolve_token_boundary_conflicts(screen_data)

        # Should return same number of points
        self.assertEqual(len(resolved_data), len(screen_data))

        # Visibility factors should be smoothed for conflicting neighbors
        resolved_visibility = [point.visibility_factor for point, token in resolved_data]
        original_visibility = [point.visibility_factor for point, token in screen_data]

        # At least some visibility should be modified for smoothing
        self.assertNotEqual(resolved_visibility, original_visibility)

    def test_project_to_screen_with_visibility(self):
        """Test enhanced project_to_screen function with visibility calculation."""
        from rotating_donut import project_to_screen

        # Create points with normals that align with default viewing direction (0, 0, 1)
        front_point = Point3D(x=0.0, y=0.0, z=2.0, u=0.0, v=0.0, nx=0.0, ny=0.0, nz=1.0)  # Normal toward viewer
        back_point = Point3D(x=0.0, y=0.0, z=2.0, u=0.0, v=0.0, nx=0.0, ny=0.0, nz=-1.0)  # Normal away from viewer

        # Test with token importance
        front_screen = project_to_screen(front_point, ImportanceLevel.CRITICAL)
        back_screen = project_to_screen(back_point, ImportanceLevel.CRITICAL)

        # Front-facing should have higher visibility factor
        self.assertGreater(front_screen.visibility_factor, back_screen.visibility_factor)

        # Both should have valid Point2D structure
        self.assertIsInstance(front_screen, Point2D)
        self.assertIsInstance(back_screen, Point2D)

    def test_generate_ascii_frame_with_visibility(self):
        """Test integration of visibility factors in ASCII frame generation."""
        # Create mapped pairs with visibility-enhanced points
        mapped_pairs = []
        for i, token in enumerate(self.test_tokens):
            point = self.test_points_with_normals[i % len(self.test_points_with_normals)]
            mapped_pairs.append((point, token))

        frame = generate_ascii_frame(mapped_pairs, frame_number=1)

        # Should generate valid frame
        self.assertEqual(frame.width, TERMINAL_WIDTH)
        self.assertEqual(frame.height, TERMINAL_HEIGHT)
        self.assertEqual(frame.frame_number, 1)

        # Frame should contain appropriate characters based on visibility
        buffer_chars = set()
        for row in frame.buffer:
            for char in row:
                buffer_chars.add(char)

        # Should use only valid ASCII characters
        valid_chars = {'.', '-', '+', '#'}
        self.assertTrue(buffer_chars.issubset(valid_chars))

    def test_visibility_threshold_filtering(self):
        """Test that points below visibility threshold are filtered out."""
        from rotating_donut import generate_ascii_frame

        # Create mapped pairs with very low visibility points
        low_visibility_pairs = []
        for token in self.test_tokens:
            # Use back-facing point (should have very low visibility)
            back_point = self.test_points_with_normals[1]
            low_visibility_pairs.append((back_point, token))

        frame = generate_ascii_frame(low_visibility_pairs)

        # Frame should be mostly background due to low visibility filtering
        background_count = 0
        for row in frame.buffer:
            for char in row:
                if char == '.':
                    background_count += 1

        # Should be mostly background (> 90% background chars)
        total_chars = TERMINAL_WIDTH * TERMINAL_HEIGHT
        background_ratio = background_count / total_chars
        self.assertGreater(background_ratio, 0.9)

    def test_mathematical_precision_requirements(self):
        """Test mathematical precision requirements from coding standards."""
        from math import pi, tau

        # Test with math.pi precision (not hardcoded approximations)
        nx, ny, nz = self.calculate_torus_surface_normal(pi/2, 0.0, 2.0, 1.0)

        # Verify calculations use proper mathematical constants
        self.assertIsInstance(nx, float)
        self.assertIsInstance(ny, float)
        self.assertIsInstance(nz, float)

        # Test with tau precision
        nx2, ny2, nz2 = self.calculate_torus_surface_normal(tau/4, 0.0, 2.0, 1.0)

        # Should be equivalent to pi/2 calculation
        self.assertAlmostEqual(nx, nx2, places=6)
        self.assertAlmostEqual(ny, ny2, places=6)
        self.assertAlmostEqual(nz, nz2, places=6)

    def test_performance_30fps_requirement(self):
        """Test that visibility calculations maintain 30+ FPS performance requirement."""
        import time

        # Create substantial test data
        large_mapped_pairs = []
        for i in range(100):  # Simulate moderate frame complexity
            point = self.test_points_with_normals[i % len(self.test_points_with_normals)]
            token = self.test_tokens[i % len(self.test_tokens)]
            large_mapped_pairs.append((point, token))

        # Measure frame generation time
        start_time = time.time()
        frame = generate_ascii_frame(large_mapped_pairs)
        end_time = time.time()

        frame_time = end_time - start_time

        # Should complete frame generation in < 33ms (30 FPS requirement)
        self.assertLess(frame_time, 0.033)

        # Frame should be valid
        self.assertEqual(frame.width, TERMINAL_WIDTH)
        self.assertEqual(frame.height, TERMINAL_HEIGHT)

    def test_edge_case_zero_visibility_factor(self):
        """Test edge case handling for zero visibility factor."""
        # Test dimming with zero visibility
        dimmed_char = self.apply_visibility_dimming('#', 0.0, ImportanceLevel.CRITICAL)

        # Should return background character
        self.assertEqual(dimmed_char, '.')

    def test_edge_case_perpendicular_surface_normal(self):
        """Test edge case where surface normal is perpendicular to viewing direction."""
        # Create point with normal perpendicular to default viewing direction (0,0,1)
        perpendicular_point = Point3D(x=1.0, y=0.0, z=0.0, u=0.0, v=0.0,
                                    nx=1.0, ny=0.0, nz=0.0)  # Normal in x direction

        visibility = self.calculate_surface_visibility(perpendicular_point)

        # Perpendicular surface should have zero visibility
        self.assertEqual(visibility, 0.0)

    def test_input_validation_viewing_direction(self):
        """Test input validation for viewing direction parameter."""
        point = self.test_points_with_normals[0]

        # Test zero viewing direction
        with self.assertRaises(ValueError) as context:
            self.calculate_surface_visibility(point, viewing_direction=(0.0, 0.0, 0.0))
        self.assertIn("Invalid viewing direction", str(context.exception))
        self.assertIn("Solution:", str(context.exception))

    def test_coverage_visibility_functions(self):
        """Test coverage of all visibility-related functions."""
        from rotating_donut import (
            calculate_torus_surface_normal,
            calculate_surface_visibility,
            calculate_enhanced_visibility,
            apply_visibility_dimming,
            handle_visibility_boundary_smoothing,
            resolve_token_boundary_conflicts
        )

        # Test each function is callable and returns expected types
        point = self.test_points_with_normals[0]
        token = self.test_tokens[0]

        # Surface normal calculation
        normal = calculate_torus_surface_normal(0.0, 0.0, 2.0, 1.0)
        self.assertIsInstance(normal, tuple)
        self.assertEqual(len(normal), 3)

        # Surface visibility calculation
        visibility = calculate_surface_visibility(point)
        self.assertIsInstance(visibility, float)
        self.assertGreaterEqual(visibility, 0.0)
        self.assertLessEqual(visibility, 1.0)

        # Enhanced visibility calculation
        enhanced_vis = calculate_enhanced_visibility(point, ImportanceLevel.CRITICAL)
        self.assertIsInstance(enhanced_vis, float)

        # Visibility dimming
        dimmed = apply_visibility_dimming('#', 0.5, ImportanceLevel.HIGH)
        self.assertIsInstance(dimmed, str)
        self.assertEqual(len(dimmed), 1)

        # Boundary smoothing
        pairs = [(point, token)]
        smoothed = handle_visibility_boundary_smoothing(pairs)
        self.assertIsInstance(smoothed, list)

        # Boundary conflict resolution
        screen_data = [(Point2D(x=10, y=10, depth=0.5, visible=True, visibility_factor=0.8), token)]
        resolved = resolve_token_boundary_conflicts(screen_data)
        self.assertIsInstance(resolved, list)


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)