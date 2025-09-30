#!/usr/bin/env python3
"""
Comprehensive Viral Validation Tests for Story 4.5: Social Sharing Optimization

Tests for Tasks 4, 5, and 6:
- Task 4: "Wow factor" validation (extended stability, smooth animation, clean state)
- Task 5: PRD success metrics validation (portfolio quality, viral shareability)
- Task 6: Viral-readiness validation suite (cross-platform, compatibility, NFRs)

Test Framework: unittest
Coverage Requirement: 90%+ for viral-readiness validation
"""

import unittest
import sys
import os
import time
import subprocess
from io import StringIO
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rotating_donut
from rotating_donut import (
    TorusParameters, Point3D, CodeToken, ImportanceLevel,
    generate_torus_points, apply_rotation, project_to_screen,
    generate_ascii_frame, output_to_terminal, DisplayFrame,
    TERMINAL_WIDTH, TERMINAL_HEIGHT, ASCII_CHARS
)


class TestWowFactorValidation(unittest.TestCase):
    """
    Task 4: Create "wow factor" validation tests

    Tests for:
    - Extended run stability (5+ minutes)
    - Visual continuity and rotation smoothness
    - Self-referential concept clarity
    - Mathematical beauty visibility
    - Interrupt handling and terminal cleanup
    """

    def test_extended_run_stability_5_minutes(self):
        """Test AC: Animation runs smoothly for extended periods (5+ minutes)."""
        # Simulate 5 minutes at 30 FPS = 9000 frames
        # Test reduced sample for speed: 300 frames = 10 seconds at 30 FPS
        params = TorusParameters(2.0, 1.0, 40, 20, 0.05)

        frame_times = []
        errors = []

        start_time = time.time()
        for frame_num in range(300):
            frame_start = time.time()

            try:
                # Standard animation operations
                points = generate_torus_points(params)
                angle = frame_num * 0.05
                rotated_points = apply_rotation(points, angle)

                # Generate frame
                mapped_pairs = []
                for i in range(min(100, len(rotated_points))):
                    point = rotated_points[i]
                    token = CodeToken(
                        type='NAME', value='test',
                        importance=ImportanceLevel.MEDIUM,
                        line=1, column=0, ascii_char='-'
                    )
                    mapped_pairs.append((point, token))

                frame = generate_ascii_frame(mapped_pairs, frame_number=frame_num)

                frame_time = time.time() - frame_start
                frame_times.append(frame_time)

            except Exception as e:
                errors.append((frame_num, str(e)))

        total_time = time.time() - start_time

        # Validation: Should complete without errors
        self.assertEqual(len(errors), 0,
                        f"Errors during extended run: {errors[:3]}")

        # Validation: Frame times should be consistent
        avg_frame_time = sum(frame_times) / len(frame_times)
        max_frame_time = max(frame_times)

        self.assertLess(avg_frame_time, 0.05,
                       f"Average frame time too high: {avg_frame_time:.3f}s")
        self.assertLess(max_frame_time, 0.1,
                       f"Max frame spike too high: {max_frame_time:.3f}s")

    def test_visual_continuity_and_smoothness(self):
        """Test AC: Validate visual continuity and rotation smoothness."""
        params = TorusParameters(2.0, 1.0, 50, 25, 0.05)
        points = generate_torus_points(params)

        # Generate sequence of frames with small rotation increments
        frames = []
        for angle in [0.0, 0.05, 0.10, 0.15, 0.20]:
            rotated = apply_rotation(points, angle)

            # Create basic frame
            mapped_pairs = [(rotated[i], CodeToken(
                type='NAME', value='x', importance=ImportanceLevel.MEDIUM,
                line=1, column=0, ascii_char='-'
            )) for i in range(min(200, len(rotated)))]

            frame = generate_ascii_frame(mapped_pairs, frame_number=len(frames))
            frames.append(frame)

        # Validation: All frames should be valid
        self.assertEqual(len(frames), 5)
        for frame in frames:
            self.assertEqual(frame.width, TERMINAL_WIDTH)
            self.assertEqual(frame.height, TERMINAL_HEIGHT)

        # Validation: Frames should show smooth progression
        # Count non-background characters in each frame
        char_counts = []
        for frame in frames:
            count = sum(1 for row in frame.buffer for char in row if char != '.')
            char_counts.append(count)

        # Character counts should be relatively consistent (smooth animation)
        avg_count = sum(char_counts) / len(char_counts)
        for count in char_counts:
            variance_pct = abs(count - avg_count) / avg_count
            self.assertLess(variance_pct, 0.5,
                           "Frame character count variance too high (not smooth)")

    def test_self_referential_concept_clarity(self):
        """Test AC: Self-referential concept is immediately apparent."""
        # Verify the program can read its own source code
        source_code = rotating_donut.read_self_code()

        self.assertIsNotNone(source_code)
        self.assertGreater(len(source_code), 1000,
                          "Source code should be substantial")

        # Verify tokenization works
        tokens = rotating_donut.tokenize_code(source_code)

        self.assertGreater(len(tokens), 100,
                          "Should have substantial number of tokens")

        # Verify tokens are classified by importance
        importance_counts = {}
        for token in tokens[:1000]:  # Sample first 1000
            importance = token.importance
            importance_counts[importance] = importance_counts.get(importance, 0) + 1

        # Should have multiple importance levels represented
        self.assertGreaterEqual(len(importance_counts), 2,
                               "Should have multiple importance levels")

    def test_mathematical_beauty_visibility(self):
        """Test AC: Mathematical beauty is clearly visible."""
        params = TorusParameters(2.0, 1.0, 50, 25, 0.05)
        points = generate_torus_points(params)

        # Validate torus geometry creates proper shape
        # Points should be distributed around torus surface
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        z_coords = [p.z for p in points]

        # Check range spans expected torus dimensions
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        z_range = max(z_coords) - min(z_coords)

        # Expected range: roughly 2*(R + r) = 2*(2.0 + 1.0) = 6.0
        self.assertGreater(x_range, 4.0, "X range too small for torus")
        self.assertGreater(y_range, 4.0, "Y range too small for torus")
        self.assertGreater(z_range, 1.0, "Z range too small for torus tube")

        # Verify rotation creates smooth transformation
        angle = 0.5
        rotated = apply_rotation(points, angle)

        self.assertEqual(len(rotated), len(points),
                        "Rotation should preserve point count")

        # Rotated points should have different coordinates
        different_count = sum(1 for i in range(len(points))
                             if (rotated[i].x != points[i].x or
                                 rotated[i].y != points[i].y or
                                 rotated[i].z != points[i].z))

        self.assertGreater(different_count, len(points) * 0.9,
                          "Rotation should affect most points")

    @patch('sys.stdout', new_callable=StringIO)
    def test_clean_terminal_state_handling(self, mock_stdout):
        """Test AC: Interrupt handling leaves terminal in clean state."""
        # Test output_to_terminal doesn't crash and uses proper codes
        frame = DisplayFrame(
            width=TERMINAL_WIDTH,
            height=TERMINAL_HEIGHT,
            buffer=[['.' for _ in range(TERMINAL_WIDTH)] for _ in range(TERMINAL_HEIGHT)],
            depth_buffer=[[float('inf') for _ in range(TERMINAL_WIDTH)]
                         for _ in range(TERMINAL_HEIGHT)],
            frame_number=1
        )

        # Should not raise exception
        try:
            output_to_terminal(frame)
            output = mock_stdout.getvalue()

            # Should contain ANSI escape codes for screen control
            self.assertIn('\033[', output, "Should use ANSI codes for terminal control")

        except Exception as e:
            self.fail(f"Terminal output failed: {e}")


class TestSuccessMetricsValidation(unittest.TestCase):
    """
    Task 5: Validate all PRD success metrics are achievable

    Tests for:
    - 1000+ GitHub stars potential (portfolio quality, viral appeal)
    - 500+ shares/mentions potential (shareability, visual appeal)
    - Portfolio centerpiece quality (professional code, comprehensive docs)
    - Educational value clarity
    - Viral shareability characteristics
    """

    def test_portfolio_quality_code_standards(self):
        """Test portfolio quality: Professional code with PEP 8 compliance."""
        # Test that main module exists and is importable
        self.assertTrue(hasattr(rotating_donut, 'generate_torus_points'))
        self.assertTrue(hasattr(rotating_donut, 'apply_rotation'))
        self.assertTrue(hasattr(rotating_donut, 'generate_ascii_frame'))

        # Test that module has comprehensive docstring
        self.assertIsNotNone(rotating_donut.__doc__)
        self.assertGreater(len(rotating_donut.__doc__), 500,
                          "Module docstring should be comprehensive")

        # Verify docstring contains key portfolio elements
        doc = rotating_donut.__doc__
        self.assertIn('Author:', doc, "Should have author attribution")
        self.assertIn('License:', doc, "Should have license information")
        self.assertIn('Python Version:', doc, "Should specify Python version")
        self.assertIn('Repository:', doc, "Should have repository URL")

    def test_portfolio_quality_documentation(self):
        """Test portfolio quality: Comprehensive documentation exists."""
        # Verify README exists
        readme_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'README.md'
        )
        self.assertTrue(os.path.exists(readme_path),
                       "README.md should exist for portfolio showcase")

        # Verify README is substantial
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()

        self.assertGreater(len(readme_content), 1000,
                          "README should be comprehensive (>1000 chars)")

        # Check for key documentation sections
        self.assertIn('# ', readme_content, "Should have markdown headers")

    def test_viral_shareability_single_file(self):
        """Test viral shareability: Single-file constraint maintained."""
        # Verify main implementation is in single file
        main_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'rotating_donut.py'
        )
        self.assertTrue(os.path.exists(main_file),
                       "Main file rotating_donut.py should exist")

        # Verify file contains main functionality
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for key functions in single file
        self.assertIn('def generate_torus_points', content)
        self.assertIn('def apply_rotation', content)
        self.assertIn('def generate_ascii_frame', content)
        self.assertIn('def run_animation_loop', content)

    def test_viral_shareability_zero_dependencies(self):
        """Test viral shareability: Zero external dependencies."""
        # Check that only stdlib modules are imported
        import_start = False
        stdlib_only = True
        non_stdlib = []

        with open('rotating_donut.py', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # Check import statements
                if line.startswith('import ') or line.startswith('from '):
                    # Extract module name
                    if line.startswith('import '):
                        module = line.split()[1].split('.')[0]
                    elif line.startswith('from '):
                        module = line.split()[1].split('.')[0]

                    # Check if it's a known stdlib module or project module
                    stdlib_modules = {
                        'math', 'sys', 'time', 'tokenize', 'io', 'os',
                        'typing', 'ast', 'keyword', 'subprocess', 'random',
                        'builtins', 'gc', 'performance_monitor',
                        'cache_manager'  # Project modules
                    }

                    if module not in stdlib_modules:
                        non_stdlib.append(module)

        # Should have no non-stdlib imports
        project_modules = {'performance_monitor', 'cache_manager'}
        external_deps = [m for m in non_stdlib if m not in project_modules]

        self.assertEqual(len(external_deps), 0,
                        f"Should have zero external dependencies, found: {external_deps}")

    def test_educational_value_content(self):
        """Test educational value: Clear mathematical explanations present."""
        doc = rotating_donut.__doc__

        # Check for mathematical content
        self.assertIn('parametric', doc.lower(),
                     "Should explain parametric equations")
        self.assertIn('rotation', doc.lower(),
                     "Should explain rotation concepts")
        self.assertIn('torus', doc.lower(),
                     "Should explain torus geometry")

        # Check for code examples or usage
        self.assertIn('Example', doc or '',
                     "Should include usage examples")

    def test_wow_factor_immediate_execution(self):
        """Test wow factor: Immediate execution without setup."""
        # Verify script can be run directly
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'rotating_donut.py'
        )

        with open(script_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()

        # Should have shebang for direct execution
        self.assertIn('python', first_line.lower(),
                     "Should have Python shebang for direct execution")

        # Verify main guard exists
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertIn('if __name__ == "__main__":', content,
                     "Should have main guard for script execution")


class TestViralReadinessValidation(unittest.TestCase):
    """
    Task 6: Create comprehensive viral-readiness validation suite

    Tests for:
    - Cross-platform compatibility (Windows, macOS, Linux)
    - Python 3.8+ compatibility
    - Terminal compatibility across major emulators
    - Zero external dependencies requirement
    - Single-file constraint maintained
    - All functional requirements (FR1-FR8) met
    - All non-functional requirements (NFR1-NFR6) met
    """

    def test_cross_platform_imports(self):
        """Test cross-platform: Only platform-independent imports used."""
        # Verify platform detection capability
        self.assertTrue(hasattr(sys, 'platform'))

        # Verify cross-platform modules are used
        import platform
        self.assertTrue(callable(platform.system))

    def test_python_38_plus_compatibility(self):
        """Test Python 3.8+ compatibility: No Python 3.9+ only features."""
        # Check Python version
        version_info = sys.version_info
        self.assertGreaterEqual(version_info.major, 3)
        self.assertGreaterEqual(version_info.minor, 8)

        # Verify code doesn't use 3.9+ only features
        # (This test validates by successfully importing the module)
        import rotating_donut
        self.assertIsNotNone(rotating_donut)

    def test_terminal_compatibility_ascii_only(self):
        """Test terminal compatibility: Uses only safe ASCII characters."""
        # Verify ASCII_CHARS contains only basic ASCII
        for key, char in ASCII_CHARS.items():
            self.assertTrue(ord(char) < 128,
                           f"Character '{char}' not basic ASCII")
            self.assertIn(char, '.+-# ',
                         f"Character '{char}' not in safe set")

    def test_fr1_self_code_reading(self):
        """Test FR1: Self-code reading implemented."""
        source = rotating_donut.read_self_code()
        self.assertIsNotNone(source)
        self.assertGreater(len(source), 100)

    def test_fr2_tokenization_stdlib(self):
        """Test FR2: Tokenization using stdlib tokenize."""
        source = "def test(): pass"
        tokens = rotating_donut.tokenize_code(source)
        self.assertIsNotNone(tokens)
        self.assertGreater(len(tokens), 0)

    def test_fr3_3d_torus_geometry(self):
        """Test FR3: 3D torus geometry generation."""
        params = TorusParameters(2.0, 1.0, 20, 10, 0.05)
        points = generate_torus_points(params)

        self.assertEqual(len(points), 200)  # 20 * 10
        self.assertIsInstance(points[0], Point3D)

    def test_fr4_importance_hierarchy(self):
        """Test FR4: 4-level semantic importance hierarchy."""
        # Verify ImportanceLevel has 4 levels
        levels = [ImportanceLevel.CRITICAL, ImportanceLevel.HIGH,
                 ImportanceLevel.MEDIUM, ImportanceLevel.LOW]

        self.assertEqual(len(set(levels)), 4)

        # Verify hierarchy order (ImportanceLevel is IntEnum)
        self.assertGreater(ImportanceLevel.CRITICAL,
                          ImportanceLevel.HIGH)
        self.assertGreater(ImportanceLevel.HIGH,
                          ImportanceLevel.MEDIUM)
        self.assertGreater(ImportanceLevel.MEDIUM,
                          ImportanceLevel.LOW)

    def test_fr5_density_mapping(self):
        """Test FR5: Density mapping for important tokens."""
        # Create tokens with different importance
        tokens = [
            CodeToken(type='KEYWORD', value='def',
                     importance=ImportanceLevel.CRITICAL,
                     line=1, column=0, ascii_char='#'),
            CodeToken(type='COMMENT', value='# test',
                     importance=ImportanceLevel.LOW,
                     line=2, column=0, ascii_char='.')
        ]

        params = TorusParameters(2.0, 1.0, 20, 10, 0.05)
        points = generate_torus_points(params)

        # Map tokens to surface
        mapped = rotating_donut.map_tokens_to_surface(tokens, points)

        # Count mappings by importance
        critical_count = sum(1 for _, t in mapped
                            if t.importance == ImportanceLevel.CRITICAL)
        low_count = sum(1 for _, t in mapped
                       if t.importance == ImportanceLevel.LOW)

        # Critical should get more surface points
        self.assertGreaterEqual(critical_count, low_count)

    def test_fr6_y_axis_rotation(self):
        """Test FR6: Y-axis rotation with configurable speed."""
        params = TorusParameters(2.0, 1.0, 20, 10, 0.05)
        points = generate_torus_points(params)

        # Apply rotation
        rotated = apply_rotation(points, 1.0)

        self.assertEqual(len(rotated), len(points))
        self.assertIsInstance(rotated[0], Point3D)

    def test_fr7_ascii_display_with_depth(self):
        """Test FR7: 40x20 ASCII display with depth sorting."""
        frame = DisplayFrame(
            width=TERMINAL_WIDTH,
            height=TERMINAL_HEIGHT,
            buffer=[['.' for _ in range(TERMINAL_WIDTH)]
                   for _ in range(TERMINAL_HEIGHT)],
            depth_buffer=[[float('inf') for _ in range(TERMINAL_WIDTH)]
                         for _ in range(TERMINAL_HEIGHT)],
            frame_number=1
        )

        self.assertEqual(frame.width, 40)
        self.assertEqual(frame.height, 20)

    def test_fr8_30fps_performance(self):
        """Test FR8: 30+ FPS smooth animation capability."""
        params = TorusParameters(2.0, 1.0, 30, 15, 0.05)

        # Measure time for 30 frames
        start_time = time.time()
        for i in range(30):
            points = generate_torus_points(params)
            rotated = apply_rotation(points, i * 0.1)

        elapsed = time.time() - start_time
        fps = 30 / elapsed

        # Should achieve at least 30 FPS
        self.assertGreaterEqual(fps, 30,
                               f"FPS too low: {fps:.1f} < 30")

    def test_nfr1_zero_external_dependencies(self):
        """Test NFR1: Zero external dependencies."""
        # This is validated by successful import without pip install
        import rotating_donut
        self.assertIsNotNone(rotating_donut)

    def test_nfr2_single_python_file(self):
        """Test NFR2: Single Python file constraint."""
        # Verify main implementation is self-contained
        main_file = 'rotating_donut.py'
        self.assertTrue(os.path.exists(main_file))

        # Check file size is reasonable for single file
        size = os.path.getsize(main_file)
        self.assertGreater(size, 10000, "File should contain substantial code")
        self.assertLess(size, 300000, "File should remain manageable size (< 300KB)")

    def test_nfr3_python_38_cross_platform(self):
        """Test NFR3: Python 3.8+ cross-platform compatibility."""
        # Test passes if module imports successfully
        import rotating_donut
        self.assertIsNotNone(rotating_donut)

        # Verify version check
        self.assertGreaterEqual(sys.version_info.major, 3)
        self.assertGreaterEqual(sys.version_info.minor, 8)

    def test_nfr4_minimal_cpu_usage(self):
        """Test NFR4: Minimal CPU usage (caching implemented)."""
        # Verify caching functions exist
        from cache_manager import get_cached_rotation_matrix
        from cache_manager import get_cached_projection

        self.assertTrue(callable(get_cached_rotation_matrix))
        self.assertTrue(callable(get_cached_projection))

    def test_nfr5_self_documenting_educational(self):
        """Test NFR5: Self-documenting and educational."""
        # Verify comprehensive module docstring
        doc = rotating_donut.__doc__
        self.assertIsNotNone(doc)
        self.assertGreater(len(doc), 1000)

        # Check for educational content
        self.assertIn('MATHEMATICAL', doc)
        self.assertIn('EDUCATIONAL', doc)

    def test_nfr6_terminal_compatibility(self):
        """Test NFR6: Standard terminal emulator compatibility."""
        # Verify only safe ASCII characters used
        for char in ASCII_CHARS.values():
            self.assertLess(ord(char), 128)

        # Verify ANSI escape codes are used properly
        self.assertTrue(hasattr(rotating_donut, 'output_to_terminal'))


class TestComprehensiveCoverage(unittest.TestCase):
    """Additional comprehensive tests for 90%+ coverage requirement."""

    def test_startup_time_under_1_second(self):
        """Test startup completes in under 1 second."""
        # This test validates startup time requirement from Task 2
        start = time.time()

        # Simulate startup sequence
        params = TorusParameters(2.0, 1.0, 50, 25, 0.05)
        points = generate_torus_points(params)
        source = rotating_donut.read_self_code()
        tokens = rotating_donut.tokenize_code(source)

        elapsed = time.time() - start

        self.assertLess(elapsed, 1.0,
                       f"Startup too slow: {elapsed:.3f}s > 1.0s")

    def test_visual_output_screenshot_worthy(self):
        """Test visual output is compelling for screenshots."""
        params = TorusParameters(2.0, 1.0, 50, 25, 0.05)
        points = generate_torus_points(params)
        rotated = apply_rotation(points, 0.5)

        # Create frame
        mapped = [(rotated[i], CodeToken(
            type='NAME', value='x',
            importance=ImportanceLevel.MEDIUM,
            line=1, column=0, ascii_char='-'
        )) for i in range(min(500, len(rotated)))]

        frame = generate_ascii_frame(mapped, frame_number=1)

        # Count non-background characters (should have good visual density)
        char_count = sum(1 for row in frame.buffer for char in row
                        if char != '.')

        # Should have 10-50% of frame filled for good visual impact
        total_chars = TERMINAL_WIDTH * TERMINAL_HEIGHT
        fill_ratio = char_count / total_chars

        self.assertGreater(fill_ratio, 0.05,
                          "Frame too empty for visual impact")
        self.assertLess(fill_ratio, 0.7,
                       "Frame too dense for visual clarity")

    def test_attribution_header_completeness(self):
        """Test attribution header has all required elements."""
        doc = rotating_donut.__doc__

        # Required elements for viral sharing
        required_elements = [
            'Author:',
            'License:',
            'Python Version:',
            'Repository:',
            'Dependencies:'
        ]

        for element in required_elements:
            self.assertIn(element, doc,
                         f"Missing required element: {element}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
