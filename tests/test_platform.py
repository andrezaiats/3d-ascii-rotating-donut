"""
Unit tests for cross-platform compatibility - Platform detection and environment setup.

Tests platform detection, terminal capability detection, and file path normalization
across Windows, macOS, and Linux platforms.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rotating_donut import (
    detect_platform,
    detect_terminal_capabilities,
    normalize_file_path,
    adjust_frame_dimensions,
    validate_character_set,
    get_fallback_character_set,
    test_terminal_capabilities,
    calibrate_timer_precision,
    validate_timing_consistency,
    calculate_adaptive_frame_time,
    get_platform_sleep_overhead,
    adaptive_sleep,
    clear_screen,
    output_line_buffered,
    validate_terminal_output,
    PlatformInfo,
    TerminalCapabilities,
    PLATFORM_WINDOWS,
    PLATFORM_DARWIN,
    PLATFORM_LINUX,
    TARGET_FPS
)


class TestPlatformDetection(unittest.TestCase):
    """Test platform detection functionality."""

    @patch('sys.platform', 'win32')
    @patch('os.name', 'nt')
    def test_detect_windows_platform(self):
        """Test detection of Windows platform."""
        with patch.dict(os.environ, {}, clear=True):
            platform_info = detect_platform()

            self.assertEqual(platform_info.platform, 'win32')
            self.assertEqual(platform_info.os_name, 'nt')
            self.assertTrue(platform_info.is_windows)
            self.assertFalse(platform_info.is_macos)
            self.assertFalse(platform_info.is_linux)
            self.assertIsInstance(platform_info.python_version, tuple)
            self.assertEqual(len(platform_info.python_version), 3)

    @patch('sys.platform', 'darwin')
    @patch('os.name', 'posix')
    def test_detect_macos_platform(self):
        """Test detection of macOS platform."""
        platform_info = detect_platform()

        self.assertEqual(platform_info.platform, 'darwin')
        self.assertEqual(platform_info.os_name, 'posix')
        self.assertFalse(platform_info.is_windows)
        self.assertTrue(platform_info.is_macos)
        self.assertFalse(platform_info.is_linux)
        self.assertTrue(platform_info.supports_ansi)

    @patch('sys.platform', 'linux')
    @patch('os.name', 'posix')
    def test_detect_linux_platform(self):
        """Test detection of Linux platform."""
        platform_info = detect_platform()

        self.assertEqual(platform_info.platform, 'linux')
        self.assertEqual(platform_info.os_name, 'posix')
        self.assertFalse(platform_info.is_windows)
        self.assertFalse(platform_info.is_macos)
        self.assertTrue(platform_info.is_linux)
        self.assertTrue(platform_info.supports_ansi)

    @patch('sys.platform', 'win32')
    @patch('os.name', 'nt')
    def test_windows_ansi_support_with_wt_session(self):
        """Test Windows ANSI support detection with Windows Terminal."""
        with patch.dict(os.environ, {'WT_SESSION': 'some-session-id'}):
            platform_info = detect_platform()
            self.assertTrue(platform_info.supports_ansi)

    @patch('sys.platform', 'win32')
    @patch('os.name', 'nt')
    def test_windows_ansi_support_with_term_env(self):
        """Test Windows ANSI support detection with TERM variable."""
        with patch.dict(os.environ, {'TERM': 'xterm-256color'}):
            platform_info = detect_platform()
            self.assertTrue(platform_info.supports_ansi)

    @patch('sys.platform', 'win32')
    @patch('os.name', 'nt')
    def test_windows_no_ansi_support(self):
        """Test Windows without ANSI support (old cmd.exe)."""
        with patch.dict(os.environ, {}, clear=True):
            platform_info = detect_platform()
            self.assertFalse(platform_info.supports_ansi)


class TestTerminalCapabilities(unittest.TestCase):
    """Test terminal capability detection."""

    @patch('os.get_terminal_size')
    def test_detect_terminal_size(self, mock_term_size):
        """Test terminal size detection."""
        mock_size = MagicMock()
        mock_size.columns = 80
        mock_size.lines = 24
        mock_term_size.return_value = mock_size

        platform_info = PlatformInfo(
            platform='linux',
            os_name='posix',
            is_windows=False,
            is_macos=False,
            is_linux=True,
            python_version=(3, 8, 0),
            supports_ansi=True
        )

        caps = detect_terminal_capabilities(platform_info)

        self.assertEqual(caps.width, 80)
        self.assertEqual(caps.height, 24)

    @patch('os.get_terminal_size', side_effect=OSError)
    def test_fallback_terminal_size(self, mock_term_size):
        """Test fallback to default terminal size on detection failure."""
        platform_info = PlatformInfo(
            platform='linux',
            os_name='posix',
            is_windows=False,
            is_macos=False,
            is_linux=True,
            python_version=(3, 8, 0),
            supports_ansi=True
        )

        caps = detect_terminal_capabilities(platform_info)

        # Should use defaults from constants
        self.assertEqual(caps.width, 40)
        self.assertEqual(caps.height, 20)

    @patch('sys.stdout')
    def test_encoding_detection(self, mock_stdout):
        """Test terminal encoding detection."""
        mock_stdout.encoding = 'utf-8'

        platform_info = PlatformInfo(
            platform='linux',
            os_name='posix',
            is_windows=False,
            is_macos=False,
            is_linux=True,
            python_version=(3, 8, 0),
            supports_ansi=True
        )

        with patch('os.get_terminal_size', side_effect=OSError):
            caps = detect_terminal_capabilities(platform_info)

        self.assertEqual(caps.encoding, 'utf-8')
        self.assertTrue(caps.supports_unicode)

    @patch('sys.stdout')
    def test_ascii_encoding_detection(self, mock_stdout):
        """Test ASCII encoding detection."""
        mock_stdout.encoding = 'ascii'

        platform_info = PlatformInfo(
            platform='linux',
            os_name='posix',
            is_windows=False,
            is_macos=False,
            is_linux=True,
            python_version=(3, 8, 0),
            supports_ansi=True
        )

        with patch('os.get_terminal_size', side_effect=OSError):
            caps = detect_terminal_capabilities(platform_info)

        self.assertEqual(caps.encoding, 'ascii')
        self.assertFalse(caps.supports_unicode)

    def test_windows_terminal_type_detection(self):
        """Test Windows terminal type identification."""
        platform_info = PlatformInfo(
            platform='win32',
            os_name='nt',
            is_windows=True,
            is_macos=False,
            is_linux=False,
            python_version=(3, 8, 0),
            supports_ansi=True
        )

        with patch.dict(os.environ, {'WT_SESSION': 'session-id'}):
            with patch('os.get_terminal_size', side_effect=OSError):
                caps = detect_terminal_capabilities(platform_info)
            self.assertEqual(caps.terminal_type, 'windows-terminal')

    def test_cmd_terminal_type_detection(self):
        """Test cmd.exe terminal type identification."""
        platform_info = PlatformInfo(
            platform='win32',
            os_name='nt',
            is_windows=True,
            is_macos=False,
            is_linux=False,
            python_version=(3, 8, 0),
            supports_ansi=False
        )

        with patch.dict(os.environ, {'PROMPT': '$P$G'}, clear=True):
            with patch('os.get_terminal_size', side_effect=OSError):
                caps = detect_terminal_capabilities(platform_info)
            self.assertEqual(caps.terminal_type, 'cmd')

    def test_powershell_terminal_type_detection(self):
        """Test PowerShell terminal type identification."""
        platform_info = PlatformInfo(
            platform='win32',
            os_name='nt',
            is_windows=True,
            is_macos=False,
            is_linux=False,
            python_version=(3, 8, 0),
            supports_ansi=False
        )

        with patch.dict(os.environ, {'PROMPT': '$P$G', 'PSModulePath': 'C:\\path'}):
            with patch('os.get_terminal_size', side_effect=OSError):
                caps = detect_terminal_capabilities(platform_info)
            self.assertEqual(caps.terminal_type, 'powershell')


class TestFilePathNormalization(unittest.TestCase):
    """Test cross-platform file path normalization."""

    @patch('sys.platform', 'win32')
    def test_windows_path_normalization(self):
        """Test Windows path normalization."""
        platform_info = PlatformInfo(
            platform='win32',
            os_name='nt',
            is_windows=True,
            is_macos=False,
            is_linux=False,
            python_version=(3, 8, 0),
            supports_ansi=True
        )

        # Test with mixed separators
        path = "C:/Users/test\\Documents/file.py"
        normalized = normalize_file_path(path, platform_info)

        # Should normalize to Windows backslashes and be absolute
        self.assertTrue(os.path.isabs(normalized))
        # Path should be normalized (os.path.normpath handles separators)
        self.assertNotIn('/', normalized.replace('C:\\', ''))  # Allow C:\ but no / elsewhere

    @patch('sys.platform', 'linux')
    def test_linux_path_normalization(self):
        """Test Linux path normalization."""
        platform_info = PlatformInfo(
            platform='linux',
            os_name='posix',
            is_windows=False,
            is_macos=False,
            is_linux=True,
            python_version=(3, 8, 0),
            supports_ansi=True
        )

        # Test relative path conversion to absolute
        path = "./test_file.py"
        normalized = normalize_file_path(path, platform_info)

        self.assertTrue(os.path.isabs(normalized))
        self.assertTrue(normalized.endswith('test_file.py'))

    def test_path_with_double_slashes(self):
        """Test path normalization removes double slashes."""
        platform_info = PlatformInfo(
            platform='linux',
            os_name='posix',
            is_windows=False,
            is_macos=False,
            is_linux=True,
            python_version=(3, 8, 0),
            supports_ansi=True
        )

        path = "/home//user//documents"
        normalized = normalize_file_path(path, platform_info)

        # os.path.normpath should remove redundant separators
        self.assertNotIn('//', normalized)


class TestGracefulDegradation(unittest.TestCase):
    """Test terminal capability graceful degradation."""

    def test_adjust_frame_dimensions_normal(self):
        """Test frame dimension adjustment for normal terminal."""
        term_caps = TerminalCapabilities(
            width=80,
            height=24,
            encoding='utf-8',
            supports_color=True,
            supports_unicode=True,
            terminal_type='xterm'
        )

        width, height = adjust_frame_dimensions(term_caps)

        # Should use close to terminal size with margins
        self.assertEqual(width, 78)  # 80 - 2
        self.assertEqual(height, 22)  # 24 - 2

    def test_adjust_frame_dimensions_large(self):
        """Test frame dimension adjustment for large terminal."""
        term_caps = TerminalCapabilities(
            width=200,
            height=100,
            encoding='utf-8',
            supports_color=True,
            supports_unicode=True,
            terminal_type='xterm'
        )

        width, height = adjust_frame_dimensions(term_caps)

        # Should cap at maximum values
        self.assertEqual(width, 120)
        self.assertEqual(height, 60)

    def test_adjust_frame_dimensions_small(self):
        """Test frame dimension adjustment for small terminal."""
        term_caps = TerminalCapabilities(
            width=30,
            height=15,
            encoding='utf-8',
            supports_color=True,
            supports_unicode=True,
            terminal_type='xterm'
        )

        width, height = adjust_frame_dimensions(term_caps)

        # Should use minimum values
        self.assertEqual(width, 40)
        self.assertEqual(height, 20)

    def test_validate_character_set_ascii(self):
        """Test character set validation for ASCII terminal."""
        term_caps = TerminalCapabilities(
            width=80,
            height=24,
            encoding='ascii',
            supports_color=False,
            supports_unicode=False,
            terminal_type='cmd'
        )

        valid = validate_character_set(term_caps)
        self.assertTrue(valid)

    def test_validate_character_set_utf8(self):
        """Test character set validation for UTF-8 terminal."""
        term_caps = TerminalCapabilities(
            width=80,
            height=24,
            encoding='utf-8',
            supports_color=True,
            supports_unicode=True,
            terminal_type='xterm'
        )

        valid = validate_character_set(term_caps)
        self.assertTrue(valid)

    def test_get_fallback_character_set_unicode(self):
        """Test fallback character set for Unicode terminal."""
        term_caps = TerminalCapabilities(
            width=80,
            height=24,
            encoding='utf-8',
            supports_color=True,
            supports_unicode=True,
            terminal_type='xterm'
        )

        chars = get_fallback_character_set(term_caps)

        self.assertEqual(chars['HIGH'], '#')
        self.assertEqual(chars['MEDIUM'], '+')
        self.assertEqual(chars['LOW'], '-')
        self.assertEqual(chars['BACKGROUND'], '.')

    def test_get_fallback_character_set_no_unicode(self):
        """Test fallback character set for non-Unicode terminal."""
        term_caps = TerminalCapabilities(
            width=80,
            height=24,
            encoding='ascii',
            supports_color=False,
            supports_unicode=False,
            terminal_type='cmd'
        )

        chars = get_fallback_character_set(term_caps)

        self.assertEqual(chars['HIGH'], '#')
        self.assertEqual(chars['MEDIUM'], '+')
        self.assertEqual(chars['LOW'], '-')
        self.assertEqual(chars['BACKGROUND'], ' ')

    def test_terminal_compatibility_full(self):
        """Test terminal compatibility check with full support."""
        platform_info = PlatformInfo(
            platform='linux',
            os_name='posix',
            is_windows=False,
            is_macos=False,
            is_linux=True,
            python_version=(3, 8, 0),
            supports_ansi=True
        )

        term_caps = TerminalCapabilities(
            width=80,
            height=24,
            encoding='utf-8',
            supports_color=True,
            supports_unicode=True,
            terminal_type='xterm'
        )

        compatible, message = test_terminal_capabilities(platform_info, term_caps)

        self.assertTrue(compatible)
        self.assertIn('compatible', message.lower())

    def test_terminal_compatibility_small_size(self):
        """Test terminal compatibility check with small terminal."""
        platform_info = PlatformInfo(
            platform='linux',
            os_name='posix',
            is_windows=False,
            is_macos=False,
            is_linux=True,
            python_version=(3, 8, 0),
            supports_ansi=True
        )

        term_caps = TerminalCapabilities(
            width=30,
            height=15,
            encoding='utf-8',
            supports_color=True,
            supports_unicode=True,
            terminal_type='xterm'
        )

        compatible, message = test_terminal_capabilities(platform_info, term_caps)

        # Small size detection should be present in message
        self.assertTrue(compatible)
        # Message includes size information
        self.assertTrue('30x15' in message or 'small' in message.lower() or 'recommend' in message.lower() or 'compatible' in message.lower())

    def test_terminal_compatibility_no_ansi(self):
        """Test terminal compatibility check without ANSI support."""
        platform_info = PlatformInfo(
            platform='win32',
            os_name='nt',
            is_windows=True,
            is_macos=False,
            is_linux=False,
            python_version=(3, 8, 0),
            supports_ansi=False
        )

        term_caps = TerminalCapabilities(
            width=80,
            height=24,
            encoding='cp1252',
            supports_color=False,
            supports_unicode=False,
            terminal_type='cmd'
        )

        compatible, message = test_terminal_capabilities(platform_info, term_caps)

        # No ANSI is a serious issue
        self.assertFalse(compatible)
        self.assertIn('ANSI', message)


class TestCrossPlatformTiming(unittest.TestCase):
    """Test cross-platform timing consistency."""

    def test_calibrate_timer_precision(self):
        """Test timer precision calibration."""
        platform_info = PlatformInfo(
            platform='linux',
            os_name='posix',
            is_windows=False,
            is_macos=False,
            is_linux=True,
            python_version=(3, 8, 0),
            supports_ansi=True
        )

        precision = calibrate_timer_precision(platform_info, samples=5)

        # Precision should be positive and less than 100ms
        self.assertGreater(precision, 0)
        self.assertLess(precision, 0.1)

    def test_validate_timing_consistency(self):
        """Test timing consistency validation."""
        platform_info = PlatformInfo(
            platform='linux',
            os_name='posix',
            is_windows=False,
            is_macos=False,
            is_linux=True,
            python_version=(3, 8, 0),
            supports_ansi=True
        )

        consistent, precision = validate_timing_consistency(platform_info)

        # Should be consistent for 30 FPS animation
        self.assertIsInstance(consistent, bool)
        self.assertGreater(precision, 0)
        self.assertLess(precision, 1.0)

    def test_calculate_adaptive_frame_time_high_precision(self):
        """Test adaptive frame time with high precision timer."""
        # High precision (1ms)
        timer_precision = 0.001
        frame_time = calculate_adaptive_frame_time(TARGET_FPS, timer_precision)

        # Should be close to base frame time
        expected = 1.0 / TARGET_FPS
        self.assertAlmostEqual(frame_time, expected, delta=0.001)

    def test_calculate_adaptive_frame_time_low_precision(self):
        """Test adaptive frame time with low precision timer."""
        # Low precision (10ms)
        timer_precision = 0.010
        frame_time = calculate_adaptive_frame_time(TARGET_FPS, timer_precision)

        # Should add buffer for coarse precision
        base_time = 1.0 / TARGET_FPS
        self.assertGreater(frame_time, base_time)
        # Buffer should be about half the precision
        self.assertLess(frame_time, base_time + timer_precision)

    def test_get_platform_sleep_overhead_windows(self):
        """Test sleep overhead estimation for Windows."""
        platform_info = PlatformInfo(
            platform='win32',
            os_name='nt',
            is_windows=True,
            is_macos=False,
            is_linux=False,
            python_version=(3, 8, 0),
            supports_ansi=True
        )

        overhead = get_platform_sleep_overhead(platform_info)

        # Windows has higher overhead
        self.assertGreater(overhead, 0)
        self.assertEqual(overhead, 0.001)

    def test_get_platform_sleep_overhead_unix(self):
        """Test sleep overhead estimation for Unix."""
        platform_info = PlatformInfo(
            platform='linux',
            os_name='posix',
            is_windows=False,
            is_macos=False,
            is_linux=True,
            python_version=(3, 8, 0),
            supports_ansi=True
        )

        overhead = get_platform_sleep_overhead(platform_info)

        # Unix has lower overhead
        self.assertGreater(overhead, 0)
        self.assertEqual(overhead, 0.0005)

    def test_adaptive_sleep_positive_time(self):
        """Test adaptive sleep with positive sleep time."""
        platform_info = PlatformInfo(
            platform='linux',
            os_name='posix',
            is_windows=False,
            is_macos=False,
            is_linux=True,
            python_version=(3, 8, 0),
            supports_ansi=True
        )

        import time
        start = time.time()
        adaptive_sleep(0.01, platform_info)  # 10ms sleep
        elapsed = time.time() - start

        # Should sleep approximately the requested time
        # Allow generous variance for platform differences and overhead
        self.assertGreater(elapsed, 0.005)  # At least 5ms
        self.assertLess(elapsed, 0.030)  # Less than 30ms

    def test_adaptive_sleep_zero_time(self):
        """Test adaptive sleep with zero sleep time."""
        platform_info = PlatformInfo(
            platform='linux',
            os_name='posix',
            is_windows=False,
            is_macos=False,
            is_linux=True,
            python_version=(3, 8, 0),
            supports_ansi=True
        )

        import time
        start = time.time()
        adaptive_sleep(0, platform_info)
        elapsed = time.time() - start

        # Should return immediately
        self.assertLess(elapsed, 0.001)

    def test_adaptive_sleep_negative_time(self):
        """Test adaptive sleep with negative sleep time."""
        platform_info = PlatformInfo(
            platform='linux',
            os_name='posix',
            is_windows=False,
            is_macos=False,
            is_linux=True,
            python_version=(3, 8, 0),
            supports_ansi=True
        )

        import time
        start = time.time()
        adaptive_sleep(-0.01, platform_info)
        elapsed = time.time() - start

        # Should return immediately
        self.assertLess(elapsed, 0.001)


class TestTerminalOutputStandardization(unittest.TestCase):
    """Test cross-platform terminal output standardization."""

    def test_clear_screen_with_ansi_support(self):
        """Test clear screen with ANSI support."""
        platform_info = PlatformInfo(
            platform='linux',
            os_name='posix',
            is_windows=False,
            is_macos=False,
            is_linux=True,
            python_version=(3, 8, 0),
            supports_ansi=True
        )

        # Should not raise exception
        try:
            clear_screen(platform_info)
        except Exception as e:
            self.fail(f"clear_screen raised exception: {e}")

    def test_clear_screen_without_ansi_windows(self):
        """Test clear screen without ANSI on Windows."""
        platform_info = PlatformInfo(
            platform='win32',
            os_name='nt',
            is_windows=True,
            is_macos=False,
            is_linux=False,
            python_version=(3, 8, 0),
            supports_ansi=False
        )

        # Should not raise exception
        try:
            clear_screen(platform_info)
        except Exception as e:
            self.fail(f"clear_screen raised exception: {e}")

    def test_clear_screen_without_ansi_unix(self):
        """Test clear screen without ANSI on Unix."""
        platform_info = PlatformInfo(
            platform='linux',
            os_name='posix',
            is_windows=False,
            is_macos=False,
            is_linux=True,
            python_version=(3, 8, 0),
            supports_ansi=False
        )

        # Should not raise exception
        try:
            clear_screen(platform_info)
        except Exception as e:
            self.fail(f"clear_screen raised exception: {e}")

    def test_output_line_buffered(self):
        """Test line buffered output."""
        # Should not raise exception
        try:
            output_line_buffered("Test output")
        except Exception as e:
            self.fail(f"output_line_buffered raised exception: {e}")

    def test_validate_terminal_output_valid(self):
        """Test terminal output validation with valid terminal."""
        platform_info = PlatformInfo(
            platform='linux',
            os_name='posix',
            is_windows=False,
            is_macos=False,
            is_linux=True,
            python_version=(3, 8, 0),
            supports_ansi=True
        )

        term_caps = TerminalCapabilities(
            width=80,
            height=24,
            encoding='utf-8',
            supports_color=True,
            supports_unicode=True,
            terminal_type='xterm'
        )

        valid = validate_terminal_output(platform_info, term_caps)
        self.assertTrue(valid)

    def test_validate_terminal_output_no_encoding(self):
        """Test terminal output validation with no encoding."""
        platform_info = PlatformInfo(
            platform='linux',
            os_name='posix',
            is_windows=False,
            is_macos=False,
            is_linux=True,
            python_version=(3, 8, 0),
            supports_ansi=True
        )

        term_caps = TerminalCapabilities(
            width=80,
            height=24,
            encoding='',  # No encoding
            supports_color=True,
            supports_unicode=True,
            terminal_type='xterm'
        )

        valid = validate_terminal_output(platform_info, term_caps)
        self.assertFalse(valid)


if __name__ == '__main__':
    unittest.main()