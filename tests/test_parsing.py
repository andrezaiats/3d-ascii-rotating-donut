#!/usr/bin/env python3
"""
Unit tests for parsing engine functionality.

Tests the self-file reading system implementation including path resolution,
file reading with various encodings, content validation, and error handling.
"""

import unittest
import unittest.mock as mock
import tempfile
import os
import sys
from pathlib import Path

# Add the project root to Python path for importing the module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rotating_donut import (
    get_script_path,
    validate_file_content,
    read_self_code
)


class TestGetScriptPath(unittest.TestCase):
    """Test the get_script_path() function for reliable path identification."""

    def test_get_script_path_success(self):
        """Test successful script path retrieval."""
        # This should work in normal execution context
        try:
            path = get_script_path()
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.isfile(path))
            self.assertTrue(os.access(path, os.R_OK))
        except Exception as e:
            self.fail(f"get_script_path() raised an exception: {e}")

    @mock.patch('rotating_donut.__file__', None, create=True)
    def test_get_script_path_no_file_variable(self):
        """Test error handling when __file__ is not available."""
        with mock.patch('rotating_donut.globals', return_value={}):
            with self.assertRaises(FileNotFoundError) as context:
                get_script_path()
            self.assertIn("__file__ not available", str(context.exception))
            self.assertIn("Solution:", str(context.exception))

    @mock.patch('os.path.exists')
    def test_get_script_path_file_not_found(self, mock_exists):
        """Test error handling when resolved file doesn't exist."""
        mock_exists.return_value = False

        with self.assertRaises(FileNotFoundError) as context:
            get_script_path()
        self.assertIn("Script file not found at resolved path", str(context.exception))
        self.assertIn("Solution:", str(context.exception))

    @mock.patch('os.access')
    @mock.patch('os.path.exists')
    def test_get_script_path_permission_denied(self, mock_exists, mock_access):
        """Test error handling when file is not readable."""
        mock_exists.return_value = True
        mock_access.return_value = False

        with self.assertRaises(PermissionError) as context:
            get_script_path()
        self.assertIn("Cannot read script file", str(context.exception))
        self.assertIn("Solution:", str(context.exception))


class TestValidateFileContent(unittest.TestCase):
    """Test the validate_file_content() function for content integrity validation."""

    def test_validate_file_content_success(self):
        """Test successful validation of valid Python content."""
        content = "#!/usr/bin/env python3\nprint('hello world')\n"
        result = validate_file_content("test.py", content)
        self.assertTrue(result)

    def test_validate_file_content_empty(self):
        """Test validation failure for empty content."""
        with self.assertRaises(ValueError) as context:
            validate_file_content("test.py", "")
        self.assertIn("File content is empty", str(context.exception))
        self.assertIn("Solution:", str(context.exception))

    def test_validate_file_content_too_short(self):
        """Test validation failure for content that's too short."""
        with self.assertRaises(ValueError) as context:
            validate_file_content("test.py", "x")
        self.assertIn("too short for a valid Python file", str(context.exception))
        self.assertIn("Solution:", str(context.exception))

    def test_validate_file_content_too_large(self):
        """Test validation failure for content that's too large."""
        large_content = "x" * (11 * 1024 * 1024)  # 11MB
        with self.assertRaises(ValueError) as context:
            validate_file_content("test.py", large_content)
        self.assertIn("exceeds maximum", str(context.exception))
        self.assertIn("Solution:", str(context.exception))

    def test_validate_file_content_syntax_error(self):
        """Test validation failure for invalid Python syntax."""
        invalid_content = "def invalid_function(\nprint('missing colon and parenthesis')"
        with self.assertRaises(ValueError) as context:
            validate_file_content("test.py", invalid_content)
        self.assertIn("Python syntax error", str(context.exception))
        self.assertIn("Solution:", str(context.exception))

    def test_validate_file_content_complex_valid_code(self):
        """Test validation success for complex valid Python code."""
        complex_content = '''
import math
import os

class TestClass:
    def __init__(self, value):
        self.value = value

    def calculate(self):
        return math.sqrt(self.value)

def main():
    test = TestClass(16)
    print(f"Result: {test.calculate()}")

if __name__ == "__main__":
    main()
'''
        result = validate_file_content("complex.py", complex_content)
        self.assertTrue(result)


class TestReadSelfCode(unittest.TestCase):
    """Test the read_self_code() function for comprehensive file reading."""

    def test_read_self_code_success(self):
        """Test successful reading of the script's own source code."""
        try:
            content = read_self_code()
            self.assertIsInstance(content, str)
            self.assertGreater(len(content), 100)  # Should be substantial content
            self.assertIn("def read_self_code", content)  # Should contain the function itself
        except Exception as e:
            self.fail(f"read_self_code() raised an exception: {e}")

    def test_read_self_code_content_validation(self):
        """Test that read_self_code() returns syntactically valid Python."""
        content = read_self_code()
        # This should not raise an exception since we validate content internally
        try:
            compile(content, "test.py", "exec")
        except SyntaxError:
            self.fail("read_self_code() returned content with syntax errors")

    @mock.patch('rotating_donut.get_script_path')
    def test_read_self_code_file_not_found(self, mock_get_script_path):
        """Test error handling when script file cannot be found."""
        mock_get_script_path.side_effect = FileNotFoundError(
            "Test file not found. Solution: Test solution"
        )

        with self.assertRaises(FileNotFoundError) as context:
            read_self_code()
        self.assertIn("Solution:", str(context.exception))

    @mock.patch('rotating_donut.get_script_path')
    def test_read_self_code_permission_error(self, mock_get_script_path):
        """Test error handling when file access is denied."""
        mock_get_script_path.side_effect = PermissionError(
            "Test permission denied. Solution: Test solution"
        )

        with self.assertRaises(PermissionError) as context:
            read_self_code()
        self.assertIn("Solution:", str(context.exception))


class TestCrossplatformPathHandling(unittest.TestCase):
    """Test cross-platform path handling functionality."""

    def test_path_normalization(self):
        """Test that paths are properly normalized across platforms."""
        try:
            path = get_script_path()
            # Path should be absolute
            self.assertTrue(os.path.isabs(path))
            # Path should be normalized (no redundant separators)
            self.assertEqual(path, os.path.normpath(path))
        except Exception as e:
            self.fail(f"Path normalization test failed: {e}")

    @mock.patch('os.path.realpath')
    @mock.patch('os.path.abspath')
    def test_symbolic_link_resolution(self, mock_abspath, mock_realpath):
        """Test that symbolic links are properly resolved."""
        test_path = "/test/script.py"
        mock_abspath.return_value = test_path
        mock_realpath.return_value = test_path

        with mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.access', return_value=True):

            try:
                get_script_path()
                mock_realpath.assert_called_once()
            except Exception as e:
                self.fail(f"Symbolic link resolution test failed: {e}")


class TestFileEncodingHandling(unittest.TestCase):
    """Test file encoding handling and BOM support."""

    def test_utf8_bom_handling(self):
        """Test that UTF-8 BOM is properly handled."""
        # Create a temporary file with UTF-8 BOM
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.py') as temp_file:
            # Simple Python content WITHOUT BOM in the string
            content_without_bom = '#!/usr/bin/env python3\nprint("hello")\n'
            # Write with utf-8-sig encoding to add BOM automatically
            temp_file.write(content_without_bom.encode('utf-8-sig'))
            temp_file_path = temp_file.name

        try:
            # Mock get_script_path to return our temp file
            with mock.patch('rotating_donut.get_script_path', return_value=temp_file_path):
                content = read_self_code()
                # BOM should be stripped by utf-8-sig encoding
                self.assertFalse(content.startswith('\ufeff'))
                self.assertIn('print("hello")', content)
                self.assertTrue(content.startswith('#!/usr/bin/env python3'))
        finally:
            os.unlink(temp_file_path)

    def test_encoding_fallback(self):
        """Test encoding fallback behavior."""
        # Create a temporary file with Latin1 content
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.py') as temp_file:
            # Content that's valid in Latin1 but might cause issues in UTF-8
            latin1_content = 'print("café")\n'  # Simple case that works in both
            temp_file.write(latin1_content.encode('latin1'))
            temp_file_path = temp_file.name

        try:
            with mock.patch('rotating_donut.get_script_path', return_value=temp_file_path):
                content = read_self_code()
                self.assertIn('print', content)
        finally:
            os.unlink(temp_file_path)


class TestErrorMessageFormat(unittest.TestCase):
    """Test that all error messages follow the required 'Solution:' format."""

    def test_error_messages_have_solutions(self):
        """Test that all error conditions provide solution guidance."""
        # Test various error conditions and verify they include "Solution:"

        # Test empty content validation
        try:
            validate_file_content("test.py", "")
        except ValueError as e:
            self.assertIn("Solution:", str(e))

        # Test file size validation
        try:
            validate_file_content("test.py", "x" * (11 * 1024 * 1024))
        except ValueError as e:
            self.assertIn("Solution:", str(e))

        # Test syntax error validation
        try:
            validate_file_content("test.py", "invalid python syntax !!!!")
        except ValueError as e:
            self.assertIn("Solution:", str(e))


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)