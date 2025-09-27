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
    read_self_code,
    tokenize_code,
    classify_importance,
    _is_builtin_function,
    _is_string_literal,
    CodeToken,
    ImportanceLevel
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


class TestTokenizeCode(unittest.TestCase):
    """Test the tokenize_code() function for comprehensive token parsing."""

    def test_tokenize_simple_code(self):
        """Test tokenization of simple Python code."""
        source = "def hello():\n    print('world')"
        tokens = tokenize_code(source)

        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

        # Check that we have tokens of different types
        token_types = [token.type for token in tokens]
        self.assertIn('KEYWORD', token_types)  # 'def'
        self.assertIn('IDENTIFIER', token_types)  # 'hello', 'print'
        self.assertIn('LITERAL', token_types)  # 'world'
        self.assertIn('OPERATOR', token_types)  # '(', ')', ':'

    def test_tokenize_keyword_classification(self):
        """Test that Python keywords are correctly classified as HIGH importance."""
        source = "def class if for while try except import"
        tokens = tokenize_code(source)

        keyword_tokens = [token for token in tokens if token.type == 'KEYWORD']
        self.assertGreater(len(keyword_tokens), 0)

        for token in keyword_tokens:
            self.assertEqual(token.importance, ImportanceLevel.CRITICAL)
            self.assertEqual(token.ascii_char, '#')

    def test_tokenize_operator_classification(self):
        """Test that operators are correctly classified as HIGH importance."""
        source = "x = 1 + 2 * 3 / 4 - 5"
        tokens = tokenize_code(source)

        operator_tokens = [token for token in tokens if token.type == 'OPERATOR']
        self.assertGreater(len(operator_tokens), 0)

        for token in operator_tokens:
            self.assertEqual(token.importance, ImportanceLevel.HIGH)
            self.assertEqual(token.ascii_char, '+')

    def test_tokenize_identifier_classification(self):
        """Test that identifiers are correctly classified as MEDIUM importance."""
        source = "variable_name = function_call(parameter)"
        tokens = tokenize_code(source)

        identifier_tokens = [token for token in tokens if token.type == 'IDENTIFIER']
        self.assertGreater(len(identifier_tokens), 0)

        for token in identifier_tokens:
            self.assertEqual(token.importance, ImportanceLevel.MEDIUM)
            self.assertEqual(token.ascii_char, '-')

    def test_tokenize_literal_classification(self):
        """Test that literals are correctly classified as MEDIUM importance."""
        source = "x = 42\ny = 'hello'\nz = 3.14"
        tokens = tokenize_code(source)

        literal_tokens = [token for token in tokens if token.type == 'LITERAL']
        self.assertGreater(len(literal_tokens), 0)

        for token in literal_tokens:
            self.assertEqual(token.importance, ImportanceLevel.MEDIUM)
            self.assertEqual(token.ascii_char, '-')

    def test_tokenize_comment_classification(self):
        """Test that comments are correctly classified as LOW importance."""
        source = "# This is a comment\nprint('hello')  # Another comment"
        tokens = tokenize_code(source)

        comment_tokens = [token for token in tokens if token.type == 'COMMENT']
        self.assertGreater(len(comment_tokens), 0)

        for token in comment_tokens:
            self.assertEqual(token.importance, ImportanceLevel.LOW)
            self.assertEqual(token.ascii_char, '.')

    def test_tokenize_whitespace_classification(self):
        """Test that whitespace tokens are correctly classified as LOW importance."""
        source = "def hello():\n    print('world')\n"
        tokens = tokenize_code(source)

        whitespace_tokens = [token for token in tokens if token.type == 'WHITESPACE']
        # Should have NEWLINE and INDENT tokens
        self.assertGreater(len(whitespace_tokens), 0)

        for token in whitespace_tokens:
            self.assertEqual(token.importance, ImportanceLevel.LOW)
            self.assertEqual(token.ascii_char, '.')

    def test_tokenize_position_information(self):
        """Test that position information is correctly extracted."""
        source = "def hello():\n    print('world')"
        tokens = tokenize_code(source)

        # Find the 'def' token
        def_token = next(token for token in tokens if token.value == 'def')
        self.assertEqual(def_token.line, 1)
        self.assertEqual(def_token.column, 0)

        # Find the 'print' token (should be on line 2)
        print_token = next(token for token in tokens if token.value == 'print')
        self.assertEqual(print_token.line, 2)
        self.assertGreaterEqual(print_token.column, 4)  # After indentation

    def test_tokenize_sequence_order(self):
        """Test that token sequence order is maintained."""
        source = "x = 1 + 2"
        tokens = tokenize_code(source)

        # Find specific tokens and verify their order
        token_values = [token.value for token in tokens if token.value.strip()]

        self.assertIn('x', token_values)
        self.assertIn('=', token_values)
        self.assertIn('1', token_values)
        self.assertIn('+', token_values)
        self.assertIn('2', token_values)

        # Verify order
        x_index = token_values.index('x')
        equals_index = token_values.index('=')
        one_index = token_values.index('1')
        plus_index = token_values.index('+')
        two_index = token_values.index('2')

        self.assertLess(x_index, equals_index)
        self.assertLess(equals_index, one_index)
        self.assertLess(one_index, plus_index)
        self.assertLess(plus_index, two_index)

    def test_tokenize_complex_code(self):
        """Test tokenization of complex Python code with various constructs."""
        source = '''
import math

class Calculator:
    """A simple calculator class."""

    def __init__(self, precision=2):
        self.precision = precision

    def add(self, a, b):
        # Addition method
        return round(a + b, self.precision)

    def multiply(self, x, y):
        """Multiply two numbers."""
        result = x * y
        return result

# Create instance
calc = Calculator(precision=3)
print(f"Result: {calc.add(1.5, 2.7)}")
'''
        tokens = tokenize_code(source)

        # Verify we have a substantial number of tokens
        self.assertGreater(len(tokens), 50)

        # Check for presence of different token types
        token_types = set(token.type for token in tokens)
        expected_types = {'KEYWORD', 'IDENTIFIER', 'LITERAL', 'OPERATOR', 'COMMENT'}
        self.assertTrue(expected_types.issubset(token_types))

        # Verify specific important tokens
        token_values = [token.value for token in tokens]
        self.assertIn('import', token_values)
        self.assertIn('class', token_values)
        self.assertIn('def', token_values)
        self.assertIn('Calculator', token_values)

    def test_tokenize_empty_source(self):
        """Test error handling for empty source code."""
        with self.assertRaises(ValueError) as context:
            tokenize_code("")
        self.assertIn("Empty or whitespace-only source", str(context.exception))
        self.assertIn("Solution:", str(context.exception))

    def test_tokenize_whitespace_only_source(self):
        """Test error handling for whitespace-only source code."""
        with self.assertRaises(ValueError) as context:
            tokenize_code("   \n  \t  \n   ")
        self.assertIn("Empty or whitespace-only source", str(context.exception))
        self.assertIn("Solution:", str(context.exception))

    def test_tokenize_invalid_syntax(self):
        """Test error handling for severe tokenization errors."""
        invalid_sources = [
            "print('unclosed string",  # Unclosed string literal
            'print("unclosed double quote',  # Unclosed double quote
            "'''unclosed triple quote",  # Unclosed triple quote
            "print(\\\n",  # Invalid line continuation
        ]

        for source in invalid_sources:
            with self.assertRaises(ValueError) as context:
                tokenize_code(source)
            self.assertIn("Tokenization failed", str(context.exception))
            self.assertIn("Solution:", str(context.exception))

    def test_tokenize_unicode_handling(self):
        """Test tokenization with Unicode characters in strings and comments."""
        source = '''
# Café function with émojis 🚀
def café():
    message = "Hello, 世界! 🌍"
    return message
'''
        tokens = tokenize_code(source)

        # Should successfully tokenize Unicode content
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

        # Find Unicode content
        token_values = [token.value for token in tokens]
        self.assertIn('café', token_values)

    def test_tokenize_special_characters(self):
        """Test tokenization of special characters and operators."""
        source = "x[0] = {1: 2, 3: 4}; y = (a, b, c)"
        tokens = tokenize_code(source)

        # Check for special character tokens
        special_chars = ['[', ']', '{', '}', '(', ')', ',', ';', ':']
        token_values = [token.value for token in tokens]

        for char in special_chars:
            self.assertIn(char, token_values)

    def test_tokenize_multiline_strings(self):
        """Test tokenization of multiline strings."""
        source = '''
text = """
This is a
multiline string
with multiple lines
"""
'''
        tokens = tokenize_code(source)

        # Should have string token with multiline content
        string_tokens = [token for token in tokens if token.type == 'LITERAL' and '"""' in token.value]
        self.assertGreater(len(string_tokens), 0)

    @mock.patch('tokenize.generate_tokens')
    def test_tokenize_error_handling_mock(self, mock_generate_tokens):
        """Test error handling using mocked tokenize module."""
        import tokenize as tokenize_module

        # Mock tokenize.TokenError
        mock_generate_tokens.side_effect = tokenize_module.TokenError("Mock tokenization error")

        with self.assertRaises(ValueError) as context:
            tokenize_code("valid code")
        self.assertIn("Tokenization failed", str(context.exception))
        self.assertIn("Solution:", str(context.exception))

    def test_tokenize_codetoken_structure(self):
        """Test that returned CodeToken objects have the correct structure."""
        source = "def test(): pass"
        tokens = tokenize_code(source)

        for token in tokens:
            self.assertIsInstance(token, CodeToken)
            self.assertIsInstance(token.type, str)
            self.assertIsInstance(token.value, str)
            self.assertIsInstance(token.importance, int)
            self.assertIsInstance(token.line, int)
            self.assertIsInstance(token.column, int)
            self.assertIsInstance(token.ascii_char, str)

            # Validate importance levels
            self.assertIn(token.importance, [ImportanceLevel.CRITICAL, ImportanceLevel.HIGH, ImportanceLevel.MEDIUM, ImportanceLevel.LOW])

            # Validate ASCII characters
            self.assertIn(token.ascii_char, ['#', '+', '-', '.'])

            # Validate line and column are positive
            self.assertGreaterEqual(token.line, 1)
            self.assertGreaterEqual(token.column, 0)


class TestClassifyImportance(unittest.TestCase):
    """Test the classify_importance() function for semantic importance hierarchy."""

    def test_classify_importance_critical_keywords(self):
        """Test that Python keywords are classified as CRITICAL importance."""
        keyword_tokens = [
            CodeToken('KEYWORD', 'def', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('KEYWORD', 'class', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('KEYWORD', 'if', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('KEYWORD', 'for', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('KEYWORD', 'while', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('KEYWORD', 'try', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('KEYWORD', 'except', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('KEYWORD', 'import', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('KEYWORD', 'return', ImportanceLevel.LOW, 1, 0, '-'),
        ]

        for token in keyword_tokens:
            importance = classify_importance(token)
            self.assertEqual(importance, ImportanceLevel.CRITICAL,
                           f"Keyword '{token.value}' should be CRITICAL importance")

    def test_classify_importance_high_operators(self):
        """Test that operators are classified as HIGH importance."""
        operator_tokens = [
            CodeToken('OPERATOR', '+', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('OPERATOR', '-', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('OPERATOR', '*', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('OPERATOR', '/', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('OPERATOR', '==', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('OPERATOR', '!=', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('OPERATOR', '>=', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('OPERATOR', '<=', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('OPERATOR', '=', ImportanceLevel.LOW, 1, 0, '-'),
        ]

        for token in operator_tokens:
            importance = classify_importance(token)
            self.assertEqual(importance, ImportanceLevel.HIGH,
                           f"Operator '{token.value}' should be HIGH importance")

    def test_classify_importance_medium_identifiers(self):
        """Test that identifiers are classified as MEDIUM importance."""
        identifier_tokens = [
            CodeToken('IDENTIFIER', 'variable_name', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('IDENTIFIER', 'function_name', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('IDENTIFIER', 'class_name', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('IDENTIFIER', 'parameter', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('IDENTIFIER', 'method', ImportanceLevel.LOW, 1, 0, '-'),
        ]

        for token in identifier_tokens:
            importance = classify_importance(token)
            self.assertEqual(importance, ImportanceLevel.MEDIUM,
                           f"Identifier '{token.value}' should be MEDIUM importance")

    def test_classify_importance_medium_literals(self):
        """Test that literals are classified as MEDIUM importance."""
        literal_tokens = [
            CodeToken('LITERAL', '42', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('LITERAL', '3.14', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('LITERAL', "'hello'", ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('LITERAL', '"world"', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('LITERAL', '"""multiline"""', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('LITERAL', 'True', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('LITERAL', 'False', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('LITERAL', 'None', ImportanceLevel.LOW, 1, 0, '-'),
        ]

        for token in literal_tokens:
            importance = classify_importance(token)
            self.assertEqual(importance, ImportanceLevel.MEDIUM,
                           f"Literal '{token.value}' should be MEDIUM importance")

    def test_classify_importance_low_comments(self):
        """Test that comments are classified as LOW importance."""
        comment_tokens = [
            CodeToken('COMMENT', '# This is a comment', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('COMMENT', '# TODO: Fix this', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('COMMENT', '# Another comment', ImportanceLevel.LOW, 1, 0, '-'),
        ]

        for token in comment_tokens:
            importance = classify_importance(token)
            self.assertEqual(importance, ImportanceLevel.LOW,
                           f"Comment '{token.value}' should be LOW importance")

    def test_classify_importance_low_whitespace(self):
        """Test that whitespace tokens are classified as LOW importance."""
        whitespace_tokens = [
            CodeToken('WHITESPACE', '\n', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('WHITESPACE', '    ', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('WHITESPACE', '\t', ImportanceLevel.LOW, 1, 0, '-'),
        ]

        for token in whitespace_tokens:
            importance = classify_importance(token)
            self.assertEqual(importance, ImportanceLevel.LOW,
                           f"Whitespace token should be LOW importance")

    def test_classify_importance_low_special_characters(self):
        """Test that special characters are classified as LOW importance."""
        special_tokens = [
            CodeToken('SPECIAL', '(', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('SPECIAL', ')', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('SPECIAL', '[', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('SPECIAL', ']', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('SPECIAL', '{', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('SPECIAL', '}', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('SPECIAL', ';', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('SPECIAL', ',', ImportanceLevel.LOW, 1, 0, '-'),
        ]

        for token in special_tokens:
            importance = classify_importance(token)
            self.assertEqual(importance, ImportanceLevel.LOW,
                           f"Special character '{token.value}' should be LOW importance")

    def test_classify_importance_high_decorators(self):
        """Test that decorators are classified as HIGH importance."""
        decorator_tokens = [
            CodeToken('OPERATOR', '@property', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('OPERATOR', '@staticmethod', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('OPERATOR', '@classmethod', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('DECORATOR', '@custom_decorator', ImportanceLevel.LOW, 1, 0, '-'),
        ]

        for token in decorator_tokens:
            importance = classify_importance(token)
            self.assertEqual(importance, ImportanceLevel.HIGH,
                           f"Decorator '{token.value}' should be HIGH importance")

    def test_classify_importance_high_builtin_functions(self):
        """Test that built-in functions are classified as HIGH importance."""
        builtin_tokens = [
            CodeToken('IDENTIFIER', 'print', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('IDENTIFIER', 'len', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('IDENTIFIER', 'str', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('IDENTIFIER', 'int', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('IDENTIFIER', 'float', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('IDENTIFIER', 'list', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('IDENTIFIER', 'dict', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('IDENTIFIER', 'range', ImportanceLevel.LOW, 1, 0, '-'),
        ]

        for token in builtin_tokens:
            importance = classify_importance(token)
            self.assertEqual(importance, ImportanceLevel.HIGH,
                           f"Built-in function '{token.value}' should be HIGH importance")

    def test_classify_importance_error_handling(self):
        """Test error handling for invalid token parameters."""
        # Test with non-CodeToken object (string)
        with self.assertRaises(ValueError) as context:
            classify_importance("not_a_token")
        self.assertIn("Invalid token parameter", str(context.exception))
        self.assertIn("Solution:", str(context.exception))

        # Test with non-CodeToken object (dict)
        with self.assertRaises(ValueError) as context:
            classify_importance({"type": "test", "value": "test"})
        self.assertIn("Invalid token parameter", str(context.exception))
        self.assertIn("Solution:", str(context.exception))

        # Test with object missing required attributes (this will also be caught by isinstance)
        class InvalidToken:
            pass

        with self.assertRaises(ValueError) as context:
            classify_importance(InvalidToken())
        self.assertIn("Invalid token parameter", str(context.exception))
        self.assertIn("Solution:", str(context.exception))

    def test_classify_importance_empty_value(self):
        """Test handling of tokens with empty values."""
        empty_token = CodeToken('IDENTIFIER', '', ImportanceLevel.LOW, 1, 0, '-')
        importance = classify_importance(empty_token)
        self.assertEqual(importance, ImportanceLevel.LOW)

        none_token = CodeToken('IDENTIFIER', None, ImportanceLevel.LOW, 1, 0, '-')
        importance = classify_importance(none_token)
        self.assertEqual(importance, ImportanceLevel.LOW)

    def test_classify_importance_unknown_token_types(self):
        """Test that unknown token types default to LOW importance."""
        unknown_tokens = [
            CodeToken('UNKNOWN_TYPE', 'value', ImportanceLevel.LOW, 1, 0, '-'),
            CodeToken('WEIRD_TOKEN', 'another_value', ImportanceLevel.LOW, 1, 0, '-'),
        ]

        for token in unknown_tokens:
            importance = classify_importance(token)
            self.assertEqual(importance, ImportanceLevel.LOW,
                           f"Unknown token type '{token.type}' should default to LOW importance")


class TestBuiltinFunctionDetection(unittest.TestCase):
    """Test the _is_builtin_function() helper function."""

    def test_is_builtin_function_python_keywords(self):
        """Test detection of Python keywords."""
        keywords = ['def', 'class', 'if', 'for', 'while', 'try', 'except', 'import']
        for keyword in keywords:
            self.assertTrue(_is_builtin_function(keyword),
                          f"'{keyword}' should be detected as keyword/builtin")

    def test_is_builtin_function_builtin_functions(self):
        """Test detection of Python built-in functions."""
        builtins = ['print', 'len', 'str', 'int', 'float', 'list', 'dict', 'range', 'enumerate']
        for builtin in builtins:
            self.assertTrue(_is_builtin_function(builtin),
                          f"'{builtin}' should be detected as built-in function")

    def test_is_builtin_function_non_builtins(self):
        """Test that regular identifiers are not detected as built-ins."""
        non_builtins = ['variable_name', 'custom_function', 'MyClass', 'parameter']
        for identifier in non_builtins:
            self.assertFalse(_is_builtin_function(identifier),
                           f"'{identifier}' should not be detected as built-in")

    def test_is_builtin_function_edge_cases(self):
        """Test edge cases for built-in function detection."""
        # Empty string
        self.assertFalse(_is_builtin_function(''))

        # None (should handle gracefully)
        self.assertFalse(_is_builtin_function(None))


class TestStringLiteralDetection(unittest.TestCase):
    """Test the _is_string_literal() helper function."""

    def test_is_string_literal_single_quotes(self):
        """Test detection of single-quoted strings."""
        single_quotes = ["'hello'", "'world'", "'test string'", "'123'"]
        for string in single_quotes:
            self.assertTrue(_is_string_literal(string),
                          f"'{string}' should be detected as string literal")

    def test_is_string_literal_double_quotes(self):
        """Test detection of double-quoted strings."""
        double_quotes = ['"hello"', '"world"', '"test string"', '"123"']
        for string in double_quotes:
            self.assertTrue(_is_string_literal(string),
                          f"'{string}' should be detected as string literal")

    def test_is_string_literal_triple_quotes(self):
        """Test detection of triple-quoted strings."""
        triple_quotes = ['"""hello"""', "'''world'''", '"""multiline\nstring"""']
        for string in triple_quotes:
            self.assertTrue(_is_string_literal(string),
                          f"'{string}' should be detected as string literal")

    def test_is_string_literal_prefixed_strings(self):
        """Test detection of prefixed strings (raw, unicode, bytes, f-strings)."""
        prefixed_strings = [
            'r"raw string"', "r'raw string'",
            'u"unicode"', "u'unicode'",
            'b"bytes"', "b'bytes'",
            'f"f-string"', "f'f-string'",
            'rf"raw f-string"', "rf'raw f-string'"
        ]
        for string in prefixed_strings:
            self.assertTrue(_is_string_literal(string),
                          f"'{string}' should be detected as string literal")

    def test_is_string_literal_non_strings(self):
        """Test that non-string tokens are not detected as string literals."""
        non_strings = ['42', '3.14', 'variable_name', 'True', 'False', 'None', '+', '==']
        for token in non_strings:
            self.assertFalse(_is_string_literal(token),
                           f"'{token}' should not be detected as string literal")

    def test_is_string_literal_edge_cases(self):
        """Test edge cases for string literal detection."""
        # Empty string
        self.assertFalse(_is_string_literal(''))

        # None
        self.assertFalse(_is_string_literal(None))

        # Incomplete strings (should not match)
        incomplete_strings = ['"incomplete', "'incomplete", 'incomplete"', "incomplete'"]
        for string in incomplete_strings:
            self.assertFalse(_is_string_literal(string),
                           f"'{string}' should not be detected as complete string literal")


class TestImportanceLevelIntegration(unittest.TestCase):
    """Integration tests for the complete importance classification system."""

    def test_tokenize_with_new_importance_system(self):
        """Test that tokenize_code() properly uses the new importance classification."""
        source = """
def example_function():
    # This is a comment
    x = 42 + 3.14
    print('Hello, world!')
    return x
"""
        tokens = tokenize_code(source)

        # Find specific tokens and verify their importance levels
        token_map = {token.value: token for token in tokens if token.value.strip()}

        # Keywords should be CRITICAL (4)
        if 'def' in token_map:
            self.assertEqual(token_map['def'].importance, ImportanceLevel.CRITICAL)
        if 'return' in token_map:
            self.assertEqual(token_map['return'].importance, ImportanceLevel.CRITICAL)

        # Operators should be HIGH (3)
        if '=' in token_map:
            self.assertEqual(token_map['='].importance, ImportanceLevel.HIGH)
        if '+' in token_map:
            self.assertEqual(token_map['+'].importance, ImportanceLevel.HIGH)

        # Built-in functions should be HIGH (3)
        if 'print' in token_map:
            self.assertEqual(token_map['print'].importance, ImportanceLevel.HIGH)

        # Identifiers should be MEDIUM (2)
        if 'example_function' in token_map:
            self.assertEqual(token_map['example_function'].importance, ImportanceLevel.MEDIUM)
        if 'x' in token_map:
            self.assertEqual(token_map['x'].importance, ImportanceLevel.MEDIUM)

        # Literals should be MEDIUM (2)
        if '42' in token_map:
            self.assertEqual(token_map['42'].importance, ImportanceLevel.MEDIUM)
        if '3.14' in token_map:
            self.assertEqual(token_map['3.14'].importance, ImportanceLevel.MEDIUM)

    def test_ascii_character_mapping_with_importance(self):
        """Test that ASCII characters are properly mapped based on importance levels."""
        source = "def test(): return 42 + x  # comment"
        tokens = tokenize_code(source)

        # Create mapping for easier testing
        token_map = {token.value: token for token in tokens if token.value.strip()}

        # CRITICAL importance (keywords) should map to '#'
        if 'def' in token_map:
            self.assertEqual(token_map['def'].ascii_char, '#')
        if 'return' in token_map:
            self.assertEqual(token_map['return'].ascii_char, '#')

        # HIGH importance (operators, built-ins) should map to '+'
        if '+' in token_map:
            self.assertEqual(token_map['+'].ascii_char, '+')

        # MEDIUM importance (identifiers, literals) should map to '-'
        if 'test' in token_map:
            self.assertEqual(token_map['test'].ascii_char, '-')
        if '42' in token_map:
            self.assertEqual(token_map['42'].ascii_char, '-')
        if 'x' in token_map:
            self.assertEqual(token_map['x'].ascii_char, '-')

        # LOW importance (comments) should map to '.'
        comment_tokens = [token for token in tokens if token.type == 'COMMENT']
        for token in comment_tokens:
            self.assertEqual(token.ascii_char, '.')

    def test_comprehensive_code_classification(self):
        """Test classification of a comprehensive code sample."""
        source = '''
@property
def calculate_area(self, radius: float) -> float:
    """Calculate circle area using math.pi."""
    import math
    area = math.pi * radius ** 2
    print(f"Area: {area}")
    return area
'''
        tokens = tokenize_code(source)

        # Count tokens by importance level
        importance_counts = {
            ImportanceLevel.CRITICAL: 0,
            ImportanceLevel.HIGH: 0,
            ImportanceLevel.MEDIUM: 0,
            ImportanceLevel.LOW: 0
        }

        for token in tokens:
            importance_counts[token.importance] += 1

        # Should have tokens at all importance levels
        self.assertGreater(importance_counts[ImportanceLevel.CRITICAL], 0, "Should have CRITICAL tokens")
        self.assertGreater(importance_counts[ImportanceLevel.HIGH], 0, "Should have HIGH tokens")
        self.assertGreater(importance_counts[ImportanceLevel.MEDIUM], 0, "Should have MEDIUM tokens")
        self.assertGreater(importance_counts[ImportanceLevel.LOW], 0, "Should have LOW tokens")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)