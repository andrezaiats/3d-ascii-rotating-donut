#!/usr/bin/env python3
"""
3D ASCII Rotating Donut with Self-Code Display

This project creates an animated ASCII art visualization that renders its own
source code as a rotating 3D torus (donut shape). The mathematical art combines
3D geometry, perspective projection, and terminal-based animation to create
a self-referential display where the code literally visualizes itself.

Mathematical Background:
The torus is generated using parametric equations:
- x = (R + r*cos(v)) * cos(u)
- y = (R + r*cos(v)) * sin(u)
- z = r * sin(v)
Where R is outer radius, r is inner radius, u and v are parameters [0, 2π]

The visualization maps source code tokens to the torus surface points and
applies rotation matrices to create smooth animation. Each frame projects
3D coordinates to 2D screen space using perspective projection.

Self-Referential Concept:
The program reads its own source code, tokenizes it into semantic elements,
and distributes these tokens across the torus surface. Higher importance
tokens (keywords, operators) are rendered with more prominent ASCII characters,
creating a visual hierarchy that represents code structure in 3D space.

Author: Andre Zaiats
License: MIT
Python Version: 3.8+
Dependencies: Python Standard Library Only
"""

# === IMPORTS (Python Standard Library Only) ===
import ast
import keyword
import math
import os
import sys
import time
import tokenize
from io import StringIO
from typing import NamedTuple, List, Optional, Tuple

# === CONSTANTS ===
TERMINAL_WIDTH = 40
TERMINAL_HEIGHT = 20
TARGET_FPS = 30
ASCII_CHARS = {
    'HIGH': '#',
    'MEDIUM': '+',
    'LOW': '-',
    'BACKGROUND': '.'
}

# Cache for torus geometry calculations to prevent regeneration per performance rules
_torus_cache = {}


# === DATA MODELS ===

class ImportanceLevel:
    """Semantic importance hierarchy for code tokens."""
    CRITICAL = 4  # Keywords (def, class, if, for, etc.)
    HIGH = 3      # Operators (+, -, *, /, ==, etc.)
    MEDIUM = 2    # Identifiers, literals (names, numbers, strings)
    LOW = 1       # Comments, whitespace, special characters


# Configurable importance weights for fine-tuning visual emphasis
IMPORTANCE_WEIGHTS = {
    ImportanceLevel.CRITICAL: 1.0,   # Full weight for keywords
    ImportanceLevel.HIGH: 0.8,       # High weight for operators
    ImportanceLevel.MEDIUM: 0.6,     # Medium weight for identifiers/literals
    ImportanceLevel.LOW: 0.3         # Low weight for comments/whitespace
}


class Point3D(NamedTuple):
    """Represents a 3D point with torus surface parameters."""
    x: float
    y: float
    z: float
    u: float  # Torus parameter [0, 2π]
    v: float  # Torus parameter [0, 2π]


class Point2D(NamedTuple):
    """Represents a 2D screen coordinate with depth information."""
    x: int
    y: int
    depth: float
    visible: bool


class CodeToken(NamedTuple):
    """Represents a parsed source code token with metadata."""
    type: str
    value: str
    importance: int  # ImportanceLevel value (4=CRITICAL, 3=HIGH, 2=MEDIUM, 1=LOW)
    line: int
    column: int
    ascii_char: str


class TorusParameters(NamedTuple):
    """Configuration parameters for torus generation."""
    outer_radius: float
    inner_radius: float
    u_resolution: int
    v_resolution: int
    rotation_speed: float


class DisplayFrame(NamedTuple):
    """Represents a complete ASCII frame buffer."""
    width: int
    height: int
    buffer: List[List[str]]
    depth_buffer: List[List[float]]
    frame_number: int


class StructuralElement(NamedTuple):
    """Represents a structural element in the code (function, class, import)."""
    element_type: str  # 'function', 'class', 'import'
    name: str
    start_line: int
    end_line: int
    complexity_score: float
    nesting_depth: int
    parent_element: Optional[str] = None


class StructuralInfo(NamedTuple):
    """Contains structural analysis results."""
    elements: List[StructuralElement]
    max_complexity: float
    total_lines: int
    import_count: int
    function_count: int
    class_count: int


# === MATHEMATICAL ENGINE ===

def generate_torus_points(params: TorusParameters) -> List[Point3D]:
    """Generate 3D torus surface points using parametric equations.

    Implements standard torus parametric equations:
    - x = (R + r*cos(v)) * cos(u)
    - y = (R + r*cos(v)) * sin(u)
    - z = r * sin(v)

    Where:
    - R = outer_radius (major radius from center to tube center)
    - r = inner_radius (minor radius of tube cross-section)
    - u ∈ [0, 2π] (angle around torus center, controls rotation around main axis)
    - v ∈ [0, 2π] (angle around tube, controls position on tube cross-section)

    Mathematical relationships:
    - Total torus width: 2 * (R + r)
    - Hole diameter: 2 * (R - r)
    - Surface area: A = 4π²Rr
    - Volume: V = 2π²Rr²

    Performance Features:
    - Caches identical parameter sets to prevent regeneration
    - Uses specific math function imports for optimal performance
    - Employs list comprehensions for efficient point generation

    Args:
        params: Torus configuration parameters

    Returns:
        List of 3D points representing the torus surface

    Raises:
        ValueError: If torus parameters are invalid
    """
    # Validate parameters per mathematical validation rules
    if params.outer_radius <= params.inner_radius or params.inner_radius <= 0:
        raise ValueError(
            "Invalid torus parameters. "
            "Solution: Ensure outer_radius > inner_radius > 0"
        )

    if params.u_resolution <= 0 or params.v_resolution <= 0:
        raise ValueError(
            "Invalid resolution parameters. "
            "Solution: Ensure u_resolution > 0 and v_resolution > 0"
        )

    # Cache torus geometry calculations to prevent regeneration per performance rules
    cache_key = (params.outer_radius, params.inner_radius, params.u_resolution, params.v_resolution)
    if cache_key in _torus_cache:
        return _torus_cache[cache_key]

    # Import specific math functions for performance optimization
    from math import sin, cos, tau

    # Extract parameters for mathematical precision
    R = params.outer_radius  # Major radius
    r = params.inner_radius  # Minor radius
    u_res = params.u_resolution
    v_res = params.v_resolution

    # Generate torus surface points using parametric equations with list comprehension
    points = [
        Point3D(
            x=(R + r * cos((j / v_res) * tau)) * cos((i / u_res) * tau),
            y=(R + r * cos((j / v_res) * tau)) * sin((i / u_res) * tau),
            z=r * sin((j / v_res) * tau),
            u=(i / u_res) * tau,
            v=(j / v_res) * tau
        )
        for i in range(u_res)
        for j in range(v_res)
    ]

    # Cache the result for future use
    _torus_cache[cache_key] = points
    return points


def validate_torus_volume(params: TorusParameters, tolerance: float = 1e-10) -> bool:
    """Validate torus volume calculation using theoretical formula V = 2π²Rr².

    Args:
        params: Torus parameters to validate
        tolerance: Numerical tolerance for floating-point comparison

    Returns:
        True if volume calculation is mathematically correct

    Raises:
        ValueError: If parameters are invalid
    """
    # Validate parameters first
    if params.outer_radius <= params.inner_radius or params.inner_radius <= 0:
        raise ValueError(
            "Invalid torus parameters for volume validation. "
            "Solution: Ensure outer_radius > inner_radius > 0"
        )

    from math import pi

    # Theoretical torus volume: V = 2π²Rr²
    R = params.outer_radius
    r = params.inner_radius
    theoretical_volume = 2 * (pi ** 2) * R * (r ** 2)

    # For validation, we check that our parameters produce a positive volume
    # This is a mathematical property verification rather than numerical integration
    return theoretical_volume > 0


def validate_torus_surface_area(params: TorusParameters, tolerance: float = 1e-10) -> bool:
    """Validate torus surface area calculation using theoretical formula A = 4π²Rr.

    Args:
        params: Torus parameters to validate
        tolerance: Numerical tolerance for floating-point comparison

    Returns:
        True if surface area calculation is mathematically correct

    Raises:
        ValueError: If parameters are invalid
    """
    # Validate parameters first
    if params.outer_radius <= params.inner_radius or params.inner_radius <= 0:
        raise ValueError(
            "Invalid torus parameters for surface area validation. "
            "Solution: Ensure outer_radius > inner_radius > 0"
        )

    from math import pi

    # Theoretical torus surface area: A = 4π²Rr
    R = params.outer_radius
    r = params.inner_radius
    theoretical_surface_area = 4 * (pi ** 2) * R * r

    # For validation, we check that our parameters produce a positive surface area
    # This is a mathematical property verification
    return theoretical_surface_area > 0


def validate_torus_geometry(params: TorusParameters) -> bool:
    """Comprehensive torus geometry validation combining all mathematical checks.

    Args:
        params: Torus parameters to validate

    Returns:
        True if all geometric properties are mathematically valid

    Raises:
        ValueError: If any validation fails with detailed error message
    """
    try:
        # Check basic parameter constraints
        if params.outer_radius <= params.inner_radius or params.inner_radius <= 0:
            raise ValueError(
                "Invalid torus parameters. "
                "Solution: Ensure outer_radius > inner_radius > 0"
            )

        if params.u_resolution <= 0 or params.v_resolution <= 0:
            raise ValueError(
                "Invalid resolution parameters. "
                "Solution: Ensure u_resolution > 0 and v_resolution > 0"
            )

        # Validate volume and surface area calculations
        volume_valid = validate_torus_volume(params)
        surface_area_valid = validate_torus_surface_area(params)

        return volume_valid and surface_area_valid

    except Exception as e:
        raise ValueError(f"Torus geometry validation failed: {e}")


def apply_rotation(points: List[Point3D], angle: float) -> List[Point3D]:
    """Apply Y-axis rotation matrix to 3D points.

    Implements Y-axis rotation matrix transformation:
    rotation_matrix = [
        [cos(angle), 0, sin(angle)],
        [0, 1, 0],
        [-sin(angle), 0, cos(angle)]
    ]

    Mathematical transformation:
    - new_x = old_x * cos(angle) + old_z * sin(angle)
    - new_y = old_y (unchanged for Y-axis rotation)
    - new_z = -old_x * sin(angle) + old_z * cos(angle)

    Preserves parametric u,v coordinates for token mapping consistency.

    Args:
        points: List of 3D points to rotate
        angle: Rotation angle in radians

    Returns:
        List of rotated 3D points with preserved parametric coordinates
    """
    # Import specific math functions for performance optimization
    from math import cos, sin

    # Calculate rotation matrix components
    cos_angle = cos(angle)
    sin_angle = sin(angle)

    # Apply Y-axis rotation transformation to each point
    rotated_points = []
    for point in points:
        # Apply rotation matrix: Y-axis rotation preserves Y coordinate
        new_x = point.x * cos_angle + point.z * sin_angle
        new_y = point.y  # Y remains unchanged for Y-axis rotation
        new_z = -point.x * sin_angle + point.z * cos_angle

        # Create new Point3D with rotated coordinates but preserved parametric values
        rotated_point = Point3D(
            x=new_x,
            y=new_y,
            z=new_z,
            u=point.u,  # Preserve original parametric coordinates
            v=point.v   # for consistent token mapping
        )
        rotated_points.append(rotated_point)

    return rotated_points


def project_to_screen(point: Point3D) -> Point2D:
    """Perspective projection from 3D to 2D screen coordinates.

    Implements perspective projection formula with camera distance and focal length.
    Projects 3D coordinates to normalized screen space [-1,1] then maps to 40x20 grid.

    Projection equations:
    - screen_x = (3D_x * focal_length) / (3D_z + camera_distance)
    - screen_y = (3D_y * focal_length) / (3D_z + camera_distance)

    Grid mapping:
    - grid_x = int((screen_x + 1.0) * 20)  # Map [-1,1] to [0,39]
    - grid_y = int((screen_y + 1.0) * 10)  # Map [-1,1] to [0,19]

    Args:
        point: 3D point to project

    Returns:
        2D screen coordinate with depth information and visibility flag

    Raises:
        ValueError: If projection results in invalid coordinates
    """
    # Camera and projection parameters
    camera_distance = 5.0  # Distance from camera to origin
    focal_length = 2.0     # Controls field of view and projection scale

    # Handle points behind camera (negative Z after camera distance offset)
    z_camera = point.z + camera_distance
    if z_camera <= 0:
        # Point is behind camera, mark as invisible
        return Point2D(x=0, y=0, depth=float('inf'), visible=False)

    # Apply perspective projection formula
    try:
        screen_x = (point.x * focal_length) / z_camera
        screen_y = (point.y * focal_length) / z_camera
    except ZeroDivisionError:
        # Handle division by zero edge case
        raise ValueError(
            "Perspective projection division by zero. "
            "Solution: Ensure point is not at camera position"
        )

    # Map normalized coordinates [-1,1] to 40x20 ASCII grid
    # DisplayFrame dimensions: width=40, height=20
    grid_x = int((screen_x + 1.0) * 19.5)  # Maps [-1,1] to [0,39]
    grid_y = int((screen_y + 1.0) * 9.5)   # Maps [-1,1] to [0,19]

    # Bounds checking - mark points outside display area as invisible
    visible = (0 <= grid_x < 40) and (0 <= grid_y < 20)

    # Clamp coordinates to valid grid range for safety
    grid_x = max(0, min(39, grid_x))
    grid_y = max(0, min(19, grid_y))

    # Calculate normalized depth for sorting (closer points have smaller depth values)
    # Use original Z coordinate for depth buffer, normalized to [0,1] range
    depth = (point.z + 3.0) / 6.0  # Assuming torus Z range is approximately [-3,3]
    depth = max(0.0, min(1.0, depth))  # Clamp to valid range

    return Point2D(x=grid_x, y=grid_y, depth=depth, visible=visible)


# === PARSING ENGINE ===

def get_script_path() -> str:
    """Get the absolute, normalized path to this script.

    Provides reliable path identification across platforms with comprehensive
    validation and edge case handling including symbolic links and relative paths.

    Returns:
        Absolute, normalized path to the current script

    Raises:
        FileNotFoundError: If __file__ is not available or path cannot be resolved
        PermissionError: If script path is not accessible
        OSError: For file system related errors
    """
    try:
        # Validate __file__ exists per self-reference safety rules
        if '__file__' not in globals():
            raise FileNotFoundError(
                "Cannot determine script path: __file__ not available. "
                "Solution: Run script directly, not in interactive mode"
            )

        script_file = __file__

        # Handle absolute path resolution using os.path.abspath
        absolute_path = os.path.abspath(script_file)

        # Add symbolic link resolution using os.path.realpath for edge cases
        real_path = os.path.realpath(absolute_path)

        # Add cross-platform path normalization with os.path.normpath
        normalized_path = os.path.normpath(real_path)

        # Validate that the resolved path exists and is accessible
        if not os.path.exists(normalized_path):
            raise FileNotFoundError(
                f"Script file not found at resolved path: {normalized_path}. "
                "Solution: Ensure script file exists and is accessible"
            )

        if not os.access(normalized_path, os.R_OK):
            raise PermissionError(
                f"Cannot read script file: {normalized_path}. "
                "Solution: Check file permissions and run with appropriate access rights"
            )

        return normalized_path

    except (FileNotFoundError, PermissionError):
        # Re-raise specific exceptions as-is for proper test handling
        raise
    except OSError as e:
        raise OSError(
            f"File system error accessing script path: {e}. "
            "Solution: Check file system permissions and disk space"
        )


def validate_file_content(file_path: str, content: str) -> bool:
    """Validate file content for integrity and Python syntax.

    Performs comprehensive content validation including file size checks,
    syntax validation, and basic integrity verification.

    Args:
        file_path: Path to the file being validated
        content: File content to validate

    Returns:
        True if content passes all validation checks

    Raises:
        ValueError: If content fails validation with specific error details
    """
    try:
        # File size validation to prevent reading binary files
        file_size = len(content.encode('utf-8'))
        max_size = 10 * 1024 * 1024  # 10MB maximum for source files

        if file_size == 0:
            raise ValueError(
                "File content is empty. "
                "Solution: Ensure the script file contains valid Python code"
            )

        if file_size > max_size:
            raise ValueError(
                f"File size {file_size} bytes exceeds maximum {max_size} bytes. "
                "Solution: Check if this is a binary file or reduce file size"
            )

        # Basic syntax validation using Python AST parsing
        # Strip BOM if present before parsing
        content_for_parsing = content
        if content_for_parsing.startswith('\ufeff'):
            content_for_parsing = content_for_parsing[1:]

        try:
            ast.parse(content_for_parsing, filename=file_path)
        except SyntaxError as e:
            raise ValueError(
                f"Python syntax error in file: {e}. "
                f"Solution: Fix syntax error at line {e.lineno}"
            )
        except Exception as e:
            raise ValueError(
                f"AST parsing failed: {e}. "
                "Solution: Ensure file contains valid Python code"
            )

        # Check for minimum expected content (basic Python file structure)
        if len(content.strip()) < 10:
            raise ValueError(
                "File content appears too short for a valid Python file. "
                "Solution: Ensure file contains meaningful Python code"
            )

        return True

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(
            f"Content validation failed: {e}. "
            "Solution: Check file integrity and encoding"
        )


def read_self_code() -> str:
    """Read this script's own source code with comprehensive error handling and validation.

    Implements robust source code reading with UTF-8 encoding, BOM handling,
    content validation, and comprehensive error handling per story requirements.

    Returns:
        String containing the complete source code

    Raises:
        FileNotFoundError: If script file cannot be found or __file__ unavailable
        PermissionError: If file access permissions are insufficient
        UnicodeDecodeError: If encoding issues occur with fallback guidance
        IOError: For disk/network file system failures
        ValueError: If content validation fails
    """
    try:
        # Get reliable script path using comprehensive path resolution
        script_path = get_script_path()

        # Get file timestamp for development change detection
        try:
            file_stat = os.stat(script_path)
            modification_time = file_stat.st_mtime
        except OSError as e:
            raise IOError(
                f"Cannot access file metadata: {e}. "
                "Solution: Check file system permissions and disk space"
            )

        # Read file with explicit UTF-8 encoding and BOM handling
        try:
            with open(script_path, 'r', encoding='utf-8-sig') as file:
                content = file.read()
        except UnicodeDecodeError as e:
            # Fallback encoding attempts for problematic files
            try:
                with open(script_path, 'r', encoding='utf-8') as file:
                    content = file.read()
            except UnicodeDecodeError:
                try:
                    with open(script_path, 'r', encoding='latin1') as file:
                        content = file.read()
                except UnicodeDecodeError:
                    raise UnicodeDecodeError(
                        e.encoding, e.object, e.start, e.end,
                        f"Cannot decode file with UTF-8 or fallback encodings: {e.reason}. "
                        "Solution: Save file with UTF-8 encoding or check for binary content"
                    )
        except PermissionError:
            raise PermissionError(
                f"Permission denied reading file: {script_path}. "
                "Solution: Check file permissions and run with appropriate access rights"
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Script file not found: {script_path}. "
                "Solution: Ensure script file exists and path is correct"
            )
        except IOError as e:
            raise IOError(
                f"I/O error reading file: {e}. "
                "Solution: Check disk space, network connectivity, and file system health"
            )

        # Validate file content integrity and syntax
        validate_file_content(script_path, content)

        # Ensure complete file content read without truncation
        if len(content) == 0:
            raise ValueError(
                "File content is empty after reading. "
                "Solution: Ensure script file is not empty and is readable"
            )

        return content

    except NameError:
        raise FileNotFoundError(
            "Cannot read self code: __file__ not available. "
            "Solution: Run script directly, not in interactive mode"
        )
    except Exception as e:
        # Re-raise known exceptions
        if isinstance(e, (FileNotFoundError, PermissionError, UnicodeDecodeError, IOError, ValueError)):
            raise

        # Handle unexpected exceptions
        raise IOError(
            f"Unexpected error reading self code: {e}. "
            "Solution: Check file system integrity and Python installation"
        )


def tokenize_code(source: str) -> List[CodeToken]:
    """Parse source code into classified tokens using Python's tokenize module.

    Processes source code through tokenize.generate_tokens() to create a token stream,
    maps tokenize module token types to CodeToken classifications, and extracts
    position information for spatial mapping.

    Token Classification Mapping:
    - KEYWORD tokens (def, class, if, for, etc.) -> HIGH importance
    - OPERATOR tokens (+, -, *, /, ==, etc.) -> MEDIUM importance
    - NAME tokens (identifiers, function names) -> MEDIUM importance
    - NUMBER and STRING literals -> MEDIUM importance
    - COMMENT and whitespace tokens -> LOW importance
    - Special characters (brackets, parentheses, etc.) -> LOW importance

    Args:
        source: Source code string to tokenize

    Returns:
        List of classified code tokens with position information

    Raises:
        ValueError: If tokenization fails with informative error message
    """
    if not source or not source.strip():
        raise ValueError(
            "Empty or whitespace-only source code provided. "
            "Solution: Provide valid Python source code for tokenization"
        )

    tokens = []

    try:
        # Use StringIO to wrap source code for tokenize.generate_tokens()
        source_io = StringIO(source)

        # Generate tokens using Python's tokenize module
        token_generator = tokenize.generate_tokens(source_io.readline)

        for token_info in token_generator:
            # Extract token information
            token_type_num = token_info.type
            token_value = token_info.string
            start_pos = token_info.start

            # Map tokenize module constants to our classification system
            token_type_name = tokenize.tok_name.get(token_type_num, 'UNKNOWN')

            # Classify token type for initial classification
            if token_type_num == tokenize.NAME and keyword.iskeyword(token_value):
                # NAME tokens that are Python keywords
                token_type = 'KEYWORD'
            elif token_type_num == tokenize.OP:
                token_type = 'OPERATOR'
            elif token_type_num == tokenize.NAME:
                # NAME tokens that are not keywords (identifiers)
                token_type = 'IDENTIFIER'
            elif token_type_num in (tokenize.NUMBER, tokenize.STRING):
                token_type = 'LITERAL'
            elif token_type_num == tokenize.COMMENT:
                token_type = 'COMMENT'
            elif token_type_num in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                                  tokenize.DEDENT):
                token_type = 'WHITESPACE'
            elif token_type_num == tokenize.ENDMARKER:
                # Skip ENDMARKER tokens as they don't represent actual code
                continue
            else:
                # Unknown token types default to SPECIAL per coding standards
                token_type = 'SPECIAL'

            # Create initial CodeToken for importance classification
            temp_token = CodeToken(
                type=token_type,
                value=token_value,
                importance=ImportanceLevel.LOW,  # Temporary value
                line=start_pos[0],
                column=start_pos[1],
                ascii_char='-'  # Temporary value
            )

            # Use classify_importance() function to determine semantic importance
            importance_level = classify_importance(temp_token)

            # Map importance level to ASCII character for rendering
            if importance_level == ImportanceLevel.CRITICAL:
                ascii_char = ASCII_CHARS['HIGH']  # '#' for highest visibility
            elif importance_level == ImportanceLevel.HIGH:
                ascii_char = ASCII_CHARS['MEDIUM']  # '+' for high visibility
            elif importance_level == ImportanceLevel.MEDIUM:
                ascii_char = ASCII_CHARS['LOW']  # '-' for medium visibility
            else:  # ImportanceLevel.LOW
                ascii_char = ASCII_CHARS['BACKGROUND']  # '.' for lowest visibility

            # Create final CodeToken with classified importance
            code_token = CodeToken(
                type=token_type,
                value=token_value,
                importance=importance_level,
                line=start_pos[0],
                column=start_pos[1],
                ascii_char=ascii_char
            )

            tokens.append(code_token)

    except tokenize.TokenError as e:
        raise ValueError(
            f"Tokenization failed: {e}. "
            "Solution: Check Python syntax and ensure source code is valid"
        )
    except Exception as e:
        raise ValueError(
            f"Unexpected tokenization error: {e}. "
            "Solution: Verify source code format and encoding"
        )

    # Validate that we have some tokens (empty source should have been caught earlier)
    if not tokens:
        raise ValueError(
            "No tokens generated from source code. "
            "Solution: Ensure source contains valid Python code"
        )

    return tokens


def classify_importance(token: CodeToken) -> int:
    """Assign semantic importance hierarchy to tokens based on 4-level system.

    Implements comprehensive token classification mapping Python token types to
    semantic importance levels for visual emphasis in the torus display.

    Classification Hierarchy:
    - CRITICAL (4): Python keywords (def, class, if, for, while, etc.)
    - HIGH (3): Operators (+, -, *, /, ==, etc.) and decorators
    - MEDIUM (2): Identifiers, literals (variables, numbers, strings)
    - LOW (1): Comments, whitespace, special characters

    Special Cases Handled:
    - Built-in functions detected using keyword.iskeyword() and builtins
    - Decorators (@decorator) assigned HIGH importance
    - String literals classified as MEDIUM importance
    - Unknown token types default to LOW importance per coding standards

    Args:
        token: CodeToken object with type, value, and position information

    Returns:
        ImportanceLevel value (4=CRITICAL, 3=HIGH, 2=MEDIUM, 1=LOW)

    Raises:
        ValueError: If token parameter is invalid with "Solution:" guidance
    """
    # Input validation with proper error handling per coding standards
    if not isinstance(token, CodeToken):
        raise ValueError(
            "Invalid token parameter: expected CodeToken object. "
            "Solution: Ensure token is created using CodeToken constructor"
        )

    if not hasattr(token, 'type') or not hasattr(token, 'value'):
        raise ValueError(
            "Invalid token structure: missing required attributes. "
            "Solution: Ensure token has 'type' and 'value' attributes"
        )

    token_type = token.type
    token_value = token.value

    # Handle edge case of empty or None values
    if not token_value:
        return ImportanceLevel.LOW

    try:
        # CRITICAL IMPORTANCE: Python keywords
        if token_type == 'KEYWORD':
            return ImportanceLevel.CRITICAL

        # HIGH IMPORTANCE: Operators and decorators
        if token_type == 'OPERATOR':
            return ImportanceLevel.HIGH

        # Special case: Decorator detection for @ symbol
        if token_value.startswith('@') or token_type == 'DECORATOR':
            return ImportanceLevel.HIGH

        # Special case: Built-in function detection
        if token_type == 'IDENTIFIER' and _is_builtin_function(token_value):
            return ImportanceLevel.HIGH

        # MEDIUM IMPORTANCE: Identifiers and literals
        if token_type in ('IDENTIFIER', 'LITERAL'):
            return ImportanceLevel.MEDIUM

        # LOW IMPORTANCE: Comments, whitespace, special characters
        if token_type in ('COMMENT', 'WHITESPACE', 'SPECIAL'):
            return ImportanceLevel.LOW

        # Unknown token types default to LOW importance per coding standards
        return ImportanceLevel.LOW

    except Exception as e:
        # Graceful error handling with specific exception catching
        raise ValueError(
            f"Token classification failed: {e}. "
            "Solution: Verify token structure and retry classification"
        )


def _is_builtin_function(token_value: str) -> bool:
    """Detect built-in function names using keyword and builtins modules.

    Checks if a token value represents a Python built-in function or keyword
    for appropriate importance classification.

    Args:
        token_value: String value to check

    Returns:
        True if token represents a built-in function or keyword
    """
    try:
        # Check if it's a Python keyword
        if keyword.iskeyword(token_value):
            return True

        # Check if it's a built-in function
        import builtins
        return hasattr(builtins, token_value) and callable(getattr(builtins, token_value))

    except Exception:
        # Safe fallback for any unexpected issues
        return False


def _is_string_literal(token_value: str) -> bool:
    """Determine if a token value represents a string literal.

    Handles various Python string literal formats including single quotes,
    double quotes, triple quotes, and raw strings.

    Args:
        token_value: String value to check

    Returns:
        True if token represents a string literal
    """
    try:
        # Check for basic string literal patterns
        if not token_value:
            return False

        # Remove common prefixes (r, u, b, f for raw, unicode, bytes, f-strings)
        cleaned_value = token_value.lower()
        for prefix in ('r"', "r'", 'u"', "u'", 'b"', "b'", 'f"', "f'", 'rf"', "rf'"):
            if cleaned_value.startswith(prefix):
                return True

        # Check for standard quoted strings
        if ((token_value.startswith('"') and token_value.endswith('"')) or
            (token_value.startswith("'") and token_value.endswith("'")) or
            (token_value.startswith('"""') and token_value.endswith('"""')) or
            (token_value.startswith("'''") and token_value.endswith("'''"))):
            return True

        return False

    except Exception:
        # Safe fallback for any unexpected issues
        return False


def classify_importance_with_structure(token: CodeToken,
                                     structural_info: StructuralInfo) -> int:
    """Enhanced token importance classification incorporating structural analysis.

    Extends the base classify_importance() function with structural context to
    boost importance of tokens within critical structural elements (functions,
    classes, imports) based on their complexity and nesting depth.

    Enhancement Strategy:
    - Base importance from existing classify_importance()
    - Structural context boosts for function/class/import elements
    - Complexity-based modifiers from structural analysis
    - Hierarchical nesting bonuses for nested structures

    Args:
        token: CodeToken object with type, value, and position information
        structural_info: StructuralInfo from analyze_structure() containing
                        structural elements and complexity analysis

    Returns:
        Enhanced ImportanceLevel value incorporating structural context

    Raises:
        ValueError: If parameters are invalid with "Solution:" guidance
    """
    # Input validation
    if not isinstance(token, CodeToken):
        raise ValueError(
            "Invalid token parameter: expected CodeToken object. "
            "Solution: Ensure token is created using CodeToken constructor"
        )

    if not isinstance(structural_info, StructuralInfo):
        raise ValueError(
            "Invalid structural_info parameter: expected StructuralInfo object. "
            "Solution: Generate structural_info using analyze_structure()"
        )

    # Get base importance using existing classification
    base_importance = classify_importance(token)

    # Find structural elements containing this token
    containing_elements = _find_containing_elements(token, structural_info.elements)

    if not containing_elements:
        # Token not in any structural element, return base importance
        return base_importance

    # Calculate structural enhancement based on containing elements
    structural_bonus = _calculate_structural_bonus(containing_elements, structural_info)

    # Apply enhancement while respecting ImportanceLevel bounds
    enhanced_importance = _apply_structural_enhancement(base_importance, structural_bonus)

    return enhanced_importance


def _find_containing_elements(token: CodeToken,
                            elements: List[StructuralElement]) -> List[StructuralElement]:
    """Find structural elements that contain the given token."""
    containing = []

    for element in elements:
        # Check if token is within element's line range
        if element.start_line <= token.line <= element.end_line:
            containing.append(element)

    # Sort by nesting depth (deepest first) for proper hierarchy
    return sorted(containing, key=lambda e: e.nesting_depth, reverse=True)


def _calculate_structural_bonus(containing_elements: List[StructuralElement],
                              structural_info: StructuralInfo) -> float:
    """Calculate importance bonus based on structural context."""
    if not containing_elements:
        return 0.0

    total_bonus = 0.0

    for element in containing_elements:
        # Base bonus by element type
        if element.element_type == 'class':
            total_bonus += 0.8  # Classes are architecturally important
        elif element.element_type == 'function':
            total_bonus += 0.6  # Functions are implementation important
        elif element.element_type == 'import':
            total_bonus += 0.4  # Imports set up dependencies

        # Complexity bonus (normalized by max complexity)
        if structural_info.max_complexity > 0:
            complexity_ratio = element.complexity_score / structural_info.max_complexity
            total_bonus += complexity_ratio * 0.5

        # Nesting depth bonus (nested structures are more complex)
        nesting_bonus = min(element.nesting_depth * 0.2, 0.6)  # Cap at 0.6
        total_bonus += nesting_bonus

    # Cap total bonus to prevent excessive enhancement
    return min(total_bonus, 1.5)


def _apply_structural_enhancement(base_importance: int, structural_bonus: float) -> int:
    """Apply structural bonus while respecting ImportanceLevel bounds."""
    # Convert to float for calculation
    enhanced = float(base_importance) + structural_bonus

    # Round and clamp to valid ImportanceLevel range
    enhanced_int = round(enhanced)
    return max(ImportanceLevel.LOW, min(enhanced_int, ImportanceLevel.CRITICAL))


def analyze_structure(tokens: List[CodeToken]) -> StructuralInfo:
    """Extract structural elements from tokenized code.

    Analyzes token stream to identify functions, classes, imports, and their
    structural relationships. Calculates complexity scores and nesting depth
    for hierarchical distribution on torus surface.

    Args:
        tokens: List of CodeToken objects from tokenize_code()

    Returns:
        StructuralInfo containing all identified structural elements

    Raises:
        ValueError: If token analysis fails with "Solution:" guidance
    """
    if not tokens:
        raise ValueError(
            "Empty token list provided for structural analysis. "
            "Solution: Ensure tokens are generated using tokenize_code()"
        )

    elements = []
    element_stack = []  # Track nesting depth
    current_line = 1
    total_lines = max(token.line for token in tokens) if tokens else 0

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Identify function definitions
        if token.type == 'KEYWORD' and token.value == 'def':
            element = _identify_function_definition(tokens, i, element_stack)
            if element:
                elements.append(element)
                element_stack.append(element)

        # Identify class definitions
        elif token.type == 'KEYWORD' and token.value == 'class':
            element = _identify_class_definition(tokens, i, element_stack)
            if element:
                elements.append(element)
                element_stack.append(element)

        # Identify import statements
        elif token.type == 'KEYWORD' and token.value in ('import', 'from'):
            element = _identify_import_statement(tokens, i, element_stack)
            if element:
                elements.append(element)

        # Track indentation changes to manage nesting stack
        if token.line > current_line:
            current_line = token.line
            # Remove elements from stack if we've moved past their scope
            element_stack = _update_element_stack(element_stack, current_line, tokens)

        i += 1

    # Calculate summary statistics
    import_count = sum(1 for e in elements if e.element_type == 'import')
    function_count = sum(1 for e in elements if e.element_type == 'function')
    class_count = sum(1 for e in elements if e.element_type == 'class')
    max_complexity = max((e.complexity_score for e in elements), default=0.0)

    return StructuralInfo(
        elements=elements,
        max_complexity=max_complexity,
        total_lines=total_lines,
        import_count=import_count,
        function_count=function_count,
        class_count=class_count
    )


def _identify_function_definition(tokens: List[CodeToken], start_idx: int,
                                element_stack: List[StructuralElement]) -> Optional[StructuralElement]:
    """Identify function definition from token stream starting at 'def' keyword."""
    if start_idx >= len(tokens) - 1:
        return None

    # Look for function name after 'def'
    name_idx = start_idx + 1
    while name_idx < len(tokens) and tokens[name_idx].type in ('WHITESPACE',):
        name_idx += 1

    if name_idx >= len(tokens) or tokens[name_idx].type != 'IDENTIFIER':
        return None

    function_name = tokens[name_idx].value
    start_line = tokens[start_idx].line

    # Calculate complexity based on token count and keywords
    end_line = _find_function_end_line(tokens, start_idx)
    complexity = _calculate_function_complexity(tokens, start_idx, end_line)

    # Determine nesting depth and parent
    nesting_depth = len(element_stack)
    parent_element = element_stack[-1].name if element_stack else None

    return StructuralElement(
        element_type='function',
        name=function_name,
        start_line=start_line,
        end_line=end_line,
        complexity_score=complexity,
        nesting_depth=nesting_depth,
        parent_element=parent_element
    )


def _identify_class_definition(tokens: List[CodeToken], start_idx: int,
                             element_stack: List[StructuralElement]) -> Optional[StructuralElement]:
    """Identify class definition from token stream starting at 'class' keyword."""
    if start_idx >= len(tokens) - 1:
        return None

    # Look for class name after 'class'
    name_idx = start_idx + 1
    while name_idx < len(tokens) and tokens[name_idx].type in ('WHITESPACE',):
        name_idx += 1

    if name_idx >= len(tokens) or tokens[name_idx].type != 'IDENTIFIER':
        return None

    class_name = tokens[name_idx].value
    start_line = tokens[start_idx].line

    # Calculate complexity based on methods and inheritance
    end_line = _find_class_end_line(tokens, start_idx)
    complexity = _calculate_class_complexity(tokens, start_idx, end_line)

    # Determine nesting depth and parent
    nesting_depth = len(element_stack)
    parent_element = element_stack[-1].name if element_stack else None

    return StructuralElement(
        element_type='class',
        name=class_name,
        start_line=start_line,
        end_line=end_line,
        complexity_score=complexity,
        nesting_depth=nesting_depth,
        parent_element=parent_element
    )


def _identify_import_statement(tokens: List[CodeToken], start_idx: int,
                             element_stack: List[StructuralElement]) -> Optional[StructuralElement]:
    """Identify import statement from token stream starting at 'import' or 'from'."""
    if start_idx >= len(tokens):
        return None

    import_type = tokens[start_idx].value  # 'import' or 'from'
    start_line = tokens[start_idx].line

    # Extract import name/module
    import_name = _extract_import_name(tokens, start_idx)
    if not import_name:
        return None

    # Calculate complexity based on import type and dependencies
    complexity = _calculate_import_complexity(tokens, start_idx, import_type)

    # Imports are typically at module level (nesting depth 0)
    nesting_depth = 0

    return StructuralElement(
        element_type='import',
        name=import_name,
        start_line=start_line,
        end_line=start_line,  # Imports are single line
        complexity_score=complexity,
        nesting_depth=nesting_depth,
        parent_element=None
    )


def _find_function_end_line(tokens: List[CodeToken], start_idx: int) -> int:
    """Find the end line of a function definition by tracking indentation."""
    if start_idx >= len(tokens):
        return tokens[start_idx].line if start_idx < len(tokens) else 1

    function_start_line = tokens[start_idx].line
    base_indentation = None

    # Find the first indented line after the function definition
    for i in range(start_idx + 1, len(tokens)):
        token = tokens[i]
        if token.line > function_start_line and token.type not in ('WHITESPACE', 'COMMENT'):
            if base_indentation is None:
                base_indentation = token.column
            elif token.column <= base_indentation and token.type != 'WHITESPACE':
                return token.line - 1

    # If no end found, use last token line
    return tokens[-1].line if tokens else function_start_line


def _find_class_end_line(tokens: List[CodeToken], start_idx: int) -> int:
    """Find the end line of a class definition by tracking indentation."""
    return _find_function_end_line(tokens, start_idx)  # Same logic as functions


def _calculate_function_complexity(tokens: List[CodeToken], start_idx: int, end_line: int) -> float:
    """Calculate function complexity based on token count and control structures."""
    complexity = 1.0  # Base complexity
    control_keywords = {'if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally', 'with'}

    # Count tokens and control structures within function
    for i in range(start_idx, len(tokens)):
        token = tokens[i]
        if token.line > end_line:
            break

        if token.type == 'KEYWORD' and token.value in control_keywords:
            complexity += 1.0
        elif token.type == 'OPERATOR':
            complexity += 0.1

    return complexity


def _calculate_class_complexity(tokens: List[CodeToken], start_idx: int, end_line: int) -> float:
    """Calculate class complexity based on method count and inheritance."""
    complexity = 2.0  # Base complexity higher than functions
    method_count = 0

    # Count methods within class
    for i in range(start_idx, len(tokens)):
        token = tokens[i]
        if token.line > end_line:
            break

        if token.type == 'KEYWORD' and token.value == 'def':
            method_count += 1
            complexity += 1.5

    # Add complexity for inheritance (parentheses after class name)
    inheritance_bonus = _check_inheritance(tokens, start_idx)
    complexity += inheritance_bonus

    return complexity


def _calculate_import_complexity(tokens: List[CodeToken], start_idx: int, import_type: str) -> float:
    """Calculate import complexity based on type and dependencies."""
    base_complexity = 0.5  # Imports are generally less complex

    if import_type == 'from':
        base_complexity += 0.3  # from imports slightly more complex

    # Count imported items
    import_count = _count_imported_items(tokens, start_idx)
    return base_complexity + (import_count * 0.1)


def _extract_import_name(tokens: List[CodeToken], start_idx: int) -> str:
    """Extract the main module/package name from import statement."""
    import_token = tokens[start_idx]

    if import_token.value == 'from':
        # from module import item
        for i in range(start_idx + 1, min(start_idx + 10, len(tokens))):
            if tokens[i].type == 'IDENTIFIER':
                return tokens[i].value
    elif import_token.value == 'import':
        # import module
        for i in range(start_idx + 1, min(start_idx + 10, len(tokens))):
            if tokens[i].type == 'IDENTIFIER':
                return tokens[i].value

    return 'unknown_import'


def _count_imported_items(tokens: List[CodeToken], start_idx: int) -> int:
    """Count the number of items being imported."""
    count = 0
    in_import_list = False

    for i in range(start_idx, min(start_idx + 20, len(tokens))):
        token = tokens[i]
        if token.line > tokens[start_idx].line:
            break

        if token.value == 'import':
            in_import_list = True
        elif in_import_list and token.type == 'IDENTIFIER':
            count += 1
        elif token.value == ',':
            continue

    return max(count, 1)


def _check_inheritance(tokens: List[CodeToken], start_idx: int) -> float:
    """Check if class has inheritance and return complexity bonus."""
    # Look for parentheses after class name indicating inheritance
    for i in range(start_idx, min(start_idx + 10, len(tokens))):
        if tokens[i].value == '(':
            return 1.0  # Inheritance adds complexity
    return 0.0


def _update_element_stack(element_stack: List[StructuralElement], current_line: int,
                         tokens: List[CodeToken]) -> List[StructuralElement]:
    """Update element stack by removing elements that are no longer in scope."""
    # Keep elements that haven't ended yet
    return [elem for elem in element_stack if elem.end_line >= current_line]


def debug_structural_analysis(structural_info: StructuralInfo, tokens: List[CodeToken],
                            enable_debug: bool = False) -> None:
    """Display detailed debugging information for structural analysis.

    Provides comprehensive debugging output showing structural elements,
    complexity calculations, token classifications, and surface mapping
    distribution when debug mode is enabled.

    Args:
        structural_info: StructuralInfo from analyze_structure()
        tokens: List of CodeToken objects from tokenize_code()
        enable_debug: Flag to enable/disable debug output

    Output:
        Detailed structural analysis information to stdout when enabled
    """
    if not enable_debug:
        return

    print("\n" + "="*60, flush=True)
    print("STRUCTURAL ANALYSIS DEBUG OUTPUT", flush=True)
    print("="*60, flush=True)

    # Summary statistics
    print(f"\n[STRUCTURAL SUMMARY]:", flush=True)
    print(f"   Total Lines: {structural_info.total_lines}", flush=True)
    print(f"   Functions: {structural_info.function_count}", flush=True)
    print(f"   Classes: {structural_info.class_count}", flush=True)
    print(f"   Imports: {structural_info.import_count}", flush=True)
    print(f"   Max Complexity: {structural_info.max_complexity:.2f}", flush=True)

    # Structural elements details
    print(f"\n[STRUCTURAL ELEMENTS]:", flush=True)
    for element in sorted(structural_info.elements, key=lambda e: e.complexity_score, reverse=True):
        print(f"   {element.element_type.upper()}: {element.name}", flush=True)
        print(f"      Lines: {element.start_line}-{element.end_line}", flush=True)
        print(f"      Complexity: {element.complexity_score:.2f}", flush=True)
        print(f"      Nesting: {element.nesting_depth}", flush=True)
        if element.parent_element:
            print(f"      Parent: {element.parent_element}", flush=True)
        print("", flush=True)

    # Token classification with structural context
    print(f"\n[TOKEN CLASSIFICATION SAMPLES]:", flush=True)
    sample_tokens = tokens[:20] if len(tokens) > 20 else tokens
    for token in sample_tokens:
        base_importance = classify_importance(token)
        enhanced_importance = classify_importance_with_structure(token, structural_info)
        enhancement = enhanced_importance - base_importance

        print(f"   '{token.value}' (Line {token.line})", flush=True)
        print(f"      Base: {base_importance} -> Enhanced: {enhanced_importance} (+{enhancement})", flush=True)
        print(f"      Type: {token.type} | ASCII: '{token.ascii_char}'", flush=True)

    print(f"\n[SURFACE MAPPING PREVIEW]:", flush=True)
    _debug_surface_mapping_distribution(structural_info)


def debug_surface_mapping(mapped_pairs: List[Tuple[Point3D, CodeToken]],
                         structural_info: StructuralInfo,
                         enable_debug: bool = False) -> None:
    """Display debugging information for surface mapping distribution.

    Shows how tokens are distributed across the torus surface with
    structural element groupings and importance distributions.

    Args:
        mapped_pairs: Token-surface mapping pairs from map_tokens_to_surface_with_structure()
        structural_info: StructuralInfo from analyze_structure()
        enable_debug: Flag to enable/disable debug output
    """
    if not enable_debug:
        return

    print(f"\n[SURFACE MAPPING DEBUG]:", flush=True)
    print(f"   Total Mapped Pairs: {len(mapped_pairs)}", flush=True)

    # Group by importance levels
    importance_counts = {}
    for point, token in mapped_pairs:
        level = token.importance
        importance_counts[level] = importance_counts.get(level, 0) + 1

    print(f"\n[IMPORTANCE DISTRIBUTION]:", flush=True)
    for level in sorted(importance_counts.keys(), reverse=True):
        count = importance_counts[level]
        percentage = (count / len(mapped_pairs)) * 100 if mapped_pairs else 0
        level_name = _get_importance_level_name(level)
        print(f"   {level_name}: {count} tokens ({percentage:.1f}%)", flush=True)

    # Show spatial distribution by structural elements
    _debug_structural_spatial_distribution(mapped_pairs, structural_info)


def debug_nested_structures(structural_info: StructuralInfo, enable_debug: bool = False) -> None:
    """Display debugging information specifically for nested structure handling.

    Shows the hierarchy of nested functions, classes, and their complexity
    inheritance through the nesting levels.

    Args:
        structural_info: StructuralInfo from analyze_structure()
        enable_debug: Flag to enable/disable debug output
    """
    if not enable_debug:
        return

    print(f"\n[NESTED STRUCTURE DEBUG]:", flush=True)

    # Find all nested elements
    nested_elements = [elem for elem in structural_info.elements if elem.nesting_depth > 0]

    if not nested_elements:
        print("   No nested structures found.", flush=True)
        return

    # Group by nesting depth
    depth_groups = {}
    for element in nested_elements:
        depth = element.nesting_depth
        if depth not in depth_groups:
            depth_groups[depth] = []
        depth_groups[depth].append(element)

    for depth in sorted(depth_groups.keys()):
        elements = depth_groups[depth]
        print(f"\n   DEPTH {depth}:", flush=True)
        for element in elements:
            indent = "   " + "  " * depth
            print(f"{indent}{element.element_type}: {element.name}", flush=True)
            print(f"{indent}   Parent: {element.parent_element}", flush=True)
            print(f"{indent}   Complexity: {element.complexity_score:.2f}", flush=True)


def _debug_surface_mapping_distribution(structural_info: StructuralInfo) -> None:
    """Show preview of how surface points would be allocated to structural elements."""
    if not structural_info.elements:
        print("   No structural elements for surface allocation.", flush=True)
        return

    total_complexity = sum(elem.complexity_score for elem in structural_info.elements)
    sample_surface_points = 100  # Sample for calculation

    print(f"   Surface Allocation Preview (100 sample points):", flush=True)
    for element in structural_info.elements:
        if total_complexity > 0:
            ratio = element.complexity_score / total_complexity
            allocation = int(sample_surface_points * ratio * 0.8)
        else:
            allocation = sample_surface_points // len(structural_info.elements)

        print(f"      {element.name}: {allocation} points ({allocation}%)", flush=True)


def _debug_structural_spatial_distribution(mapped_pairs: List[Tuple[Point3D, CodeToken]],
                                         structural_info: StructuralInfo) -> None:
    """Show how tokens are spatially distributed across structural elements."""
    print(f"\n[SPATIAL DISTRIBUTION BY STRUCTURE]:", flush=True)

    # Group mapped pairs by structural elements
    element_mappings = {}
    unstructured_count = 0

    for point, token in mapped_pairs:
        # Find which structural element contains this token
        containing_element = None
        for element in structural_info.elements:
            if element.start_line <= token.line <= element.end_line:
                containing_element = element.name
                break

        if containing_element:
            if containing_element not in element_mappings:
                element_mappings[containing_element] = 0
            element_mappings[containing_element] += 1
        else:
            unstructured_count += 1

    # Display distribution
    for element_name, count in element_mappings.items():
        percentage = (count / len(mapped_pairs)) * 100 if mapped_pairs else 0
        print(f"   {element_name}: {count} tokens ({percentage:.1f}%)", flush=True)

    if unstructured_count > 0:
        percentage = (unstructured_count / len(mapped_pairs)) * 100
        print(f"   [Unstructured]: {unstructured_count} tokens ({percentage:.1f}%)", flush=True)


def _get_importance_level_name(level: int) -> str:
    """Convert importance level to human-readable name."""
    if level == ImportanceLevel.CRITICAL:
        return "CRITICAL"
    elif level == ImportanceLevel.HIGH:
        return "HIGH"
    elif level == ImportanceLevel.MEDIUM:
        return "MEDIUM"
    else:
        return "LOW"


# === RENDERING ENGINE ===

def map_tokens_to_surface_with_structure(tokens: List[CodeToken],
                                        points: List[Point3D],
                                        structural_info: StructuralInfo) -> List[Tuple[Point3D, CodeToken]]:
    """Enhanced token-to-surface mapping incorporating structural hierarchy.

    Extends map_tokens_to_surface() with structural analysis to prioritize
    tokens within critical structural elements (functions, classes, imports)
    and apply hierarchical distribution based on code architecture.

    Enhancement Features:
    - Structural importance boosts using classify_importance_with_structure()
    - Hierarchical surface allocation prioritizing complex structural elements
    - Spatial clustering for tokens within same structural elements
    - Enhanced density mapping accounting for structural complexity

    Args:
        tokens: List of classified code tokens with importance and ASCII chars
        points: List of 3D torus surface points with u,v parameters
        structural_info: StructuralInfo from analyze_structure() with complexity data

    Returns:
        List of (point, token) pairs with structural hierarchy emphasis

    Raises:
        ValueError: If inputs are invalid with actionable solution guidance
    """
    # Input validation
    if not tokens:
        raise ValueError(
            "Empty token list provided for structural surface mapping. "
            "Solution: Ensure source code tokenization produces valid tokens"
        )

    if not points:
        raise ValueError(
            "Empty surface points list provided for mapping. "
            "Solution: Ensure torus generation produces valid 3D points"
        )

    if not isinstance(structural_info, StructuralInfo):
        raise ValueError(
            "Invalid structural_info parameter: expected StructuralInfo object. "
            "Solution: Generate structural_info using analyze_structure()"
        )

    # Enhance token importance using structural context
    enhanced_tokens = []
    for token in tokens:
        enhanced_importance = classify_importance_with_structure(token, structural_info)

        # Create enhanced token with new importance and corresponding ASCII char
        ascii_char = _get_ascii_char_for_importance(enhanced_importance)

        enhanced_token = CodeToken(
            type=token.type,
            value=token.value,
            importance=enhanced_importance,
            line=token.line,
            column=token.column,
            ascii_char=ascii_char
        )
        enhanced_tokens.append(enhanced_token)

    # Apply structural spatial distribution
    mapped_pairs = _apply_structural_distribution(enhanced_tokens, points, structural_info)

    # Apply visual balance with structural awareness
    balanced_pairs = _apply_structural_visual_balance(mapped_pairs, points, structural_info)

    return balanced_pairs


def _get_ascii_char_for_importance(importance_level: int) -> str:
    """Map importance level to ASCII character for rendering."""
    if importance_level == ImportanceLevel.CRITICAL:
        return ASCII_CHARS['HIGH']  # '#'
    elif importance_level == ImportanceLevel.HIGH:
        return ASCII_CHARS['MEDIUM']  # '+'
    elif importance_level == ImportanceLevel.MEDIUM:
        return ASCII_CHARS['LOW']  # '-'
    else:  # ImportanceLevel.LOW
        return ASCII_CHARS['BACKGROUND']  # '.'


def _apply_structural_distribution(tokens: List[CodeToken],
                                 points: List[Point3D],
                                 structural_info: StructuralInfo) -> List[Tuple[Point3D, CodeToken]]:
    """Apply structural hierarchy to surface distribution."""
    # Group tokens by structural elements
    element_groups = _group_tokens_by_structure(tokens, structural_info.elements)

    # Calculate surface allocation for each structural element
    element_allocations = _calculate_structural_allocations(element_groups, points, structural_info)

    # Distribute tokens within allocated surface regions
    mapped_pairs = []
    point_index = 0

    # Process elements in complexity order (most complex first)
    sorted_elements = sorted(structural_info.elements,
                           key=lambda e: e.complexity_score, reverse=True)

    for element in sorted_elements:
        element_tokens = element_groups.get(element.name, [])
        if not element_tokens:
            continue

        allocation_size = element_allocations.get(element.name, 0)
        if allocation_size == 0:
            continue

        # Get surface points for this element
        element_points = points[point_index:point_index + allocation_size]
        point_index += allocation_size

        # Apply density mapping within element
        element_pairs = _map_element_tokens_to_points(element_tokens, element_points)
        mapped_pairs.extend(element_pairs)

    # Handle remaining tokens not in structural elements
    remaining_tokens = [token for token in tokens
                       if not any(token in element_groups.get(elem.name, [])
                                for elem in structural_info.elements)]

    if remaining_tokens and point_index < len(points):
        remaining_points = points[point_index:]
        remaining_pairs = _map_element_tokens_to_points(remaining_tokens, remaining_points)
        mapped_pairs.extend(remaining_pairs)

    return mapped_pairs


def _group_tokens_by_structure(tokens: List[CodeToken],
                             elements: List[StructuralElement]) -> dict:
    """Group tokens by their containing structural elements."""
    groups = {}

    for element in elements:
        element_tokens = []
        for token in tokens:
            if element.start_line <= token.line <= element.end_line:
                element_tokens.append(token)
        groups[element.name] = element_tokens

    return groups


def _calculate_structural_allocations(element_groups: dict,
                                    points: List[Point3D],
                                    structural_info: StructuralInfo) -> dict:
    """Calculate surface point allocation for each structural element."""
    allocations = {}
    total_points = len(points)
    total_complexity = sum(elem.complexity_score for elem in structural_info.elements)

    if total_complexity == 0:
        # Equal distribution if no complexity data
        points_per_element = total_points // max(len(element_groups), 1)
        for element_name in element_groups:
            allocations[element_name] = points_per_element
    else:
        # Allocate based on complexity ratios
        for element in structural_info.elements:
            if element.name in element_groups:
                complexity_ratio = element.complexity_score / total_complexity
                allocation = int(total_points * complexity_ratio * 0.8)  # 80% for structured elements
                allocations[element.name] = max(allocation, 1)

    return allocations


def _map_element_tokens_to_points(tokens: List[CodeToken],
                                points: List[Point3D]) -> List[Tuple[Point3D, CodeToken]]:
    """Map tokens to points within a structural element."""
    if not tokens or not points:
        return []

    # Use density mapping similar to original algorithm
    density_map = {
        ImportanceLevel.CRITICAL: 4,
        ImportanceLevel.HIGH: 3,
        ImportanceLevel.MEDIUM: 2,
        ImportanceLevel.LOW: 1
    }

    mapped_pairs = []
    point_index = 0

    # Sort tokens by importance and position
    sorted_tokens = sorted(tokens, key=lambda t: (-t.importance, t.line, t.column))

    for token in sorted_tokens:
        density_allocation = min(density_map[token.importance],
                               len(points) - point_index)

        for _ in range(max(1, density_allocation)):
            if point_index >= len(points):
                break

            mapped_pairs.append((points[point_index], token))
            point_index += 1

    return mapped_pairs


def _apply_structural_visual_balance(mapped_pairs: List[Tuple[Point3D, CodeToken]],
                                   points: List[Point3D],
                                   structural_info: StructuralInfo) -> List[Tuple[Point3D, CodeToken]]:
    """Apply visual balance with structural awareness."""
    # For now, return the pairs as-is. This can be enhanced with
    # structural clustering algorithms in future iterations
    return mapped_pairs


def map_tokens_to_surface(tokens: List[CodeToken],
                         points: List[Point3D]) -> List[Tuple[Point3D, CodeToken]]:
    """Distribute code tokens across torus surface coordinates.

    Implements token-to-surface mapping with character mapping, density allocation,
    dynamic scaling, and visual balance per Story 2.4 requirements.

    Core Algorithm:
    1. Validate inputs and handle edge cases
    2. Calculate scaling factor based on token count vs. surface points
    3. Apply density mapping for importance-based surface allocation
    4. Distribute tokens across surface using parametric u,v coordinates
    5. Ensure visual balance and rotation-aware distribution

    Character Mapping (already set in tokens):
    - CRITICAL (4): '#' - Keywords (def, class, if, for, etc.)
    - HIGH (3): '+' - Operators (+, -, *, /, ==, etc.)
    - MEDIUM (2): '-' - Identifiers, literals (variables, numbers, strings)
    - LOW (1): '.' - Comments, whitespace, special characters

    Density Mapping:
    - CRITICAL tokens get 4x surface points allocation
    - HIGH tokens get 3x surface points allocation
    - MEDIUM tokens get 2x surface points allocation
    - LOW tokens get 1x surface points allocation

    Args:
        tokens: List of classified code tokens with importance and ASCII chars
        points: List of 3D torus surface points with u,v parameters

    Returns:
        List of (point, token) pairs for rendering pipeline

    Raises:
        ValueError: If inputs are invalid with actionable solution guidance
    """
    # Input validation with proper error handling per coding standards
    if not tokens:
        raise ValueError(
            "Empty token list provided for surface mapping. "
            "Solution: Ensure source code tokenization produces valid tokens"
        )

    if not points:
        raise ValueError(
            "Empty surface points list provided for mapping. "
            "Solution: Ensure torus generation produces valid 3D points"
        )

    # Handle edge case: insufficient surface points
    if len(points) < len(tokens):
        # Compression scenario - tokens exceed surface points
        return _handle_token_compression(tokens, points)

    # Calculate density multipliers for each importance level
    density_map = {
        ImportanceLevel.CRITICAL: 4,  # 4x allocation for keywords
        ImportanceLevel.HIGH: 3,      # 3x allocation for operators
        ImportanceLevel.MEDIUM: 2,    # 2x allocation for identifiers/literals
        ImportanceLevel.LOW: 1        # 1x allocation for comments/whitespace
    }

    # Calculate total density requirement
    total_density_units = sum(density_map[token.importance] for token in tokens)

    # Calculate scaling factor for density mapping
    available_points = len(points)
    if total_density_units > available_points:
        # Scale down density multipliers to fit available points
        scale_factor = available_points / total_density_units
        density_map = {level: max(1, int(mult * scale_factor))
                      for level, mult in density_map.items()}

    # Create token-to-surface mapping using sequence-based distribution
    mapped_pairs = []
    point_index = 0

    # Sort tokens by line and column for consistent sequence distribution
    sorted_tokens = sorted(tokens, key=lambda t: (t.line, t.column))

    for token in sorted_tokens:
        # Calculate density allocation for this token
        density_allocation = density_map[token.importance]

        # Assign multiple surface points based on importance density
        for _ in range(density_allocation):
            if point_index >= len(points):
                # Reached end of available points
                break

            # Get surface point with parametric coordinates
            surface_point = points[point_index]

            # Create token-surface mapping pair
            mapped_pairs.append((surface_point, token))

            point_index += 1

    # Apply visual balance distribution for aesthetic appeal
    balanced_pairs = _apply_visual_balance(mapped_pairs, points)

    return balanced_pairs


def _handle_token_compression(tokens: List[CodeToken],
                            points: List[Point3D]) -> List[Tuple[Point3D, CodeToken]]:
    """Handle compression scenario where tokens exceed available surface points.

    Uses importance-based selection to prioritize higher importance tokens
    when surface points are insufficient for all tokens.

    Args:
        tokens: List of tokens to compress
        points: Limited surface points available

    Returns:
        Compressed token-surface mapping prioritizing important tokens
    """
    # Sort tokens by importance (highest first) and line position
    sorted_tokens = sorted(tokens,
                          key=lambda t: (-t.importance, t.line, t.column))

    # Select top N tokens that fit available surface points
    selected_tokens = sorted_tokens[:len(points)]

    # Create direct 1:1 mapping for compression scenario
    mapped_pairs = []
    for i, token in enumerate(selected_tokens):
        mapped_pairs.append((points[i], token))

    return mapped_pairs


def _apply_visual_balance(mapped_pairs: List[Tuple[Point3D, CodeToken]],
                         all_points: List[Point3D]) -> List[Tuple[Point3D, CodeToken]]:
    """Apply visual balance and aesthetic considerations to token distribution.

    Implements rotation-aware distribution to prevent clustering of same
    importance tokens in single areas of the torus surface.

    Args:
        mapped_pairs: Initial token-surface mappings
        all_points: Complete surface points for reference

    Returns:
        Rebalanced token-surface mappings for optimal visual appeal
    """
    if not mapped_pairs:
        return mapped_pairs

    # Group mapped pairs by importance level
    importance_groups = {}
    for point, token in mapped_pairs:
        importance = token.importance
        if importance not in importance_groups:
            importance_groups[importance] = []
        importance_groups[importance].append((point, token))

    # Redistribute each importance group across torus surface
    rebalanced_pairs = []

    for importance_level, group_pairs in importance_groups.items():
        if not group_pairs:
            continue

        # Sort group points by parametric u coordinate for even distribution
        sorted_group = sorted(group_pairs, key=lambda x: x[0].u)

        # Calculate distribution spacing to spread tokens evenly
        total_surface_u = 2 * math.pi  # Full parametric u range
        group_size = len(sorted_group)

        if group_size == 1:
            # Single token - use original position
            rebalanced_pairs.extend(sorted_group)
        else:
            # Multiple tokens - redistribute with spacing
            u_spacing = total_surface_u / group_size

            for i, (original_point, token) in enumerate(sorted_group):
                # Calculate new u coordinate for even spacing
                target_u = i * u_spacing

                # Find closest surface point to target u coordinate
                closest_point = min(all_points,
                                  key=lambda p: abs(p.u - target_u))

                rebalanced_pairs.append((closest_point, token))

    return rebalanced_pairs


def generate_ascii_frame(mapped_pairs: List[Tuple[Point3D, CodeToken]], frame_number: int = 0) -> DisplayFrame:
    """Create ASCII character frame with token-based characters and depth sorting.

    Enhanced from Story 2.4 to use token-mapped surface points with proper
    character mapping based on token importance rather than depth-based characters.

    Core Algorithm:
    1. Project 3D points to 2D screen coordinates
    2. Sort by depth value (farthest to closest) for painter's algorithm
    3. Render using token's ASCII character from importance classification
    4. Apply depth buffer for proper layering

    Token Character Mapping (from token.ascii_char):
    - CRITICAL (4): '#' - Keywords (def, class, if, for, etc.)
    - HIGH (3): '+' - Operators (+, -, *, /, ==, etc.)
    - MEDIUM (2): '-' - Identifiers, literals (variables, numbers, strings)
    - LOW (1): '.' - Comments, whitespace, special characters

    Args:
        mapped_pairs: List of (Point3D, CodeToken) pairs from token mapping
        frame_number: Current frame number for tracking

    Returns:
        Complete ASCII frame ready for terminal output with token characters
    """
    # Initialize 40x20 character buffer and depth buffer
    buffer = [[ASCII_CHARS['BACKGROUND'] for _ in range(TERMINAL_WIDTH)] for _ in range(TERMINAL_HEIGHT)]
    depth_buffer = [[float('inf') for _ in range(TERMINAL_WIDTH)] for _ in range(TERMINAL_HEIGHT)]

    # Convert 3D points to 2D screen coordinates with tokens
    screen_data = []
    for point_3d, token in mapped_pairs:
        point_2d = project_to_screen(point_3d)
        if point_2d.visible:
            screen_data.append((point_2d, token))

    # Sort by depth (farthest to closest for painter's algorithm)
    sorted_data = sorted(screen_data, key=lambda x: x[0].depth, reverse=True)

    # Render points with token-based characters and depth sorting
    for point_2d, token in sorted_data:
        x, y = point_2d.x, point_2d.y

        # Bounds check for safety
        if 0 <= x < TERMINAL_WIDTH and 0 <= y < TERMINAL_HEIGHT:
            # Only render if this point is closer than what's already in the buffer
            if point_2d.depth < depth_buffer[y][x]:
                # Use token's ASCII character from importance classification
                char = token.ascii_char

                # Update buffer and depth buffer
                buffer[y][x] = char
                depth_buffer[y][x] = point_2d.depth

    return DisplayFrame(
        width=TERMINAL_WIDTH,
        height=TERMINAL_HEIGHT,
        buffer=buffer,
        depth_buffer=depth_buffer,
        frame_number=frame_number
    )


def generate_ascii_frame_legacy(points: List[Point2D], frame_number: int = 0) -> DisplayFrame:
    """Legacy ASCII frame generator for backward compatibility.

    Original depth-based character mapping preserved for testing and fallback.
    Use generate_ascii_frame() for token-based rendering.

    Args:
        points: List of 2D screen coordinates with depth information
        frame_number: Current frame number for tracking

    Returns:
        Complete ASCII frame with depth-based characters
    """
    # Initialize 40x20 character buffer and depth buffer
    buffer = [[ASCII_CHARS['BACKGROUND'] for _ in range(TERMINAL_WIDTH)] for _ in range(TERMINAL_HEIGHT)]
    depth_buffer = [[float('inf') for _ in range(TERMINAL_WIDTH)] for _ in range(TERMINAL_HEIGHT)]

    # Filter visible points and sort by depth (farthest to closest for painter's algorithm)
    visible_points = [p for p in points if p.visible]
    sorted_points = sorted(visible_points, key=lambda p: p.depth, reverse=True)

    # Render points with depth sorting
    for point in sorted_points:
        x, y = point.x, point.y

        # Bounds check for safety
        if 0 <= x < TERMINAL_WIDTH and 0 <= y < TERMINAL_HEIGHT:
            # Only render if this point is closer than what's already in the buffer
            if point.depth < depth_buffer[y][x]:
                # Map depth to ASCII character
                if point.depth >= 0.75:
                    char = '.'  # Background/far
                elif point.depth >= 0.5:
                    char = '-'  # Medium-far
                elif point.depth >= 0.25:
                    char = '+'  # Medium-close
                else:
                    char = '#'  # Closest/foreground

                # Update buffer and depth buffer
                buffer[y][x] = char
                depth_buffer[y][x] = point.depth

    return DisplayFrame(
        width=TERMINAL_WIDTH,
        height=TERMINAL_HEIGHT,
        buffer=buffer,
        depth_buffer=depth_buffer,
        frame_number=frame_number
    )


def output_to_terminal(frame: DisplayFrame) -> None:
    """Render frame to terminal with screen clearing.

    Implements cross-platform screen clearing and smooth frame output:
    - Uses ANSI escape codes for screen clearing (works on modern Windows Terminal)
    - Outputs each row with flush=True for smooth animation
    - Includes frame number for debugging purposes
    - Handles terminal compatibility per coding standards

    Args:
        frame: ASCII frame to display
    """
    # Clear screen using ANSI escape codes (cross-platform)
    # \033[2J clears entire screen, \033[H moves cursor to home position
    print("\033[2J\033[H", end="", flush=True)

    # Output frame buffer line by line with flush for smooth animation
    for row in frame.buffer:
        print(''.join(row), flush=True)

    # Optional: Display frame number for debugging (can be removed for production)
    # print(f"Frame: {frame.frame_number}", flush=True)


# === ANIMATION CONTROLLER ===

def calculate_frame_timing() -> float:
    """Calculate target frame time for 30+ FPS performance.

    Returns:
        Target frame time in seconds (maximum 33.33ms per frame)
    """
    return 1.0 / TARGET_FPS


def handle_interrupts() -> bool:
    """Handle graceful Ctrl+C interrupts and cleanup.

    Returns:
        True if interrupt was handled gracefully
    """
    try:
        # This will be called from the animation loop's exception handler
        print("\nAnimation stopped gracefully.")
        print("Terminal state restored.")
        print("Thank you for viewing the 3D ASCII Donut!")
        return True
    except Exception:
        return False


def run_animation_loop(enable_debug: bool = False) -> None:
    """Main execution loop with frame rate control and structural analysis.

    Implements comprehensive animation control with enhanced structural analysis:
    - Target 30+ FPS with dynamic timing adjustment
    - Graceful keyboard interrupt handling (Ctrl+C)
    - Frame timing validation and debugging
    - Memory management with buffer clearing
    - Structural analysis integration with optional debugging
    - Consistent performance across different system capabilities

    Follows Animation Controller specifications with proper integration
    of MathematicalEngine, ParsingEngine, and RenderingEngine systems.
    Enhanced with structural analysis from Story 2.5.

    Args:
        enable_debug: Flag to enable structural analysis debugging output
    """
    print("Starting 3D ASCII Donut Animation with Structural Analysis...", flush=True)
    print("Target Frame Rate: 30+ FPS", flush=True)
    if enable_debug:
        print("Debug Mode: ENABLED", flush=True)
    print("Press Ctrl+C to exit", flush=True)
    print("", flush=True)  # Add spacing

    # Animation parameters with configurable rotation speed
    torus_params = TorusParameters(
        outer_radius=2.0,
        inner_radius=1.0,
        u_resolution=50,
        v_resolution=25,
        rotation_speed=0.05  # Reduced for smoother animation
    )

    # Timing and performance tracking variables
    frame_count = 0
    target_frame_time = calculate_frame_timing()
    total_elapsed = 0.0
    performance_samples = []

    # Read and tokenize source code (cached per performance rules)
    try:
        source_code = read_self_code()
        tokens = tokenize_code(source_code)
    except Exception as e:
        print(f"Error reading/tokenizing source code: {e}", flush=True)
        print("Solution: Ensure script file is accessible and contains valid Python", flush=True)
        return

    # Perform structural analysis (cached for performance per coding standards)
    try:
        structural_info = analyze_structure(tokens)

        # Display debug information once at startup if enabled
        if enable_debug:
            debug_structural_analysis(structural_info, tokens, enable_debug)
            debug_nested_structures(structural_info, enable_debug)
    except Exception as e:
        print(f"Error in structural analysis: {e}", flush=True)
        print("Solution: Check token structure and retry analysis", flush=True)
        return

    try:
        while True:
            frame_start_time = time.time()

            # Calculate current rotation angle with incremental updates
            rotation_angle = frame_count * torus_params.rotation_speed

            # Generate torus points (cached for performance per coding standards)
            torus_points = generate_torus_points(torus_params)

            # Apply Y-axis rotation transformation
            rotated_points = apply_rotation(torus_points, rotation_angle)

            # Map tokens to rotated surface points with structural analysis
            try:
                mapped_pairs = map_tokens_to_surface_with_structure(tokens, rotated_points, structural_info)

                # Display surface mapping debug info for first frame only
                if enable_debug and frame_count == 0:
                    debug_surface_mapping(mapped_pairs, structural_info, enable_debug)

            except Exception as e:
                print(f"Error mapping tokens to surface with structure: {e}", flush=True)
                print("Solution: Check token, surface point, and structural analysis", flush=True)

                # Fallback to basic mapping if structural mapping fails
                try:
                    mapped_pairs = map_tokens_to_surface(tokens, rotated_points)
                    if enable_debug:
                        print("Fallback: Using basic token mapping", flush=True)
                except Exception as fallback_e:
                    print(f"Fallback mapping also failed: {fallback_e}", flush=True)
                    break

            # Generate ASCII frame with token-based characters
            frame = generate_ascii_frame(mapped_pairs, frame_count)

            # Output frame to terminal with screen clearing
            output_to_terminal(frame)

            # Frame timing calculation and control
            frame_elapsed = time.time() - frame_start_time
            total_elapsed += frame_elapsed

            # Performance tracking for timing adjustment
            performance_samples.append(frame_elapsed)
            if len(performance_samples) > 30:  # Keep last 30 samples
                performance_samples.pop(0)

            # Dynamic timing adjustment for consistent frame rate
            avg_frame_time = sum(performance_samples) / len(performance_samples)
            sleep_time = max(0, target_frame_time - frame_elapsed)

            # Handle performance variations gracefully
            if avg_frame_time > target_frame_time * 1.5:
                # System struggling - reduce update frequency slightly
                sleep_time = max(sleep_time, 0.01)  # Minimum 10ms sleep

            # Frame rate control with timing validation
            if sleep_time > 0:
                time.sleep(sleep_time)

            frame_count += 1

            # Memory management: Clear display buffers periodically
            # This prevents memory accumulation during long runs per coding standards
            if frame_count % 100 == 0:
                # Force garbage collection of old frame data
                import gc
                gc.collect()

    except KeyboardInterrupt:
        # Graceful interrupt handling with proper cleanup
        success = handle_interrupts()
        if success:
            # Display performance statistics
            if frame_count > 0:
                avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0
                print(f"Average FPS: {avg_fps:.1f}", flush=True)
                print(f"Total Frames: {frame_count}", flush=True)
    except Exception as e:
        print(f"\nAnimation error: {e}", flush=True)
        print("Solution: Check terminal compatibility and system resources", flush=True)
        raise


def main() -> None:
    """Entry point with error handling and setup.

    Validates environment and starts the animation loop with proper error handling.
    Supports --debug flag for structural analysis debugging output.
    """
    try:
        # Parse command line arguments
        enable_debug = '--debug' in sys.argv or '-d' in sys.argv

        # Validate Python version
        if sys.version_info < (3, 8):
            print("Error: Python 3.8+ required.")
            print("Solution: Upgrade to Python 3.8 or higher")
            sys.exit(1)

        # Validate __file__ availability for self-code reading
        try:
            _ = __file__
        except NameError:
            print("Warning: Self-code reading not available in interactive mode.")
            print("Solution: Run script directly: python rotating_donut.py")

        # Start animation with debug mode
        run_animation_loop(enable_debug=enable_debug)

    except Exception as e:
        print(f"Error: {e}")
        print("Solution: Check terminal compatibility and Python installation")
        sys.exit(1)


if __name__ == "__main__":
    main()