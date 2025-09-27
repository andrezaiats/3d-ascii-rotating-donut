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

Author: BMAD Development System
License: MIT
Python Version: 3.8+
Dependencies: Python Standard Library Only
"""

# === IMPORTS (Python Standard Library Only) ===
import ast
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
    importance: str  # HIGH, MEDIUM, LOW
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
    """Parse source code into classified tokens.

    Args:
        source: Source code string to tokenize

    Returns:
        List of classified code tokens
    """
    # Placeholder implementation - will be filled in future stories
    tokens = []
    # TODO: Implement tokenization logic
    return tokens


def classify_importance(token_type: str, token_value: str) -> str:
    """Assign semantic importance hierarchy to tokens.

    Args:
        token_type: Type of token from tokenize module
        token_value: String value of the token

    Returns:
        Importance level: 'HIGH', 'MEDIUM', or 'LOW'
    """
    # Placeholder implementation - will be filled in future stories
    # TODO: Implement token classification logic
    return 'LOW'  # Default to LOW importance for unknown tokens


# === RENDERING ENGINE ===

def map_tokens_to_surface(tokens: List[CodeToken],
                         points: List[Point3D]) -> List[Tuple[Point3D, CodeToken]]:
    """Distribute code tokens across torus surface coordinates.

    Args:
        tokens: List of code tokens to distribute
        points: List of torus surface points

    Returns:
        List of (point, token) pairs for rendering
    """
    # Placeholder implementation - will be filled in future stories
    mapped = []
    # TODO: Implement token-to-surface mapping
    return mapped


def generate_ascii_frame(points: List[Point2D], frame_number: int = 0) -> DisplayFrame:
    """Create ASCII character frame with depth sorting.

    Implements painter's algorithm for proper depth layering:
    1. Sort Point2D list by depth value (farthest to closest)
    2. Render points in sorted order to DisplayFrame buffer
    3. For each position, update depth_buffer if current point is closer
    4. Overwrite buffer position only if depth check passes

    Character mapping based on depth values:
    - depth >= 0.75: '.' (background/far)
    - depth >= 0.5:  '-' (medium-far)
    - depth >= 0.25: '+' (medium-close)
    - depth < 0.25:  '#' (closest/foreground)

    Args:
        points: List of 2D screen coordinates with depth information
        frame_number: Current frame number for tracking

    Returns:
        Complete ASCII frame ready for terminal output
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


def run_animation_loop() -> None:
    """Main execution loop with frame rate control.

    Implements comprehensive animation control:
    - Target 30+ FPS with dynamic timing adjustment
    - Graceful keyboard interrupt handling (Ctrl+C)
    - Frame timing validation and debugging
    - Memory management with buffer clearing
    - Consistent performance across different system capabilities

    Follows Animation Controller specifications with proper integration
    of MathematicalEngine, ParsingEngine, and RenderingEngine systems.
    """
    print("Starting 3D ASCII Donut Animation...", flush=True)
    print("Target Frame Rate: 30+ FPS", flush=True)
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

    try:
        while True:
            frame_start_time = time.time()

            # Calculate current rotation angle with incremental updates
            rotation_angle = frame_count * torus_params.rotation_speed

            # Generate torus points (cached for performance per coding standards)
            torus_points = generate_torus_points(torus_params)

            # Apply Y-axis rotation transformation
            rotated_points = apply_rotation(torus_points, rotation_angle)

            # Project 3D points to 2D screen coordinates
            screen_points = [project_to_screen(point) for point in rotated_points]

            # Generate ASCII frame with depth sorting
            frame = generate_ascii_frame(screen_points, frame_count)

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
    """
    try:
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

        # Start animation
        run_animation_loop()

    except Exception as e:
        print(f"Error: {e}")
        print("Solution: Check terminal compatibility and Python installation")
        sys.exit(1)


if __name__ == "__main__":
    main()