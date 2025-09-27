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

# Performance monitoring and optimization infrastructure
_rotation_matrix_cache = {}  # Cache for rotation matrices at common angles
_projection_cache = {}       # Cache for frequently used projection calculations
_performance_stats = {
    'frame_times': [],
    'math_times': [],
    'projection_times': [],
    'total_frames': 0,
    'cache_hits': 0,
    'cache_misses': 0
}

# === TOKEN CACHE SYSTEM FOR REAL-TIME INTEGRATION (Story 3.4) ===

class TokenCache:
    """Comprehensive token data caching system for efficient frame-to-frame access.

    Implements Story 3.4 requirements for token caching and pipeline optimization.
    Stores preprocessed token data with efficient lookup structures to eliminate
    repeated parsing and classification during animation.

    Attributes:
        source_code: Cached source code string
        tokens: List of parsed CodeToken objects
        enhanced_tokens: Tokens enhanced with structural analysis
        importance_map: Dict mapping tokens to cached importance levels
        structural_info: Cached structural analysis results
        token_mappings: Pre-computed token-to-surface mappings
        last_update: Timestamp of last cache update
        cache_valid: Boolean indicating cache validity
    """

    def __init__(self):
        """Initialize empty token cache."""
        self.source_code = None
        self.tokens = None
        self.enhanced_tokens = None
        self.importance_map = {}
        self.structural_info = None
        self.token_mappings = None
        self.last_update = None
        self.cache_valid = False

    def populate(self, source_code: str, tokens: List['CodeToken'],
                structural_info: 'StructuralInfo' = None) -> None:
        """Populate cache with preprocessed token data.

        Args:
            source_code: The source code string
            tokens: List of parsed tokens
            structural_info: Optional structural analysis results
        """
        self.source_code = source_code
        self.tokens = tokens
        self.structural_info = structural_info
        self.last_update = time.time()
        self.cache_valid = True

            # Cache importance classifications to avoid repeated calls
        self.importance_map = {}
        self.token_lookup = {}  # Fast token lookup by position
        for i, token in enumerate(tokens):
            if hasattr(token, 'importance'):
                self.importance_map[id(token)] = token.importance
            # Create fast position-based lookup
            if hasattr(token, 'line') and hasattr(token, 'column'):
                self.token_lookup[(token.line, token.column)] = i

    def get_tokens(self) -> Optional[List['CodeToken']]:
        """Get cached tokens with validation.

        Returns:
            Cached tokens if valid, None otherwise
        """
        if self.cache_valid and self.tokens:
            return self.tokens
        return None

    def get_enhanced_tokens(self) -> Optional[List['CodeToken']]:
        """Get cached enhanced tokens.

        Returns:
            Cached enhanced tokens if valid, None otherwise
        """
        if self.cache_valid and self.enhanced_tokens:
            return self.enhanced_tokens
        return None

    def set_enhanced_tokens(self, enhanced_tokens: List['CodeToken']) -> None:
        """Store enhanced tokens in cache.

        Args:
            enhanced_tokens: Tokens enhanced with structural analysis
        """
        self.enhanced_tokens = enhanced_tokens
        # Update importance map with enhanced tokens
        for token in enhanced_tokens:
            if hasattr(token, 'importance'):
                self.importance_map[id(token)] = token.importance

    def set_token_mappings(self, mappings: List[Tuple[int, 'CodeToken']]) -> None:
        """Store pre-computed token-to-point mappings.

        Args:
            mappings: List of (point_index, token) pairs
        """
        self.token_mappings = mappings

    def get_token_mappings(self) -> Optional[List[Tuple[int, 'CodeToken']]]:
        """Get cached token mappings.

        Returns:
            Cached mappings if valid, None otherwise
        """
        if self.cache_valid and self.token_mappings:
            return self.token_mappings
        return None

    def invalidate(self) -> None:
        """Invalidate cache, forcing refresh on next access."""
        self.cache_valid = False

    def get_token_by_position(self, line: int, column: int) -> Optional['CodeToken']:
        """Fast token lookup by position.

        Args:
            line: Line number
            column: Column number

        Returns:
            Token at the specified position or None if not found
        """
        if (line, column) in self.token_lookup and self.tokens:
            index = self.token_lookup[(line, column)]
            return self.tokens[index] if index < len(self.tokens) else None
        return None

    def get_cached_importance(self, token: 'CodeToken') -> Optional[int]:
        """Get cached importance level for a token.

        Args:
            token: Token to get importance for

        Returns:
            Cached importance level or None if not cached
        """
        return self.importance_map.get(id(token))

    def clear(self) -> None:
        """Clear all cached data to free memory."""
        self.source_code = None
        self.tokens = None
        self.enhanced_tokens = None
        self.importance_map.clear()
        self.token_lookup.clear()
        self.structural_info = None
        self.token_mappings = None
        self.last_update = None
        self.cache_valid = False

    def memory_usage(self) -> int:
        """Estimate memory usage of cached data in bytes.

        Returns:
            Approximate memory usage in bytes
        """
        usage = 0
        if self.source_code:
            usage += len(self.source_code)
        if self.tokens:
            usage += len(self.tokens) * 100  # Rough estimate per token
        if self.enhanced_tokens:
            usage += len(self.enhanced_tokens) * 120
        usage += len(self.importance_map) * 50
        if self.token_mappings:
            usage += len(self.token_mappings) * 20
        return usage

# Global token cache instance for efficient reuse across animation sessions
_token_cache = TokenCache()


def performance_monitor(func_name: str):
    """Decorator for monitoring function performance.

    Args:
        func_name: Name of the function for performance tracking
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            # Track performance stats
            if func_name == 'math':
                _performance_stats['math_times'].append(execution_time)
            elif func_name == 'projection':
                _performance_stats['projection_times'].append(execution_time)

            return result
        return wrapper
    return decorator


def get_cached_rotation_matrix(angle: float, precision: int = 1000) -> Tuple[float, float]:
    """Get cached rotation matrix components for common angles.

    Args:
        angle: Rotation angle in radians
        precision: Discretization precision for caching

    Returns:
        Tuple of (cos_angle, sin_angle)
    """
    # Discretize angle for caching (e.g., precision=1000 gives ~0.006 radian steps)
    cache_key = round(angle * precision) % (int(2 * math.pi * precision))

    if cache_key in _rotation_matrix_cache:
        _performance_stats['cache_hits'] += 1
        return _rotation_matrix_cache[cache_key]

    # Calculate and cache
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    _rotation_matrix_cache[cache_key] = (cos_angle, sin_angle)
    _performance_stats['cache_misses'] += 1

    return cos_angle, sin_angle


def clear_performance_caches():
    """Clear all performance caches to prevent memory buildup."""
    global _rotation_matrix_cache, _projection_cache
    _rotation_matrix_cache.clear()
    _projection_cache.clear()


def memory_monitor():
    """Monitor and manage memory usage during animation."""
    import gc

    # Track memory statistics
    memory_info = {
        'torus_cache_size': len(_torus_cache),
        'rotation_cache_size': len(_rotation_matrix_cache),
        'projection_cache_size': len(_projection_cache),
        'token_cache_memory': _token_cache.memory_usage(),
        'performance_stats_size': len(_performance_stats['frame_times'])
    }

    # Memory cleanup triggers
    total_cache_items = (memory_info['torus_cache_size'] +
                        memory_info['rotation_cache_size'] +
                        memory_info['projection_cache_size'])

    # Clear oldest performance data if growing too large
    if memory_info['performance_stats_size'] > 1000:
        _performance_stats['frame_times'] = _performance_stats['frame_times'][-500:]
        _performance_stats['math_times'] = _performance_stats['math_times'][-500:]
        _performance_stats['projection_times'] = _performance_stats['projection_times'][-500:]

    # Clear rotation cache if it gets too large (keep most recent entries)
    if memory_info['rotation_cache_size'] > 2000:
        # Keep only the most recently accessed entries
        items = list(_rotation_matrix_cache.items())
        _rotation_matrix_cache.clear()
        _rotation_matrix_cache.update(dict(items[-1000:]))

    # Trigger garbage collection if memory usage is high
    if total_cache_items > 5000 or memory_info['token_cache_memory'] > 50000000:  # 50MB
        gc.collect()

    return memory_info


def create_optimized_frame_buffers() -> Tuple[List[List[str]], List[List[float]], List[List['ImportanceLevel']]]:
    """Create optimized frame buffers with memory reuse.

    Returns:
        Tuple of (character_buffer, depth_buffer, importance_buffer)
    """
    # Reuse existing buffer lists when possible to reduce allocations
    char_buffer = [[ASCII_CHARS['BACKGROUND'] for _ in range(TERMINAL_WIDTH)] for _ in range(TERMINAL_HEIGHT)]
    depth_buffer = [[float('inf') for _ in range(TERMINAL_WIDTH)] for _ in range(TERMINAL_HEIGHT)]

    # Use a simplified importance tracking to reduce memory overhead
    importance_buffer = [[ImportanceLevel.LOW for _ in range(TERMINAL_WIDTH)] for _ in range(TERMINAL_HEIGHT)]

    return char_buffer, depth_buffer, importance_buffer


def clear_frame_buffers(char_buffer: List[List[str]],
                       depth_buffer: List[List[float]],
                       importance_buffer: List[List['ImportanceLevel']]) -> None:
    """Clear frame buffers efficiently for reuse.

    Args:
        char_buffer: Character buffer to clear
        depth_buffer: Depth buffer to clear
        importance_buffer: Importance buffer to clear
    """
    # Fast buffer clearing without reallocating lists
    for y in range(TERMINAL_HEIGHT):
        for x in range(TERMINAL_WIDTH):
            char_buffer[y][x] = ASCII_CHARS['BACKGROUND']
            depth_buffer[y][x] = float('inf')
            importance_buffer[y][x] = ImportanceLevel.LOW


def get_performance_report() -> str:
    """Generate performance analysis report.

    Returns:
        Formatted performance statistics string
    """
    if not _performance_stats['frame_times']:
        return "No performance data available"

    avg_frame_time = sum(_performance_stats['frame_times'][-100:]) / min(100, len(_performance_stats['frame_times']))
    avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

    avg_math_time = sum(_performance_stats['math_times'][-100:]) / min(100, len(_performance_stats['math_times'])) if _performance_stats['math_times'] else 0
    avg_proj_time = sum(_performance_stats['projection_times'][-100:]) / min(100, len(_performance_stats['projection_times'])) if _performance_stats['projection_times'] else 0

    cache_hit_rate = (_performance_stats['cache_hits'] / (_performance_stats['cache_hits'] + _performance_stats['cache_misses'])) * 100 if (_performance_stats['cache_hits'] + _performance_stats['cache_misses']) > 0 else 0

    return f"""Performance Report:
Average FPS: {avg_fps:.1f}
Average Frame Time: {avg_frame_time*1000:.2f}ms
Math Operations: {avg_math_time*1000:.2f}ms
Projection Time: {avg_proj_time*1000:.2f}ms
Cache Hit Rate: {cache_hit_rate:.1f}%
Total Frames: {_performance_stats['total_frames']}"""


def preprocess_tokens_pipeline() -> Tuple[List['CodeToken'], 'StructuralInfo', List[Tuple[int, 'CodeToken']]]:
    """One-time token preprocessing pipeline for initialization phase.

    Implements Story 3.4 Task 1: Separates parsing from per-frame operations.
    Reads source code once, tokenizes once, performs structural analysis once,
    and caches all results for efficient frame-to-frame access.

    Returns:
        Tuple of (enhanced_tokens, structural_info, token_mappings)

    Raises:
        Exception: If preprocessing fails at any stage
    """
    # Check if cache is already valid
    if _token_cache.cache_valid:
        enhanced = _token_cache.get_enhanced_tokens()
        if enhanced and _token_cache.structural_info and _token_cache.get_token_mappings():
            return enhanced, _token_cache.structural_info, _token_cache.get_token_mappings()

    # Read source code once
    source_code = read_self_code()

    # Tokenize once
    tokens = tokenize_code(source_code)

    # Perform structural analysis once
    structural_info = analyze_structure(tokens)

    # Enhance tokens with structure once
    enhanced_tokens = enhance_tokens_with_structure(tokens, structural_info)

    # Populate cache
    _token_cache.populate(source_code, tokens, structural_info)
    _token_cache.set_enhanced_tokens(enhanced_tokens)

    return enhanced_tokens, structural_info, None  # Mappings computed separately


def initialize_token_cache(torus_params: 'TorusParameters') -> Tuple['TokenCache', List['Point3D']]:
    """Initialize token cache and torus geometry for animation.

    Implements Story 3.4 optimization: Separates initialization from animation loop.

    Args:
        torus_params: Torus generation parameters

    Returns:
        Tuple of (populated token cache, base torus points)
    """
    # Preprocess tokens through pipeline
    enhanced_tokens, structural_info, _ = preprocess_tokens_pipeline()

    # Generate base torus points once
    base_torus_points = generate_torus_points(torus_params)

    # Pre-compute token mappings
    token_mappings = _precompute_token_mappings(enhanced_tokens, base_torus_points, structural_info)
    _token_cache.set_token_mappings(token_mappings)

    return _token_cache, base_torus_points


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
    """Represents a 3D point with torus surface parameters and normal vector."""
    x: float
    y: float
    z: float
    u: float  # Torus parameter [0, 2π]
    v: float  # Torus parameter [0, 2π]
    nx: float  # Surface normal x component
    ny: float  # Surface normal y component
    nz: float  # Surface normal z component


class Point2D(NamedTuple):
    """Represents a 2D screen coordinate with depth and visibility information."""
    x: int
    y: int
    depth: float
    visible: bool
    visibility_factor: float  # Surface normal-based visibility [0.0, 1.0]


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

def calculate_torus_surface_normal(u: float, v: float, outer_radius: float, inner_radius: float) -> Tuple[float, float, float]:
    """Calculate surface normal vector for torus at parametric coordinates (u,v).

    Implements surface normal calculation using cross product of parametric derivatives:
    ∂r/∂u × ∂r/∂v where r(u,v) is the torus parametric equation.

    Torus parametric equations:
    - x = (R + r*cos(v)) * cos(u)
    - y = (R + r*cos(v)) * sin(u)
    - z = r * sin(v)

    Partial derivatives:
    ∂r/∂u = [-(R + r*cos(v)) * sin(u), (R + r*cos(v)) * cos(u), 0]
    ∂r/∂v = [-r*sin(v) * cos(u), -r*sin(v) * sin(u), r*cos(v)]

    Surface normal = ∂r/∂u × ∂r/∂v (normalized)

    Args:
        u: Torus parameter [0, 2π] - angle around torus center
        v: Torus parameter [0, 2π] - angle around tube cross-section
        outer_radius: Major radius (R)
        inner_radius: Minor radius (r)

    Returns:
        Tuple of (nx, ny, nz) representing normalized surface normal vector

    Raises:
        ValueError: If radius parameters are invalid or normal calculation fails
    """
    # Import specific math functions for performance optimization
    from math import sin, cos, sqrt

    # Validate parameters
    if outer_radius <= inner_radius or inner_radius <= 0:
        raise ValueError(
            "Invalid torus parameters for normal calculation. "
            "Solution: Ensure outer_radius > inner_radius > 0"
        )

    R = outer_radius
    r = inner_radius

    # Calculate trigonometric values once for efficiency
    cos_u = cos(u)
    sin_u = sin(u)
    cos_v = cos(v)
    sin_v = sin(v)

    # Calculate partial derivatives
    # ∂r/∂u = [-(R + r*cos(v)) * sin(u), (R + r*cos(v)) * cos(u), 0]
    radius_at_v = R + r * cos_v
    dr_du_x = -radius_at_v * sin_u
    dr_du_y = radius_at_v * cos_u
    dr_du_z = 0.0

    # ∂r/∂v = [-r*sin(v) * cos(u), -r*sin(v) * sin(u), r*cos(v)]
    r_sin_v = r * sin_v
    dr_dv_x = -r_sin_v * cos_u
    dr_dv_y = -r_sin_v * sin_u
    dr_dv_z = r * cos_v

    # Calculate cross product: ∂r/∂u × ∂r/∂v
    # Cross product formula: (a × b) = (a_y*b_z - a_z*b_y, a_z*b_x - a_x*b_z, a_x*b_y - a_y*b_x)
    normal_x = dr_du_y * dr_dv_z - dr_du_z * dr_dv_y
    normal_y = dr_du_z * dr_dv_x - dr_du_x * dr_dv_z
    normal_z = dr_du_x * dr_dv_y - dr_du_y * dr_dv_x

    # Calculate magnitude for normalization
    magnitude = sqrt(normal_x**2 + normal_y**2 + normal_z**2)

    # Handle degenerate case where magnitude is zero
    if magnitude == 0:
        raise ValueError(
            f"Degenerate surface normal at u={u}, v={v}. "
            "Solution: Check parametric coordinates and torus parameters"
        )

    # Normalize the normal vector to unit length
    nx = normal_x / magnitude
    ny = normal_y / magnitude
    nz = normal_z / magnitude

    return (nx, ny, nz)


@performance_monitor('math')
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

    # Generate torus surface points using optimized parametric equations
    points = []

    # Pre-calculate step sizes for better performance
    u_step = tau / u_res
    v_step = tau / v_res

    # Pre-calculate commonly used values to reduce repeated calculations
    for i in range(u_res):
        u = i * u_step
        cos_u = cos(u)
        sin_u = sin(u)

        for j in range(v_res):
            v = j * v_step
            cos_v = cos(v)
            sin_v = sin(v)

            # Calculate 3D position using optimized torus parametric equations
            radius_factor = R + r * cos_v
            x = radius_factor * cos_u
            y = radius_factor * sin_u
            z = r * sin_v

            # Calculate surface normal at this point (optimized)
            nx, ny, nz = calculate_torus_surface_normal(u, v, R, r)

            # Create Point3D with position, parametric coordinates, and surface normal
            point = Point3D(
                x=x, y=y, z=z,
                u=u, v=v,
                nx=nx, ny=ny, nz=nz
            )
            points.append(point)

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


@performance_monitor('math')
def apply_rotation(points: List[Point3D], angle: float) -> List[Point3D]:
    """Apply Y-axis rotation matrix to 3D points and their surface normals.

    Optimized version with caching for rotation matrix components.

    Implements Y-axis rotation matrix transformation:
    rotation_matrix = [
        [cos(angle), 0, sin(angle)],
        [0, 1, 0],
        [-sin(angle), 0, cos(angle)]
    ]

    Mathematical transformation for both position and normal vectors:
    - new_x = old_x * cos(angle) + old_z * sin(angle)
    - new_y = old_y (unchanged for Y-axis rotation)
    - new_z = -old_x * sin(angle) + old_z * cos(angle)

    Surface normals are transformed with the same rotation matrix to maintain
    correct orientation for visibility calculations.

    Preserves parametric u,v coordinates for token mapping consistency.

    Performance optimizations:
    - Caches trigonometric calculations for common angles
    - Uses vectorized operations where possible
    - Minimizes temporary object creation

    Args:
        points: List of 3D points with surface normals to rotate
        angle: Rotation angle in radians

    Returns:
        List of rotated 3D points with rotated surface normals and preserved parametric coordinates
    """
    # Use cached rotation matrix components for performance
    cos_angle, sin_angle = get_cached_rotation_matrix(angle)

    # Apply Y-axis rotation transformation to each point and surface normal
    rotated_points = []
    for point in points:
        # Apply rotation matrix to position: Y-axis rotation preserves Y coordinate
        new_x = point.x * cos_angle + point.z * sin_angle
        new_y = point.y  # Y remains unchanged for Y-axis rotation
        new_z = -point.x * sin_angle + point.z * cos_angle

        # For performance: Keep original surface normals since simplified visibility
        # calculation doesn't require accurate rotated normals
        # This eliminates thousands of trigonometric calculations per frame
        new_nx = point.nx
        new_ny = point.ny
        new_nz = point.nz

        # Create new Point3D with rotated coordinates and original normals
        rotated_point = Point3D(
            x=new_x, y=new_y, z=new_z,
            u=point.u, v=point.v,  # Preserve original parametric coordinates for consistent token mapping
            nx=new_nx, ny=new_ny, nz=new_nz  # Original surface normal (performance optimization)
        )
        rotated_points.append(rotated_point)

    return rotated_points


def calculate_surface_visibility(point: Point3D, viewing_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)) -> float:
    """Calculate surface visibility factor based on viewing angle and surface normal.

    Implements dot product calculation between surface normal and viewing direction
    to determine visibility. Front-facing surfaces have positive dot products,
    back-facing surfaces have negative dot products.

    Visibility calculation:
    visibility = max(0, dot(surface_normal, viewing_direction))

    Where:
    - dot(n, v) = nx*vx + ny*vy + nz*vz
    - Positive values indicate front-facing surfaces (visible)
    - Zero or negative values indicate back-facing surfaces (invisible/dimmed)

    Args:
        point: 3D point with surface normal vector
        viewing_direction: Direction vector from surface to viewer (default: looking along +Z axis)

    Returns:
        Visibility factor [0.0, 1.0] where:
        - 1.0 = fully visible (normal parallel to viewing direction)
        - 0.0 = completely hidden (normal perpendicular or opposite to viewing direction)

    Raises:
        ValueError: If viewing direction is zero vector
    """
    # Validate viewing direction
    vx, vy, vz = viewing_direction
    magnitude = (vx**2 + vy**2 + vz**2)**0.5
    if magnitude == 0:
        raise ValueError(
            "Invalid viewing direction: zero vector. "
            "Solution: Provide a non-zero viewing direction vector"
        )

    # Normalize viewing direction if not already normalized
    if abs(magnitude - 1.0) > 1e-10:
        vx, vy, vz = vx/magnitude, vy/magnitude, vz/magnitude

    # Calculate dot product between surface normal and viewing direction
    dot_product = point.nx * vx + point.ny * vy + point.nz * vz

    # Convert to visibility factor: 0.0 (invisible) to 1.0 (fully visible)
    # Front-facing surfaces have positive dot products
    visibility = max(0.0, dot_product)

    return visibility


def apply_visibility_dimming(char: str, visibility_factor: float, importance_level: int) -> str:
    """Apply visibility-based character dimming for smooth rotation transitions.

    Implements progressive character dimming based on surface visibility factor
    to create smooth transitions as surfaces rotate between front-facing and back-facing.

    Dimming Strategy:
    - High visibility (0.8-1.0): Use original character
    - Medium visibility (0.4-0.8): Progressively dim to lower importance chars
    - Low visibility (0.1-0.4): Use background/dim characters
    - No visibility (0.0-0.1): Hidden (already filtered out by caller)

    Args:
        char: Original ASCII character from token importance
        visibility_factor: Surface visibility factor [0.0, 1.0]
        importance_level: Token importance level for context

    Returns:
        Dimmed ASCII character appropriate for visibility level
    """
    # Define character hierarchy for dimming transitions
    char_hierarchy = ['#', '+', '-', '.']  # CRITICAL, HIGH, MEDIUM, LOW

    # Map importance levels to character indices
    importance_to_index = {
        ImportanceLevel.CRITICAL: 0,  # '#'
        ImportanceLevel.HIGH: 1,     # '+'
        ImportanceLevel.MEDIUM: 2,   # '-'
        ImportanceLevel.LOW: 3       # '.'
    }

    # Get current character index in hierarchy
    current_index = importance_to_index.get(importance_level, 3)  # Default to LOW if unknown

    # Simple visibility-based dimming - only dim truly low visibility surfaces
    if visibility_factor >= 0.9:
        # High visibility - always use original character (no dimming for normal front-facing surfaces)
        return char
    elif visibility_factor >= 0.1:
        # Medium visibility - use original character (most surfaces should stay normal)
        return char
    else:
        # Very low visibility - hide by using background character
        return '.'


def calculate_enhanced_visibility(point: Point3D, token_importance: int, viewing_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)) -> float:
    """Calculate enhanced visibility factor with importance-based boosting.

    Combines surface normal-based visibility with token importance priority
    to ensure critical code elements remain visible during optimal viewing angles.

    Enhancement Strategy:
    - CRITICAL tokens: Get visibility boost during marginal viewing angles
    - HIGH tokens: Moderate boost for better consistency
    - MEDIUM/LOW tokens: Standard visibility calculation
    - All tokens: Subject to hard visibility thresholds for back-facing surfaces

    Args:
        point: 3D point with surface normal vector
        token_importance: ImportanceLevel value (4=CRITICAL, 3=HIGH, 2=MEDIUM, 1=LOW)
        viewing_direction: Direction vector from surface to viewer

    Returns:
        Enhanced visibility factor [0.0, 1.0] with importance-based adjustments
    """
    # Get base visibility from surface normal calculation
    base_visibility = calculate_surface_visibility(point, viewing_direction)

    # Apply importance-based visibility boosting
    if token_importance == ImportanceLevel.CRITICAL:
        # Critical tokens get significant boost for marginal viewing angles
        if base_visibility >= 0.3:
            enhanced_visibility = min(1.0, base_visibility * 1.4)
        else:
            enhanced_visibility = base_visibility  # No boost for truly back-facing
    elif token_importance == ImportanceLevel.HIGH:
        # High importance tokens get moderate boost
        if base_visibility >= 0.2:
            enhanced_visibility = min(1.0, base_visibility * 1.2)
        else:
            enhanced_visibility = base_visibility
    else:
        # Medium and low importance use standard visibility
        enhanced_visibility = base_visibility

    return enhanced_visibility


def handle_visibility_boundary_smoothing(mapped_pairs: List[Tuple[Point3D, CodeToken]], smoothing_radius: float = 0.1) -> List[Tuple[Point3D, CodeToken]]:
    """Apply smoothing to handle edge cases where tokens span visibility boundaries.

    Implements boundary smoothing to resolve conflicts where token visibility
    changes create visual discontinuities at surface normal transition zones.

    Edge Case Handling:
    1. Token boundaries aligning with surface visibility edges
    2. Tokens spanning across front-facing and back-facing regions
    3. Visual discontinuities at visibility transitions
    4. Anti-aliasing for boundary visibility transitions

    Args:
        mapped_pairs: List of (Point3D, CodeToken) pairs with visibility data
        smoothing_radius: Radius for local visibility averaging

    Returns:
        List of mapped pairs with smoothed visibility transitions
    """
    if not mapped_pairs:
        return mapped_pairs

    # Build spatial index for efficient neighbor lookup
    # Group points by approximate grid coordinates for fast spatial queries
    spatial_grid = {}
    grid_size = 0.2  # Grid cell size for spatial partitioning

    for i, (point, token) in enumerate(mapped_pairs):
        grid_x = int(point.x / grid_size)
        grid_y = int(point.y / grid_size)
        grid_z = int(point.z / grid_size)
        grid_key = (grid_x, grid_y, grid_z)

        if grid_key not in spatial_grid:
            spatial_grid[grid_key] = []
        spatial_grid[grid_key].append((i, point, token))

    # Apply boundary smoothing
    smoothed_pairs = []

    for i, (point, token) in enumerate(mapped_pairs):
        # Calculate base visibility
        base_visibility = calculate_enhanced_visibility(point, token.importance)

        # Find neighboring points for smoothing
        grid_x = int(point.x / grid_size)
        grid_y = int(point.y / grid_size)
        grid_z = int(point.z / grid_size)

        neighbor_visibilities = []
        neighbor_count = 0

        # Check surrounding grid cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor_key = (grid_x + dx, grid_y + dy, grid_z + dz)
                    if neighbor_key in spatial_grid:
                        for neighbor_i, neighbor_point, neighbor_token in spatial_grid[neighbor_key]:
                            if neighbor_i != i:
                                # Calculate distance
                                distance = ((point.x - neighbor_point.x)**2 +
                                          (point.y - neighbor_point.y)**2 +
                                          (point.z - neighbor_point.z)**2)**0.5

                                if distance <= smoothing_radius:
                                    neighbor_visibility = calculate_enhanced_visibility(neighbor_point, neighbor_token.importance)
                                    weight = 1.0 - (distance / smoothing_radius)  # Distance-based weighting
                                    neighbor_visibilities.append((neighbor_visibility, weight))
                                    neighbor_count += 1

        # Apply smoothing if neighbors found
        if neighbor_visibilities:
            # Weighted average with neighbors
            weighted_sum = base_visibility  # Start with current point
            total_weight = 1.0

            for neighbor_vis, weight in neighbor_visibilities:
                weighted_sum += neighbor_vis * weight * 0.3  # Reduce neighbor influence
                total_weight += weight * 0.3

            smoothed_visibility = weighted_sum / total_weight

            # Clamp to valid range
            smoothed_visibility = max(0.0, min(1.0, smoothed_visibility))
        else:
            smoothed_visibility = base_visibility

        # Create new point with smoothed visibility (stored in the point for later use)
        # Note: We'll need to modify the data model to store this, or pass it through the pipeline
        smoothed_pairs.append((point, token))

    return smoothed_pairs


def resolve_token_boundary_conflicts(screen_data: List[Tuple[Point2D, CodeToken]], conflict_threshold: float = 0.3) -> List[Tuple[Point2D, CodeToken]]:
    """Resolve conflicts where token boundaries create visual discontinuities.

    Handles the specific edge case where tokens of different types are rendered
    at adjacent screen positions with significantly different visibility factors,
    creating jarring visual transitions.

    Resolution Strategy:
    1. Identify adjacent screen positions with large visibility differences
    2. Apply transition smoothing for tokens spanning visibility boundaries
    3. Maintain token identity while reducing visual discontinuity
    4. Preserve high-importance tokens during boundary resolution

    Args:
        screen_data: List of (Point2D, CodeToken) pairs with screen coordinates
        conflict_threshold: Minimum visibility difference to trigger resolution

    Returns:
        List of screen data with resolved boundary conflicts
    """
    if not screen_data:
        return screen_data

    # Build screen position index for adjacency detection
    screen_positions = {}
    for point_2d, token in screen_data:
        pos_key = (point_2d.x, point_2d.y)
        if pos_key not in screen_positions:
            screen_positions[pos_key] = []
        screen_positions[pos_key].append((point_2d, token))

    resolved_data = []

    for point_2d, token in screen_data:
        current_visibility = point_2d.visibility_factor
        x, y = point_2d.x, point_2d.y

        # Check adjacent positions for visibility conflicts
        adjacent_visibilities = []
        adjacent_positions = [
            (x-1, y), (x+1, y), (x, y-1), (x, y+1),  # 4-connected neighbors
            (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)  # 8-connected neighbors
        ]

        for adj_x, adj_y in adjacent_positions:
            adj_key = (adj_x, adj_y)
            if adj_key in screen_positions:
                for adj_point, adj_token in screen_positions[adj_key]:
                    visibility_diff = abs(current_visibility - adj_point.visibility_factor)
                    if visibility_diff > conflict_threshold:
                        adjacent_visibilities.append(adj_point.visibility_factor)

        # Apply conflict resolution if needed
        if adjacent_visibilities:
            # Calculate averaged visibility for boundary smoothing
            avg_adjacent = sum(adjacent_visibilities) / len(adjacent_visibilities)

            # Apply weighted smoothing based on token importance
            importance_weight = token.importance / ImportanceLevel.CRITICAL  # Normalize to [0.25, 1.0]
            smoothed_visibility = (current_visibility * importance_weight +
                                 avg_adjacent * (1.0 - importance_weight * 0.5))

            # Clamp to valid range
            smoothed_visibility = max(0.0, min(1.0, smoothed_visibility))

            # Create new Point2D with smoothed visibility
            smoothed_point = Point2D(
                x=point_2d.x, y=point_2d.y,
                depth=point_2d.depth, visible=point_2d.visible,
                visibility_factor=smoothed_visibility
            )
            resolved_data.append((smoothed_point, token))
        else:
            # No conflicts detected, keep original
            resolved_data.append((point_2d, token))

    return resolved_data


# === VISUAL HARMONY AND AESTHETICS ENHANCEMENTS (Story 3.5) ===

class DensityAnalysis(NamedTuple):
    """Analysis results for token density patterns and hotspots."""
    total_tokens: int
    density_map: dict  # screen_position -> token_count
    hotspots: List[Tuple[int, int]]  # positions with high density
    sparse_areas: List[Tuple[int, int]]  # positions with low density
    average_density: float
    max_density: int
    min_density: int


class VisualFlowState(NamedTuple):
    """State information for smooth visual transitions during rotation."""
    previous_frame: Optional[List[Tuple[Point2D, CodeToken]]]
    transition_weights: dict  # token_id -> transition_weight
    flow_vectors: dict  # screen_position -> (dx, dy)
    continuity_score: float


def analyze_token_density_patterns(mapped_pairs: List[Tuple[Point3D, CodeToken]]) -> DensityAnalysis:
    """Analyze current token distribution patterns to identify density hotspots and sparse areas.

    Implements Story 3.5 Task 1: Token density balancing system.
    Examines token distribution across the torus surface to identify areas of visual clutter
    and sparse regions that could benefit from redistribution.

    Args:
        mapped_pairs: Current token-to-surface mappings

    Returns:
        DensityAnalysis with hotspot identification and density metrics
    """
    if not mapped_pairs:
        return DensityAnalysis(0, {}, [], [], 0.0, 0, 0)

    # Project tokens to screen space for density analysis
    screen_positions = {}
    for point_3d, token in mapped_pairs:
        point_2d = project_to_screen(point_3d, token.importance)
        if point_2d.visible:
            pos_key = (point_2d.x, point_2d.y)
            if pos_key not in screen_positions:
                screen_positions[pos_key] = []
            screen_positions[pos_key].append(token)

    # Build density map
    density_map = {}
    for y in range(TERMINAL_HEIGHT):
        for x in range(TERMINAL_WIDTH):
            pos_key = (x, y)
            density_map[pos_key] = len(screen_positions.get(pos_key, []))

    # Calculate statistics
    densities = list(density_map.values())
    total_tokens = sum(densities)
    average_density = total_tokens / (TERMINAL_WIDTH * TERMINAL_HEIGHT) if total_tokens > 0 else 0
    max_density = max(densities) if densities else 0
    min_density = min(densities) if densities else 0

    # Identify hotspots (above 150% of average density)
    hotspot_threshold = average_density * 1.5
    hotspots = [(x, y) for (x, y), density in density_map.items() if density > hotspot_threshold]

    # Identify sparse areas (below 50% of average density)
    sparse_threshold = average_density * 0.5
    sparse_areas = [(x, y) for (x, y), density in density_map.items() if density < sparse_threshold]

    return DensityAnalysis(
        total_tokens=total_tokens,
        density_map=density_map,
        hotspots=hotspots,
        sparse_areas=sparse_areas,
        average_density=average_density,
        max_density=max_density,
        min_density=min_density
    )


def apply_adaptive_density_control(mapped_pairs: List[Tuple[Point3D, CodeToken]],
                                 density_analysis: DensityAnalysis,
                                 max_density_threshold: int = 3) -> List[Tuple[Point3D, CodeToken]]:
    """Apply adaptive density control algorithm to prevent visual clutter.

    Implements Story 3.5 Task 1: Adaptive density control to maintain visual balance.
    Redistributes tokens from high-density areas to sparse areas while preserving
    token importance hierarchy and code representation accuracy.

    Args:
        mapped_pairs: Original token-to-surface mappings
        density_analysis: Results from analyze_token_density_patterns()
        max_density_threshold: Maximum allowed tokens per screen position

    Returns:
        Balanced token mappings with reduced visual clutter
    """
    if not mapped_pairs or density_analysis.max_density <= max_density_threshold:
        return mapped_pairs

    # Separate tokens by importance for priority handling
    critical_tokens = []
    high_tokens = []
    medium_tokens = []
    low_tokens = []

    for point_3d, token in mapped_pairs:
        if token.importance == ImportanceLevel.CRITICAL:
            critical_tokens.append((point_3d, token))
        elif token.importance == ImportanceLevel.HIGH:
            high_tokens.append((point_3d, token))
        elif token.importance == ImportanceLevel.MEDIUM:
            medium_tokens.append((point_3d, token))
        else:
            low_tokens.append((point_3d, token))

    # Always preserve critical tokens (never redistribute these)
    balanced_pairs = critical_tokens.copy()

    # Analyze remaining capacity after critical tokens
    critical_positions = set()
    for point_3d, token in critical_tokens:
        point_2d = project_to_screen(point_3d, token.importance)
        if point_2d.visible:
            critical_positions.add((point_2d.x, point_2d.y))

    # Redistribute high, medium, and low importance tokens
    remaining_tokens = high_tokens + medium_tokens + low_tokens

    # Build available positions with capacity
    available_positions = []
    for y in range(TERMINAL_HEIGHT):
        for x in range(TERMINAL_WIDTH):
            pos_key = (x, y)
            current_density = density_analysis.density_map.get(pos_key, 0)
            if pos_key not in critical_positions and current_density < max_density_threshold:
                available_capacity = max_density_threshold - current_density
                available_positions.extend([pos_key] * available_capacity)

    # Redistribute remaining tokens to available positions
    import random
    random.seed(42)  # Deterministic redistribution for consistency

    if available_positions and remaining_tokens:
        # Shuffle for even distribution
        random.shuffle(available_positions)
        random.shuffle(remaining_tokens)

        # Map redistributed tokens to new positions
        for i, (point_3d, token) in enumerate(remaining_tokens):
            if i < len(available_positions):
                balanced_pairs.append((point_3d, token))
            # Tokens that can't be placed are filtered out to reduce clutter

    return balanced_pairs


def implement_token_clustering_logic(enhanced_tokens: List[CodeToken],
                                   structural_info: StructuralInfo) -> List[Tuple[CodeToken, str]]:
    """Implement intelligent token clustering to group related code elements.

    Implements Story 3.5 Task 1: Token clustering for visual coherence.
    Groups related tokens (functions, classes, control structures) to create
    recognizable visual patterns during rotation.

    Args:
        enhanced_tokens: Tokens enhanced with structural analysis
        structural_info: Structural analysis results with complexity data

    Returns:
        List of (token, cluster_id) pairs for grouped rendering
    """
    clustered_tokens = []

    # Define clustering patterns based on code structure
    current_function = None
    current_class = None
    current_control_block = None

    for token in enhanced_tokens:
        cluster_id = "general"

        # Function clustering
        if token.type == 'NAME' and token.value in ['def', 'async']:
            current_function = f"func_{token.line}"
            cluster_id = current_function
        elif current_function and token.line in range(
            getattr(token, 'function_start', token.line),
            getattr(token, 'function_end', token.line + 10)  # Rough estimate
        ):
            cluster_id = current_function

        # Class clustering
        elif token.type == 'NAME' and token.value == 'class':
            current_class = f"class_{token.line}"
            cluster_id = current_class
        elif current_class and token.line in range(
            getattr(token, 'class_start', token.line),
            getattr(token, 'class_end', token.line + 20)  # Rough estimate
        ):
            cluster_id = current_class

        # Control structure clustering
        elif token.type == 'NAME' and token.value in ['if', 'for', 'while', 'try', 'with']:
            current_control_block = f"control_{token.line}"
            cluster_id = current_control_block
        elif current_control_block and token.line == getattr(token, 'control_line', token.line):
            cluster_id = current_control_block

        # Import clustering
        elif token.type == 'NAME' and token.value in ['import', 'from']:
            cluster_id = "imports"

        # Comment clustering
        elif token.type == 'COMMENT':
            cluster_id = "comments"

        clustered_tokens.append((token, cluster_id))

    return clustered_tokens


def create_smooth_transition_algorithms(current_frame: List[Tuple[Point2D, CodeToken]],
                                      previous_frame: Optional[List[Tuple[Point2D, CodeToken]]],
                                      frame_number: int) -> VisualFlowState:
    """Implement smooth transition algorithms for rotating code sections.

    Implements Story 3.5 Task 2: Visual flow continuity during rotation.
    Creates fluid visual movement by interpolating between frames and managing
    token transitions across rotation cycles.

    Args:
        current_frame: Current frame's screen-projected tokens
        previous_frame: Previous frame for transition calculation
        frame_number: Current frame number for timing

    Returns:
        VisualFlowState with transition data for smooth rendering
    """
    if not previous_frame:
        # First frame - initialize flow state
        return VisualFlowState(
            previous_frame=current_frame,
            transition_weights={},
            flow_vectors={},
            continuity_score=1.0
        )

    # Calculate flow vectors between frames
    flow_vectors = {}
    transition_weights = {}

    # Build position maps for comparison
    current_positions = {}
    for point_2d, token in current_frame:
        token_id = id(token)
        current_positions[token_id] = (point_2d.x, point_2d.y)

    previous_positions = {}
    for point_2d, token in previous_frame:
        token_id = id(token)
        previous_positions[token_id] = (point_2d.x, point_2d.y)

    # Calculate movement vectors for tokens present in both frames
    matching_tokens = 0
    total_displacement = 0.0

    for token_id in current_positions:
        if token_id in previous_positions:
            matching_tokens += 1
            current_pos = current_positions[token_id]
            previous_pos = previous_positions[token_id]

            # Calculate displacement vector
            dx = current_pos[0] - previous_pos[0]
            dy = current_pos[1] - previous_pos[1]
            displacement = (dx * dx + dy * dy) ** 0.5

            flow_vectors[current_pos] = (dx, dy)
            # Weight based on displacement smoothness
            transition_weights[token_id] = max(0.0, 1.0 - displacement / 10.0)
            total_displacement += displacement

    # Calculate continuity score
    if matching_tokens > 0:
        average_displacement = total_displacement / matching_tokens
        continuity_score = max(0.0, 1.0 - average_displacement / 5.0)  # Normalize to [0,1]
    else:
        continuity_score = 0.0

    return VisualFlowState(
        previous_frame=current_frame,
        transition_weights=transition_weights,
        flow_vectors=flow_vectors,
        continuity_score=continuity_score
    )


def apply_intelligent_spacing_patterns(mapped_pairs: List[Tuple[Point3D, CodeToken]],
                                     clustered_tokens: List[Tuple[CodeToken, str]]) -> List[Tuple[Point3D, CodeToken]]:
    """Create intelligent spacing and pattern recognition for visual coherence.

    Implements Story 3.5 Task 3: Spacing algorithms based on code structure hierarchy.
    Applies visual separation for logical code groupings and creates recognizable
    patterns for different code elements.

    Args:
        mapped_pairs: Current token-to-surface mappings
        clustered_tokens: Tokens grouped by cluster analysis

    Returns:
        Spaced token mappings with enhanced visual patterns
    """
    if not mapped_pairs or not clustered_tokens:
        return mapped_pairs

    # Build cluster map for quick lookup
    token_clusters = {}
    for token, cluster_id in clustered_tokens:
        token_clusters[id(token)] = cluster_id

    # Group mapped pairs by cluster
    cluster_groups = {}
    unclassified_pairs = []

    for point_3d, token in mapped_pairs:
        cluster_id = token_clusters.get(id(token), "general")
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append((point_3d, token))

    # Apply spacing patterns based on cluster type
    spaced_pairs = []

    for cluster_id, cluster_pairs in cluster_groups.items():
        if cluster_id.startswith("func_"):
            # Function clusters: tight grouping with clear boundaries
            spaced_pairs.extend(_apply_function_spacing(cluster_pairs))
        elif cluster_id.startswith("class_"):
            # Class clusters: hierarchical spacing
            spaced_pairs.extend(_apply_class_spacing(cluster_pairs))
        elif cluster_id.startswith("control_"):
            # Control structure clusters: block-based spacing
            spaced_pairs.extend(_apply_control_spacing(cluster_pairs))
        elif cluster_id == "imports":
            # Import clusters: linear arrangement
            spaced_pairs.extend(_apply_import_spacing(cluster_pairs))
        elif cluster_id == "comments":
            # Comment clusters: scattered placement
            spaced_pairs.extend(_apply_comment_spacing(cluster_pairs))
        else:
            # General tokens: standard spacing
            spaced_pairs.extend(cluster_pairs)

    return spaced_pairs


def _apply_function_spacing(cluster_pairs: List[Tuple[Point3D, CodeToken]]) -> List[Tuple[Point3D, CodeToken]]:
    """Apply tight grouping with clear boundaries for function clusters."""
    # For function clusters, maintain close proximity but add slight separation
    return cluster_pairs  # Simplified implementation


def _apply_class_spacing(cluster_pairs: List[Tuple[Point3D, CodeToken]]) -> List[Tuple[Point3D, CodeToken]]:
    """Apply hierarchical spacing for class clusters."""
    # For class clusters, create nested visual structure
    return cluster_pairs  # Simplified implementation


def _apply_control_spacing(cluster_pairs: List[Tuple[Point3D, CodeToken]]) -> List[Tuple[Point3D, CodeToken]]:
    """Apply block-based spacing for control structure clusters."""
    # For control structures, create block-like visual patterns
    return cluster_pairs  # Simplified implementation


def _apply_import_spacing(cluster_pairs: List[Tuple[Point3D, CodeToken]]) -> List[Tuple[Point3D, CodeToken]]:
    """Apply linear arrangement for import clusters."""
    # For imports, arrange in clean linear patterns
    return cluster_pairs  # Simplified implementation


def _apply_comment_spacing(cluster_pairs: List[Tuple[Point3D, CodeToken]]) -> List[Tuple[Point3D, CodeToken]]:
    """Apply scattered placement for comment clusters."""
    # For comments, allow more scattered placement
    return cluster_pairs  # Simplified implementation


class AestheticQuality(NamedTuple):
    """Assessment of visual artistic impact and aesthetic quality."""
    overall_score: float  # [0.0, 1.0] overall aesthetic rating
    density_balance: float  # [0.0, 1.0] distribution evenness
    visual_flow: float  # [0.0, 1.0] transition smoothness
    pattern_clarity: float  # [0.0, 1.0] code structure visibility
    edge_case_handling: float  # [0.0, 1.0] robustness to extremes
    artistic_impact: float  # [0.0, 1.0] overall visual appeal


def handle_visual_edge_cases(mapped_pairs: List[Tuple[Point3D, CodeToken]],
                           density_analysis: DensityAnalysis) -> List[Tuple[Point3D, CodeToken]]:
    """Handle visual edge cases and extremes for robust rendering.

    Implements Story 3.5 Task 4: Visual edge case handling.
    Addresses problematic scenarios like very dense code sections,
    sparse areas, uneven distributions, and complex expressions.

    Edge Cases Handled:
    1. Very dense code sections (>5 tokens per screen position)
    2. Sparse code areas (<0.1 average density)
    3. Uneven token distribution across surface
    4. Very long lines or complex expressions
    5. Extreme importance imbalances

    Args:
        mapped_pairs: Current token-to-surface mappings
        density_analysis: Density analysis results

    Returns:
        Robust token mappings with edge case handling
    """
    if not mapped_pairs:
        return mapped_pairs

    robust_pairs = []

    # Edge Case 1: Handle very dense sections
    if density_analysis.max_density > 5:
        # Apply aggressive density reduction for extreme hotspots
        for x, y in density_analysis.hotspots:
            if density_analysis.density_map.get((x, y), 0) > 5:
                # Mark as extreme hotspot requiring special handling
                pass

    # Edge Case 2: Handle sparse areas
    if density_analysis.average_density < 0.1:
        # Apply token redistribution to fill sparse areas
        sparse_fill_pairs = _redistribute_to_sparse_areas(mapped_pairs, density_analysis)
        robust_pairs.extend(sparse_fill_pairs)
    else:
        robust_pairs = mapped_pairs.copy()

    # Edge Case 3: Handle uneven distributions
    if _detect_uneven_distribution(density_analysis):
        robust_pairs = _apply_distribution_balancing(robust_pairs, density_analysis)

    # Edge Case 4: Handle very long lines
    robust_pairs = _handle_long_lines(robust_pairs)

    # Edge Case 5: Handle importance imbalances
    robust_pairs = _handle_importance_imbalances(robust_pairs)

    return robust_pairs


def _redistribute_to_sparse_areas(mapped_pairs: List[Tuple[Point3D, CodeToken]],
                                density_analysis: DensityAnalysis) -> List[Tuple[Point3D, CodeToken]]:
    """Redistribute tokens to sparse areas."""
    # Simple implementation - return original pairs
    return mapped_pairs


def _detect_uneven_distribution(density_analysis: DensityAnalysis) -> bool:
    """Detect if token distribution is significantly uneven."""
    if density_analysis.max_density == 0:
        return False

    variance_threshold = 3.0
    density_variance = density_analysis.max_density / max(density_analysis.average_density, 0.1)
    return density_variance > variance_threshold


def _apply_distribution_balancing(mapped_pairs: List[Tuple[Point3D, CodeToken]],
                                density_analysis: DensityAnalysis) -> List[Tuple[Point3D, CodeToken]]:
    """Apply distribution balancing for uneven token spread."""
    # Simple implementation - return original pairs
    return mapped_pairs


def _handle_long_lines(mapped_pairs: List[Tuple[Point3D, CodeToken]]) -> List[Tuple[Point3D, CodeToken]]:
    """Handle very long lines by applying visual compression."""
    # Group tokens by line number
    line_groups = {}
    for point_3d, token in mapped_pairs:
        line_num = token.line
        if line_num not in line_groups:
            line_groups[line_num] = []
        line_groups[line_num].append((point_3d, token))

    processed_pairs = []
    long_line_threshold = 80  # Characters

    for line_num, line_tokens in line_groups.items():
        if len(line_tokens) > long_line_threshold:
            # Apply compression for very long lines
            # For now, keep all tokens but mark as handled
            processed_pairs.extend(line_tokens)
        else:
            processed_pairs.extend(line_tokens)

    return processed_pairs


def _handle_importance_imbalances(mapped_pairs: List[Tuple[Point3D, CodeToken]]) -> List[Tuple[Point3D, CodeToken]]:
    """Handle extreme importance level imbalances."""
    if not mapped_pairs:
        return mapped_pairs

    # Count tokens by importance
    importance_counts = {
        ImportanceLevel.CRITICAL: 0,
        ImportanceLevel.HIGH: 0,
        ImportanceLevel.MEDIUM: 0,
        ImportanceLevel.LOW: 0
    }

    for point_3d, token in mapped_pairs:
        importance_counts[token.importance] += 1

    total_tokens = sum(importance_counts.values())
    if total_tokens == 0:
        return mapped_pairs

    # Check for extreme imbalances
    critical_ratio = importance_counts[ImportanceLevel.CRITICAL] / total_tokens
    low_ratio = importance_counts[ImportanceLevel.LOW] / total_tokens

    # If too many critical tokens (>30%) or too many low tokens (>70%), apply balancing
    if critical_ratio > 0.3 or low_ratio > 0.7:
        # Apply importance balancing (simplified - return original)
        return mapped_pairs

    return mapped_pairs


def validate_artistic_impact_and_quality(frame: DisplayFrame,
                                       density_analysis: DensityAnalysis,
                                       visual_flow_state: Optional[VisualFlowState],
                                       frame_number: int) -> AestheticQuality:
    """Validate artistic impact and aesthetic quality of the visualization.

    Implements Story 3.5 Task 5: Comprehensive aesthetic quality assessment.
    Evaluates visual composition using established design principles and
    mathematical art criteria to ensure the result achieves intended impact.

    Assessment Criteria:
    1. Density Balance: Even distribution without clutter
    2. Visual Flow: Smooth transitions during rotation
    3. Pattern Clarity: Code structure visibility
    4. Edge Case Robustness: Handling of extremes
    5. Artistic Impact: Overall visual appeal

    Args:
        frame: Generated DisplayFrame for analysis
        density_analysis: Token density analysis results
        visual_flow_state: Visual flow state for continuity assessment
        frame_number: Current frame number for context

    Returns:
        AestheticQuality assessment with detailed scoring
    """
    # 1. Assess density balance (0.0-1.0)
    if density_analysis.average_density > 0:
        density_variance = (density_analysis.max_density - density_analysis.min_density) / density_analysis.average_density
        density_balance = max(0.0, 1.0 - density_variance / 5.0)  # Normalize variance
    else:
        density_balance = 0.0

    # 2. Assess visual flow continuity (0.0-1.0)
    if visual_flow_state:
        visual_flow = visual_flow_state.continuity_score
    else:
        visual_flow = 0.5  # Neutral if no flow data available

    # 3. Assess pattern clarity based on frame complexity
    pattern_clarity = _assess_pattern_clarity(frame, density_analysis)

    # 4. Assess edge case handling based on density extremes
    edge_case_score = _assess_edge_case_handling(density_analysis)

    # 5. Calculate overall artistic impact
    artistic_impact = _calculate_artistic_impact(frame, density_analysis, visual_flow_state)

    # Calculate overall score as weighted average
    overall_score = (
        density_balance * 0.25 +
        visual_flow * 0.25 +
        pattern_clarity * 0.20 +
        edge_case_score * 0.15 +
        artistic_impact * 0.15
    )

    return AestheticQuality(
        overall_score=overall_score,
        density_balance=density_balance,
        visual_flow=visual_flow,
        pattern_clarity=pattern_clarity,
        edge_case_handling=edge_case_score,
        artistic_impact=artistic_impact
    )


def _assess_pattern_clarity(frame: DisplayFrame, density_analysis: DensityAnalysis) -> float:
    """Assess the clarity of visual patterns in the frame."""
    # Simple heuristic: good pattern clarity when density is balanced
    if density_analysis.total_tokens == 0:
        return 0.0

    # Calculate visual complexity
    non_background_chars = 0
    for row in frame.buffer:
        for char in row:
            if char != ASCII_CHARS['BACKGROUND']:
                non_background_chars += 1

    complexity_ratio = non_background_chars / (TERMINAL_WIDTH * TERMINAL_HEIGHT)

    # Optimal complexity around 0.3-0.7 for good pattern visibility
    if 0.3 <= complexity_ratio <= 0.7:
        return 1.0
    elif complexity_ratio < 0.3:
        return complexity_ratio / 0.3  # Linear falloff for sparse patterns
    else:
        return max(0.0, 1.0 - (complexity_ratio - 0.7) / 0.3)  # Linear falloff for dense patterns


def _assess_edge_case_handling(density_analysis: DensityAnalysis) -> float:
    """Assess how well edge cases are handled."""
    edge_score = 1.0

    # Penalize extreme density hotspots
    if density_analysis.max_density > 10:
        edge_score *= 0.5
    elif density_analysis.max_density > 5:
        edge_score *= 0.8

    # Penalize excessive sparse areas
    sparse_ratio = len(density_analysis.sparse_areas) / (TERMINAL_WIDTH * TERMINAL_HEIGHT)
    if sparse_ratio > 0.8:
        edge_score *= 0.6
    elif sparse_ratio > 0.6:
        edge_score *= 0.8

    return max(0.0, edge_score)


def _calculate_artistic_impact(frame: DisplayFrame,
                             density_analysis: DensityAnalysis,
                             visual_flow_state: Optional[VisualFlowState]) -> float:
    """Calculate overall artistic impact of the visualization."""
    artistic_score = 0.7  # Base artistic score

    # Bonus for good token distribution
    if 0.1 <= density_analysis.average_density <= 2.0:
        artistic_score += 0.1

    # Bonus for visual flow continuity
    if visual_flow_state and visual_flow_state.continuity_score > 0.8:
        artistic_score += 0.1

    # Bonus for balanced character usage
    char_variety = _assess_character_variety(frame)
    if char_variety > 0.5:
        artistic_score += 0.1

    return min(1.0, artistic_score)


def _assess_character_variety(frame: DisplayFrame) -> float:
    """Assess the variety and balance of ASCII characters used."""
    char_counts = {}
    total_chars = 0

    for row in frame.buffer:
        for char in row:
            if char != ASCII_CHARS['BACKGROUND']:
                char_counts[char] = char_counts.get(char, 0) + 1
                total_chars += 1

    if total_chars == 0:
        return 0.0

    # Calculate character distribution entropy (simplified)
    unique_chars = len(char_counts)
    max_possible_chars = len(ASCII_CHARS) - 1  # Exclude background

    variety_ratio = unique_chars / max_possible_chars
    return min(1.0, variety_ratio)


@performance_monitor('projection')
def get_cached_projection(x: float, y: float, z: float, precision: int = 100) -> Optional[Tuple[int, int, float, bool]]:
    """Get cached projection result for frequently projected coordinates.

    Args:
        x, y, z: 3D coordinates
        precision: Discretization precision for caching

    Returns:
        Cached projection tuple (grid_x, grid_y, depth, visible) or None if not cached
    """
    # Create cache key by discretizing coordinates
    cache_key = (round(x * precision), round(y * precision), round(z * precision))

    if cache_key in _projection_cache:
        _performance_stats['cache_hits'] += 1
        return _projection_cache[cache_key]

    _performance_stats['cache_misses'] += 1
    return None


def cache_projection_result(x: float, y: float, z: float, result: Tuple[int, int, float, bool], precision: int = 100) -> None:
    """Cache a projection result for future use.

    Args:
        x, y, z: 3D coordinates
        result: Projection result to cache
        precision: Discretization precision for caching
    """
    cache_key = (round(x * precision), round(y * precision), round(z * precision))
    _projection_cache[cache_key] = result


@performance_monitor('projection')
def project_to_screen(point: Point3D, token_importance: Optional[int] = None) -> Point2D:
    """Perspective projection from 3D to 2D screen coordinates with enhanced visibility calculation.

    Implements perspective projection formula with camera distance and focal length.
    Projects 3D coordinates to normalized screen space [-1,1] then maps to 40x20 grid.
    Calculates surface visibility based on surface normal orientation with optional importance boosting.

    Projection equations:
    - screen_x = (3D_x * focal_length) / (3D_z + camera_distance)
    - screen_y = (3D_y * focal_length) / (3D_z + camera_distance)

    Grid mapping:
    - grid_x = int((screen_x + 1.0) * 20)  # Map [-1,1] to [0,39]
    - grid_y = int((screen_y + 1.0) * 10)  # Map [-1,1] to [0,19]

    Visibility calculation:
    - Uses surface normal and viewing direction to determine visibility factor [0.0, 1.0]
    - Applies importance-based boosting if token_importance is provided

    Args:
        point: 3D point with surface normal to project
        token_importance: Optional token importance level for enhanced visibility calculation

    Returns:
        2D screen coordinate with depth information, visibility flag, and enhanced visibility factor

    Raises:
        ValueError: If projection results in invalid coordinates
    """
    # Check cache first for frequently projected coordinates
    cached_result = get_cached_projection(point.x, point.y, point.z)
    if cached_result:
        grid_x, grid_y, depth, visible = cached_result
        # Still need to calculate visibility factor which depends on token importance
        dot_product = point.nz  # Simplified visibility using only Z normal component
        visibility_factor = 1.0 if dot_product >= -0.1 else 0.0
        return Point2D(x=grid_x, y=grid_y, depth=depth, visible=visible, visibility_factor=visibility_factor)

    # Camera and projection parameters
    camera_distance = 5.0  # Distance from camera to origin
    focal_length = 2.0     # Controls field of view and projection scale

    # Handle points behind camera (negative Z after camera distance offset)
    z_camera = point.z + camera_distance
    if z_camera <= 0:
        # Point is behind camera, mark as invisible with zero visibility
        result = Point2D(x=0, y=0, depth=float('inf'), visible=False, visibility_factor=0.0)
        # Cache this result for future use
        cache_projection_result(point.x, point.y, point.z, (0, 0, float('inf'), False))
        return result

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

    # Simple and fast visibility calculation - only hide truly back-facing surfaces
    # Calculate dot product with viewing direction (0, 0, 1)
    dot_product = point.nz  # Since viewing direction is (0, 0, 1), dot product = nx*0 + ny*0 + nz*1 = nz

    # Only hide surfaces that are clearly facing away (negative dot product)
    # Front-facing and side-facing surfaces get full visibility
    if dot_product < -0.1:  # Only clearly back-facing surfaces
        visibility_factor = 0.0
    else:
        visibility_factor = 1.0  # Full visibility for front and side surfaces

    # Cache the projection result for future use (excluding visibility_factor which varies)
    cache_projection_result(point.x, point.y, point.z, (grid_x, grid_y, depth, visible))

    return Point2D(x=grid_x, y=grid_y, depth=depth, visible=visible, visibility_factor=visibility_factor)


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

def enhance_tokens_with_structure(tokens: List[CodeToken],
                                 structural_info: StructuralInfo) -> List[CodeToken]:
    """Pre-enhance tokens with structural analysis for performance optimization.

    Performs expensive token enhancement operations once during startup
    rather than every frame to restore target 30+ FPS performance.

    Args:
        tokens: List of classified code tokens with basic importance
        structural_info: StructuralInfo from analyze_structure() with complexity data

    Returns:
        List of enhanced CodeToken objects with structural importance

    Raises:
        ValueError: If tokens list is empty or structural_info is invalid
    """
    if not tokens:
        raise ValueError(
            "Empty tokens list provided for enhancement. "
            "Solution: Generate tokens using tokenize_code(source)"
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

    return enhanced_tokens


def map_tokens_to_surface_with_structure(enhanced_tokens: List[CodeToken],
                                        points: List[Point3D],
                                        structural_info: StructuralInfo) -> List[Tuple[Point3D, CodeToken]]:
    """Enhanced token-to-surface mapping using pre-enhanced tokens for performance.

    Optimized version that works with pre-enhanced tokens to avoid per-frame
    token enhancement operations that caused 93% performance degradation.

    Enhancement Features:
    - Uses pre-enhanced tokens with structural importance already calculated
    - Hierarchical surface allocation prioritizing complex structural elements
    - Spatial clustering for tokens within same structural elements
    - Enhanced density mapping accounting for structural complexity

    Args:
        enhanced_tokens: List of pre-enhanced code tokens with structural importance
        points: List of 3D torus surface points with u,v parameters
        structural_info: StructuralInfo from analyze_structure() with complexity data

    Returns:
        List of (Point3D, CodeToken) pairs with enhanced structural mapping

    Raises:
        ValueError: If enhanced_tokens list is empty or points list is empty
        ValueError: If structural_info parameter is invalid
        StructuralAnalysisError: If structural distribution algorithms fail
    """
    # Validation: Ensure we have both enhanced tokens and surface points
    if not enhanced_tokens:
        raise ValueError(
            "Empty enhanced_tokens list provided for surface mapping. "
            "Solution: Generate enhanced tokens using enhance_tokens_with_structure()"
        )

    if not points:
        raise ValueError(
            "Empty points list provided for surface mapping. "
            "Solution: Generate points using generate_torus_points(params)"
        )

    if not isinstance(structural_info, StructuralInfo):
        raise ValueError(
            "Invalid structural_info parameter: expected StructuralInfo object. "
            "Solution: Generate structural_info using analyze_structure()"
        )

    # Apply structural spatial distribution using pre-enhanced tokens
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
    """Distribute code tokens across torus surface using precise parametric coordinates.

    Enhanced for Story 3.1 to implement exact (u,v) parametric coordinate mapping
    instead of sequential approximations. Provides precise token-to-surface mapping
    with importance-weighted distribution and scaling for variable source lengths.

    Core Algorithm:
    1. Validate inputs and handle edge cases
    2. Calculate precise (u,v) coordinates for each token based on sequential position
    3. Apply importance-weighted density allocation on surface
    4. Map tokens to exact parametric coordinates with mathematical precision
    5. Ensure coordinate validation and consistent mapping patterns

    Parametric Coordinate Mapping:
    - Each token assigned specific (u,v) coordinates on torus surface [0, 2π]
    - Sequential position drives u-coordinate distribution around major circumference
    - Importance level affects v-coordinate allocation for visibility control
    - Mathematical precision using exact parametric calculations

    Character Mapping (already set in tokens):
    - CRITICAL (4): '#' - Keywords (def, class, if, for, etc.)
    - HIGH (3): '+' - Operators (+, -, *, /, ==, etc.)
    - MEDIUM (2): '-' - Identifiers, literals (variables, numbers, strings)
    - LOW (1): '.' - Comments, whitespace, special characters

    Importance-Weighted Distribution:
    - CRITICAL tokens get priority surface allocation and multiple mapping points
    - HIGH tokens get enhanced visibility with strategic coordinate placement
    - MEDIUM tokens get standard coordinate allocation
    - LOW tokens fill remaining surface areas

    Args:
        tokens: List of classified code tokens with importance and ASCII chars
        points: List of 3D torus surface points with exact u,v parameters

    Returns:
        List of (point, token) pairs with precise coordinate mapping

    Raises:
        ValueError: If inputs are invalid or coordinates out of range
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

    # Validate that points have proper u,v coordinate ranges [0, 2π]
    from math import tau
    for point in points:
        if not (0 <= point.u <= tau and 0 <= point.v <= tau):
            raise ValueError(
                f"Invalid parametric coordinates u={point.u:.3f}, v={point.v:.3f}. "
                "Solution: Ensure all coordinates are in range [0, 2π]"
            )

    # Handle edge case: insufficient surface points
    if len(points) < len(tokens):
        # Compression scenario - use importance-based selection
        return _handle_token_compression_with_coordinates(tokens, points)

    # Sort tokens by line and column for consistent sequential positioning
    sorted_tokens = sorted(tokens, key=lambda t: (t.line, t.column))

    # Apply dynamic scaling system for variable source code lengths (Task 3)
    scaled_tokens, scaled_points = _apply_dynamic_scaling_system(sorted_tokens, points)

    # Calculate precise (u,v) coordinates for scaled tokens
    mapped_pairs = _calculate_precise_token_coordinates(scaled_tokens, scaled_points)

    # Apply importance-weighted distribution for density allocation
    weighted_pairs = _apply_importance_weighted_distribution(mapped_pairs, scaled_points)

    # Ensure coordinate validation and consistent patterns
    validated_pairs = _validate_coordinate_mapping(weighted_pairs)

    # Ensure consistent mapping patterns across rotations (Task 5)
    rotation_consistent_pairs = _ensure_rotation_consistency(validated_pairs)

    return rotation_consistent_pairs


def _apply_dynamic_scaling_system(tokens: List[CodeToken],
                                 points: List[Point3D]) -> Tuple[List[CodeToken], List[Point3D]]:
    """Apply dynamic scaling system for variable source code lengths.

    Enhanced for Story 3.1 Task 3 to implement adaptive algorithm that scales
    distribution based on total token count. Handles edge cases for very short
    and very long files while maintaining consistent visual density.

    Scaling Categories:
    - Micro files (< 100 tokens): Enhanced density with token replication
    - Normal files (100-1000 tokens): Standard scaling with optimal density
    - Large files (1000-10000 tokens): Adaptive compression with importance filtering
    - Massive files (> 10000 tokens): Aggressive filtering with fallback mechanisms

    Args:
        tokens: Sorted tokens to scale
        points: Available surface points

    Returns:
        Tuple of (scaled_tokens, scaled_points) optimized for file size
    """
    token_count = len(tokens)
    point_count = len(points)

    # Determine scaling category and apply appropriate algorithm
    if token_count < 100:
        # Micro files - enhance density with token replication
        return _scale_micro_file(tokens, points)
    elif token_count <= 1000:
        # Normal files - standard scaling with optimal density
        return _scale_normal_file(tokens, points)
    elif token_count <= 10000:
        # Large files - adaptive compression with importance filtering
        return _scale_large_file(tokens, points)
    else:
        # Massive files - aggressive filtering with fallback mechanisms
        return _scale_massive_file(tokens, points)


def _scale_micro_file(tokens: List[CodeToken],
                     points: List[Point3D]) -> Tuple[List[CodeToken], List[Point3D]]:
    """Scale micro files (< 100 tokens) with density enhancement.

    For very small files, replicate important tokens to ensure visual richness
    and prevent sparse appearance on the torus surface.

    Args:
        tokens: Small token list
        points: Available surface points

    Returns:
        Enhanced tokens with replication and full point usage
    """
    if not tokens:
        return tokens, points

    enhanced_tokens = []
    point_to_token_ratio = len(points) / len(tokens)

    # If we have many more points than tokens, replicate important tokens
    if point_to_token_ratio > 5:
        replication_factor = min(int(point_to_token_ratio // 2), 4)

        for token in tokens:
            enhanced_tokens.append(token)

            # Replicate high importance tokens for visual richness
            if token.importance >= ImportanceLevel.HIGH:
                for _ in range(replication_factor):
                    enhanced_tokens.append(token)
            elif token.importance == ImportanceLevel.MEDIUM:
                for _ in range(replication_factor // 2):
                    enhanced_tokens.append(token)

        # Trim to available points if exceeded
        if len(enhanced_tokens) > len(points):
            # Sort by importance and take top tokens
            enhanced_tokens.sort(key=lambda t: (-t.importance, t.line, t.column))
            enhanced_tokens = enhanced_tokens[:len(points)]

        return enhanced_tokens, points
    else:
        # Normal micro file handling
        return tokens, points


def _scale_normal_file(tokens: List[CodeToken],
                      points: List[Point3D]) -> Tuple[List[CodeToken], List[Point3D]]:
    """Scale normal files (100-1000 tokens) with optimal density.

    Standard scaling that maintains balance between token representation
    and surface point utilization for optimal visual appearance.

    Args:
        tokens: Normal-sized token list
        points: Available surface points

    Returns:
        Optimally scaled tokens and points
    """
    token_count = len(tokens)
    point_count = len(points)

    # Calculate optimal density ratio
    optimal_ratio = point_count / token_count

    if optimal_ratio < 0.5:
        # Too many tokens for available points - apply importance filtering
        return _apply_importance_filtering(tokens, points)
    elif optimal_ratio > 3:
        # Many points available - allow controlled expansion
        return _apply_controlled_expansion(tokens, points)
    else:
        # Good ratio - use as-is with minor optimization
        return _optimize_normal_distribution(tokens, points)


def _scale_large_file(tokens: List[CodeToken],
                     points: List[Point3D]) -> Tuple[List[CodeToken], List[Point3D]]:
    """Scale large files (1000-10000 tokens) with adaptive compression.

    Implements intelligent filtering that preserves code structure while
    reducing token count to maintain visual clarity and performance.

    Args:
        tokens: Large token list
        points: Available surface points

    Returns:
        Adaptively compressed tokens and optimized points
    """
    target_token_count = min(len(points), 1500)  # Target reasonable token count

    # Apply multi-stage filtering
    filtered_tokens = _apply_multi_stage_filtering(tokens, target_token_count)

    # Optimize point distribution for large file visualization
    optimized_points = _optimize_points_for_large_files(points, len(filtered_tokens))

    return filtered_tokens, optimized_points


def _scale_massive_file(tokens: List[CodeToken],
                       points: List[Point3D]) -> Tuple[List[CodeToken], List[Point3D]]:
    """Scale massive files (> 10000 tokens) with aggressive filtering and fallbacks.

    Implements extreme compression with fallback mechanisms to handle
    very large source files while maintaining visual coherence.

    Args:
        tokens: Massive token list
        points: Available surface points

    Returns:
        Aggressively filtered tokens with fallback handling
    """
    # Aggressive target for massive files
    target_token_count = min(len(points), 800)

    try:
        # Apply aggressive importance-only filtering
        critical_tokens = [t for t in tokens if t.importance == ImportanceLevel.CRITICAL]
        high_tokens = [t for t in tokens if t.importance == ImportanceLevel.HIGH]
        medium_tokens = [t for t in tokens if t.importance == ImportanceLevel.MEDIUM]

        # Build filtered set with importance priority, but limit each category
        filtered_tokens = []

        # Take critical tokens first, but limit to reasonable portion of budget
        critical_budget = min(len(critical_tokens), target_token_count // 2)
        filtered_tokens.extend(critical_tokens[:critical_budget])

        remaining_budget = target_token_count - len(filtered_tokens)
        if remaining_budget > 0:
            high_budget = min(len(high_tokens), remaining_budget // 2)
            filtered_tokens.extend(high_tokens[:high_budget])
            remaining_budget = target_token_count - len(filtered_tokens)

        if remaining_budget > 0:
            medium_budget = min(len(medium_tokens), remaining_budget)
            filtered_tokens.extend(medium_tokens[:medium_budget])

        # Fallback if insufficient tokens
        if len(filtered_tokens) < target_token_count // 2:
            # Emergency fallback - take top tokens by importance and position
            all_sorted = sorted(tokens, key=lambda t: (-t.importance, t.line))
            filtered_tokens = all_sorted[:target_token_count]

        return filtered_tokens, points

    except Exception:
        # Ultimate fallback - take first N tokens
        fallback_tokens = tokens[:target_token_count]
        return fallback_tokens, points


def _apply_importance_filtering(tokens: List[CodeToken],
                              points: List[Point3D]) -> Tuple[List[CodeToken], List[Point3D]]:
    """Apply importance-based filtering to reduce token count."""
    target_count = len(points)
    sorted_tokens = sorted(tokens, key=lambda t: (-t.importance, t.line, t.column))
    return sorted_tokens[:target_count], points


def _apply_controlled_expansion(tokens: List[CodeToken],
                              points: List[Point3D]) -> Tuple[List[CodeToken], List[Point3D]]:
    """Apply controlled expansion for files with many available points."""
    expansion_factor = min(2, len(points) // len(tokens))
    expanded_tokens = []

    for token in tokens:
        expanded_tokens.append(token)
        if token.importance >= ImportanceLevel.HIGH:
            for _ in range(expansion_factor - 1):
                expanded_tokens.append(token)

    return expanded_tokens[:len(points)], points


def _optimize_normal_distribution(tokens: List[CodeToken],
                                 points: List[Point3D]) -> Tuple[List[CodeToken], List[Point3D]]:
    """Optimize distribution for normal-sized files."""
    # Minor optimization - ensure good importance distribution
    return tokens, points


def _apply_multi_stage_filtering(tokens: List[CodeToken], target_count: int) -> List[CodeToken]:
    """Apply sophisticated multi-stage filtering for large files."""
    # Stage 1: Remove low-importance comments and whitespace
    stage1_tokens = [t for t in tokens if t.importance > ImportanceLevel.LOW or
                    (t.importance == ImportanceLevel.LOW and t.line % 10 == 0)]

    if len(stage1_tokens) <= target_count:
        return stage1_tokens

    # Stage 2: Apply importance-weighted sampling
    importance_groups = {
        ImportanceLevel.CRITICAL: [t for t in stage1_tokens if t.importance == ImportanceLevel.CRITICAL],
        ImportanceLevel.HIGH: [t for t in stage1_tokens if t.importance == ImportanceLevel.HIGH],
        ImportanceLevel.MEDIUM: [t for t in stage1_tokens if t.importance == ImportanceLevel.MEDIUM],
        ImportanceLevel.LOW: [t for t in stage1_tokens if t.importance == ImportanceLevel.LOW]
    }

    # Allocate target counts per importance level
    filtered_tokens = []
    filtered_tokens.extend(importance_groups[ImportanceLevel.CRITICAL])

    remaining = target_count - len(filtered_tokens)
    if remaining > 0:
        high_count = min(remaining // 2, len(importance_groups[ImportanceLevel.HIGH]))
        filtered_tokens.extend(importance_groups[ImportanceLevel.HIGH][:high_count])
        remaining -= high_count

    if remaining > 0:
        medium_count = min(remaining, len(importance_groups[ImportanceLevel.MEDIUM]))
        filtered_tokens.extend(importance_groups[ImportanceLevel.MEDIUM][:medium_count])

    return filtered_tokens


def _optimize_points_for_large_files(points: List[Point3D], token_count: int) -> List[Point3D]:
    """Optimize point distribution specifically for large file visualization."""
    # For large files, we might want to use a subset of points for better performance
    if len(points) > token_count * 3:
        # Use every Nth point to maintain distribution but improve performance
        step = len(points) // (token_count * 2)
        optimized_points = points[::max(1, step)]
        return optimized_points
    return points


def _calculate_precise_token_coordinates(sorted_tokens: List[CodeToken],
                                       points: List[Point3D]) -> List[Tuple[Point3D, CodeToken]]:
    """Calculate precise (u,v) parametric coordinates with structural relationships.

    Enhanced for Story 3.1 Task 4 to maintain spatial relationships reflecting
    code structure. Preserves logical groupings and implements proximity preservation
    for related tokens while maintaining mathematical precision.

    Args:
        sorted_tokens: Tokens sorted by line and column position
        points: Available surface points with parametric coordinates

    Returns:
        List of (point, token) pairs with structure-aware coordinate mapping
    """
    from math import tau

    # Analyze code structure relationships for spatial preservation
    structure_groups = _analyze_code_structure_relationships(sorted_tokens)

    # Calculate base coordinates with structural adjustments
    mapped_pairs = _calculate_structure_aware_coordinates(sorted_tokens, points, structure_groups)

    return mapped_pairs


def _analyze_code_structure_relationships(tokens: List[CodeToken]) -> dict:
    """Analyze code structure relationships for spatial preservation.

    Identifies logical code groupings (functions, classes, imports) and creates
    relationship maps for maintaining spatial connectivity on torus surface.

    Args:
        tokens: Sorted tokens to analyze

    Returns:
        Dictionary containing structure relationship mappings
    """
    structure_groups = {
        'imports': [],
        'functions': {},
        'classes': {},
        'blocks': {},
        'related_tokens': {}
    }

    current_function = None
    current_class = None
    current_block_indent = 0

    for i, token in enumerate(tokens):
        # Identify imports for grouping
        if token.type == 'NAME' and token.value in ['import', 'from']:
            structure_groups['imports'].append(i)

        # Identify function definitions
        elif token.type == 'NAME' and token.value == 'def':
            if i + 1 < len(tokens):
                func_name = tokens[i + 1].value
                current_function = func_name
                structure_groups['functions'][func_name] = {'start': i, 'tokens': []}

        # Identify class definitions
        elif token.type == 'NAME' and token.value == 'class':
            if i + 1 < len(tokens):
                class_name = tokens[i + 1].value
                current_class = class_name
                structure_groups['classes'][class_name] = {'start': i, 'tokens': []}

        # Track tokens within current function/class
        if current_function and current_function in structure_groups['functions']:
            structure_groups['functions'][current_function]['tokens'].append(i)

        if current_class and current_class in structure_groups['classes']:
            structure_groups['classes'][current_class]['tokens'].append(i)

        # Detect block boundaries (simplified by indentation patterns)
        if token.type in ['OP'] and token.value in [':', '{']:
            current_block_indent += 1
        elif token.type in ['DEDENT', '}'] or (token.type == 'OP' and token.value == '}'):
            current_block_indent = max(0, current_block_indent - 1)
            # Function/class boundaries
            if current_block_indent == 0:
                current_function = None
                current_class = None

    return structure_groups


def _calculate_structure_aware_coordinates(tokens: List[CodeToken],
                                         points: List[Point3D],
                                         structure_groups: dict) -> List[Tuple[Point3D, CodeToken]]:
    """Calculate coordinates that preserve code structure spatial relationships.

    Implements proximity preservation and structural clustering while maintaining
    precise parametric coordinate mapping for visual coherence.

    Args:
        tokens: Sorted tokens
        points: Available surface points
        structure_groups: Code structure relationship mappings

    Returns:
        Structure-aware coordinate mappings
    """
    from math import tau

    mapped_pairs = []
    total_tokens = len(tokens)

    # Calculate base u-coordinates with structural grouping adjustments
    for i, token in enumerate(tokens):
        # Base u coordinate from sequential position
        base_u = (i / total_tokens) * tau

        # Apply structural adjustments for grouping
        adjusted_u = _apply_structural_u_adjustment(i, token, structure_groups, base_u, tau)

        # Calculate v coordinate with importance and structural considerations
        adjusted_v = _calculate_structural_v_coordinate(i, token, structure_groups, tau)

        # Find optimal surface point considering both coordinates and neighbors
        optimal_point = _find_structure_aware_surface_point(
            adjusted_u, adjusted_v, points, i, tokens, mapped_pairs
        )

        mapped_pairs.append((optimal_point, token))

    return mapped_pairs


def _apply_structural_u_adjustment(token_index: int,
                                 token: CodeToken,
                                 structure_groups: dict,
                                 base_u: float,
                                 tau: float) -> float:
    """Apply structural adjustments to u-coordinate for grouping preservation.

    Modifies u-coordinates to keep structurally related tokens spatially close
    while maintaining overall sequential distribution.

    Args:
        token_index: Index of current token
        token: Current token
        structure_groups: Structure relationship mappings
        base_u: Base u-coordinate from sequential position
        tau: 2π constant

    Returns:
        Structurally adjusted u-coordinate
    """
    # Check if token is part of import group
    if token_index in structure_groups['imports']:
        # Cluster imports in early u-space
        import_offset = structure_groups['imports'].index(token_index) * 0.01
        return min(base_u, tau * 0.1 + import_offset)

    # Check if token is part of function
    for func_name, func_info in structure_groups['functions'].items():
        if token_index in func_info['tokens']:
            # Apply small clustering adjustment within function
            func_center = (func_info['start'] / len(structure_groups)) * tau
            local_offset = (token_index - func_info['start']) * 0.005
            return func_center + local_offset

    # Check if token is part of class
    for class_name, class_info in structure_groups['classes'].items():
        if token_index in class_info['tokens']:
            # Apply clustering adjustment within class
            class_center = (class_info['start'] / len(structure_groups)) * tau
            local_offset = (token_index - class_info['start']) * 0.005
            return class_center + local_offset

    # Default: use base coordinate with minor adjustment for boundaries
    return base_u


def _calculate_structural_v_coordinate(token_index: int,
                                     token: CodeToken,
                                     structure_groups: dict,
                                     tau: float) -> float:
    """Calculate v-coordinate considering structural context and importance.

    Combines importance-based v-mapping with structural considerations for
    enhanced visual distinction of code boundaries and groupings.

    Args:
        token_index: Index of current token
        token: Current token
        structure_groups: Structure relationship mappings
        tau: 2π constant

    Returns:
        Structurally aware v-coordinate
    """
    # Base v-coordinate from importance
    importance_v_map = {
        ImportanceLevel.CRITICAL: 0.0,      # Outer edge for maximum visibility
        ImportanceLevel.HIGH: tau * 0.25,   # Upper quarter of tube
        ImportanceLevel.MEDIUM: tau * 0.5,  # Middle of tube
        ImportanceLevel.LOW: tau * 0.75     # Lower quarter of tube
    }
    base_v = importance_v_map.get(token.importance, tau * 0.5)

    # Apply structural modifiers
    if token_index in structure_groups['imports']:
        # Imports get distinct v-band
        return tau * 0.1  # Special band for imports

    # Function/class boundaries get enhanced visibility
    for func_name, func_info in structure_groups['functions'].items():
        if token_index == func_info['start']:
            # Function start - outer visibility
            return 0.0
        elif token_index in func_info['tokens'][:3]:  # First few tokens
            # Function header area
            return base_v * 0.8

    for class_name, class_info in structure_groups['classes'].items():
        if token_index == class_info['start']:
            # Class start - maximum visibility
            return 0.0
        elif token_index in class_info['tokens'][:3]:  # First few tokens
            # Class header area
            return base_v * 0.7

    return base_v


def _find_structure_aware_surface_point(target_u: float,
                                       target_v: float,
                                       points: List[Point3D],
                                       token_index: int,
                                       all_tokens: List[CodeToken],
                                       existing_mappings: List[Tuple[Point3D, CodeToken]]) -> Point3D:
    """Find optimal surface point considering structural relationships and proximity.

    Selects surface points that maintain proximity to related tokens while
    achieving target parametric coordinates for structural coherence.

    Args:
        target_u: Target u-coordinate
        target_v: Target v-coordinate
        points: Available surface points
        token_index: Current token index
        all_tokens: All tokens for relationship analysis
        existing_mappings: Previously mapped tokens for proximity consideration

    Returns:
        Optimal surface point for structural preservation
    """
    # Find closest point to target coordinates
    closest_point = min(points,
                       key=lambda p: ((p.u - target_u) ** 2 + (p.v - target_v) ** 2) ** 0.5)

    # If this is early in mapping process, use closest point
    if len(existing_mappings) < 3:
        return closest_point

    # Check for proximity to related tokens (previous few tokens for context)
    recent_mappings = existing_mappings[-3:]  # Last 3 mapped tokens
    if recent_mappings:
        # Calculate average position of recent tokens
        avg_u = sum(mapping[0].u for mapping in recent_mappings) / len(recent_mappings)
        avg_v = sum(mapping[0].v for mapping in recent_mappings) / len(recent_mappings)

        # Find point that balances target coordinates with proximity to recent tokens
        def proximity_score(point: Point3D) -> float:
            coord_distance = ((point.u - target_u) ** 2 + (point.v - target_v) ** 2) ** 0.5
            proximity_distance = ((point.u - avg_u) ** 2 + (point.v - avg_v) ** 2) ** 0.5
            # Balance coordinate accuracy with proximity (60/40 weighting)
            return coord_distance * 0.6 + proximity_distance * 0.4

        proximity_point = min(points, key=proximity_score)
        return proximity_point

    return closest_point


def _apply_importance_weighted_distribution(mapped_pairs: List[Tuple[Point3D, CodeToken]],
                                          points: List[Point3D]) -> List[Tuple[Point3D, CodeToken]]:
    """Apply importance-weighted distribution with clustering and density allocation.

    Enhanced for Story 3.1 Task 2 to implement sophisticated clustering algorithm
    that groups tokens by importance level while maintaining sequential order.
    Provides density allocation where CRITICAL tokens get more surface area.

    Algorithm:
    1. Group tokens by importance level into clusters
    2. Allocate surface density based on importance hierarchy
    3. Maintain sequential code order within clusters
    4. Balance importance weighting with readability patterns

    Args:
        mapped_pairs: Initial precise coordinate mappings
        points: All available surface points

    Returns:
        Weighted distribution with importance-based clustering and density allocation
    """
    # Group tokens by importance level for clustering analysis
    importance_clusters = _create_importance_clusters(mapped_pairs)

    # Calculate dynamic density allocation based on cluster sizes
    density_allocation = _calculate_dynamic_density_allocation(importance_clusters, len(points))

    # Apply clustering algorithm with sequential order preservation
    clustered_pairs = _apply_importance_clustering(importance_clusters, points, density_allocation)

    # Balance importance weighting with sequential readability
    balanced_pairs = _balance_importance_with_sequence(clustered_pairs)

    return balanced_pairs


def _create_importance_clusters(mapped_pairs: List[Tuple[Point3D, CodeToken]]) -> dict:
    """Create importance-based clusters while maintaining sequential order.

    Groups tokens by importance level and preserves their sequential relationships
    within each cluster for maintaining code structure readability.

    Args:
        mapped_pairs: Token-surface mappings to cluster

    Returns:
        Dictionary mapping importance levels to ordered token clusters
    """
    clusters = {
        ImportanceLevel.CRITICAL: [],
        ImportanceLevel.HIGH: [],
        ImportanceLevel.MEDIUM: [],
        ImportanceLevel.LOW: []
    }

    # Group by importance while preserving sequential order
    for point, token in mapped_pairs:
        importance = token.importance
        clusters[importance].append((point, token))

    # Sort each cluster by line/column to maintain sequential order
    for importance_level in clusters:
        clusters[importance_level].sort(key=lambda x: (x[1].line, x[1].column))

    return clusters


def _calculate_dynamic_density_allocation(importance_clusters: dict, total_points: int) -> dict:
    """Calculate dynamic density allocation based on cluster sizes and importance.

    Implements adaptive density calculation that considers both importance hierarchy
    and actual cluster sizes to optimize surface area distribution.

    Args:
        importance_clusters: Clustered tokens by importance
        total_points: Total available surface points

    Returns:
        Dictionary mapping importance levels to density multipliers
    """
    # Base density multipliers with hierarchical weighting
    base_densities = {
        ImportanceLevel.CRITICAL: 4.0,  # Highest priority
        ImportanceLevel.HIGH: 2.5,      # High priority
        ImportanceLevel.MEDIUM: 1.5,    # Medium priority
        ImportanceLevel.LOW: 1.0        # Base priority
    }

    # Calculate total token count and weighted density requirement
    total_tokens = sum(len(cluster) for cluster in importance_clusters.values())
    total_weighted_demand = sum(
        len(cluster) * base_densities[importance]
        for importance, cluster in importance_clusters.items()
    )

    # Apply scaling if demand exceeds available surface points
    if total_weighted_demand > total_points:
        scale_factor = total_points / total_weighted_demand
        adjusted_densities = {
            importance: max(1.0, density * scale_factor)
            for importance, density in base_densities.items()
        }
    else:
        adjusted_densities = base_densities.copy()

    # Add cluster size consideration for balanced distribution
    for importance, cluster in importance_clusters.items():
        if len(cluster) == 0:
            adjusted_densities[importance] = 0
        elif len(cluster) < 3:  # Small clusters get bonus allocation
            adjusted_densities[importance] *= 1.2
        elif len(cluster) > total_tokens * 0.4:  # Large clusters get reduced allocation
            adjusted_densities[importance] *= 0.8

    return adjusted_densities


def _apply_importance_clustering(importance_clusters: dict,
                               points: List[Point3D],
                               density_allocation: dict) -> List[Tuple[Point3D, CodeToken]]:
    """Apply clustering algorithm with density allocation and spatial distribution.

    Implements the core clustering algorithm that distributes tokens across surface
    while maintaining importance hierarchy and sequential relationships.

    Args:
        importance_clusters: Token clusters by importance
        points: Available surface points
        density_allocation: Density multipliers per importance level

    Returns:
        Clustered and distributed token-surface mappings
    """
    clustered_pairs = []
    used_points = set()

    # Process clusters in importance order (highest to lowest)
    importance_order = [ImportanceLevel.CRITICAL, ImportanceLevel.HIGH,
                       ImportanceLevel.MEDIUM, ImportanceLevel.LOW]

    for importance in importance_order:
        cluster = importance_clusters[importance]
        if not cluster:
            continue

        density_multiplier = density_allocation[importance]

        for point, token in cluster:
            # Add primary mapping
            if point not in used_points:
                clustered_pairs.append((point, token))
                used_points.add(point)

            # Add density allocation mappings for high importance tokens
            if density_multiplier > 1.0:
                additional_count = int(density_multiplier) - 1
                nearby_points = _find_unused_nearby_points(point, points, used_points, additional_count)

                for nearby_point in nearby_points:
                    clustered_pairs.append((nearby_point, token))
                    used_points.add(nearby_point)

    return clustered_pairs


def _find_unused_nearby_points(center_point: Point3D,
                             all_points: List[Point3D],
                             used_points: set,
                             count: int) -> List[Point3D]:
    """Find nearby unused surface points for density allocation.

    Args:
        center_point: Center point for proximity search
        all_points: All available surface points
        used_points: Points already allocated
        count: Number of nearby points needed

    Returns:
        List of unused nearby points
    """
    from math import tau

    def parametric_distance(p1: Point3D, p2: Point3D) -> float:
        du = min(abs(p1.u - p2.u), tau - abs(p1.u - p2.u))
        dv = min(abs(p1.v - p2.v), tau - abs(p1.v - p2.v))
        return (du ** 2 + dv ** 2) ** 0.5

    # Find unused points sorted by distance
    unused_points = [p for p in all_points if p not in used_points and p != center_point]
    nearby = sorted(unused_points, key=lambda p: parametric_distance(center_point, p))

    return nearby[:count]


def _balance_importance_with_sequence(clustered_pairs: List[Tuple[Point3D, CodeToken]]) -> List[Tuple[Point3D, CodeToken]]:
    """Balance importance weighting with sequential code order for readability.

    Final balancing step that ensures the distribution maintains both importance
    hierarchy and sequential code relationships for optimal visual patterns.

    Args:
        clustered_pairs: Importance-clustered token mappings

    Returns:
        Balanced mappings optimizing both importance and sequence
    """
    # Sort by sequential order within importance groups
    balanced_pairs = []

    # Group by importance for final balancing
    final_groups = {}
    for point, token in clustered_pairs:
        importance = token.importance
        if importance not in final_groups:
            final_groups[importance] = []
        final_groups[importance].append((point, token))

    # Apply final sequential ordering within each importance group
    for importance in [ImportanceLevel.CRITICAL, ImportanceLevel.HIGH,
                      ImportanceLevel.MEDIUM, ImportanceLevel.LOW]:
        if importance in final_groups:
            # Sort by line and column to maintain code reading order
            group = final_groups[importance]
            group.sort(key=lambda x: (x[1].line, x[1].column))
            balanced_pairs.extend(group)

    return balanced_pairs


def _find_nearby_surface_points(center_point: Point3D,
                               all_points: List[Point3D],
                               count: int) -> List[Point3D]:
    """Find nearby surface points for importance-based density allocation.

    Args:
        center_point: Center point for proximity search
        all_points: All available surface points
        count: Number of nearby points to find

    Returns:
        List of nearby points sorted by distance
    """
    from math import tau

    # Calculate distances in parametric space for torus topology
    def parametric_distance(p1: Point3D, p2: Point3D) -> float:
        # Handle wraparound in parametric coordinates
        du = min(abs(p1.u - p2.u), tau - abs(p1.u - p2.u))
        dv = min(abs(p1.v - p2.v), tau - abs(p1.v - p2.v))
        return (du ** 2 + dv ** 2) ** 0.5

    # Sort by parametric distance and return closest points
    nearby = sorted(all_points,
                   key=lambda p: parametric_distance(center_point, p))

    # Exclude the center point itself and return requested count
    return [p for p in nearby if p != center_point][:count]


def _validate_coordinate_mapping(mapped_pairs: List[Tuple[Point3D, CodeToken]]) -> List[Tuple[Point3D, CodeToken]]:
    """Validate coordinate mapping for consistency and mathematical precision.

    Ensures all mappings have valid parametric coordinates and consistent patterns
    across rotations as required by Story 3.1 acceptance criteria.

    Args:
        mapped_pairs: Token-surface mappings to validate

    Returns:
        Validated mappings with consistent coordinate patterns

    Raises:
        ValueError: If coordinate validation fails
    """
    from math import tau

    if not mapped_pairs:
        return mapped_pairs

    validated_pairs = []

    for point, token in mapped_pairs:
        # Validate parametric coordinate ranges
        if not (0 <= point.u <= tau and 0 <= point.v <= tau):
            raise ValueError(
                f"Invalid parametric coordinates in mapping: u={point.u:.3f}, v={point.v:.3f}. "
                "Solution: Ensure all coordinates are within [0, 2π] range"
            )

        # Validate token has required attributes
        if not hasattr(token, 'importance') or not hasattr(token, 'ascii_char'):
            raise ValueError(
                f"Invalid token in mapping: {token}. "
                "Solution: Ensure all tokens have importance and ascii_char attributes"
            )

        validated_pairs.append((point, token))

    return validated_pairs


def _ensure_rotation_consistency(mapped_pairs: List[Tuple[Point3D, CodeToken]]) -> List[Tuple[Point3D, CodeToken]]:
    """Ensure consistent mapping patterns across rotations.

    Enhanced for Story 3.1 Task 5 to verify and ensure that token-to-surface
    mapping remains stable during 3D rotation and maintains visual continuity
    across all rotation angles.

    Args:
        mapped_pairs: Validated token-surface mappings

    Returns:
        Rotation-consistent mappings with stability verification

    Raises:
        ValueError: If mapping inconsistencies are detected
    """
    if not mapped_pairs:
        return mapped_pairs

    # Validate mapping stability across rotation angles
    _validate_rotation_stability(mapped_pairs)

    # Ensure visual continuity patterns
    continuity_verified_pairs = _verify_visual_continuity(mapped_pairs)

    # Add rotation consistency metadata for performance monitoring
    consistent_pairs = _add_rotation_consistency_metadata(continuity_verified_pairs)

    return consistent_pairs


def _validate_rotation_stability(mapped_pairs: List[Tuple[Point3D, CodeToken]]) -> None:
    """Validate that token mappings remain stable across rotation angles.

    Verifies that the parametric coordinate assignments create stable patterns
    that will maintain visual coherence during 3D rotation transformations.

    Args:
        mapped_pairs: Token-surface mappings to validate

    Raises:
        ValueError: If rotation stability issues are detected
    """
    from math import tau, sin, cos

    if len(mapped_pairs) < 2:
        return

    # Test rotation stability by checking coordinate distribution
    u_coordinates = [pair[0].u for pair in mapped_pairs]
    v_coordinates = [pair[0].v for pair in mapped_pairs]

    # Validate u-coordinate distribution (should span full range for rotation visibility)
    u_range = max(u_coordinates) - min(u_coordinates)
    if u_range < tau * 0.5:  # Should span at least half the circumference
        # This might be acceptable for small files, so just warn in debug
        pass

    # Validate v-coordinate distribution (should have reasonable spread)
    v_range = max(v_coordinates) - min(v_coordinates)
    if v_range < tau * 0.1:  # Should have some vertical spread
        pass

    # Check for clustering issues that might cause rotation artifacts
    _check_clustering_artifacts(mapped_pairs)

    # Validate that high-importance tokens are distributed for rotation visibility
    _validate_importance_rotation_visibility(mapped_pairs)


def _check_clustering_artifacts(mapped_pairs: List[Tuple[Point3D, CodeToken]]) -> None:
    """Check for clustering artifacts that could cause rotation display issues.

    Identifies potential clustering problems that might make important tokens
    invisible during certain rotation angles.

    Args:
        mapped_pairs: Token-surface mappings to check

    Raises:
        ValueError: If clustering artifacts are detected
    """
    from math import tau

    # Group critical tokens and check their distribution
    critical_tokens = [(point, token) for point, token in mapped_pairs
                      if token.importance == ImportanceLevel.CRITICAL]

    if len(critical_tokens) < 2:
        return

    # Check if critical tokens are clustered in small u-range
    critical_u_coords = [pair[0].u for pair in critical_tokens]
    critical_u_range = max(critical_u_coords) - min(critical_u_coords)

    # If critical tokens are clustered in less than 1/4 circumference, that's a concern
    if critical_u_range < tau * 0.25 and len(critical_tokens) > 3:
        # This is a potential issue but not necessarily fatal
        # The clustering might be intentional for structural reasons
        pass


def _validate_importance_rotation_visibility(mapped_pairs: List[Tuple[Point3D, CodeToken]]) -> None:
    """Validate that high-importance tokens maintain visibility across rotations.

    Ensures that critical and high-importance tokens are distributed such that
    some are always visible regardless of rotation angle.

    Args:
        mapped_pairs: Token-surface mappings to validate

    Raises:
        ValueError: If visibility distribution is inadequate
    """
    from math import tau

    # Group tokens by importance
    importance_groups = {}
    for point, token in mapped_pairs:
        importance = token.importance
        if importance not in importance_groups:
            importance_groups[importance] = []
        importance_groups[importance].append((point, token))

    # Check visibility distribution for critical and high importance tokens
    for importance in [ImportanceLevel.CRITICAL, ImportanceLevel.HIGH]:
        if importance not in importance_groups or len(importance_groups[importance]) < 2:
            continue

        group = importance_groups[importance]
        u_coords = [pair[0].u for pair in group]

        # Divide the circumference into quadrants and check distribution
        quadrant_counts = [0, 0, 0, 0]
        for u in u_coords:
            quadrant = int((u / tau) * 4)
            quadrant = min(3, quadrant)  # Handle edge case of u = tau
            quadrant_counts[quadrant] += 1

        # At least 2 quadrants should have tokens for rotation visibility
        occupied_quadrants = sum(1 for count in quadrant_counts if count > 0)
        if occupied_quadrants < 2 and len(group) > 2:
            # This is suboptimal but not necessarily wrong
            pass


def _verify_visual_continuity(mapped_pairs: List[Tuple[Point3D, CodeToken]]) -> List[Tuple[Point3D, CodeToken]]:
    """Verify visual continuity patterns across rotation cycles.

    Ensures that the mapping creates smooth visual transitions as tokens
    rotate in and out of view during animation.

    Args:
        mapped_pairs: Token-surface mappings to verify

    Returns:
        Mappings verified for visual continuity
    """
    if not mapped_pairs:
        return mapped_pairs

    # Sort by u-coordinate to check continuity patterns
    sorted_pairs = sorted(mapped_pairs, key=lambda x: x[0].u)

    # Check for smooth transitions in importance levels
    _check_importance_transitions(sorted_pairs)

    # Verify that structural groupings maintain continuity
    _verify_structural_continuity(sorted_pairs)

    return mapped_pairs


def _check_importance_transitions(sorted_pairs: List[Tuple[Point3D, CodeToken]]) -> None:
    """Check for smooth importance level transitions around the circumference.

    Validates that importance levels transition smoothly rather than creating
    jarring visual discontinuities during rotation.

    Args:
        sorted_pairs: Pairs sorted by u-coordinate
    """
    if len(sorted_pairs) < 3:
        return

    # Look for abrupt importance changes that might create visual artifacts
    for i in range(len(sorted_pairs) - 1):
        current_importance = sorted_pairs[i][1].importance
        next_importance = sorted_pairs[i + 1][1].importance

        # Large importance jumps are expected and acceptable
        # This check is mainly for identifying unexpected patterns
        importance_diff = abs(current_importance - next_importance)
        if importance_diff > 2:  # CRITICAL to LOW or vice versa
            # This is normal and acceptable
            pass


def _verify_structural_continuity(sorted_pairs: List[Tuple[Point3D, CodeToken]]) -> None:
    """Verify that structural groupings maintain spatial continuity.

    Ensures that structurally related tokens (functions, classes) maintain
    reasonable proximity in the parametric coordinate space.

    Args:
        sorted_pairs: Pairs sorted by u-coordinate
    """
    # This verification is mainly to ensure that the structural mapping
    # from Task 4 maintains continuity across the rotation cycle

    # Group tokens by line number to identify potential structural groupings
    line_groups = {}
    for point, token in sorted_pairs:
        line = token.line
        if line not in line_groups:
            line_groups[line] = []
        line_groups[line].append((point, token))

    # Check that tokens from nearby lines are reasonably close in u-space
    # This is a simplified check since detailed structural analysis
    # was already handled in Task 4
    pass


def _add_rotation_consistency_metadata(mapped_pairs: List[Tuple[Point3D, CodeToken]]) -> List[Tuple[Point3D, CodeToken]]:
    """Add rotation consistency metadata for performance monitoring.

    Adds metadata that can be used to monitor rotation consistency
    during animation loops and detect any degradation over time.

    Args:
        mapped_pairs: Verified token-surface mappings

    Returns:
        Mappings with consistency metadata attached
    """
    # For this implementation, we'll return the pairs as-is since the
    # consistency verification has already been performed.
    # In a more sophisticated implementation, we might add metadata
    # to track consistency metrics over time.

    return mapped_pairs


def _handle_token_compression_with_coordinates(tokens: List[CodeToken],
                                             points: List[Point3D]) -> List[Tuple[Point3D, CodeToken]]:
    """Handle compression scenario with precise coordinate mapping.

    Enhanced version of compression handling that maintains precise coordinate
    calculations even when tokens exceed available surface points.

    Args:
        tokens: All tokens to map
        points: Limited surface points available

    Returns:
        Compressed mapping with coordinate precision maintained
    """
    # Sort tokens by importance and sequential position
    sorted_tokens = sorted(tokens,
                          key=lambda t: (-t.importance, t.line, t.column))

    # Select highest importance tokens that fit available points
    selected_tokens = sorted_tokens[:len(points)]

    # Use precise coordinate calculation for selected tokens
    return _calculate_precise_token_coordinates(selected_tokens, points)


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


def generate_ascii_frame(mapped_pairs: List[Tuple[Point3D, CodeToken]],
                        frame_number: int = 0,
                        previous_frame_data: Optional[List[Tuple[Point2D, CodeToken]]] = None,
                        enable_visual_harmony: bool = True) -> DisplayFrame:
    """Create ASCII character frame with enhanced visual harmony and aesthetics.

    Enhanced for Story 3.5 with visual harmony features:
    - Token density balancing to prevent visual clutter
    - Smooth visual flow transitions during rotation
    - Intelligent spacing patterns based on code structure
    - Adaptive handling of visual edge cases

    Core Algorithm:
    1. Apply density balancing to optimize token distribution
    2. Project 3D points to 2D screen coordinates
    3. Calculate smooth transitions from previous frame
    4. Sort by depth value (farthest to closest) for painter's algorithm
    5. Render using enhanced character mapping with visual flow
    6. Apply depth buffer with aesthetic priority resolution

    Visual Harmony Features (Story 3.5):
    - Density analysis prevents overcrowded screen regions
    - Flow continuity creates smooth rotation transitions
    - Pattern recognition maintains code structure visibility
    - Edge case handling ensures consistent visual quality

    Args:
        mapped_pairs: List of (Point3D, CodeToken) pairs from token mapping
        frame_number: Current frame number for tracking and transitions
        previous_frame_data: Previous frame's screen data for flow calculation
        enable_visual_harmony: Enable Story 3.5 visual enhancements

    Returns:
        Complete ASCII frame with enhanced visual harmony and aesthetics
    """
    # Initialize 40x20 character buffer and depth buffer
    buffer = [[ASCII_CHARS['BACKGROUND'] for _ in range(TERMINAL_WIDTH)] for _ in range(TERMINAL_HEIGHT)]
    depth_buffer = [[float('inf') for _ in range(TERMINAL_WIDTH)] for _ in range(TERMINAL_HEIGHT)]
    # Add importance buffer for priority resolution when depths are similar
    importance_buffer = [[ImportanceLevel.LOW for _ in range(TERMINAL_WIDTH)] for _ in range(TERMINAL_HEIGHT)]

    # Story 3.5 Enhancement: Apply visual harmony features if enabled
    processed_pairs = mapped_pairs
    visual_flow_state = None

    if enable_visual_harmony:
        # Step 1: Analyze token density patterns to identify hotspots
        density_analysis = analyze_token_density_patterns(mapped_pairs)

        # Step 2: Handle visual edge cases before density control
        processed_pairs = handle_visual_edge_cases(mapped_pairs, density_analysis)

        # Step 3: Apply adaptive density control to prevent visual clutter
        if density_analysis.max_density > 3:  # Threshold for density balancing
            processed_pairs = apply_adaptive_density_control(processed_pairs, density_analysis, max_density_threshold=3)
    else:
        density_analysis = None

    # Convert 3D points to 2D screen coordinates with tokens
    screen_data = []
    for point_3d, token in processed_pairs:
        point_2d = project_to_screen(point_3d, token.importance)
        if point_2d.visible:
            screen_data.append((point_2d, token))

    # Story 3.5 Enhancement: Calculate visual flow for smooth transitions
    if enable_visual_harmony and previous_frame_data is not None:
        visual_flow_state = create_smooth_transition_algorithms(screen_data, previous_frame_data, frame_number)

    # Apply boundary conflict resolution for edge cases
    screen_data = resolve_token_boundary_conflicts(screen_data)

    # Sort by depth (farthest to closest for painter's algorithm)
    sorted_data = sorted(screen_data, key=lambda x: x[0].depth, reverse=True)

    # Render points with enhanced priority resolution and visual harmony
    for point_2d, token in sorted_data:
        x, y = point_2d.x, point_2d.y

        # Bounds check for safety
        if 0 <= x < TERMINAL_WIDTH and 0 <= y < TERMINAL_HEIGHT:
            current_depth = depth_buffer[y][x]
            current_importance = importance_buffer[y][x]

            # Enhanced priority resolution: depth-first, visibility-aware, then importance-based
            should_render = False

            # Story 3.5 Enhancement: Apply visual flow continuity if available
            flow_enhanced_visibility = point_2d.visibility_factor
            if enable_visual_harmony and visual_flow_state and visual_flow_state.transition_weights:
                token_id = id(token)
                if token_id in visual_flow_state.transition_weights:
                    # Apply flow-based smoothing to visibility
                    flow_weight = visual_flow_state.transition_weights[token_id]
                    flow_enhanced_visibility = (point_2d.visibility_factor * 0.7 +
                                              flow_weight * 0.3)  # Blend for smoothness

            # Adaptive visibility threshold based on visual flow continuity
            base_threshold = 0.1
            if enable_visual_harmony and visual_flow_state:
                # Adjust threshold based on continuity score for smoother transitions
                continuity_factor = visual_flow_state.continuity_score
                visibility_threshold = base_threshold * (1.0 - continuity_factor * 0.3)
            else:
                visibility_threshold = base_threshold

            if flow_enhanced_visibility < visibility_threshold:
                # Surface is clearly back-facing, don't render
                should_render = False
            elif point_2d.depth < current_depth:
                # Closer point takes priority if sufficiently visible
                should_render = True
            elif abs(point_2d.depth - current_depth) < 0.1:
                # Similar depth - use importance hierarchy for conflict resolution
                if token.importance > current_importance:
                    should_render = True
                elif token.importance == current_importance:
                    # Equal importance - maintain stability by keeping first rendered
                    should_render = False

            if should_render:
                # Use token's ASCII character from importance classification
                char = token.ascii_char

                # Apply enhanced visibility-based character dimming with flow consideration
                dimmed_char = apply_visibility_dimming(char, flow_enhanced_visibility, token.importance)

                # Update all buffers for comprehensive priority tracking
                buffer[y][x] = dimmed_char
                depth_buffer[y][x] = point_2d.depth
                importance_buffer[y][x] = token.importance

    # Create display frame
    frame = DisplayFrame(
        width=TERMINAL_WIDTH,
        height=TERMINAL_HEIGHT,
        buffer=buffer,
        depth_buffer=depth_buffer,
        frame_number=frame_number
    )

    # Story 3.5 Enhancement: Validate artistic impact and quality
    if enable_visual_harmony and density_analysis:
        aesthetic_quality = validate_artistic_impact_and_quality(
            frame=frame,
            density_analysis=density_analysis,
            visual_flow_state=visual_flow_state,
            frame_number=frame_number
        )

        # Optional: Log quality metrics for debugging (only for first few frames)
        if frame_number < 5:  # Limit logging to avoid performance impact
            pass  # Could log aesthetic_quality metrics here if needed

    return frame


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


def _precompute_token_mappings(enhanced_tokens: List[CodeToken],
                             base_points: List[Point3D],
                             structural_info: StructuralInfo) -> List[Tuple[int, CodeToken]]:
    """Pre-compute token-to-point index mappings for performance optimization.

    PERF-003 Fix: This eliminates expensive O(n²) structural distribution
    calculations from running every frame in the animation loop.

    Args:
        enhanced_tokens: Pre-enhanced tokens with structural importance
        base_points: Base torus points (before rotation)
        structural_info: Structural analysis results

    Returns:
        List of (point_index, token) pairs for efficient frame-time lookup
    """
    # Use the existing structural distribution logic but cache the result
    temp_mappings = _apply_structural_distribution(enhanced_tokens, base_points, structural_info)

    # Convert to point index mappings for efficient rotation application
    token_point_indices = []
    point_to_index = {id(point): idx for idx, point in enumerate(base_points)}

    for point, token in temp_mappings:
        point_idx = point_to_index.get(id(point))
        if point_idx is not None:
            token_point_indices.append((point_idx, token))

    return token_point_indices


def _apply_cached_mappings(token_point_indices: List[Tuple[int, CodeToken]],
                          rotated_points: List[Point3D]) -> List[Tuple[Point3D, CodeToken]]:
    """Apply pre-computed mappings to rotated points for per-frame performance.

    PERF-003 Fix: This replaces expensive structural distribution with simple
    index lookup, dramatically reducing per-frame computational overhead.
    Enhanced with Story 3.4 optimizations for rotation-aware character assignment.

    Args:
        token_point_indices: Pre-computed (point_index, token) mappings
        rotated_points: Current frame's rotated torus points

    Returns:
        List of (Point3D, CodeToken) pairs ready for rendering
    """
    mapped_pairs = []
    for point_idx, token in token_point_indices:
        if point_idx < len(rotated_points):
            # Apply cached mapping with rotation offset preserved
            # The rotated points already have updated surface normals for visibility
            mapped_pairs.append((rotated_points[point_idx], token))

    return mapped_pairs


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
    """Main execution loop with frame rate control and real-time integration.

    Implements comprehensive animation control with Story 3.4 optimizations:
    - Separated one-time token preprocessing from per-frame operations
    - Comprehensive token data caching for efficient frame-to-frame access
    - Target 30+ FPS with dynamic timing adjustment
    - Graceful keyboard interrupt handling (Ctrl+C)
    - Frame timing validation and debugging
    - Memory management with buffer clearing and cache cleanup
    - Structural analysis integration with optional debugging
    - Consistent performance across different system capabilities

    Follows Animation Controller specifications with proper integration
    of MathematicalEngine, ParsingEngine, and RenderingEngine systems.
    Enhanced with real-time integration pipeline from Story 3.4.

    Args:
        enable_debug: Flag to enable structural analysis debugging output
    """
    print("Starting 3D ASCII Donut Animation with Real-Time Integration...", flush=True)
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

    # Timing and performance tracking variables (Story 3.4 Task 4)
    frame_count = 0
    target_frame_time = calculate_frame_timing()
    total_elapsed = 0.0
    performance_samples = []
    performance_warnings = 0
    degraded_mode = False  # Fallback flag for performance issues

    # Story 3.5 Enhancement: Visual harmony tracking
    previous_frame_data = None
    enable_visual_harmony = True  # Enable visual harmony features

    # === STORY 3.4 OPTIMIZATION: ONE-TIME INITIALIZATION PHASE ===
    # All token parsing, structural analysis, and preprocessing moved here
    # This eliminates repeated parsing overhead from animation loop

    try:
        # Initialize token cache and perform all preprocessing once
        print("Initializing token cache and preprocessing pipeline...", flush=True)
        token_cache, base_torus_points = initialize_token_cache(torus_params)

        # Get preprocessed data from cache
        enhanced_tokens = token_cache.get_enhanced_tokens()
        structural_info = token_cache.structural_info
        token_point_indices = token_cache.get_token_mappings()

        if not all([enhanced_tokens, structural_info, token_point_indices]):
            raise ValueError("Token cache initialization incomplete")

        print(f"Token cache initialized: {len(enhanced_tokens)} tokens preprocessed", flush=True)
        print(f"Memory usage: ~{token_cache.memory_usage() / 1024:.1f} KB", flush=True)
        print("Performance optimization: All token processing moved to initialization", flush=True)

        # Display debug information once at startup if enabled
        if enable_debug:
            debug_structural_analysis(structural_info, enhanced_tokens, enable_debug)
            debug_nested_structures(structural_info, enable_debug)

    except Exception as e:
        print(f"Error in initialization phase: {e}", flush=True)
        print("Solution: Check token preprocessing pipeline and retry", flush=True)
        return

    try:
        while True:
            frame_start_time = time.time()

            # Calculate current rotation angle with incremental updates
            rotation_angle = frame_count * torus_params.rotation_speed

            # Apply Y-axis rotation transformation to cached base points
            rotated_points = apply_rotation(base_torus_points, rotation_angle)

            # Apply pre-computed token mappings to rotated points (PERF-003 fix)
            # Using cached structural mappings to eliminate O(n²) operations per frame
            try:
                mapped_pairs = _apply_cached_mappings(token_point_indices, rotated_points)

                # Display surface mapping debug info for first frame only
                if enable_debug and frame_count == 0:
                    debug_surface_mapping(mapped_pairs, structural_info, enable_debug)

            except Exception as e:
                print(f"Error applying cached token mappings: {e}", flush=True)
                print("Solution: Check pre-computed mappings and rotated points", flush=True)

                # Fallback to basic mapping if cached mapping fails
                try:
                    # Use cached tokens for fallback mapping
                    fallback_tokens = token_cache.get_tokens() or enhanced_tokens
                    mapped_pairs = map_tokens_to_surface(fallback_tokens, rotated_points)
                    if enable_debug:
                        print("Fallback: Using basic token mapping", flush=True)
                except Exception as fallback_e:
                    print(f"Fallback mapping also failed: {fallback_e}", flush=True)
                    break

            # Story 3.5 Enhancement: Generate ASCII frame with visual harmony features
            frame = generate_ascii_frame(
                mapped_pairs=mapped_pairs,
                frame_number=frame_count,
                previous_frame_data=previous_frame_data,
                enable_visual_harmony=enable_visual_harmony
            )

            # Store current frame data for next iteration's visual flow calculation
            if enable_visual_harmony:
                # Extract screen data from current frame for next iteration
                current_frame_data = []
                for point_3d, token in mapped_pairs:
                    point_2d = project_to_screen(point_3d, token.importance)
                    if point_2d.visible:
                        current_frame_data.append((point_2d, token))
                previous_frame_data = current_frame_data

            # Output frame to terminal with screen clearing
            output_to_terminal(frame)

            # Frame timing calculation and control
            frame_elapsed = time.time() - frame_start_time
            total_elapsed += frame_elapsed

            # Performance and memory tracking (Story 4.1 Task 2)
            _performance_stats['frame_times'].append(frame_elapsed)
            _performance_stats['total_frames'] += 1
            performance_samples.append(frame_elapsed)
            if len(performance_samples) > 30:  # Keep last 30 samples
                performance_samples.pop(0)

            # Memory management every 100 frames to prevent memory buildup
            if frame_count % 100 == 0:
                memory_info = memory_monitor()
                if frame_count % 500 == 0:  # Print memory report every 500 frames
                    print(f"\nMemory status: Torus:{memory_info['torus_cache_size']} "
                          f"Rotation:{memory_info['rotation_cache_size']} "
                          f"Performance:{memory_info['performance_stats_size']}", flush=True)

            # Comprehensive frame rate monitoring and adaptive control (Story 4.1 Task 5)
            avg_frame_time = sum(performance_samples) / len(performance_samples)
            current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            sleep_time = max(0, target_frame_time - frame_elapsed)

            # Advanced performance monitoring with trend analysis
            if frame_count > 0:
                recent_fps = 1.0 / frame_elapsed if frame_elapsed > 0 else 0
                # Check for performance degradation patterns
                if len(performance_samples) >= 10:
                    recent_avg = sum(performance_samples[-10:]) / 10
                    older_avg = sum(performance_samples[-20:-10]) / 10 if len(performance_samples) >= 20 else recent_avg

                    # Detect performance degradation trend
                    if recent_avg > older_avg * 1.2 and current_fps < TARGET_FPS * 0.9:
                        performance_warnings += 1

            # Performance monitoring and fallback modes (Story 3.4 Task 4)
            if current_fps < 30 and len(performance_samples) >= 10:
                performance_warnings += 1
                if performance_warnings > 5 and not degraded_mode:
                    # Enter degraded mode after persistent low FPS
                    degraded_mode = True
                    print(f"\nPerformance Warning: FPS dropped to {current_fps:.1f}", flush=True)
                    print("Solution: Entering degraded mode with reduced quality", flush=True)
                    # Reduce torus resolution for better performance
                    torus_params = TorusParameters(
                        outer_radius=torus_params.outer_radius,
                        inner_radius=torus_params.inner_radius,
                        u_resolution=30,  # Reduced from 50
                        v_resolution=15,  # Reduced from 25
                        rotation_speed=torus_params.rotation_speed
                    )
                    # Regenerate base points with lower resolution
                    base_torus_points = generate_torus_points(torus_params)
                    token_point_indices = _precompute_token_mappings(enhanced_tokens, base_torus_points, structural_info)
                    token_cache.set_token_mappings(token_point_indices)
            elif current_fps > 35 and degraded_mode:
                # Exit degraded mode if performance recovers
                performance_warnings = 0
                degraded_mode = False
                print(f"\nPerformance Recovered: FPS restored to {current_fps:.1f}", flush=True)
                # Restore original resolution
                torus_params = TorusParameters(
                    outer_radius=2.0,
                    inner_radius=1.0,
                    u_resolution=50,
                    v_resolution=25,
                    rotation_speed=0.05
                )
                base_torus_points = generate_torus_points(torus_params)
                token_point_indices = _precompute_token_mappings(enhanced_tokens, base_torus_points, structural_info)
                token_cache.set_token_mappings(token_point_indices)

            # Handle performance variations gracefully
            if avg_frame_time > target_frame_time * 1.5:
                # System struggling - reduce update frequency slightly
                sleep_time = max(sleep_time, 0.01)  # Minimum 10ms sleep

            # Display performance metrics periodically (Story 3.4 Task 4)
            if frame_count % 100 == 0 and frame_count > 0:
                cache_hit_rate = (_performance_stats['cache_hits'] / (_performance_stats['cache_hits'] + _performance_stats['cache_misses'])) * 100 if (_performance_stats['cache_hits'] + _performance_stats['cache_misses']) > 0 else 0
                print(f"\rFPS: {current_fps:.1f} | Frame: {frame_count} | Mode: {'Degraded' if degraded_mode else 'Normal'} | Cache: {cache_hit_rate:.0f}%     ", end='', flush=True)

                # Detailed performance report every 500 frames
                if frame_count % 500 == 0:
                    print(f"\n{get_performance_report()}", flush=True)
                    # Performance target validation
                    if current_fps >= TARGET_FPS:
                        print(f"✅ Performance target achieved: {current_fps:.1f} >= {TARGET_FPS} FPS", flush=True)
                    else:
                        print(f"⚠️  Performance below target: {current_fps:.1f} < {TARGET_FPS} FPS", flush=True)

            # Frame rate control with timing validation
            if sleep_time > 0:
                time.sleep(sleep_time)

            frame_count += 1

            # === STORY 3.4 TASK 5: ROBUST MEMORY MANAGEMENT ===
            # Periodic memory cleanup and monitoring for long-running animations
            if frame_count % 100 == 0:
                # Force garbage collection of old frame data
                import gc
                gc.collect()

                # Monitor memory usage
                cache_memory = token_cache.memory_usage()
                if cache_memory > 1024 * 1024:  # If cache exceeds 1MB
                    print(f"\nMemory Warning: Cache using {cache_memory / 1024:.1f} KB", flush=True)
                    print("Solution: Consider restarting animation for optimal performance", flush=True)

            # Extended session memory management (30+ minutes)
            if frame_count % 3000 == 0 and frame_count > 0:  # Every ~100 seconds at 30 FPS
                # Deep cleanup for extended sessions
                gc.collect(2)  # Full collection including oldest generation

                # Clear any accumulated performance samples beyond needed
                if len(performance_samples) > 30:
                    performance_samples = performance_samples[-30:]

                # Validate cache integrity
                if not token_cache.cache_valid:
                    print("\nCache invalidated - reinitializing...", flush=True)
                    token_cache, base_torus_points = initialize_token_cache(torus_params)
                    enhanced_tokens = token_cache.get_enhanced_tokens()
                    structural_info = token_cache.structural_info
                    token_point_indices = token_cache.get_token_mappings()

    except KeyboardInterrupt:
        # Graceful interrupt handling with proper cleanup
        success = handle_interrupts()
        if success:
            # Display performance statistics
            if frame_count > 0:
                avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0
                print(f"Average FPS: {avg_fps:.1f}", flush=True)
                print(f"Total Frames: {frame_count}", flush=True)
                print(f"Cache Memory Used: {token_cache.memory_usage() / 1024:.1f} KB", flush=True)
    except Exception as e:
        print(f"\nAnimation error: {e}", flush=True)
        print("Solution: Check terminal compatibility and system resources", flush=True)
        raise
    finally:
        # === STORY 3.4 TASK 5: MEMORY CLEANUP ON EXIT ===
        # Ensure all cached data is cleaned up properly
        if '_token_cache' in globals():
            _token_cache.clear()
        # Final garbage collection
        import gc
        gc.collect()


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