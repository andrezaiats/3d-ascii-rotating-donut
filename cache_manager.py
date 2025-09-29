"""Cache management system for rotating donut animation.

Extracted from rotating_donut.py as part of refactoring to address file size concerns.
Provides caching for rotation matrices, projections, torus geometry, and token data.

Author: Andre Zaiats
License: MIT
Python Version: 3.8+
"""

import math
import time
from typing import Dict, List, Optional, Tuple, Any


# Global cache dictionaries
_torus_cache = {}
_rotation_matrix_cache = {}
_projection_cache = {}


class TokenCache:
    """Comprehensive token data caching system for efficient frame-to-frame access.

    Implements token caching and pipeline optimization.
    Stores preprocessed token data with efficient lookup structures to eliminate
    repeated parsing and classification during animation.
    """

    def __init__(self):
        """Initialize empty token cache."""
        self.source_code = None
        self.tokens = None
        self.enhanced_tokens = None
        self.importance_map = {}
        self.token_lookup = {}
        self.structural_info = None
        self.token_mappings = None
        self.last_update = None
        self.cache_valid = False

    def populate(self, source_code: str, tokens: List[Any], structural_info: Any = None) -> None:
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
        self.token_lookup = {}
        for i, token in enumerate(tokens):
            if hasattr(token, 'importance'):
                self.importance_map[id(token)] = token.importance
            # Create fast position-based lookup
            if hasattr(token, 'line') and hasattr(token, 'column'):
                self.token_lookup[(token.line, token.column)] = i

    def get_tokens(self) -> Optional[List[Any]]:
        """Get cached tokens with validation."""
        if self.cache_valid and self.tokens:
            return self.tokens
        return None

    def get_enhanced_tokens(self) -> Optional[List[Any]]:
        """Get cached enhanced tokens."""
        if self.cache_valid and self.enhanced_tokens:
            return self.enhanced_tokens
        return None

    def set_enhanced_tokens(self, enhanced_tokens: List[Any]) -> None:
        """Store enhanced tokens in cache."""
        self.enhanced_tokens = enhanced_tokens
        # Update importance map with enhanced tokens
        for token in enhanced_tokens:
            if hasattr(token, 'importance'):
                self.importance_map[id(token)] = token.importance

    def set_token_mappings(self, mappings: List[Tuple[int, Any]]) -> None:
        """Store pre-computed token-to-point mappings."""
        self.token_mappings = mappings

    def get_token_mappings(self) -> Optional[List[Tuple[int, Any]]]:
        """Get cached token mappings."""
        if self.cache_valid and self.token_mappings:
            return self.token_mappings
        return None

    def invalidate(self) -> None:
        """Invalidate cache, forcing refresh on next access."""
        self.cache_valid = False

    def get_token_by_position(self, line: int, column: int) -> Optional[Any]:
        """Fast token lookup by position."""
        if (line, column) in self.token_lookup and self.tokens:
            index = self.token_lookup[(line, column)]
            return self.tokens[index] if index < len(self.tokens) else None
        return None

    def get_cached_importance(self, token: Any) -> Optional[int]:
        """Get cached importance level for a token."""
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
        """Estimate memory usage of cached data in bytes."""
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


# Global token cache instance
_token_cache = TokenCache()


def get_token_cache() -> TokenCache:
    """Get the global token cache instance."""
    return _token_cache


def get_cached_rotation_matrix(angle: float, precision: int = 1000) -> Tuple[float, float]:
    """Get cached rotation matrix components for common angles.

    Args:
        angle: Rotation angle in radians
        precision: Discretization precision for caching

    Returns:
        Tuple of (cos_angle, sin_angle)
    """
    from performance_monitor import record_cache_hit, record_cache_miss

    # Discretize angle for caching
    cache_key = round(angle * precision) % (int(2 * math.pi * precision))

    if cache_key in _rotation_matrix_cache:
        record_cache_hit()
        return _rotation_matrix_cache[cache_key]

    # Calculate and cache
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    _rotation_matrix_cache[cache_key] = (cos_angle, sin_angle)
    record_cache_miss()

    return cos_angle, sin_angle


def get_cached_projection(x: float, y: float, z: float, precision: int = 100) -> Optional[Tuple[int, int, float, bool]]:
    """Get cached projection result for frequently projected coordinates.

    Args:
        x, y, z: 3D coordinates
        precision: Discretization precision for caching

    Returns:
        Cached projection tuple (grid_x, grid_y, depth, visible) or None
    """
    from performance_monitor import record_cache_hit, record_cache_miss

    # Create cache key by discretizing coordinates
    cache_key = (round(x * precision), round(y * precision), round(z * precision))

    if cache_key in _projection_cache:
        record_cache_hit()
        return _projection_cache[cache_key]

    record_cache_miss()
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


def cache_torus_geometry(cache_key: Tuple[float, float, int, int], points: List[Any]) -> None:
    """Cache torus geometry for reuse.

    Args:
        cache_key: Tuple of (outer_radius, inner_radius, u_resolution, v_resolution)
        points: List of generated torus points
    """
    _torus_cache[cache_key] = points


def get_cached_torus_geometry(cache_key: Tuple[float, float, int, int]) -> Optional[List[Any]]:
    """Get cached torus geometry.

    Args:
        cache_key: Tuple of (outer_radius, inner_radius, u_resolution, v_resolution)

    Returns:
        Cached torus points or None
    """
    return _torus_cache.get(cache_key)


def clear_performance_caches():
    """Clear all performance caches to prevent memory buildup."""
    global _rotation_matrix_cache, _projection_cache
    _rotation_matrix_cache.clear()
    _projection_cache.clear()


def clear_rotation_cache():
    """Clear rotation matrix cache."""
    _rotation_matrix_cache.clear()


def clear_projection_cache():
    """Clear projection cache."""
    _projection_cache.clear()


def clear_torus_cache():
    """Clear torus geometry cache."""
    _torus_cache.clear()


def get_torus_cache():
    """Get the torus cache dictionary for testing purposes."""
    return _torus_cache


def get_cache_sizes() -> Dict[str, int]:
    """Get sizes of all caches.

    Returns:
        Dictionary with cache sizes
    """
    return {
        'torus': len(_torus_cache),
        'rotation': len(_rotation_matrix_cache),
        'projection': len(_projection_cache),
        'token': _token_cache.memory_usage()
    }


def manage_rotation_cache_size(max_size: int = 2000):
    """Manage rotation cache size by keeping most recent entries.

    Args:
        max_size: Maximum number of cache entries to keep
    """
    if len(_rotation_matrix_cache) > max_size:
        # Keep only the most recently accessed entries
        items = list(_rotation_matrix_cache.items())
        _rotation_matrix_cache.clear()
        _rotation_matrix_cache.update(dict(items[-1000:]))