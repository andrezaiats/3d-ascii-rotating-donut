"""Performance monitoring and optimization infrastructure for rotating donut animation.

Extracted from rotating_donut.py as part of refactoring to address file size concerns.
Provides performance tracking, monitoring decorators, and statistics reporting.

Author: Andre Zaiats
License: MIT
Python Version: 3.8+
"""

import time
from typing import Dict, List, Any


# Performance monitoring statistics
_performance_stats = {
    'frame_times': [],
    'math_times': [],
    'projection_times': [],
    'total_frames': 0,
    'cache_hits': 0,
    'cache_misses': 0
}


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


def get_performance_stats() -> Dict[str, Any]:
    """Get current performance statistics.

    Returns:
        Dictionary containing performance metrics
    """
    return _performance_stats


def record_cache_hit():
    """Record a cache hit event."""
    _performance_stats['cache_hits'] += 1


def record_cache_miss():
    """Record a cache miss event."""
    _performance_stats['cache_misses'] += 1


def record_frame_time(frame_time: float):
    """Record frame rendering time.

    Args:
        frame_time: Time taken to render frame in seconds
    """
    _performance_stats['frame_times'].append(frame_time)
    _performance_stats['total_frames'] += 1


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


def memory_monitor() -> Dict[str, int]:
    """Monitor and manage memory usage during animation.

    Returns:
        Dictionary containing memory usage statistics
    """
    import gc
    from cache_manager import get_cache_sizes

    # Get cache sizes
    cache_sizes = get_cache_sizes()

    # Track memory statistics
    memory_info = {
        'torus_cache_size': cache_sizes['torus'],
        'rotation_cache_size': cache_sizes['rotation'],
        'projection_cache_size': cache_sizes['projection'],
        'token_cache_memory': cache_sizes['token'],
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

    # Trigger garbage collection if memory usage is high
    if total_cache_items > 5000 or memory_info['token_cache_memory'] > 50000000:  # 50MB
        gc.collect()

    return memory_info


def clear_performance_stats():
    """Clear all performance statistics."""
    _performance_stats['frame_times'].clear()
    _performance_stats['math_times'].clear()
    _performance_stats['projection_times'].clear()
    _performance_stats['total_frames'] = 0
    _performance_stats['cache_hits'] = 0
    _performance_stats['cache_misses'] = 0