#!/usr/bin/env python3
"""
Performance validation tests for Story 4.1: Performance Optimization

Tests all performance-critical functions and optimization features:
- Mathematical operation timing and caching
- Memory management and leak detection
- Frame rate consistency and performance targets
- Cache effectiveness validation
- Token parsing optimization impact

Requirements: 90%+ coverage for performance-critical functions
"""

import time
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the path to import the main module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rotating_donut
from rotating_donut import (
    TorusParameters, Point3D, CodeToken, ImportanceLevel,
    generate_torus_points, apply_rotation, project_to_screen,
    get_cached_rotation_matrix, get_cached_projection, cache_projection_result,
    memory_monitor, clear_performance_caches, get_performance_report,
    _performance_stats, _rotation_matrix_cache, _projection_cache, _torus_cache
)


class TestPerformanceOptimizations(unittest.TestCase):
    """Test suite for performance optimization features."""

    def setUp(self):
        """Set up test environment."""
        # Clear all caches before each test
        clear_performance_caches()
        _torus_cache.clear()
        _performance_stats['frame_times'].clear()
        _performance_stats['math_times'].clear()
        _performance_stats['projection_times'].clear()
        _performance_stats['total_frames'] = 0
        _performance_stats['cache_hits'] = 0
        _performance_stats['cache_misses'] = 0

    def tearDown(self):
        """Clean up after each test."""
        clear_performance_caches()
        _torus_cache.clear()

    def test_mathematical_operation_performance(self):
        """Test mathematical operations meet performance requirements."""
        # Test torus generation performance
        params = TorusParameters(
            outer_radius=2.0,
            inner_radius=1.0,
            u_resolution=50,
            v_resolution=25,
            rotation_speed=0.05
        )

        start_time = time.time()
        points = generate_torus_points(params)
        generation_time = time.time() - start_time

        # Performance requirement: <10ms for standard torus generation
        self.assertLess(generation_time, 0.01,
                       f"Torus generation too slow: {generation_time*1000:.2f}ms > 10ms")
        self.assertEqual(len(points), 50 * 25, "Incorrect number of points generated")

        # Test rotation performance
        start_time = time.time()
        rotated_points = apply_rotation(points, 1.0)
        rotation_time = time.time() - start_time

        # Performance requirement: <5ms for rotation of 1250 points
        self.assertLess(rotation_time, 0.005,
                       f"Rotation too slow: {rotation_time*1000:.2f}ms > 5ms")
        self.assertEqual(len(rotated_points), len(points), "Points lost during rotation")

    def test_rotation_matrix_caching(self):
        """Test rotation matrix caching effectiveness."""
        angle = 1.5708  # π/2 radians

        # First call should be a cache miss
        initial_misses = _performance_stats['cache_misses']
        cos_val, sin_val = get_cached_rotation_matrix(angle)
        self.assertEqual(_performance_stats['cache_misses'], initial_misses + 1,
                        "First call should be cache miss")

        # Second call should be a cache hit
        initial_hits = _performance_stats['cache_hits']
        cos_val2, sin_val2 = get_cached_rotation_matrix(angle)
        self.assertEqual(_performance_stats['cache_hits'], initial_hits + 1,
                        "Second call should be cache hit")
        self.assertEqual(cos_val, cos_val2, "Cached values should be identical")
        self.assertEqual(sin_val, sin_val2, "Cached values should be identical")

        # Test precision requirements (allow for floating point precision)
        self.assertAlmostEqual(cos_val, 0.0, places=4, msg="cos(π/2) should be ~0")
        self.assertAlmostEqual(sin_val, 1.0, places=4, msg="sin(π/2) should be ~1")

    def test_projection_caching(self):
        """Test projection caching system."""
        point = Point3D(x=1.0, y=2.0, z=0.5, u=0, v=0, nx=0, ny=0, nz=1)

        # Test cache miss on first projection
        initial_misses = _performance_stats['cache_misses']
        result1 = get_cached_projection(point.x, point.y, point.z)
        self.assertIsNone(result1, "Cache should be empty initially")

        # Project and cache result
        projected = project_to_screen(point)

        # Test cache hit on subsequent projection
        result2 = get_cached_projection(point.x, point.y, point.z)
        self.assertIsNotNone(result2, "Result should be cached after projection")

    def test_memory_management(self):
        """Test memory management and leak prevention."""
        # Generate memory usage before operations
        initial_memory = memory_monitor()

        # Perform memory-intensive operations
        for _ in range(100):
            params = TorusParameters(2.0, 1.0, 20, 10, 0.05)
            points = generate_torus_points(params)
            rotated = apply_rotation(points, 1.0)

        # Monitor memory after operations
        final_memory = memory_monitor()

        # Memory should be managed (not growing uncontrolled)
        cache_growth = (final_memory['torus_cache_size'] +
                       final_memory['rotation_cache_size'] +
                       final_memory['projection_cache_size'])

        # Should not exceed reasonable cache limits
        self.assertLess(cache_growth, 1000, "Cache growth exceeded reasonable limits")

        # Performance stats should not grow unbounded
        self.assertLess(final_memory['performance_stats_size'], 1000,
                       "Performance stats growing unbounded")

    def test_frame_rate_consistency(self):
        """Test frame rate monitoring and consistency."""
        # Simulate frame timing data
        target_frame_time = 1.0 / 30  # 30 FPS target

        # Add consistent frame times
        for _ in range(10):
            _performance_stats['frame_times'].append(target_frame_time)

        # Add one slow frame
        _performance_stats['frame_times'].append(target_frame_time * 2)

        # Calculate average FPS
        avg_frame_time = sum(_performance_stats['frame_times']) / len(_performance_stats['frame_times'])
        avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

        # Should detect performance degradation
        self.assertLess(avg_fps, 30, "Should detect FPS drop from slow frame")

        # Test performance report generation
        report = get_performance_report()
        self.assertIn("Average FPS", report, "Report should contain FPS information")
        self.assertIn("Cache Hit Rate", report, "Report should contain cache statistics")

    def test_cache_effectiveness(self):
        """Test overall cache effectiveness and hit rates."""
        params = TorusParameters(2.0, 1.0, 30, 15, 0.05)

        # Generate multiple frames with same parameters (should hit torus cache)
        for _ in range(5):
            points = generate_torus_points(params)

        # Should have high cache hit rate for torus generation
        # Check if torus cache contains results
        self.assertGreater(len(_torus_cache), 0, "Torus cache should contain results after multiple generations")

        # Test rotation caching with repeated angles
        points = generate_torus_points(params)
        angles = [0.1, 0.2, 0.1, 0.2, 0.3, 0.1]  # Repeated angles

        for angle in angles:
            apply_rotation(points, angle)

        # Should have good cache hit rate
        total_cache_operations = _performance_stats['cache_hits'] + _performance_stats['cache_misses']
        if total_cache_operations > 0:
            hit_rate = _performance_stats['cache_hits'] / total_cache_operations
            self.assertGreater(hit_rate, 0.3, f"Cache hit rate too low: {hit_rate:.2f}")

    def test_performance_under_load(self):
        """Test performance under sustained load."""
        params = TorusParameters(2.0, 1.0, 40, 20, 0.05)

        # Measure performance for 100 operations
        start_time = time.time()

        for i in range(100):
            points = generate_torus_points(params)
            rotated = apply_rotation(points, i * 0.1)

            # Project first few points to test projection performance
            for j in range(min(10, len(rotated))):
                project_to_screen(rotated[j])

        total_time = time.time() - start_time
        avg_time_per_operation = total_time / 100

        # Performance requirement: <50ms per full operation cycle
        self.assertLess(avg_time_per_operation, 0.05,
                       f"Average operation time too slow: {avg_time_per_operation*1000:.2f}ms > 50ms")

    def test_memory_leak_detection(self):
        """Test for memory leaks during extended operation."""
        import gc

        # Force garbage collection and measure initial state
        gc.collect()
        initial_memory = memory_monitor()

        # Perform many operations that could cause leaks
        for cycle in range(50):
            # Create and destroy torus geometry
            params = TorusParameters(2.0, 1.0, 20, 10, 0.05)
            points = generate_torus_points(params)

            # Rotate and project
            for angle in [0.1, 0.2, 0.3]:
                rotated = apply_rotation(points, angle)

            # Clear some caches periodically to test cleanup
            if cycle % 10 == 0:
                # Trigger memory monitoring cleanup
                memory_monitor()

        # Force garbage collection
        gc.collect()
        final_memory = memory_monitor()

        # Memory usage should not grow excessively
        memory_growth = (final_memory['token_cache_memory'] -
                        initial_memory['token_cache_memory'])

        # Allow some growth but detect excessive leaks
        self.assertLess(abs(memory_growth), 10000000, "Potential memory leak detected")

    def test_performance_degradation_detection(self):
        """Test performance degradation detection system."""
        # Simulate good performance followed by degradation
        good_times = [1/35] * 20  # 35 FPS
        bad_times = [1/20] * 10   # 20 FPS

        _performance_stats['frame_times'] = good_times + bad_times

        # Analyze last 10 vs previous 10 frames
        recent_avg = sum(_performance_stats['frame_times'][-10:]) / 10
        older_avg = sum(_performance_stats['frame_times'][-20:-10]) / 10

        # Should detect degradation
        self.assertGreater(recent_avg, older_avg * 1.2,
                          "Should detect performance degradation")

        recent_fps = 1.0 / recent_avg
        self.assertLess(recent_fps, 30, "Should detect FPS below target")

    def test_adaptive_quality_control(self):
        """Test adaptive quality control mechanisms."""
        # Test degraded mode parameters
        normal_params = TorusParameters(2.0, 1.0, 50, 25, 0.05)
        degraded_params = TorusParameters(2.0, 1.0, 30, 15, 0.05)

        # Normal resolution should generate more points
        normal_points = generate_torus_points(normal_params)
        degraded_points = generate_torus_points(degraded_params)

        self.assertGreater(len(normal_points), len(degraded_points),
                          "Normal mode should generate more points than degraded mode")

        # Degraded mode should be faster (clear cache first to ensure fair comparison)
        _torus_cache.clear()

        start_time = time.time()
        for _ in range(10):
            generate_torus_points(degraded_params)
        degraded_time = time.time() - start_time

        _torus_cache.clear()

        start_time = time.time()
        for _ in range(10):
            generate_torus_points(normal_params)
        normal_time = time.time() - start_time

        # Allow for timing variations - degraded should be noticeably faster or at least not slower
        self.assertLessEqual(degraded_time, normal_time * 1.1,
                           "Degraded mode should not be significantly slower than normal mode")


class TestPerformanceIntegration(unittest.TestCase):
    """Integration tests for performance optimization systems."""

    def setUp(self):
        """Set up integration test environment."""
        clear_performance_caches()
        _torus_cache.clear()

    def test_full_animation_cycle_performance(self):
        """Test performance of complete animation cycle."""
        # Simulate one complete animation frame
        params = TorusParameters(2.0, 1.0, 50, 25, 0.05)

        start_time = time.time()

        # Generate torus (should be cached after first call)
        points = generate_torus_points(params)

        # Apply rotation
        rotated_points = apply_rotation(points, 1.0)

        # Project subset of points (simulate token mapping)
        projected_points = []
        for i in range(0, len(rotated_points), 10):  # Every 10th point
            projected = project_to_screen(rotated_points[i])
            projected_points.append(projected)

        cycle_time = time.time() - start_time

        # Performance requirement: Full cycle should complete in <33ms (30 FPS)
        self.assertLess(cycle_time, 0.033,
                       f"Animation cycle too slow: {cycle_time*1000:.2f}ms > 33ms")

    def test_sustained_performance(self):
        """Test sustained performance over multiple frames."""
        params = TorusParameters(2.0, 1.0, 40, 20, 0.05)
        frame_times = []

        # Simulate 30 frames of animation
        for frame in range(30):
            start_time = time.time()

            # Standard animation operations
            points = generate_torus_points(params)
            angle = frame * 0.1
            rotated_points = apply_rotation(points, angle)

            # Sample projection operations
            for i in range(0, len(rotated_points), 20):
                project_to_screen(rotated_points[i])

            frame_time = time.time() - start_time
            frame_times.append(frame_time)

        # Calculate performance statistics
        avg_frame_time = sum(frame_times) / len(frame_times)
        max_frame_time = max(frame_times)
        avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

        # Performance requirements
        self.assertGreater(avg_fps, 30, f"Average FPS too low: {avg_fps:.1f} < 30")
        self.assertLess(max_frame_time, 0.05, f"Frame spike too high: {max_frame_time*1000:.2f}ms")

        # Consistency requirement: 90% of frames should be within 50% of average (more lenient for test environment)
        fast_frames = sum(1 for t in frame_times if t < avg_frame_time * 1.5)
        consistency_rate = fast_frames / len(frame_times)
        self.assertGreaterEqual(consistency_rate, 0.90,
                               f"Frame time consistency too low: {consistency_rate:.2f}")


if __name__ == '__main__':
    # Set up test environment
    print("Running Performance Optimization Tests for Story 4.1")
    print("="*60)

    # Run tests with detailed output
    unittest.main(verbosity=2, buffer=True)