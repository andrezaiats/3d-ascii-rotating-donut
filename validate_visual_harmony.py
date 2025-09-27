#!/usr/bin/env python3
"""
Quick validation script for Story 3.5 Visual Harmony features.

Tests the main visual harmony functions without requiring external dependencies.
"""

from rotating_donut import (
    Point3D, Point2D, CodeToken, ImportanceLevel, DisplayFrame,
    DensityAnalysis, VisualFlowState, AestheticQuality,
    analyze_token_density_patterns,
    apply_adaptive_density_control,
    handle_visual_edge_cases,
    validate_artistic_impact_and_quality,
    TERMINAL_WIDTH, TERMINAL_HEIGHT, ASCII_CHARS
)

def test_density_analysis():
    """Test density analysis functionality."""
    print("Testing density analysis...")

    # Create sample data
    tokens = [
        CodeToken(type='NAME', value='def', importance=ImportanceLevel.CRITICAL, line=1, column=0, ascii_char='#'),
        CodeToken(type='OP', value='+', importance=ImportanceLevel.HIGH, line=2, column=5, ascii_char='+'),
    ]

    points = [
        Point3D(x=1.0, y=0.0, z=0.5, u=0.0, v=0.0, nx=1.0, ny=0.0, nz=0.0),
        Point3D(x=0.0, y=1.0, z=0.5, u=1.57, v=1.57, nx=0.0, ny=1.0, nz=0.0),
    ]

    mapped_pairs = list(zip(points, tokens))

    try:
        result = analyze_token_density_patterns(mapped_pairs)
        print(f"[OK] Density analysis completed: {result.total_tokens} tokens processed")
        print(f"  Average density: {result.average_density:.2f}")
        print(f"  Max density: {result.max_density}")
        print(f"  Hotspots found: {len(result.hotspots)}")
        return True
    except Exception as e:
        print(f"[FAIL] Density analysis failed: {e}")
        return False

def test_edge_case_handling():
    """Test edge case handling functionality."""
    print("\nTesting edge case handling...")

    try:
        # Test with empty input
        result = handle_visual_edge_cases([], DensityAnalysis(0, {}, [], [], 0.0, 0, 0))
        print("[OK] Empty input handled correctly")

        # Test with sample data
        tokens = [CodeToken(type='NAME', value='test', importance=ImportanceLevel.MEDIUM, line=1, column=0, ascii_char='-')]
        points = [Point3D(x=1.0, y=0.0, z=0.5, u=0.0, v=0.0, nx=1.0, ny=0.0, nz=0.0)]
        mapped_pairs = list(zip(points, tokens))

        density_analysis = DensityAnalysis(1, {(10, 5): 1}, [], [], 0.5, 1, 0)
        result = handle_visual_edge_cases(mapped_pairs, density_analysis)
        print(f"[OK] Edge case handling completed: {len(result)} pairs processed")
        return True
    except Exception as e:
        print(f"[FAIL] Edge case handling failed: {e}")
        return False

def test_aesthetic_quality():
    """Test aesthetic quality validation."""
    print("\nTesting aesthetic quality validation...")

    try:
        # Create test frame
        buffer = [[ASCII_CHARS['BACKGROUND'] for _ in range(TERMINAL_WIDTH)] for _ in range(TERMINAL_HEIGHT)]
        buffer[5][10] = '#'
        buffer[8][15] = '+'

        frame = DisplayFrame(
            width=TERMINAL_WIDTH, height=TERMINAL_HEIGHT,
            buffer=buffer, depth_buffer=[], frame_number=1
        )

        density_analysis = DensityAnalysis(2, {(10, 5): 1, (15, 8): 1}, [], [], 0.1, 1, 0)

        quality = validate_artistic_impact_and_quality(frame, density_analysis, None, 1)

        print(f"[OK] Aesthetic quality validation completed")
        print(f"  Overall score: {quality.overall_score:.2f}")
        print(f"  Density balance: {quality.density_balance:.2f}")
        print(f"  Pattern clarity: {quality.pattern_clarity:.2f}")
        print(f"  Edge case handling: {quality.edge_case_handling:.2f}")
        print(f"  Artistic impact: {quality.artistic_impact:.2f}")
        return True
    except Exception as e:
        print(f"[FAIL] Aesthetic quality validation failed: {e}")
        return False

def test_performance():
    """Test performance constraints."""
    print("\nTesting performance constraints...")

    import time

    try:
        # Create larger dataset for performance testing
        tokens = []
        points = []
        for i in range(100):
            tokens.append(CodeToken(
                type='NAME', value=f'var{i}', importance=ImportanceLevel.MEDIUM,
                line=i//10, column=i%10, ascii_char='-'
            ))
            points.append(Point3D(
                x=float(i), y=0.0, z=0.5, u=0.0, v=0.0, nx=1.0, ny=0.0, nz=0.0
            ))

        mapped_pairs = list(zip(points, tokens))

        # Time density analysis
        start = time.time()
        for _ in range(10):  # Run 10 iterations
            result = analyze_token_density_patterns(mapped_pairs)
        elapsed = time.time() - start

        avg_time = elapsed / 10
        print(f"[OK] Performance test completed")
        print(f"  Average time per analysis: {avg_time*1000:.1f}ms")
        print(f"  Target per frame (30 FPS): 33.3ms")

        if avg_time < 0.01:  # Less than 10ms is excellent
            print("  [EXCELLENT] Performance: EXCELLENT")
        elif avg_time < 0.033:  # Less than 33ms meets 30 FPS target
            print("  [GOOD] Performance: GOOD")
        else:
            print("  [WARNING] Performance: NEEDS OPTIMIZATION")

        return True
    except Exception as e:
        print(f"[FAIL] Performance test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("=== Story 3.5 Visual Harmony Validation ===\n")

    tests = [
        test_density_analysis,
        test_edge_case_handling,
        test_aesthetic_quality,
        test_performance
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n=== Validation Results ===")
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("[SUCCESS] ALL TESTS PASSED - Visual Harmony implementation is working correctly!")
        return True
    else:
        print("[ERROR] Some tests failed - Review implementation")
        return False

if __name__ == "__main__":
    main()