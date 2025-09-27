# Test Strategy and Standards

## Testing Philosophy
- **Approach:** Test-after development with mathematical accuracy validation and visual output verification
- **Coverage Goals:** 90%+ for mathematical functions, 70%+ for rendering logic, 100% for critical path (animation loop)
- **Test Pyramid:** 70% unit tests (math/parsing), 20% integration tests (component interaction), 10% visual validation tests

## Test Types and Organization

### Unit Tests
- **Framework:** pytest 7.4+ with built-in fixtures
- **File Convention:** `test_{component_name}.py` (e.g., `test_mathematical.py`, `test_parsing.py`)
- **Location:** `tests/` directory separate from main script
- **Mocking Library:** unittest.mock (Python stdlib) for file I/O and terminal output
- **Coverage Requirement:** 90%+ for all mathematical calculations

**AI Agent Requirements:**
- Generate tests for all public mathematical functions with known geometric properties
- Cover edge cases: zero radius, negative values, extreme rotation angles, empty token lists
- Follow AAA pattern (Arrange, Act, Assert) with clear test names
- Mock file system operations and terminal output for deterministic testing
- Validate mathematical properties: torus volume, surface area, point distribution

### Integration Tests
- **Scope:** Component interaction, end-to-end animation pipeline, cross-platform compatibility
- **Location:** `tests/test_integration.py`
- **Test Infrastructure:**
  - **Terminal Output:** Capture stdout for visual validation
  - **File System:** Temporary test files for self-reading scenarios
  - **Performance:** Frame rate measurement and timing validation

### Visual Validation Tests
- **Framework:** Custom visual comparison utilities
- **Scope:** ASCII output correctness, rotation continuity, visual quality
- **Environment:** Controlled test environment with known source code
- **Test Data:** Predefined token sets and expected visual patterns

## Test Data Management
- **Strategy:** Generated test data using mathematical properties and controlled source code samples
- **Fixtures:** `pytest` fixtures for torus parameters, token samples, and expected results
- **Factories:** Helper functions to generate test cases with various complexity levels
- **Cleanup:** Automatic cleanup of temporary files and terminal state restoration

## Continuous Testing
- **CI Integration:** GitHub Actions matrix testing across Python 3.8, 3.9, 3.10, 3.11 on Ubuntu, Windows, macOS
- **Performance Tests:** Automated frame rate validation and memory usage monitoring
- **Security Tests:** Static analysis for potential code injection through self-reading mechanisms
