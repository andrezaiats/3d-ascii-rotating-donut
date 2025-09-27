# Coding Standards

**IMPORTANT:** These standards are MANDATORY for AI agents developing this project. They focus on project-specific conventions and potential pitfalls rather than general best practices.

## Core Standards
- **Languages & Runtimes:** Python 3.8+ only, no version compatibility shims
- **Style & Linting:** PEP 8 compliance with 88-character line length (Black formatter style)
- **Test Organization:** Tests in separate files, mathematical accuracy tests required for all geometry functions

## Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Functions | snake_case with descriptive math terms | `generate_torus_points()`, `apply_rotation_matrix()` |
| Constants | UPPER_SNAKE_CASE for mathematical constants | `TORUS_OUTER_RADIUS`, `FRAME_RATE_TARGET` |
| Classes/NamedTuples | PascalCase for data models | `Point3D`, `CodeToken`, `TorusParameters` |

## Critical Rules

- **Mathematical Precision:** Always use `math.pi` and `math.tau`, never hardcoded decimal approximations
- **Performance Critical:** Cache torus geometry calculations - never regenerate identical point sets
- **Self-Reference Safety:** Validate `__file__` exists before attempting self-code reading
- **Terminal Compatibility:** Use `print(..., flush=True)` for all animation output to prevent buffering issues
- **Error Message Format:** All user-facing errors must include "Solution:" with actionable guidance
- **Frame Rate Enforcement:** Every animation loop iteration must include timing control - never run unbounded
- **Memory Management:** Clear display buffers after each frame - prevent memory accumulation during long runs
- **Mathematical Validation:** All torus parameters must be validated (outer_radius > inner_radius > 0)
- **Token Classification:** Unknown token types must default to LOW importance, never fail parsing
- **ASCII Character Safety:** Only use characters guaranteed available in basic terminal: . - + # (no Unicode)

## Language-Specific Guidelines

### Python Specifics
- **Type Hints:** Use for all function signatures involving mathematical calculations
- **f-strings:** Required for all string formatting (leverages Python 3.8+ requirement)
- **Math Module Usage:** Import specific functions (`from math import sin, cos, pi`) for performance
- **List Comprehensions:** Prefer for point generation and transformations when readable
- **Exception Handling:** Catch specific exceptions, never bare `except:` clauses
- **Main Guard:** Always use `if __name__ == "__main__":` pattern for executable scripts
