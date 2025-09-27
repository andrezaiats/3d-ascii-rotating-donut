# Requirements

## Functional

1. **FR1:** The system shall read its own source code using `__file__` and Python's file I/O
2. **FR2:** The system shall parse source code tokens using Python's built-in tokenize module
3. **FR3:** The system shall generate 3D torus geometry with configurable inner/outer radius parameters
4. **FR4:** The system shall map code tokens to ASCII characters based on a 4-level semantic importance hierarchy
5. **FR5:** The system shall implement density mapping where important tokens receive more surface area
6. **FR6:** The system shall render continuous Y-axis rotation with configurable speed
7. **FR7:** The system shall output clean 40x20 character ASCII display with proper depth sorting
8. **FR8:** The system shall maintain 30+ FPS smooth animation performance

## Non Functional

1. **NFR1:** The system must use zero external dependencies (Python stdlib only) for maximum portability
2. **NFR2:** The implementation must remain a single Python file for shareability and simplicity
3. **NFR3:** The system must be compatible with Python 3.6+ across Windows, macOS, and Linux
4. **NFR4:** The system must maintain minimal CPU usage to prevent system slowdown
5. **NFR5:** The code must be self-documenting and educational for intermediate Python developers
6. **NFR6:** Terminal output must be compatible with standard terminal emulators
