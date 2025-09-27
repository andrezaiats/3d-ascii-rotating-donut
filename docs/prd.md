# 3D ASCII Rotating Donut with Self-Code Display Product Requirements Document (PRD)

## Goals and Background Context

### Goals
- Create a viral, shareable piece of creative code that generates 1000+ GitHub stars within 6 months
- Establish reputation for innovative creative coding projects combining math, art, and programming
- Generate substantial social media engagement (500+ shares/mentions within 3 months)
- Serve as portfolio centerpiece demonstrating mathematical and parsing skills
- Provide educational value for understanding 3D mathematics and code tokenization
- Create a self-referential artwork that transforms source code into mathematical visualization

### Background Context

This project addresses a gap in code visualization tools that are typically static, disconnected from runtime experience, and fail to create engaging artistic representations of code structure. Current solutions like IDE syntax highlighting or separate diagramming tools miss opportunities for enhanced code comprehension through visual metaphors and creative expression.

The solution leverages Python's self-referential capabilities and mathematical beauty to create a script that reads its own source code, parses it semantically, and renders it as a continuously rotating 3D ASCII donut. Different code elements (keywords, variables, operators, comments) are represented by ASCII characters based on their semantic importance, creating a meditative, artistic visualization that serves both educational and aesthetic purposes.

### Change Log
| Date | Version | Description | Author |
|------|---------|-------------|---------|
| 2025-09-26 | 1.0 | Initial PRD creation from Project Brief | John (PM) |

## Requirements

### Functional

1. **FR1:** The system shall read its own source code using `__file__` and Python's file I/O
2. **FR2:** The system shall parse source code tokens using Python's built-in tokenize module
3. **FR3:** The system shall generate 3D torus geometry with configurable inner/outer radius parameters
4. **FR4:** The system shall map code tokens to ASCII characters based on a 4-level semantic importance hierarchy
5. **FR5:** The system shall implement density mapping where important tokens receive more surface area
6. **FR6:** The system shall render continuous Y-axis rotation with configurable speed
7. **FR7:** The system shall output clean 40x20 character ASCII display with proper depth sorting
8. **FR8:** The system shall maintain 30+ FPS smooth animation performance

### Non Functional

1. **NFR1:** The system must use zero external dependencies (Python stdlib only) for maximum portability
2. **NFR2:** The implementation must remain a single Python file for shareability and simplicity
3. **NFR3:** The system must be compatible with Python 3.6+ across Windows, macOS, and Linux
4. **NFR4:** The system must maintain minimal CPU usage to prevent system slowdown
5. **NFR5:** The code must be self-documenting and educational for intermediate Python developers
6. **NFR6:** Terminal output must be compatible with standard terminal emulators

## User Interface Design Goals

### Overall UX Vision
The user experience should be immediate and mesmerizing - a single command execution that transforms into a continuously rotating mathematical artwork. The interface is purely visual through terminal display, requiring no user interaction beyond starting the script. The experience should evoke contemplation and appreciation for the intersection of mathematics, programming, and art.

### Key Interaction Paradigms
- **Zero-interaction execution** - Run script and immediately see results
- **Contemplative viewing** - Designed for extended observation and appreciation
- **Educational exploration** - Code serves as learning material for mathematical concepts
- **Social sharing** - Visual output optimized for screenshots and demonstrations

### Core Screens and Views
- **Single rotating donut display** - The primary and only user interface
- **Terminal output window** - 40x20 character ASCII rendering space
- **Startup message** (optional) - Brief explanation of what user is viewing

### Accessibility: None
This is a visual art piece focused on ASCII terminal output. Accessibility considerations are limited by the fundamental nature of the visual mathematical art concept.

### Branding
Mathematical elegance meets programming artistry - clean, minimal aesthetic that emphasizes the beauty of code and mathematics. No corporate branding required; the code itself serves as the artistic statement.

### Target Device and Platforms: Web Responsive
Cross-platform terminal compatibility targeting any system with Python 3.6+ and basic ASCII terminal support, including Windows Command Prompt, macOS Terminal, and Linux terminal emulators.

## Technical Assumptions

### Repository Structure: Monorepo
Single repository containing one Python file with comprehensive documentation

### Service Architecture
**Monolith** - Single self-contained Python script with no external services or dependencies

### Testing Requirements
**Unit + Integration** - Need both mathematical accuracy testing and visual output verification methods

### Additional Technical Assumptions and Requests

**Language & Version:**
- **Python 3.6+** (minimum for f-string support and modern features)
- **Pure Python implementation** - no C extensions or external libraries

**Development Environment:**
- **Cross-platform compatibility** - Windows, macOS, Linux terminal support
- **Standard terminal requirements** - basic ASCII character rendering capability
- **No special terminal features** - avoid ANSI colors, cursor positioning, or advanced terminal control

**Code Organization:**
- **Single file structure** with clear functional separation:
  - Mathematical functions (torus geometry)
  - Parsing logic (tokenization)
  - Rendering engine (ASCII output)
  - Main animation loop
- **Self-documenting code** with mathematical explanations in comments
- **Educational code style** - readable over hyper-optimized

**Performance & Deployment:**
- **No build process** - direct Python execution
- **Immediate execution** - no installation or setup steps
- **Graceful degradation** - handle different terminal capabilities
- **Resource efficient** - minimal memory footprint for smooth animation

## Epic List

### Epic 1: Foundation & Mathematical Core
**Goal:** Establish project infrastructure and implement core 3D torus mathematics with basic ASCII rendering capability.

### Epic 2: Self-Referential Code Analysis
**Goal:** Implement tokenization system that reads and parses the script's own source code, creating semantic importance hierarchy.

### Epic 3: Dynamic Visualization Engine
**Goal:** Integrate code parsing with 3D rendering to create the rotating donut display where code tokens drive visual representation.

### Epic 4: Performance & Polish
**Goal:** Optimize animation performance, enhance visual quality, and ensure cross-platform compatibility for production release.

## Epic 1: Foundation & Mathematical Core

**Epic Goal:** Establish project infrastructure and implement core 3D torus mathematics with basic ASCII rendering capability, delivering a working rotating donut that validates the mathematical approach and visual concept.

### Story 1.1: Project Setup and Basic Structure
**As a developer,**
**I want a well-organized Python file with clear module structure,**
**so that the codebase is maintainable and educational.**

**Acceptance Criteria:**
1. Single Python file created with clear function organization and imports
2. File includes comprehensive header documentation explaining the project concept
3. Basic project structure includes placeholders for math, rendering, and main loop functions
4. Code follows PEP 8 style guidelines for educational readability
5. File can be executed without errors (even if minimal functionality)

### Story 1.2: 3D Torus Mathematical Foundation
**As a developer,**
**I want accurate 3D torus geometry generation,**
**so that the visual foundation supports realistic donut shape rendering.**

**Acceptance Criteria:**
1. Implement parametric torus equations with configurable inner radius (r) and outer radius (R)
2. Generate 3D points (x, y, z) across torus surface with proper mathematical precision
3. Support configurable resolution for surface point density
4. Include mathematical documentation explaining torus equations in code comments
5. Validate mathematical accuracy with known torus properties (volume, surface area)

### Story 1.3: 3D Projection and Depth Sorting
**As a developer,**
**I want 3D points projected to 2D screen coordinates with proper depth handling,**
**so that the donut appears three-dimensional in ASCII output.**

**Acceptance Criteria:**
1. Implement perspective projection from 3D coordinates to 2D screen space
2. Map projected coordinates to 40x20 ASCII character grid
3. Calculate Z-depth for each surface point to enable proper depth sorting
4. Handle edge cases where points project outside the display area
5. Maintain mathematical precision to prevent visual artifacts

### Story 1.4: Basic ASCII Rendering Engine
**As a developer,**
**I want a clean ASCII character output system,**
**so that 3D geometry can be visualized in terminal display.**

**Acceptance Criteria:**
1. Create 40x20 character buffer for frame rendering
2. Implement depth sorting to render closest points last (painter's algorithm)
3. Use basic ASCII characters (., -, +, #) to represent different depth/brightness levels
4. Clear screen and render new frame for smooth animation
5. Handle terminal compatibility for character output

### Story 1.5: Rotation and Animation Loop
**As a developer,**
**I want smooth Y-axis rotation with configurable speed,**
**so that the donut continuously rotates creating animated visual effect.**

**Acceptance Criteria:**
1. Implement Y-axis rotation matrix transformation for 3D points
2. Create main animation loop with configurable frame rate control
3. Apply rotation incrementally each frame to create smooth motion
4. Maintain consistent timing across different system performance levels
5. Support graceful exit with keyboard interrupt (Ctrl+C)

## Epic 2: Self-Referential Code Analysis

**Epic Goal:** Implement tokenization system that reads and parses the script's own source code, creating semantic importance hierarchy and mapping tokens to visual elements for code-driven representation.

### Story 2.1: Self-File Reading System
**As a developer,**
**I want the script to read its own source code reliably,**
**so that it can analyze and visualize its own structure.**

**Acceptance Criteria:**
1. Use `__file__` to identify the script's own path reliably across platforms
2. Read complete source code content with proper encoding handling (UTF-8)
3. Handle edge cases like symbolic links, relative paths, and different file systems
4. Validate that file content matches current executing code
5. Gracefully handle file access errors with informative messages

### Story 2.2: Token Parsing and Classification
**As a developer,**
**I want semantic analysis of source code using Python's tokenize module,**
**so that different code elements can be identified and categorized.**

**Acceptance Criteria:**
1. Parse source code into tokens using Python's built-in tokenize module
2. Classify tokens into categories: keywords, operators, identifiers, literals, comments, whitespace
3. Extract token position information (line, column) for spatial mapping
4. Handle all Python token types including strings, numbers, and special characters
5. Maintain token sequence order for proper code reconstruction

### Story 2.3: Semantic Importance Hierarchy
**As a developer,**
**I want a 4-level importance ranking system for code tokens,**
**so that visually significant elements can be emphasized in the donut display.**

**Acceptance Criteria:**
1. Define 4-level brightness hierarchy: Critical (keywords), High (operators), Medium (identifiers), Low (comments/whitespace)
2. Map Python token types to appropriate importance levels based on semantic value
3. Create configurable importance weights for fine-tuning visual emphasis
4. Handle special cases like built-in functions, decorators, and string literals
5. Document importance classification rationale in code comments

### Story 2.4: Token-to-ASCII Character Mapping
**As a developer,**
**I want tokens mapped to specific ASCII characters based on importance,**
**so that code structure drives visual representation on the torus surface.**

**Acceptance Criteria:**
1. Create character mapping: Critical=#, High=+, Medium=-, Low=. (or similar progression)
2. Implement density mapping where important tokens occupy more surface points
3. Distribute tokens across torus surface based on token sequence and importance
4. Handle varying source code lengths by scaling distribution appropriately
5. Ensure visual balance between different token types for aesthetic appeal

### Story 2.5: Code Structure Analysis
**As a developer,**
**I want analysis of code structural elements like functions, classes, and imports,**
**so that architectural patterns can influence visual representation.**

**Acceptance Criteria:**
1. Identify structural elements: function definitions, class definitions, import statements
2. Calculate relative importance of different code sections based on complexity
3. Map structural hierarchy to spatial distribution on torus surface
4. Handle nested structures (methods within classes, nested functions)
5. Provide debugging output showing token classification and mapping results

## Epic 3: Dynamic Visualization Engine

**Epic Goal:** Integrate code parsing with 3D rendering to create the rotating donut display where code tokens drive visual representation, delivering the complete product vision of self-referential mathematical art.

### Story 3.1: Token-Surface Integration
**As a developer,**
**I want code tokens mapped to specific torus surface coordinates,**
**so that source code structure directly drives the visual pattern on the rotating donut.**

**Acceptance Criteria:**
1. Map each token to specific (u,v) parametric coordinates on torus surface
2. Distribute tokens based on importance weighting and sequential order
3. Handle variable source code lengths by scaling distribution algorithms
4. Ensure tokens maintain spatial relationships that reflect code structure
5. Provide consistent mapping that produces recognizable patterns across rotations

### Story 3.2: Dynamic Character Assignment
**As a developer,**
**I want torus surface points to display ASCII characters based on underlying tokens,**
**so that code semantics are visually represented in the rotating display.**

**Acceptance Criteria:**
1. Replace static ASCII characters with token-driven character selection
2. Apply importance hierarchy (Critical=#, High=+, Medium=-, Low=.) to surface rendering
3. Handle multiple tokens mapping to same surface area through priority resolution
4. Maintain smooth visual transitions as rotation reveals different code sections
5. Ensure character density reflects code complexity and token distribution

### Story 3.3: Rotation-Aware Code Display
**As a developer,**
**I want code token visibility to change realistically as the donut rotates,**
**so that different parts of the source code become visible based on 3D orientation.**

**Acceptance Criteria:**
1. Calculate token visibility based on 3D rotation and surface normal orientation
2. Hide tokens on back-facing surfaces or apply appropriate depth-based dimming
3. Smoothly transition token visibility as surface orientation changes during rotation
4. Maintain consistent visual quality across all rotation angles
5. Handle edge cases where token boundaries align with surface visibility boundaries

### Story 3.4: Real-Time Integration Pipeline
**As a developer,**
**I want seamless real-time integration between code analysis and 3D rendering,**
**so that the complete visualization performs smoothly during continuous rotation.**

**Acceptance Criteria:**
1. Integrate token parsing pipeline with 3D rendering loop efficiently
2. Cache parsed token data to avoid re-parsing on every frame
3. Update surface character assignments based on current rotation angle
4. Maintain target frame rate (30+ FPS) with complete code-driven rendering
5. Handle memory management to prevent performance degradation over time

### Story 3.5: Visual Harmony and Aesthetics
**As a developer,**
**I want balanced visual composition that creates engaging artistic effect,**
**so that the result is both technically impressive and aesthetically pleasing.**

**Acceptance Criteria:**
1. Balance token density to avoid visual clutter while maintaining code representation
2. Ensure smooth visual flow as different code sections rotate into view
3. Apply appropriate spacing and clustering to create recognizable patterns
4. Handle visual edge cases like very dense or very sparse code sections
5. Validate that the complete visualization achieves the intended artistic impact

## Epic 4: Performance & Polish

**Epic Goal:** Optimize animation performance, enhance visual quality, and ensure cross-platform compatibility for production release, delivering a viral-ready showcase piece that meets all success criteria.

### Story 4.1: Performance Optimization
**As a developer,**
**I want optimized algorithms and efficient resource usage,**
**so that the animation runs smoothly on various system configurations.**

**Acceptance Criteria:**
1. Profile and optimize mathematical calculations to minimize CPU usage per frame
2. Implement efficient memory management to prevent memory leaks during long runs
3. Cache frequently calculated values (torus geometry, projection matrices) where possible
4. Optimize token parsing to run once at startup rather than per frame
5. Achieve consistent 30+ FPS performance on typical development machines

### Story 4.2: Cross-Platform Compatibility
**As a developer,**
**I want reliable operation across Windows, macOS, and Linux terminals,**
**so that the project works for the broadest possible audience.**

**Acceptance Criteria:**
1. Test and validate functionality across major operating systems and terminal emulators
2. Handle platform-specific differences in file path handling and terminal capabilities
3. Implement graceful degradation for terminals with limited character support
4. Ensure consistent timing behavior across different system performance levels
5. Document known compatibility limitations and recommended terminal settings

### Story 4.3: Error Handling and Robustness
**As a developer,**
**I want comprehensive error handling and graceful failure modes,**
**so that users have a reliable experience even when issues occur.**

**Acceptance Criteria:**
1. Handle file reading errors with informative messages (permissions, missing files)
2. Gracefully handle keyboard interrupts and provide clean exit behavior
3. Validate mathematical inputs and handle edge cases in geometry calculations
4. Provide helpful error messages that guide users toward solutions
5. Implement fallback behaviors for non-critical features that might fail

### Story 4.4: Documentation and Educational Value
**As a developer,**
**I want comprehensive inline documentation and educational explanations,**
**so that the code serves as a learning resource for mathematical and parsing concepts.**

**Acceptance Criteria:**
1. Add detailed mathematical explanations for torus equations and 3D transformations
2. Document the tokenization process and semantic importance hierarchy decisions
3. Include performance considerations and optimization explanations in comments
4. Provide clear examples of how to modify key parameters for different effects
5. Create comprehensive README with usage instructions and mathematical background

### Story 4.5: Social Sharing Optimization
**As a developer,**
**I want the project optimized for viral sharing and portfolio showcase,**
**so that it achieves maximum social media impact and professional recognition.**

**Acceptance Criteria:**
1. Ensure visual output is compelling and screenshot-worthy for social media
2. Optimize startup time to create immediate impressive effect when demonstrated
3. Include clear attribution and project description in code header
4. Test that the project creates "wow factor" appropriate for viral content
5. Validate that all success metrics from the Project Brief are achievable