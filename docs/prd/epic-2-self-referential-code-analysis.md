# Epic 2: Self-Referential Code Analysis

**Epic Goal:** Implement tokenization system that reads and parses the script's own source code, creating semantic importance hierarchy and mapping tokens to visual elements for code-driven representation.

## Story 2.1: Self-File Reading System
**As a developer,**
**I want the script to read its own source code reliably,**
**so that it can analyze and visualize its own structure.**

**Acceptance Criteria:**
1. Use `__file__` to identify the script's own path reliably across platforms
2. Read complete source code content with proper encoding handling (UTF-8)
3. Handle edge cases like symbolic links, relative paths, and different file systems
4. Validate that file content matches current executing code
5. Gracefully handle file access errors with informative messages

## Story 2.2: Token Parsing and Classification
**As a developer,**
**I want semantic analysis of source code using Python's tokenize module,**
**so that different code elements can be identified and categorized.**

**Acceptance Criteria:**
1. Parse source code into tokens using Python's built-in tokenize module
2. Classify tokens into categories: keywords, operators, identifiers, literals, comments, whitespace
3. Extract token position information (line, column) for spatial mapping
4. Handle all Python token types including strings, numbers, and special characters
5. Maintain token sequence order for proper code reconstruction

## Story 2.3: Semantic Importance Hierarchy
**As a developer,**
**I want a 4-level importance ranking system for code tokens,**
**so that visually significant elements can be emphasized in the donut display.**

**Acceptance Criteria:**
1. Define 4-level brightness hierarchy: Critical (keywords), High (operators), Medium (identifiers), Low (comments/whitespace)
2. Map Python token types to appropriate importance levels based on semantic value
3. Create configurable importance weights for fine-tuning visual emphasis
4. Handle special cases like built-in functions, decorators, and string literals
5. Document importance classification rationale in code comments

## Story 2.4: Token-to-ASCII Character Mapping
**As a developer,**
**I want tokens mapped to specific ASCII characters based on importance,**
**so that code structure drives visual representation on the torus surface.**

**Acceptance Criteria:**
1. Create character mapping: Critical=#, High=+, Medium=-, Low=. (or similar progression)
2. Implement density mapping where important tokens occupy more surface points
3. Distribute tokens across torus surface based on token sequence and importance
4. Handle varying source code lengths by scaling distribution appropriately
5. Ensure visual balance between different token types for aesthetic appeal

## Story 2.5: Code Structure Analysis
**As a developer,**
**I want analysis of code structural elements like functions, classes, and imports,**
**so that architectural patterns can influence visual representation.**

**Acceptance Criteria:**
1. Identify structural elements: function definitions, class definitions, import statements
2. Calculate relative importance of different code sections based on complexity
3. Map structural hierarchy to spatial distribution on torus surface
4. Handle nested structures (methods within classes, nested functions)
5. Provide debugging output showing token classification and mapping results
