# Epic 5: Code Maintainability - File Size Refactoring

**Epic Goal:** Refactor rotating_donut.py (5920 lines) into maintainable modular architecture to comply with project guidelines (<1000 lines per file) while preserving all functionality, test coverage, and performance characteristics, enabling sustainable long-term development and reducing technical debt.

## Story 5.1: Extract Mathematical and Parsing Engines
**As a developer,**
**I want mathematical and parsing components extracted into dedicated modules,**
**so that the codebase becomes more maintainable while preserving all functionality.**

**Acceptance Criteria:**
1. Extract mathematical_engine.py (~800 lines) with geometry, rotation, and projection functions
2. Extract parsing_engine.py (~600 lines) with tokenization and classification logic
3. Update imports in rotating_donut.py to use new modules
4. Adapt self-referential code reading to work from new module location
5. All mathematical tests (test_mathematical.py, test_mathematical_simple.py) pass unchanged
6. All parsing tests (test_parsing.py) pass unchanged
7. Performance benchmarks maintained (FPS ≥30, startup ≤0.6s)
8. Zero external dependencies preserved (NFR1)

## Story 5.2: Extract Rendering and Animation Controllers
**As a developer,**
**I want rendering and animation components extracted into dedicated modules,**
**so that the modular architecture is complete and maintainable.**

**Acceptance Criteria:**
1. Extract rendering_engine.py (~900 lines) with token mapping and frame generation
2. Extract animation_controller.py (~400 lines) with main loop and timing control
3. Update imports in rotating_donut.py to integrate all extracted modules
4. Integrate with existing cache_manager and performance_monitor modules
5. All rendering tests (test_rendering.py, test_visual_harmony.py) pass unchanged
6. All animation tests (test_integration.py, test_performance.py) pass unchanged
7. Module dependency hierarchy is clean (no circular imports)
8. Performance characteristics maintained or improved

## Story 5.3: Final Integration and Validation
**As a developer,**
**I want complete validation of the refactored architecture,**
**so that production deployment can proceed with confidence.**

**Acceptance Criteria:**
1. rotating_donut.py reduced to orchestration layer only (~400-500 lines)
2. All 13 test files pass with 100% success rate
3. Performance validation: FPS ≥30, startup ≤0.6s, memory within 10% baseline
4. Cross-platform smoke testing completed (Windows, macOS, Linux)
5. All 8 functional requirements (FR1-FR8) validated
6. All 6 non-functional requirements (NFR1-NFR6) validated
7. Architecture documentation updated (source-tree.md, components.md)
8. QA validation approved and production readiness confirmed
