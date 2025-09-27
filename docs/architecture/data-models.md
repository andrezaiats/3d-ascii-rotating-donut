# Data Models

## Point3D

**Purpose:** Represents a 3D coordinate in space for torus geometry and transformations

**Key Attributes:**
- x: float - X-coordinate in 3D space
- y: float - Y-coordinate in 3D space
- z: float - Z-coordinate (depth) for proper rendering order
- u: float - Parametric u coordinate on torus surface (0-2π)
- v: float - Parametric v coordinate on torus surface (0-2π)

**Relationships:**
- Projected to Point2D for screen display
- Generated from TorusParameters during surface calculation
- Modified by RotationMatrix during animation

## Point2D

**Purpose:** Represents projected screen coordinates for ASCII terminal display

**Key Attributes:**
- x: int - Screen column position (0-39 for 40-char width)
- y: int - Screen row position (0-19 for 20-char height)
- depth: float - Z-depth for sorting closest-to-farthest rendering
- visible: bool - Whether point is within screen bounds

**Relationships:**
- Derived from Point3D through perspective projection
- Mapped to DisplayFrame grid coordinates
- Associated with CodeToken for character rendering

## CodeToken

**Purpose:** Represents a parsed element of the source code with semantic classification

**Key Attributes:**
- type: TokenType - Classification (KEYWORD, OPERATOR, IDENTIFIER, LITERAL, COMMENT)
- value: str - Actual token text content
- importance: ImportanceLevel - Semantic weight (CRITICAL=4, HIGH=3, MEDIUM=2, LOW=1)
- line: int - Source line number for debugging
- column: int - Source column position for debugging
- ascii_char: str - Associated ASCII character for rendering

**Relationships:**
- Multiple tokens mapped to Point3D surface coordinates
- Drives character selection in DisplayFrame
- Classified by SemanticAnalyzer parsing results

## TorusParameters

**Purpose:** Defines the mathematical parameters for 3D torus generation

**Key Attributes:**
- outer_radius: float - Major radius (R) of the torus (typically 2.0)
- inner_radius: float - Minor radius (r) of the torus (typically 1.0)
- u_resolution: int - Number of sample points around major circumference
- v_resolution: int - Number of sample points around minor circumference
- rotation_speed: float - Y-axis rotation increment per frame (radians)

**Relationships:**
- Used by Point3D generation algorithms
- Drives surface density for token distribution
- Modified by animation loop for rotation effects

## DisplayFrame

**Purpose:** Represents the 40x20 ASCII character buffer for terminal output

**Key Attributes:**
- width: int - Character grid width (40)
- height: int - Character grid height (20)
- buffer: List[List[str]] - 2D array of ASCII characters
- depth_buffer: List[List[float]] - Z-depth values for proper layering
- frame_number: int - Current animation frame for debugging

**Relationships:**
- Populated by Point2D projections
- Characters determined by associated CodeToken
- Cleared and regenerated each animation frame
