# Source Tree

```
3d-ascii-donut/
├── rotating_donut.py           # Single main executable file containing all functionality
├── README.md                   # Project description, usage instructions, mathematical background
├── docs/
│   ├── architecture.md         # This architecture document
│   ├── prd.md                  # Product requirements document
│   └── examples/
│       ├── screenshot.txt      # Example ASCII output for documentation
│       └── performance.md      # Performance benchmarks and optimization notes
├── tests/
│   ├── test_mathematical.py    # Unit tests for 3D math functions
│   ├── test_parsing.py         # Unit tests for tokenization logic
│   ├── test_rendering.py       # Unit tests for ASCII rendering
│   └── test_integration.py     # Full system integration tests
├── .gitignore                  # Python-specific gitignore
└── LICENSE                     # Open source license (MIT recommended)
```

**Core Implementation Structure Within `rotating_donut.py`:**

```python
#!/usr/bin/env python3
"""
3D ASCII Rotating Donut with Self-Code Display
Mathematical art that visualizes its own source code as a rotating torus
"""

# === IMPORTS (Python Standard Library Only) ===
import math
import sys
import time
import tokenize
from io import StringIO

# === DATA MODELS ===
# Point3D, Point2D, CodeToken, TorusParameters, DisplayFrame classes/namedtuples

# === MATHEMATICAL ENGINE ===
def generate_torus_points(params):
    """Generate 3D torus surface points using parametric equations"""

def apply_rotation(points, angle):
    """Apply Y-axis rotation matrix to 3D points"""

def project_to_screen(point):
    """Perspective projection from 3D to 2D screen coordinates"""

# === PARSING ENGINE ===
def read_self_code():
    """Read this script's own source code"""

def tokenize_code(source):
    """Parse source code into classified tokens"""

def classify_importance(token):
    """Assign semantic importance hierarchy to tokens"""

# === RENDERING ENGINE ===
def map_tokens_to_surface(tokens, points):
    """Distribute code tokens across torus surface coordinates"""

def generate_ascii_frame(points):
    """Create ASCII character frame with depth sorting"""

def output_to_terminal(frame):
    """Render frame to terminal with screen clearing"""

# === ANIMATION CONTROLLER ===
def run_animation_loop():
    """Main execution loop with frame rate control"""

def main():
    """Entry point with error handling and setup"""

if __name__ == "__main__":
    main()
```
