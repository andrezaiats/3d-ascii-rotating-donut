# Tech Stack

## Cloud Infrastructure
- **Provider:** None (local execution only)
- **Key Services:** None (standalone Python script)
- **Deployment Regions:** Global (runs anywhere with Python)

## Technology Stack Table

| Category | Technology | Version | Purpose | Rationale |
|----------|------------|---------|---------|-----------|
| **Language** | Python | 3.8+ | Primary development language | Balances modern features (walrus operator, positional-only params) with broad compatibility |
| **Runtime** | Python Interpreter | 3.8+ | Script execution environment | Wide availability, mature stdlib, excellent math support |
| **Mathematics** | math module | stdlib | 3D calculations, trigonometry | Built-in, optimized, covers all torus/rotation needs |
| **Parsing** | tokenize module | stdlib | Source code analysis | Perfect for self-referential token extraction and classification |
| **I/O** | sys.stdout | stdlib | Terminal output | Cross-platform, reliable, no buffering issues |
| **Timing** | time module | stdlib | Animation frame control | Precise timing for smooth 30+ FPS animation |
| **File Access** | __file__ + open() | stdlib | Self-code reading | Standard approach for script self-inspection |
| **Error Handling** | Built-in exceptions | stdlib | Graceful failure modes | No external logging needed for single-file constraint |
| **Terminal Control** | print() with flush | stdlib | Screen clearing and output | Maximum compatibility across terminal types |
| **Development** | Function-based design | Python | Code organization | Maintains single-file readability and educational value |
