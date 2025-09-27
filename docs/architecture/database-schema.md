# Database Schema

**Database: Not Applicable**

This project operates entirely in-memory with no persistent data storage requirements. The self-referential design means:

- **Source Code** serves as the "database" - the script reads its own source for token data
- **All state is transient** - each animation frame is computed fresh from mathematical functions
- **No data persistence needed** - the artistic visualization is ephemeral and continuous
- **Zero external dependencies** - no database drivers, files, or storage systems

**Data Flow Architecture:**
```
Source Code (Static) → Tokens (Parsed Once) → 3D Points (Computed Per Frame) → ASCII Display (Ephemeral)
```

The architectural decision to avoid databases aligns perfectly with the viral sharing goals - users can run the script immediately without any setup, configuration, or data initialization steps.
