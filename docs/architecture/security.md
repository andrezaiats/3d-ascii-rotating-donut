# Security

## Input Validation
- **Validation Library:** Built-in Python validation (no external dependencies)
- **Validation Location:** At mathematical function entry points and file reading operations
- **Required Rules:**
  - All torus parameters MUST be validated (outer_radius > inner_radius > 0)
  - File path validation when reading self-code to prevent path traversal
  - Mathematical input bounds checking to prevent overflow/underflow

## Authentication & Authorization
**Not Applicable** - Single-user local application with no network interfaces or user accounts

## Secrets Management
- **Development:** No secrets required (mathematical constants and file paths only)
- **Production:** No secrets required (self-contained execution)
- **Code Requirements:**
  - No hardcoded sensitive data (none exists in this project)
  - No external service credentials needed
  - Mathematical constants and file paths are public by design

## API Security
**Not Applicable** - No network APIs or external service interfaces

## Data Protection
- **Encryption at Rest:** Not applicable (no persistent data storage)
- **Encryption in Transit:** Not applicable (no network communication)
- **PII Handling:** No personal data collected or processed
- **Logging Restrictions:** No sensitive data to log (mathematical calculations and code tokens only)

## Dependency Security
- **Scanning Tool:** GitHub Dependabot (for test dependencies only)
- **Update Policy:** Python stdlib only - no external dependencies to manage
- **Approval Process:** No external dependencies allowed (architectural constraint)

## Security Testing
- **SAST Tool:** Built-in Python security checks via `bandit` (development only)
- **DAST Tool:** Not applicable (no web interfaces)
- **Penetration Testing:** Not required (no attack surface)
