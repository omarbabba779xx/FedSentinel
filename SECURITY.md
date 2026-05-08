# Security Policy

## Supported Versions

FedSentinel is a **research prototype**. Only the latest commit on `main` receives security attention.

| Version | Supported |
|---------|-----------|
| latest (`main`) | ✅ |
| older commits   | ❌ |

---

## Scope

### In Scope

Security issues in FedSentinel code that could affect users who deploy or extend this framework:

- Cryptographic implementation bugs (e.g., broken commitment scheme, incorrect HMAC usage)
- Differential privacy accounting errors (privacy budget underestimated)
- Dependency vulnerabilities that affect the core attack/defense modules
- Authentication/authorization issues in the FastAPI REST API
- Unsafe deserialization or code execution in model loading

### Out of Scope

- Attacks that require physical access to the machine running FedSentinel
- Vulnerabilities in third-party libraries (report to the library maintainer directly; note them here if they affect FedSentinel users)
- Theoretical attacks against the FL protocol design (use GitHub Discussions for research conversations)
- Issues in the intentional attack simulation modules (`attacks/`) — these are designed to be adversarial

---

## Reporting a Vulnerability

**Do NOT open a public GitHub issue for security vulnerabilities.**

Please report privately via email:

**omarbabba27@gmail.com**

Include in your report:
1. Description of the vulnerability
2. Affected file(s) and line numbers
3. Steps to reproduce (or proof-of-concept code)
4. Potential impact
5. Suggested fix (optional but appreciated)

### Response Timeline

| Stage | Target |
|-------|--------|
| Acknowledgement | Within 72 hours |
| Initial assessment | Within 7 days |
| Patch / fix | Within 30 days for critical issues |
| Public disclosure | After patch is released (coordinated disclosure) |

---

## Cryptographic Disclaimer

FedSentinel implements several cryptographic constructs at **research scale**:

| Component | What it IS | What it is NOT |
|-----------|-----------|----------------|
| `crypto/zkp.py` | SHA-256 + HMAC Sigma-protocol (commit-challenge-respond) | A formal zero-knowledge proof (no zk-SNARK/STARK circuit) |
| `blockchain/audit_chain.py` | SHA-256 hash-linked append-only log | A decentralized blockchain (no consensus, no distributed nodes) |
| `privacy/differential_privacy.py` | DP-SGD with RDP+GDP accounting | A certified compliance tool |
| `server/mtls_config.py` | mTLS with self-signed certificates | A PKI with trusted CA certificates |

These are appropriate for **research and prototyping**. Production deployments require formal security review and likely replacement of these components with certified libraries.

---

## Dependency Security

To audit dependencies for known CVEs:

```bash
pip install pip-audit
pip-audit -r requirements.txt
```

Or with Safety:

```bash
pip install safety
safety check -r requirements.txt
```

---

## Acknowledgements

We thank the security research community for responsible disclosure. Reporters of valid vulnerabilities will be credited in the release notes (unless they prefer anonymity).
