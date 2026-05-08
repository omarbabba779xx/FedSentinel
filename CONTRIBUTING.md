# Contributing to FedSentinel

Thank you for your interest in contributing! FedSentinel is a research-grade federated learning IDS framework. Contributions are welcome — bug fixes, new features, better tests, and documentation improvements.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Branch & PR Workflow](#branch--pr-workflow)
- [Code Style](#code-style)
- [Writing Tests](#writing-tests)
- [Commit Messages](#commit-messages)
- [Reporting Issues](#reporting-issues)

---

## Code of Conduct

Be respectful and constructive. This is a research project — disagreements about methodology are welcome; personal attacks are not.

---

## Getting Started

```bash
# Fork + clone
git clone https://github.com/<your-username>/FedSentinel.git
cd FedSentinel

# Virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install in editable mode + dev deps
pip install -r requirements.txt
pip install pytest pytest-cov flake8 bandit
```

---

## How to Contribute

### Bug Fixes
1. Open an issue describing the bug (include stack trace + Python version).
2. Fork, fix, add a regression test.
3. Open a PR referencing the issue: `Fixes #<issue>`.

### New Features / Modules
1. Open a Discussion or issue to describe the proposed feature **before** writing code.
2. Keep new modules self-contained in their own directory (e.g., `new_module/`).
3. Add unit tests under `tests/test_<module>.py`.
4. Update `README.md` — add a row to the relevant feature table.

### Research Extensions
Extensions that add new FL algorithms, attack types, or privacy mechanisms are especially welcome. Please:
- Cite the paper your implementation is based on in the module docstring.
- Be precise about what the implementation **does** and **does not** provide (see `crypto/zkp.py` and `blockchain/audit_chain.py` for examples of honest capability statements).

---

## Branch & PR Workflow

```
main          ← stable release branch
develop       ← integration branch for new features
feature/<name> ← your feature branch (branch from develop)
fix/<name>    ← your bugfix branch (branch from main or develop)
```

1. Branch from `develop` for features, `main` for hotfixes.
2. Keep PRs focused — one feature / fix per PR.
3. All CI checks must pass before merge (lint + tests).
4. Request review from at least one maintainer.

---

## Code Style

- **PEP 8**, max line length **120 characters**.
- Type hints on all public functions.
- Docstrings on all public classes and functions (Google style preferred).
- No `print()` in library code — use `utils/logger.py` (`get_logger(__name__)`).

```bash
# Check style before pushing
flake8 . --max-line-length=120 --extend-ignore=E203,W503 --exclude=venv,build
```

---

## Writing Tests

Tests live in `tests/`. Each module should have a corresponding `test_<module>.py`.

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=term-missing
```

Guidelines:
- Use `unittest.TestCase` (consistent with existing tests).
- Mock heavy dependencies (TenSEAL, torch_geometric) with `unittest.mock` so tests run in CI without optional deps.
- Test both happy path and failure/edge cases.
- Keep each test fast (< 5s); mark slow tests with `@pytest.mark.slow`.

---

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add FedAsync client with staleness decay
fix: correct RDP composition for Poisson subsampling
test: add unit tests for GradientCommitmentScheme
docs: update README with NSL-KDD dataset caveat
refactor: extract privacy accountant into separate class
```

---

## Reporting Issues

When reporting a bug, please include:
- Python version (`python --version`)
- OS + version
- Full stack trace
- Minimal reproducible example (MRE)

For **security vulnerabilities**, do **not** open a public issue — see [SECURITY.md](SECURITY.md).
