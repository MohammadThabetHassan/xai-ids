# Contributing to XAI-IDS

Thank you for your interest in contributing to the XAI-IDS project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Bugs

- Check existing [issues](https://github.com/MohammadThabetHassan/xai-ids/issues) first
- Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
- Include: Python version, OS, steps to reproduce, expected vs actual behavior

### Suggesting Features

- Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md)
- Explain the use case and why it would benefit the project

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests: `python -m pytest tests/ -v`
5. Commit with a descriptive message
6. Push to your fork and submit a PR

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/xai-ids.git
cd xai-ids

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v
```

## Code Style

- Follow PEP 8 conventions
- Use NumPy-style docstrings for all functions
- Add type hints to function signatures
- Keep functions focused and under 50 lines when possible

## Testing

- All PRs must pass existing tests
- Add tests for new functionality
- Run: `python -m pytest tests/ -v`

## Commit Messages

Use conventional commit format:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation changes
- `refactor:` code refactoring
- `test:` test additions/changes

## Questions?

Open an issue or contact the maintainers.
