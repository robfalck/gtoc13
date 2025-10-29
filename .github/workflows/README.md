# GitHub Actions Workflows

## test.yml - Automated Testing

This workflow runs the unittest suite on every push and pull request to ensure code quality and compatibility.

### Test Matrix

The workflow tests against multiple configurations:

- **Python versions**: 3.10, 3.11, 3.12
- **Dependency versions**:
  - **pinned**: Uses versions specified in `pyproject.toml` (tested on all Python versions)
  - **latest**: Uses latest available versions from PyPI (tested only on Python 3.12)

### What it does

1. **Checkout code**: Gets the latest code from the repository
2. **Setup Python**: Installs the specified Python version with pip caching
3. **Install dependencies**:
   - For **pinned** versions: Installs from `pyproject.toml` with `pip install -e ".[dev]"`
   - For **latest** versions: Installs latest packages with `pip install --upgrade`
4. **Display versions**: Shows all installed package versions for debugging
5. **Run tests**: Executes `pytest gtoc13/ -v --tb=short`
6. **Check for updates**: If latest deps fail, warns about updating `pyproject.toml`

### When it runs

- On push to `main`, `develop`, or `testing` branches
- On pull requests to `main` or `develop`
- Manually via workflow_dispatch

### Why test with latest dependencies?

The "latest" dependency test helps you:
- Detect breaking changes in dependencies early
- Know when `pyproject.toml` version constraints need updating
- Ensure forward compatibility with newer package versions

If tests fail with latest dependencies but pass with pinned versions, it indicates that:
1. A dependency has introduced a breaking change
2. You may need to update your code or pin to an older version in `pyproject.toml`

### Badge

The workflow status badge in README.md shows the current test status:

```markdown
[![Tests](https://github.com/robfalck/gtoc13/actions/workflows/test.yml/badge.svg)](https://github.com/robfalck/gtoc13/actions/workflows/test.yml)
```
