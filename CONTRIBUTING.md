# Contributing to Project Chocolate

<!-- TODO: Expand contribution guidelines as project matures -->

## Development Setup

1. Fork and clone the repository
2. Create a conda environment: `conda env create -f environment.yml`
3. Install in editable mode: `pip install -e .`
4. Run tests to verify: `pytest tests/`

## Code Style

<!-- TODO: Define code style guidelines -->

**Recommended:**
- Follow PEP 8 for Python code
- Use type hints for function signatures
- Write docstrings for public functions (Google-style preferred)
- Keep functions focused and modular

## Testing

- Write unit tests for new features
- Run `pytest tests/` before submitting changes
- Aim for meaningful test coverage of core functionality

## Logging

<!-- TODO: Standardize logging configuration -->

- Use Python's `logging` module (avoid `print` statements)
- Log levels: DEBUG for development, INFO for key events, WARNING/ERROR for issues

## Docstring Style

<!-- TODO: Standardize on docstring format -->

**Preferred format: Google-style**

```python
def example_function(param1: int, param2: str) -> bool:
    """Brief description of function.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When invalid input is provided
    """
    pass
```

## Branching Strategy

<!-- TODO: Define branching strategy if accepting contributions -->

- Main branch: `main` (or `refactor/public-release` for current work)
- Feature branches: `feature/description`
- Bug fixes: `fix/description`

## Pull Request Process

<!-- TODO: Define PR process -->

1. Ensure all tests pass
2. Update documentation if needed
3. Add entry to experiment log if relevant
4. Request review from maintainers

## Questions?

<!-- TODO: Add contact information or discussion forum link -->

For questions or discussions, please open an issue on GitHub.
