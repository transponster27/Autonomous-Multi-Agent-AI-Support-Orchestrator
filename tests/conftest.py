# # Run all tests
# uv run pytest

# # Run with verbose output
# uv run pytest -v

# # Run a specific test file
# uv run pytest tests/unit/test_cleaner.py

# # Run a specific test function
# uv run pytest tests/unit/test_cleaner.py::TestTextCleaner::test_empty_string_returns_empty

# # Run tests matching a pattern
# uv run pytest -k "cleaner"

# # Stop on first failure
# uv run pytest -x

# # Show print statements in output
# uv run pytest -s

# # Run with coverage
# uv run pytest --cov=src --cov-report=term-missing

# # Generate HTML coverage report
# uv run pytest --cov=src --cov-report=html
# # Then open htmlcov/index.html in browser