"""Test module for core functionality verification."""
import pytest


class TestCoreFunctionality:
    """Basic tests to verify the testing framework is working."""

    def test_example_assertion(self):
        """Test that basic assertions work."""
        assert True

    def test_example_math(self):
        """Test basic mathematical operations."""
        assert 1 + 1 == 2
        assert 2 * 3 == 6
        assert 10 / 2 == 5

    def test_example_string_operations(self):
        """Test basic string operations."""
        s = "hello"
        assert len(s) == 5
        assert s.upper() == "HELLO"
        assert s.capitalize() == "Hello"


def test_simple_function():
    """Test a simple standalone function."""
    def add(a, b):
        return a + b
    
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0


@pytest.mark.parametrize("input_val,expected", [
    (1, 1),
    (2, 4),
    (3, 9),
    (4, 16),
])
def test_parametrized_square(input_val, expected):
    """Test parametrized test cases."""
    assert input_val ** 2 == expected
