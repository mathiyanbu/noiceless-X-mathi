import sys
import pytest

def test_python_version():
    """Verify that Python version is 3.11 or greater."""
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version_info}"

def test_sanity():
    """Basic sanity test to verify pytest framework execution."""
    val_a = 10
    val_b = 20
    assert val_a + val_b == 30
