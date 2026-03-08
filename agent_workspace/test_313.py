#!/usr/bin/env python3
"""
Python 3.13 Features Demo Script

This script demonstrates the new features introduced in Python 3.13.
Note: Some features require Python 3.13+ and will be skipped on older versions.
"""

import sys
import warnings
import copy
import base64
import argparse
import json

# Check Python version
PYTHON_VERSION = sys.version_info
print(f"Python Version: {PYTHON_VERSION.major}.{PYTHON_VERSION.minor}.{PYTHON_VERSION.micro}")

# =============================================================================
# Type Imports with Version Checks
# =============================================================================

# TypeIs and ReadOnly are only available in Python 3.13+
try:
    from typing import TypeVar, ParamSpec, TypeVarTuple, TypedDict, TypeIs, ReadOnly
    HAS_TYPEIS = True
except ImportError:
    HAS_TYPEIS = False
    from typing import TypeVar, ParamSpec, TypeVarTuple, TypedDict

# Type parameter defaults (PEP 696) - Python 3.13+
try:
    from typing import TypeVar
    HAS_TYPEVAR_DEFAULT = True
except:
    HAS_TYPEVAR_DEFAULT = False

# =============================================================================
# 1. Type Parameter Defaults (PEP 696) - Python 3.13+
# =============================================================================

def demonstrate_type_parameter_defaults():
    """Demonstrate type parameters with default values."""
    print("\n" + "="*60)
    print("1. Type Parameter Defaults (PEP 696)")
    print("="*60)
    
    if not HAS_TYPEVAR_DEFAULT:
        print("  SKIPPED: Type parameter defaults require Python 3.13+")
        print("  This feature allows TypeVar to have default values.")
        print("  Example: T = TypeVar('T', default=int)")
        return
    
    # TypeVar with default
    T = TypeVar('T', default=int)
    
    # Generic function with default type parameter
    def identity(x: T) -> T:
        return x
    
    # Can call without specifying type parameter
    result = identity(42)
    print(f"  identity(42) = {result} (type: {type(result).__name__})")
    
    # Can still specify explicitly
    result_str = identity[str]("hello")
    print(f"  identity[str]('hello') = {result_str}")

# =============================================================================
# 2. Deprecated Decorator (PEP 614) - Python 3.13+
# =============================================================================

def demonstrate_deprecated_decorator():
    """Demonstrate the @deprecated decorator."""
    print("\n" + "="*60)
    print("2. Deprecated Decorator (PEP 614)")
    print("="*60)
    
    if PYTHON_VERSION < (3, 13):
        print("  SKIPPED: @deprecated decorator requires Python 3.13+")
        print("  This decorator marks functions as deprecated.")
        print("  Example: @deprecated('Use new_function instead')")
        return
    
    from typing import deprecated
    
    @deprecated("Use new_function instead")
    def old_function(x: int) -> int:
        return x * 2
    
    print("  old_function(5) = ", end="")
    try:
        result = old_function(5)
        print(result)
    except Exception as e:
        print(f"Error: {e}")
    
    def new_function(x: int) -> int:
        return x * 2
    
    print("  new_function(5) = ", end="")
    try:
        result = new_function(5)
        print(result)
    except Exception as e:
        print(f"Error: {e}")

# =============================================================================
# 3. ReadOnly Type Annotation (PEP 705) - Python 3.13+
# =============================================================================

def demonstrate_readonly():
    """Demonstrate the ReadOnly type annotation."""
    print("\n" + "="*60)
    print("3. ReadOnly Type Annotation (PEP 705)")
    print("="*60)
    
    if not HAS_TYPEIS:
        print("  SKIPPED: ReadOnly requires Python 3.13+")
        print("  This annotation marks fields as read-only.")
        print("  Example: data: dict[str, ReadOnly[int]]")
        return
    
    from typing import ReadOnly
    
    class Config:
        # ReadOnly field - cannot be reassigned after initialization
        max_size: ReadOnly[int] = 100
    
    config = Config()
    print(f"  Config.max_size = {config.max_size}")
    
    # This would raise an error in a real implementation
    # config.max_size = 200  # Type error: cannot assign to ReadOnly

# =============================================================================
# 4. TypeIs (PEP 742) - Python 3.13+
# =============================================================================

def demonstrate_type_is():
    """Demonstrate TypeIs for type narrowing."""
    print("\n" + "="*60)
    print("4. TypeIs (PEP 742)")
    print("="*60)
    
    if not HAS_TYPEIS:
        print("  SKIPPED: TypeIs requires Python 3.13+")
        print("  This allows type narrowing based on runtime checks.")
        print("  Example: def is_positive(x: int | None) -> TypeIs[int]:")
        print("              if x is not None: return x")
        return
    
    from typing import TypeIs
    
    def is_positive(x: int | None) -> TypeIs[int]:
        """Return x if positive, else None."""
        if x is not None and x > 0:
            return x
        return None
    
    result = is_positive(42)
    print(f"  is_positive(42) = {result}")
    print(f"  Type of result: {type(result).__name__}")

# =============================================================================
# 5. copy.replace() (Python 3.13+)
# =============================================================================

def demonstrate_copy_replace():
    """Demonstrate the new copy.replace() function."""
    print("\n" + "="*60)
    print("5. copy.replace() (Python 3.13+)")
    print("="*60)
    
    if PYTHON_VERSION < (3, 13):
        print("  SKIPPED: copy.replace() requires Python 3.13+")
        print("  This creates a shallow copy with specified changes.")
        print("  Example: new_dict = copy.replace(old_dict, key='new_value')")
        return
    
    # Create a dictionary
    original = {"a": 1, "b": 2, "c": 3}
    print(f"  Original: {original}")
    
    # Use copy.replace() to create a modified copy
    modified = copy.replace(original, "b", 20)
    print(f"  Modified (b=20): {modified}")
    print(f"  Original unchanged: {original}")

# =============================================================================
# 6. base64.z85encode/decode (Python 3.13+)
# =============================================================================

def demonstrate_z85():
    """Demonstrate base64.z85encode/decode."""
    print("\n" + "="*60)
    print("6. base64.z85encode/decode (Python 3.13+)")
    print("="*60)
    
    if PYTHON_VERSION < (3, 13):
        print("  SKIPPED: base64.z85encode/decode requires Python 3.13+")
        print("  This provides Z85 encoding/decoding for base64.")
        print("  Example: base64.z85encode(b'hello')")
        return
    
    data = b"Hello, Python 3.13!"
    encoded = base64.z85encode(data)
    decoded = base64.z85decode(encoded)
    
    print(f"  Original: {data}")
    print(f"  Encoded: {encoded}")
    print(f"  Decoded: {decoded}")
    print(f"  Match: {data == decoded}")

# =============================================================================
# 7. argparse deprecation support (Python 3.13+)
# =============================================================================

def demonstrate_argparse_deprecation():
    """Demonstrate argparse's deprecation support."""
    print("\n" + "="*60)
    print("7. argparse deprecation support (Python 3.13+)")
    print("="*60)
    
    if PYTHON_VERSION < (3, 13):
        print("  SKIPPED: argparse deprecation support requires Python 3.13+")
        print("  This allows marking arguments as deprecated.")
        print("  Example: parser.add_argument('--old', deprecated=True)")
        return
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--new', help='New argument')
    parser.add_argument('--old', help='Deprecated argument', deprecated=True)
    
    args = parser.parse_args([])
    print(f"  Parsed arguments: {args}")
    print("  The 'deprecated' attribute can be checked on arguments")

# =============================================================================
# 8. Improved error messages (Python 3.13+)
# =============================================================================

def demonstrate_improved_errors():
    """Demonstrate improved error messages in Python 3.13."""
    print("\n" + "="*60)
    print("8. Improved error messages (Python 3.13+)")
    print("="*60)
    
    if PYTHON_VERSION < (3, 13):
        print("  SKIPPED: Improved error messages require Python 3.13+")
        print("  Python 3.13 provides better error messages for:")
        print("  - Type errors with more context")
        print("  - Syntax errors with better location info")
        print("  - Import errors with clearer paths")
        return
    
    # Example: Better type error messages
    try:
        x: int = "not an int"
    except TypeError as e:
        print(f"  Type error example: {e}")
    
    # Example: Better syntax error messages
    try:
        exec("x = 1\ny = 2\nz = 3\n")
    except SyntaxError as e:
        print(f"  Syntax error example: {e}")

# =============================================================================
# 9. locals() mutation semantics (PEP 667) - Python 3.13+
# =============================================================================

def demonstrate_locals_mutation():
    """Demonstrate improved locals() mutation semantics."""
    print("\n" + "="*60)
    print("9. locals() mutation semantics (PEP 667)")
    print("="*60)
    
    if PYTHON_VERSION < (3, 13):
        print("  SKIPPED: locals() mutation improvements require Python 3.13+")
        print("  PEP 667 clarifies that modifying locals() in a function")
        print("  does NOT affect the local variables.")
        return
    
    def test_locals():
        x = 1
        locals()['x'] = 2
        return x
    
    result = test_locals()
    print(f"  test_locals() = {result}")
    print("  Modifying locals() does NOT affect local variables")

# =============================================================================
# 10. Free-threaded interpreter (Python 3.13+)
# =============================================================================

def demonstrate_free_threaded():
    """Demonstrate the free-threaded interpreter option."""
    print("\n" + "="*60)
    print("10. Free-threaded interpreter (Python 3.13+)")
    print("="*60)
    
    if PYTHON_VERSION < (3, 13):
        print("  SKIPPED: Free-threaded interpreter requires Python 3.13+")
        print("  Use: python3.13 -X faulthandler your_script.py")
        print("  This removes the GIL for better parallelism.")
        return
    
    print(f"  Python version: {sys.version_info}")
    print("  Python 3.13 introduces a free-threaded interpreter option")
    print("  Use: python3.13 -X faulthandler your_script.py")
    print("  This removes the GIL for better parallelism")

# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "#"*60)
    print("# Python 3.13 Features Demo")
    print("#"*60)
    
    demonstrate_type_parameter_defaults()
    demonstrate_deprecated_decorator()
    demonstrate_readonly()
    demonstrate_type_is()
    demonstrate_copy_replace()
    demonstrate_z85()
    demonstrate_argparse_deprecation()
    demonstrate_improved_errors()
    demonstrate_locals_mutation()
    demonstrate_free_threaded()
    
    print("\n" + "#"*60)
    print("# Demo Complete!")
    print("#"*60 + "\n")

if __name__ == "__main__":
    main()
