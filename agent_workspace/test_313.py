#!/usr/bin/env python3
"""
test_313.py - Demonstrates Python 3.13 New Features

This script showcases the major new features introduced in Python 3.13:
- Free-threaded mode support (PEP 703)
- JIT compiler support (PEP 744)
- Defined semantics for locals() (PEP 667)
- copy.replace() function
- base64.z85encode/decode
- typing.ReadOnly and typing.TypeIs
- warnings.deprecated() decorator
- Type parameters with defaults (PEP 696)
- New data model attributes (__static_attributes__, __firstlineno__)
"""

import sys
import copy
import base64
import warnings
from typing import TypeVar, TypeVarTuple, ParamSpec, TypeIs, ReadOnly, TypedDict


def demonstrate_locals_semantics():
    """
    Demonstrate PEP 667: Defined semantics for locals() mutation.
    In Python 3.13, mutating the returned mapping from locals() has defined behavior.
    """
    print("=" * 60)
    print("1. PEP 667: Defined semantics for locals()")
    print("=" * 60)
    
    def example_function():
        x = 10
        y = 20
        
        # In Python 3.13+, we can mutate the returned mapping
        # and it will affect local variables in non-optimized scopes
        locs = locals()
        locs['x'] = 100  # This mutation is now well-defined
        
        return x, y
    
    result = example_function()
    print(f"After mutating locals()['x'] = 100:")
    print(f"  Returned values: x={result[0]}, y={result[1]}")
    print(f"  Note: In optimized scopes, locals() returns a snapshot")
    print()


def demonstrate_copy_replace():
    """
    Demonstrate copy.replace() - new function in Python 3.13.
    """
    print("=" * 60)
    print("2. copy.replace() - New function in Python 3.13")
    print("=" * 60)
    
    class Point:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
        
        def __repr__(self):
            return f"Point(x={self.x}, y={self.y}, z={self.z})"
    
    original = Point(1, 2, 3)
    print(f"Original: {original}")
    
    # Using copy.replace() to create a modified copy
    modified = copy.replace(original, x=10, z=30)
    print(f"Modified (x=10, z=30): {modified}")
    print(f"Original unchanged: {original}")
    print()


def demonstrate_base64_z85():
    """
    Demonstrate base64.z85encode() and base64.z85decode() - new in 3.13.
    """
    print("=" * 60)
    print("3. base64.z85encode() and base64.z85decode() - New in 3.13")
    print("=" * 60)
    
    # Z85 is a variant of Base64 that uses 4 characters to encode 32 bits
    data = b"Hello, Python 3.13!"
    
    encoded = base64.z85encode(data)
    print(f"Original data: {data}")
    print(f"Z85 encoded: {encoded}")
    
    decoded = base64.z85decode(encoded)
    print(f"Z85 decoded: {decoded}")
    print(f"Round-trip successful: {data == decoded}")
    print()


def demonstrate_typing_features():
    """
    Demonstrate new typing features in Python 3.13.
    """
    print("=" * 60)
    print("4. New Typing Features in Python 3.13")
    print("=" * 60)
    
    # PEP 696: Type parameters with defaults
    T = TypeVar('T', default=int)
    U = TypeVarTuple('U', default=(int, str))
    P = ParamSpec('P', default=P.empty)
    
    def function_with_type_defaults(value: T) -> T:
        return value
    
    print(f"TypeVar with default (T=int): {function_with_type_defaults(42)}")
    print(f"TypeVar with default (T=str): {function_with_type_defaults('hello')}")
    
    # PEP 705: ReadOnly for TypedDict
    class Person(TypedDict, total=False):
        name: str
        age: ReadOnly[int]
        email: str
    
    person: Person = {"name": "Alice", "age": 30, "email": "alice@example.com"}
    print(f"\nTypedDict with ReadOnly field: {person}")
    
    # PEP 742: TypeIs for more precise type narrowing
    def is_positive(x: int) -> TypeIs[int]:
        """TypeIs provides more intuitive type narrowing."""
        return x > 0
    
    value: int = 5
    if is_positive(value):
        # Type checker knows value is positive here
        print(f"Positive value: {value}")
    
    print()


def demonstrate_warnings_deprecated():
    """
    Demonstrate warnings.deprecated() decorator - new in 3.13.
    """
    print("=" * 60)
    print("5. warnings.deprecated() - New decorator in Python 3.13")
    print("=" * 60)
    
    @warnings.deprecated("Use new_function() instead", category=DeprecationWarning)
    def old_function(x: int) -> int:
        return x * 2
    
    @warnings.deprecated("This is deprecated", category=FutureWarning)
    class OldClass:
        def __init__(self):
            self.value = 42
    
    print("Calling deprecated function:")
    try:
        result = old_function(21)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\nCalling deprecated class:")
    try:
        obj = OldClass()
        print(f"  Created: {obj}")
    except Exception as e:
        print(f"  Error: {e}")
    print()


def demonstrate_data_model_attributes():
    """
    Demonstrate new data model attributes in Python 3.13.
    """
    print("=" * 60)
    print("6. New Data Model Attributes in Python 3.13")
    print("=" * 60)
    
    class MyClass:
        def __init__(self):
            self.x = 10
            self.y = 20
        
        def method(self):
            return self.x + self.y
    
    obj = MyClass()
    
    # __firstlineno__ - records the first line number of a class definition
    print(f"Class first line number: {MyClass.__firstlineno__}")
    
    # __static_attributes__ - stores names of attributes accessed through self.X
    # in any function in a class body
    print(f"Static attributes: {MyClass.__static_attributes__}")
    
    # Accessing an attribute in a method adds it to __static_attributes__
    obj.method()
    print(f"After calling method(): {MyClass.__static_attributes__}")
    print()


def demonstrate_free_threaded_support():
    """
    Demonstrate free-threaded mode support (PEP 703).
    """
    print("=" * 60)
    print("7. Free-threaded Mode Support (PEP 703)")
    print("=" * 60)
    
    # Check if this is a free-threaded build
    if hasattr(sys, '_is_gil_enabled'):
        gil_enabled = sys._is_gil_enabled()
        print(f"GIL enabled: {gil_enabled}")
        print(f"Free-threaded build: {'experimental free-threading build' in sys.version}")
    else:
        print("sys._is_gil_enabled() not available (not a free-threaded build)")
    
    # Check for JIT support
    if hasattr(sys, '_is_jit_enabled'):
        jit_enabled = sys._is_jit_enabled()
        print(f"JIT enabled: {jit_enabled}")
    else:
        print("JIT support not available or not enabled")
    
    print()


def demonstrate_improved_error_messages():
    """
    Demonstrate improved error messages in Python 3.13.
    """
    print("=" * 60)
    print("8. Improved Error Messages in Python 3.13")
    print("=" * 60)
    
    # Example: Incorrect keyword argument suggestion
    try:
        "test".split(max_split=1)  # Should be maxsplit
    except TypeError as e:
        print(f"TypeError with suggestion: {e}")
    
    # Example: Module name conflict warning
    print("\nNote: Python 3.13 now provides helpful error messages")
    print("when a script has the same name as a standard library module.")
    print()


def demonstrate_argparse_deprecation():
    """
    Demonstrate argparse deprecation support (new in 3.13).
    """
    print("=" * 60)
    print("9. argparse Deprecation Support (New in 3.13)")
    print("=" * 60)
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # Deprecate a command-line option
    parser.add_argument(
        '--old-option',
        dest='new_option',
        deprecated=True,
        help='This option is deprecated, use --new-option instead'
    )
    
    parser.add_argument(
        '--new-option',
        help='The new option'
    )
    
    args = parser.parse_args([])
    print(f"Parsed args: {args}")
    print("Note: argparse now supports deprecating CLI options.")
    print()


def demonstrate_random_cli():
    """
    Demonstrate random module CLI (new in 3.13).
    """
    print("=" * 60)
    print("10. random Module CLI (New in 3.13)")
    print("=" * 60)
    
    import random
    
    # The random module now has a command-line interface
    # Usage: python -m random [options]
    print("The random module now supports command-line usage:")
    print("  python -m random [options]")
    print("  Options include: --seed, --version, --help")
    print()


def main():
    """Main function to demonstrate all Python 3.13 features."""
    print("\n" + "=" * 60)
    print("PYTHON 3.13 NEW FEATURES DEMONSTRATION")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print()
    
    # Run all demonstrations
    demonstrate_locals_semantics()
    demonstrate_copy_replace()
    demonstrate_base64_z85()
    demonstrate_typing_features()
    demonstrate_warnings_deprecated()
    demonstrate_data_model_attributes()
    demonstrate_free_threaded_support()
    demonstrate_improved_error_messages()
    demonstrate_argparse_deprecation()
    demonstrate_random_cli()
    
    print("=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nKey Python 3.13 Features Summary:")
    print("  • PEP 667: Defined semantics for locals() mutation")
    print("  • PEP 703: Free-threaded mode support (experimental)")
    print("  • PEP 744: JIT compiler (experimental)")
    print("  • copy.replace(): New function for creating modified copies")
    print("  • base64.z85encode/decode(): New encoding functions")
    print("  • typing.ReadOnly: Mark TypedDict items as read-only")
    print("  • typing.TypeIs: More intuitive type narrowing")
    print("  • warnings.deprecated(): New decorator for deprecations")
    print("  • Type parameters with defaults (PEP 696)")
    print("  • __static_attributes__ and __firstlineno__ attributes")
    print("  • Improved error messages and interactive interpreter")
    print("  • argparse deprecation support")
    print("  • random module CLI")
    print()


if __name__ == "__main__":
    main()
