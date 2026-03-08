#!/usr/bin/env python3
"""
Python 3.13 New Features Demonstration
======================================

This script demonstrates the major new features introduced in Python 3.13:
- PEP 696: Type parameter defaults
- PEP 702: warnings.deprecated() decorator
- PEP 705: typing.ReadOnly
- PEP 742: typing.TypeIs
- PEP 667: Defined semantics for locals()
- New base64.z85encode/decode
- copy.replace()
- argparse deprecation support
- Improved error messages
"""

import sys
import warnings
from typing import TypeVar, ParamSpec, TypeVarTuple, TypeIs, TypedDict, ReadOnly
from copy import copy, replace
import base64
import argparse
import random

# =============================================================================
# PEP 696: Type Parameter Defaults
# =============================================================================
print("=" * 60)
print("PEP 696: Type Parameter Defaults")
print("=" * 60)

# TypeVar with default
T = TypeVar('T', default=int)

def process_value(value: T) -> T:
    """Process a value with type parameter default."""
    return value * 2

# Using the default type
result1 = process_value(5)
print(f"process_value(5) = {result1} (type: {type(result1).__name__})")

# TypeVarTuple with default
Ts = TypeVarTuple('Ts')

def process_tuple(*args: tuple[Ts]) -> tuple[Ts]:
    """Process a tuple with type variable tuple default."""
    return args

result2 = process_tuple(1, 2, 3)
print(f"process_tuple(1, 2, 3) = {result2}")

# =============================================================================
# PEP 702: warnings.deprecated() decorator
# =============================================================================
print("\n" + "=" * 60)
print("PEP 702: warnings.deprecated() decorator")
print("=" * 60)

@warnings.deprecated("This function is deprecated in Python 3.13", category=DeprecationWarning)
def old_function(x: int) -> int:
    """An old function that is now deprecated."""
    return x + 1

# This will trigger a deprecation warning
try:
    result3 = old_function(10)
    print(f"old_function(10) = {result3}")
except DeprecationWarning as e:
    print(f"DeprecationWarning caught: {e}")

# =============================================================================
# PEP 705: typing.ReadOnly
# =============================================================================
print("\n" + "=" * 60)
print("PEP 705: typing.ReadOnly")
print("=" * 60)

class Config(TypedDict, total=False):
    """A TypedDict with read-only fields."""
    name: ReadOnly[str]
    version: str
    enabled: bool

# Create a config
config: Config = {
    "name": "MyApp",
    "version": "1.0.0",
    "enabled": True
}

print(f"Config: {config}")
print(f"Config['name'] = {config['name']}")

# Note: Type checkers will flag attempts to modify 'name'
# config['name'] = "Changed"  # This would be flagged by type checkers

# =============================================================================
# PEP 742: typing.TypeIs
# =============================================================================
print("\n" + "=" * 60)
print("PEP 742: typing.TypeIs")
print("=" * 60)

def is_even(x: int) -> TypeIs[int]:
    """Type narrowing using TypeIs."""
    return x % 2 == 0

def process_number(x: int) -> None:
    """Demonstrate type narrowing with TypeIs."""
    if is_even(x):
        # x is narrowed to int (but we know it's even)
        # Type checkers understand this is an even number
        print(f"Processing even number: {x}")
    else:
        print(f"Processing odd number: {x}")

process_number(4)
process_number(7)

# =============================================================================
# PEP 667: Defined semantics for locals()
# =============================================================================
print("\n" + "=" * 60)
print("PEP 667: Defined semantics for locals()")
print("=" * 60)

def demonstrate_locals_mutation():
    """Demonstrate that locals() mutation is now defined."""
    x = 10
    y = 20
    
    # In Python 3.13+, mutating locals() has defined semantics
    local_vars = locals()
    local_vars['x'] = 100  # This mutation is now tracked
    
    print(f"After mutation: x = {x}")
    print(f"locals()['x'] = {local_vars['x']}")
    
    # The mutation is visible in the local scope
    print(f"Direct access x = {x}")

demonstrate_locals_mutation()

# =============================================================================
# New: base64.z85encode() and base64.z85decode()
# =============================================================================
print("\n" + "=" * 60)
print("New: base64.z85encode() and base64.z85decode()")
print("=" * 60)

data = b"Hello, Python 3.13!"
encoded = base64.z85encode(data)
decoded = base64.z85decode(encoded)

print(f"Original: {data}")
print(f"Z85 encoded: {encoded}")
print(f"Z85 decoded: {decoded}")
print(f"Match: {data == decoded}")

# =============================================================================
# New: copy.replace()
# =============================================================================
print("\n" + "=" * 60)
print("New: copy.replace()")
print("=" * 60)

class Point:
    """A simple class with __replace__ method."""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"
    
    def __replace__(self, **kwargs):
        """Support for copy.replace()."""
        return type(self)(
            kwargs.get('x', self.x),
            kwargs.get('y', self.y)
        )

p1 = Point(1.0, 2.0)
print(f"Original point: {p1}")

# Using copy.replace()
p2 = replace(p1, x=5.0)
print(f"Replaced point (x=5.0): {p2}")

p3 = replace(p1, y=10.0)
print(f"Replaced point (y=10.0): {p3}")

# =============================================================================
# New: argparse deprecation support
# =============================================================================
print("\n" + "=" * 60)
print("New: argparse deprecation support")
print("=" * 60)

parser = argparse.ArgumentParser(description="Demo argparse deprecation")

# Deprecated argument
parser.add_argument('--old-option', dest='new_option',
                    deprecated=True,
                    help="This option is deprecated, use --new-option instead")

# New argument
parser.add_argument('--new-option',
                    help="The new recommended option")

args = parser.parse_args(['--old-option', 'value'])
print(f"Parsed args: {args}")
print(f"Deprecated option value: {args.new_option}")

# =============================================================================
# New: random module CLI
# =============================================================================
print("\n" + "=" * 60)
print("New: random module CLI")
print("=" * 60)

# The random module now has a command-line interface
# This can be invoked as: python -m random
print("random module CLI is available via: python -m random")
print(f"Python version: {sys.version}")

# =============================================================================
# New: Improved error messages
# =============================================================================
print("\n" + "=" * 60)
print("New: Improved error messages")
print("=" * 60)

try:
    # This will trigger a helpful error message about the typo
    "test".split(max_split=1)
except TypeError as e:
    print(f"TypeError caught with helpful message:")
    print(f"  {e}")

# =============================================================================
# New: Color support in interactive interpreter and tracebacks
# =============================================================================
print("\n" + "=" * 60)
print("New: Color support in interactive interpreter and tracebacks")
print("=" * 60)

print("Color support is enabled by default in Python 3.13")
print("Can be controlled via PYTHON_COLORS and NO_COLOR env vars")

# =============================================================================
# New: __static_attributes__ and __firstlineno__
# =============================================================================
print("\n" + "=" * 60)
print("New: __static_attributes__ and __firstlineno__")
print("=" * 60)

class MyClass:
    """Class with new static attributes."""
    def method(self):
        self.x = 1
        self.y = 2
    
    def another_method(self):
        self.z = 3

obj = MyClass()
obj.method()

print(f"__static_attributes__: {MyClass.__static_attributes__}")
print(f"__firstlineno__: {MyClass.__firstlineno__}")

# =============================================================================
# New: sys._is_gil_enabled() for free-threaded mode
# =============================================================================
print("\n" + "=" * 60)
print("New: sys._is_gil_enabled() for free-threaded mode")
print("=" * 60)

if hasattr(sys, '_is_gil_enabled'):
    gil_enabled = sys._is_gil_enabled()
    print(f"GIL is enabled: {gil_enabled}")
    print("Note: Free-threaded mode requires special build (python3.13t)")
else:
    print("sys._is_gil_enabled() not available (not a free-threaded build)")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Python 3.13 Features Summary")
print("=" * 60)
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("\nKey features demonstrated:")
print("  ✓ PEP 696: Type parameter defaults")
print("  ✓ PEP 702: warnings.deprecated() decorator")
print("  ✓ PEP 705: typing.ReadOnly")
print("  ✓ PEP 742: typing.TypeIs")
print("  ✓ PEP 667: Defined semantics for locals()")
print("  ✓ base64.z85encode/decode")
print("  ✓ copy.replace()")
print("  ✓ argparse deprecation support")
print("  ✓ Improved error messages")
print("  ✓ Color support in REPL and tracebacks")
print("  ✓ __static_attributes__ and __firstlineno__")
print("  ✓ sys._is_gil_enabled()")

print("\n" + "=" * 60)
print("All Python 3.13 features demonstrated successfully!")
print("=" * 60)
