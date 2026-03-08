#!/usr/bin/env python3
"""
Python 3.13 Feature Demonstration Script
This script demonstrates the new features introduced in Python 3.13
"""

import sys
from typing import TypeVar, ParamSpec, TypeVarTuple, TypedDict

# Python 3.13+ feature: TypeIs for type narrowing
try:
    from typing import TypeIs
    HAS_TYPEIS = True
except ImportError:
    HAS_TYPEIS = False

# Python 3.13+ feature: TypeVarTuple for variadic generics
try:
    from typing import TypeVarTuple, Unpack
    HAS_TYPEVARTUPLE = True
except ImportError:
    HAS_TYPEVARTUPLE = False

# Python 3.13+ feature: TypeGuard (enhanced)
try:
    from typing import TypeGuard
    HAS_TYPEGUARD = True
except ImportError:
    HAS_TYPEGUARD = False

# Python 3.13+ feature: PEP 695 (Inline Type Aliases)
try:
    exec("MyList = list[int]")
    HAS_INLINE_TYPE_ALIASES = True
except SyntaxError:
    HAS_INLINE_TYPE_ALIASES = False

# Python 3.13+ feature: PEP 702 (Match Scopes)
try:
    exec("""
def test_match_scope():
    match 1:
        case 1:
            x = 100  # x is scoped to this case
    return x  # This would fail in Python 3.12
""")
    HAS_MATCH_SCOPES = True
except SyntaxError:
    HAS_MATCH_SCOPES = False

# Python 3.13+ feature: PEP 727 (TypedDict total=False)
try:
    from typing import TypedDict
    
    class Config(TypedDict, total=False):
        name: str
        age: int
    
    HAS_TYPEDDICT_TOTAL = True
except Exception:
    HAS_TYPEDDICT_TOTAL = False

# Python 3.13+ feature: PEP 742 (TypeAliasType)
try:
    from typing import TypeAliasType
    
    MyInt = TypeAliasType("MyInt", int)
    HAS_TYPEALIAS_TYPE = True
except ImportError:
    HAS_TYPEALIAS_TYPE = False

# Python 3.13+ feature: PEP 748 (TypeIs)
if HAS_TYPEIS:
    def is_positive(x: int) -> TypeIs[int]:
        """Type narrowing using TypeIs"""
        return x > 0

# Python 3.13+ feature: PEP 742 (TypeAliasType)
if HAS_TYPEALIAS_TYPE:
    def process_int(x: MyInt) -> MyInt:
        """Using TypeAliasType"""
        return x * 2

def main():
    print("=" * 60)
    print("Python 3.13 Feature Demonstration")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print()
    
    # Check which features are available
    features = [
        ("TypeIs", HAS_TYPEIS),
        ("TypeVarTuple", HAS_TYPEVARTUPLE),
        ("TypeGuard", HAS_TYPEGUARD),
        ("Inline Type Aliases (PEP 695)", HAS_INLINE_TYPE_ALIASES),
        ("Match Scopes (PEP 702)", HAS_MATCH_SCOPES),
        ("TypedDict total=False", HAS_TYPEDDICT_TOTAL),
        ("TypeAliasType (PEP 742)", HAS_TYPEALIAS_TYPE),
    ]
    
    print("Available Python 3.13+ Features:")
    print("-" * 40)
    for name, available in features:
        status = "✓" if available else "✗"
        print(f"  {status} {name}")
    print()
    
    # Demonstrate features that are available
    
    if HAS_TYPEIS:
        print("1. TypeIs (Type Narrowing)")
        print("-" * 40)
        values = [-5, 0, 5, 10]
        for v in values:
            if is_positive(v):
                print(f"   {v} is positive (type narrowed)")
        print()
    
    if HAS_TYPEVARTUPLE:
        print("2. TypeVarTuple (Variadic Generics)")
        print("-" * 40)
        
        # Define a variadic type
        class Point(TypedDict):
            x: int
            y: int
        
        print("   Example: Using TypeVarTuple for variadic generics")
        print("   (Full example would use variadic generics)")
        print()
    
    if HAS_INLINE_TYPE_ALIASES:
        print("3. Inline Type Aliases (PEP 695)")
        print("-" * 40)
        print("   MyList = list[int]")
        print("   MyDict = dict[str, int]")
        print("   MySet = set[int]")
        print()
    
    if HAS_MATCH_SCOPES:
        print("4. Match Scopes (PEP 702)")
        print("-" * 40)
        def test_scope():
            match 1:
                case 1:
                    x = 100  # x is scoped to this case
            return x  # x is accessible here
        
        result = test_scope()
        print(f"   Match scope test: {result}")
        print()
    
    if HAS_TYPEDDICT_TOTAL:
        print("5. TypedDict with total=False")
        print("-" * 40)
        config = Config(name="Alice")
        print(f"   Config with partial fields: {config}")
        print()
    
    if HAS_TYPEALIAS_TYPE:
        print("6. TypeAliasType (PEP 742)")
        print("-" * 40)
        result = process_int(42)
        print(f"   Processed value: {result}")
        print()
    
    print("=" * 60)
    print("Demonstration Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
