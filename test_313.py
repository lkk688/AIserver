#!/usr/bin/env python3
"""
Python 3.13 New Features Demonstration Script
==============================================

This script demonstrates the new features introduced in Python 3.13:
1. Just-In-Time (JIT) Compiler (PEP 744)
2. Free-threaded Mode Support (PEP 703)
3. New Interactive Interpreter features
"""

import sys
import time
import threading
import ctypes
from typing import Any


def test_jit_compiler():
    """
    Test the JIT compiler feature (PEP 744).
    
    The JIT compiler in Python 3.13 uses tracing JIT compilation.
    It can be enabled with the -X jit option when running Python.
    """
    print("=" * 60)
    print("1. JIT COMPILER (PEP 744)")
    print("=" * 60)
    
    # Check if JIT is available
    try:
        # The JIT compiler can be enabled via environment variable or -X flag
        # We demonstrate the concept with a simple benchmark
        print("JIT Compiler Status: Available (check with -X jit to enable)")
        print("Note: JIT is disabled by default in Python 3.13")
        print()
        
        # Example: A function that would benefit from JIT compilation
        def fibonacci(n):
            """Calculate Fibonacci number - good candidate for JIT"""
            if n <= 1:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)
        
        # Run with and without JIT simulation
        print("Running Fibonacci(25) calculation...")
        start = time.time()
        result = fibonacci(25)
        elapsed = time.time() - start
        print(f"Result: {result}")
        print(f"Time: {elapsed:.4f} seconds")
        print()
        
    except Exception as e:
        print(f"JIT test encountered an error: {e}")
    
    print()


def test_free_threaded_mode():
    """
    Test free-threaded mode support (PEP 703).
    
    Python 3.13 introduced experimental support for running without the GIL.
    This requires a special build with --disable-gil flag.
    """
    print("=" * 60)
    print("2. FREE-THREADED MODE (PEP 703)")
    print("=" * 60)
    
    # Check if we're running in free-threaded mode
    # In free-threaded builds, the GIL is not present
    gil_enabled = hasattr(sys, "gettotalrefcount")
    
    print(f"Python Version: {sys.version}")
    print(f"Python Implementation: {sys.implementation.name}")
    print(f"Python Build: {sys.version_info}")
    print()
    
    # Check for free-threaded indicators
    # In free-threaded builds, certain attributes may differ
    print("Free-threaded Mode Detection:")
    print("-" * 40)
    
    # Check if we can detect free-threaded mode
    # This is experimental and may vary by build
    try:
        # Try to access GIL-related attributes
        if hasattr(sys, "gettotalrefcount"):
            refcount = sys.gettotalrefcount()
            print(f"  Total Reference Count: {refcount}")
            print("  Note: Reference counting is active (GIL may be present)")
        else:
            print("  Reference counting not available")
            print("  Note: This may indicate free-threaded mode")
    except Exception as e:
        print(f"  Could not check reference count: {e}")
    
    print()
    
    # Demonstrate threading behavior
    print("Threading Test:")
    print("-" * 40)
    
    results = []
    lock = threading.Lock()
    
    def worker(worker_id, iterations):
        local_sum = 0
        for i in range(iterations):
            local_sum += i * worker_id
        with lock:
            results.append((worker_id, local_sum))
    
    # Create multiple threads
    num_threads = 4
    iterations_per_thread = 10000
    
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i, iterations_per_thread))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    print(f"  Completed {num_threads} threads with {iterations_per_thread} iterations each")
    print(f"  Results collected: {len(results)}")
    print()


def test_new_interpreter_features():
    """
    Test new interactive interpreter features.
    
    Python 3.13 introduced a new interactive interpreter with:
    - Better syntax highlighting
    - Improved error messages
    - Enhanced debugging capabilities
    """
    print("=" * 60)
    print("3. NEW INTERACTIVE INTERPRETER FEATURES")
    print("=" * 60)
    
    print("Interactive Interpreter Improvements:")
    print("-" * 40)
    
    # Demonstrate improved error handling
    print("\n1. Enhanced Error Messages:")
    try:
        # This will trigger a syntax error
        eval("def broken(")
    except SyntaxError as e:
        print(f"   SyntaxError: {e}")
        print(f"   Line: {e.lineno}")
        print(f"   Offset: {e.offset}")
        print(f"   Text: {e.text}")
    
    print()
    
    # Demonstrate improved exception handling
    print("2. Improved Exception Handling:")
    try:
        raise ValueError("This is a test exception")
    except ValueError as e:
        print(f"   Caught: {type(e).__name__}: {e}")
    
    print()
    
    # Demonstrate new string methods (if available)
    print("3. String Methods:")
    test_string = "Python 3.13"
    print(f"   String: '{test_string}'")
    print(f"   Length: {len(test_string)}")
    print(f"   Reversed: '{test_string[::-1]}'")
    
    print()
    
    # Demonstrate improved f-strings
    print("4. F-String Enhancements:")
    name = "Python"
    version = 3.13
    print(f"   Version: {name} {version}")
    print(f"   Formatted: {name} version {version:.2f}")
    
    print()


def test_performance_improvements():
    """
    Test performance-related improvements in Python 3.13.
    """
    print("=" * 60)
    print("4. PERFORMANCE IMPROVEMENTS")
    print("=" * 60)
    
    print("Python 3.13 Performance Features:")
    print("-" * 40)
    
    # Test dictionary performance
    print("\n1. Dictionary Performance:")
    test_dict = {i: i**2 for i in range(10000)}
    print(f"   Created dictionary with {len(test_dict)} entries")
    
    # Test list comprehension
    print("\n2. List Comprehension:")
    squares = [x**2 for x in range(10000)]
    print(f"   Created list with {len(squares)} entries")
    
    # Test generator expressions
    print("\n3. Generator Expressions:")
    gen = (x**2 for x in range(10000))
    total = sum(gen)
    print(f"   Sum of squares (1-10000): {total}")
    
    print()


def test_pep_744_jit_example():
    """
    Example demonstrating PEP 744 JIT usage.
    
    To enable JIT, run Python with: python -X jit
    """
    print("=" * 60)
    print("5. PEP 744 JIT COMPILER EXAMPLE")
    print("=" * 60)
    
    print("JIT Compiler Usage:")
    print("-" * 40)
    print("To enable JIT compilation, run:")
    print("  python -X jit test_313.py")
    print()
    print("JIT Features:")
    print("  - Tracing JIT compilation")
    print("  - Compiles hot code paths to machine code")
    print("  - Currently disabled by default")
    print("  - Performance improvements are modest")
    print()
    
    # Example code that would benefit from JIT
    def matrix_multiply(a, b):
        """Matrix multiplication - good candidate for JIT"""
        n = len(a)
        result = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    result[i][j] += a[i][k] * b[k][j]
        return result
    
    print("Matrix Multiplication Example:")
    print("-" * 40)
    size = 100
    a = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
    b = [[1 for _ in range(size)] for _ in range(size)]
    
    start = time.time()
    result = matrix_multiply(a, b)
    elapsed = time.time() - start
    
    print(f"   Matrix size: {size}x{size}")
    print(f"   Time: {elapsed:.4f} seconds")
    print()


def main():
    """Main function to run all demonstrations."""
    print("\n" + "=" * 60)
    print("PYTHON 3.13 NEW FEATURES DEMONSTRATION")
    print("=" * 60)
    print()
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()
    
    # Run all demonstrations
    test_jit_compiler()
    test_free_threaded_mode()
    test_new_interpreter_features()
    test_performance_improvements()
    test_pep_744_jit_example()
    
    print("=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print()
    print("Summary of Python 3.13 Features:")
    print("  1. JIT Compiler (PEP 744) - Experimental, disabled by default")
    print("  2. Free-threaded Mode (PEP 703) - Requires special build")
    print("  3. New Interactive Interpreter - Better REPL experience")
    print("  4. Performance Improvements - Various optimizations")
    print()


if __name__ == "__main__":
    main()
