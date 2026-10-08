"""Pytest configuration for NFL2027 test suite."""

import sys
import os

# Pre-import standard library code module from Python stdlib to prevent local code.py from shadowing it in pdb/torch
stdlib_lib = [p for p in sys.path if ('Lib' in p or 'lib' in p) and 'site-packages' not in p]
if stdlib_lib:
    sys.path.insert(0, stdlib_lib[0])
    import code
    sys.path.pop(0)
else:
    import code

# Ensure package directory is always on sys.path
package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if package_root not in sys.path:
    sys.path.insert(0, package_root)
