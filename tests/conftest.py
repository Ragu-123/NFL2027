"""Pytest configuration for NFL2027 test suite."""

import sys
import os

# Ensure package directory is always on sys.path
package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if package_root not in sys.path:
    sys.path.insert(0, package_root)
