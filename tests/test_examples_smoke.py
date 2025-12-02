# tests/test_examples_smoke.py
"""
Simple smoke tests ensuring that example scripts run without import errors.

These do NOT validate model quality — only that the example modules
import and execute their top-level code.
"""

import importlib


def test_import_gaussian_missing_y_example():
    """
    Ensure examples.gaussian_missing_y module imports and runs
    without raising errors.
    """
    mod = importlib.import_module(
        "multibin_cvae.examples.gaussian_missing_y"
    )
    assert mod is not None
