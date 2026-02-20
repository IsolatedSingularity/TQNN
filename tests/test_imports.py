"""
Smoke tests: verify all visualization modules import cleanly.

These are intentionally lightweight -- they confirm the dependency chain is
intact without running any expensive plotting or animation logic.
"""

import importlib

import pytest


# Use Agg backend so tests don't open windows
import matplotlib
matplotlib.use('Agg')


MODULES = [
    'tqnn.helpers',
    'tqnn.visualization.static',
    'tqnn.visualization.animated',
    'tqnn.visualization.sandbox',
    'tqnn.classifier.gui',
    'tqnn.cobordism.gui',
]


@pytest.mark.parametrize('module_name', MODULES)
def test_module_imports(module_name: str) -> None:
    """Each visualization module should import without errors."""
    mod = importlib.import_module(module_name)
    assert mod is not None
