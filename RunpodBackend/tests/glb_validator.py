"""GLB validation utilities for RunpodBackend tests.

The canonical implementation lives in conftest.py so pytest can inject
validate_glb / UNITY_HUMANOID_BONES directly into test functions.

This module re-exports those symbols so integration tests and test_glb_validator.py
can import them directly without going through conftest machinery.
"""

from __future__ import annotations

# Re-export the canonical implementations from conftest.
# conftest.py is always on sys.path when pytest runs (tests/ is on pythonpath).
from conftest import UNITY_HUMANOID_BONES, make_minimal_glb, validate_glb

__all__ = ["UNITY_HUMANOID_BONES", "validate_glb", "make_minimal_glb"]
