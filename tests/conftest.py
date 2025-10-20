"""
Pytest-wide configuration: make the repository importable without installation.

By pushing the project root onto ``sys.path`` we allow ``import citylearn`` to
work even when the package has not been installed into the active environment.
This mirrors the behaviour of several legacy scripts and keeps local runs
simple.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
