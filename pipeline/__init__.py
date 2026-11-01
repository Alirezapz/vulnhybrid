"""
Research pipeline (Phases 1-5) for the hybrid C/C++ vulnerability-detection
thesis. The modules in this directory use *flat* imports among themselves
(``from schema import Sample``, ``from static_analysis.semgrep_runner import
...``) exactly as they were written and validated for the thesis -- see
``docs/PIPELINE.md`` for the full five-phase description.

Importing this package puts its own directory on ``sys.path`` so those flat
imports keep resolving whether the project is run from a source checkout, an
editable install, or a normal wheel install. The practical scanner
(``vulnhybrid``) reuses this exact logic through the same flat names, so both
share a single set of module objects (one ``Finding`` class, one
``hybrid_score`` implementation) rather than duplicating them.
"""

import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
