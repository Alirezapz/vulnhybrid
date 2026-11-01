"""
Makes the validated research pipeline (Phases 2-4) importable by the tool
regardless of how the project is being run.

The pipeline modules live in ``<repo>/pipeline`` and import each other by
bare name (``from hybrid_scoring import ...``), the layout they were written
and tested in. Rather than fork or rewrite that validated code, the scanner
reuses it verbatim -- this shim just puts the pipeline directory on
``sys.path`` first, so those bare names resolve. It is deliberately tolerant:

  * source checkout / editable install -> ``<repo>/pipeline`` sits next to
    ``<repo>/vulnhybrid``;
  * wheel install -> ``pipeline`` is shipped as a sibling package, so the
    same ``../pipeline`` relative path still points at it.

Importing pipeline logic only ever through these flat names (never
``pipeline.hybrid_scoring``) guarantees the tool and the pipeline share the
same ``Finding``/``Verdict`` classes, which matters because objects are
handed back and forth between them.
"""

from __future__ import annotations

import os
import sys


def activate() -> str:
    """Idempotently add the pipeline directory to ``sys.path``. Returns the
    resolved pipeline path (whether or not it existed) for diagnostics."""
    here = os.path.dirname(os.path.abspath(__file__))      # .../vulnhybrid
    parent = os.path.dirname(here)                          # repo root / site-packages
    pipeline_dir = os.path.join(parent, "pipeline")
    if os.path.isdir(pipeline_dir) and pipeline_dir not in sys.path:
        sys.path.insert(0, pipeline_dir)
    return pipeline_dir
