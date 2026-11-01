"""
vulnhybrid -- a practical hybrid vulnerability scanner for C/C++ source.

This is the "product" layer on top of the thesis pipeline: instead of running
against a pre-labeled dataset, it points the *same* validated Phase 2-4 logic
(static analysis -> LLM semantic verification -> calibrated probabilistic
scoring) at an arbitrary file or directory of real C/C++ code and produces a
ranked, explainable report.

Public entry points:
    vulnhybrid.scanner.scan(...)     -- run a scan, get structured results
    vulnhybrid.cli.main()            -- the ``vulnhybrid`` command-line tool
"""

from . import _bootstrap as _bootstrap

# Put the pipeline on the import path as early as possible, so any
# ``import vulnhybrid.scanner`` (which pulls in the pipeline's flat modules)
# works without the caller having to set PYTHONPATH.
_bootstrap.activate()

__version__ = "1.2.0"

__all__ = ["__version__"]
