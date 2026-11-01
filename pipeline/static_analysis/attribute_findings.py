"""
Attributes raw static-analysis results (file:line based) to the correct
extracted Sample when multiple samples share one original file -- Juliet's
bad() and good() functions live in the same physical file, so a finding's
line number alone doesn't say which sample it belongs to without checking
against each sample's recorded line_ranges (see sard_manifest_parser.py).

Covers both tools wired into this project: semgrep (run_static_analysis_on_*)
and cppcheck (run_cppcheck_on_*), with identical attribution logic shared
via _snippet_line_for -- the line-mapping problem is the same regardless of
which tool produced the file:line result.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from findings_schema import Finding
from schema import Sample
from . import cppcheck_runner
from .semgrep_runner import RULES_DIR, result_to_finding, run_on_directory, run_on_snippets_batch

logger = logging.getLogger("attribute_findings")


def _snippet_line_for(file_line: int, line_ranges: List[Tuple[int, int]]) -> Optional[int]:
    """Convert a 1-indexed line number IN THE ORIGINAL FILE to a 1-indexed
    line number within the sample's own (possibly multi-block) extracted
    text, accounting for the single blank-line separator inserted between
    concatenated blocks (see extract_bad_and_good's "\\n\\n".join)."""
    offset = 0
    for (start, end) in line_ranges:
        block_len = end - start + 1
        if start <= file_line <= end:
            return offset + (file_line - start + 1)
        offset += block_len + 1  # +1 for the blank-line separator
    return None


def run_static_analysis_on_juliet(samples: List[Sample], root_dir: str,
                                   rules_dir: Path = RULES_DIR,
                                   timeout: int = 3600) -> List[Finding]:
    """ONE semgrep invocation across the whole tree under root_dir (see
    run_on_directory's docstring for why -- one-subprocess-per-file was
    tried first and timed out), then fans each result out to whichever
    sample(s) actually contain that line.

    Samples missing file_path/line_ranges (e.g. loaded from a Phase-1 run
    predating this metadata) are skipped with a warning, not silently
    dropped without explanation.
    """
    root = Path(root_dir)
    by_file: Dict[str, List[Sample]] = defaultdict(list)
    skipped = 0
    for s in samples:
        if s.dataset != "juliet":
            continue
        fp = s.extra.get("file_path")
        if not fp or "line_ranges" not in s.extra:
            skipped += 1
            continue
        by_file[fp].append(s)

    if skipped:
        logger.warning(
            "%d Juliet sample(s) missing file_path/line_ranges metadata "
            "(re-run build_dataset.py with the current sard_manifest_parser.py "
            "to fix this) -- skipped, not analyzed.", skipped,
        )

    logger.info("Scanning %s (one semgrep invocation, %d files of interest)...",
                root, len(by_file))
    raw_results = run_on_directory(str(root), rules_dir, timeout=timeout)
    logger.info("semgrep returned %d raw results across the whole tree", len(raw_results))

    results_by_path: Dict[str, List[dict]] = defaultdict(list)
    for r in raw_results:
        results_by_path[r.get("path", "")].append(r)

    all_findings: List[Finding] = []
    n_unmatched = 0
    for fp, file_samples in by_file.items():
        for result in results_by_path.get(fp, []):
            file_line = result.get("start", {}).get("line")
            if file_line is None:
                continue
            matched_any = False
            for sample in file_samples:
                ranges = [tuple(r) for r in sample.extra.get("line_ranges", [])]
                snippet_line = _snippet_line_for(file_line, ranges)
                if snippet_line is not None:
                    idx = len(all_findings)
                    finding = result_to_finding(result, sample.sample_id, idx,
                                                 snippet_line=snippet_line)
                    all_findings.append(finding)
                    matched_any = True
            if not matched_any:
                n_unmatched += 1

    if n_unmatched:
        logger.info(
            "%d semgrep finding(s) fell outside any extracted sample's line "
            "range (shared scaffolding/main() code, not bad()/good() bodies "
            "-- expected, not an error)", n_unmatched,
        )
    return all_findings


def run_static_analysis_on_snippets(samples: List[Sample],
                                     rules_dir: Path = RULES_DIR,
                                     timeout: int = 3600) -> List[Finding]:
    """For samples with no original file on disk (Devign, or any future
    dataset that only ever provides isolated function text). Uses ONE
    batched semgrep invocation across all snippets -- looping run_on_snippet
    per sample was the actual cause of a real run taking 2+ hours and
    timing out on individual files; this is the fix, not a tuning knob."""
    relevant = [s for s in samples if not s.extra.get("file_path")]
    if not relevant:
        return []
    return run_on_snippets_batch(relevant, rules_dir=rules_dir, timeout=timeout)


def run_cppcheck_on_juliet(samples: List[Sample], root_dir: str,
                            include_dir: Optional[str] = None,
                            timeout: int = 3600) -> List[Finding]:
    """cppcheck equivalent of run_static_analysis_on_juliet -- same
    one-invocation-per-tree shape, same line-range attribution logic,
    different tool. include_dir should point at Juliet's testcasesupport/
    directory (defines std_testcase.h) if available; cppcheck still ran
    fine without it in testing, this just avoids missingInclude noise."""
    root = Path(root_dir)
    by_file: Dict[str, List[Sample]] = defaultdict(list)
    skipped = 0
    for s in samples:
        if s.dataset != "juliet":
            continue
        fp = s.extra.get("file_path")
        if not fp or "line_ranges" not in s.extra:
            skipped += 1
            continue
        by_file[fp].append(s)

    if skipped:
        logger.warning("%d Juliet sample(s) missing file_path/line_ranges -- skipped.", skipped)

    logger.info("cppcheck scanning %s (one invocation, %d files of interest)...",
                root, len(by_file))
    raw_results = cppcheck_runner.run_on_directory(str(root), include_dir=include_dir,
                                                     timeout=timeout)
    logger.info("cppcheck returned %d raw results across the whole tree", len(raw_results))

    results_by_path: Dict[str, List[dict]] = defaultdict(list)
    for r in raw_results:
        path = (r.get("path") or "").lstrip("./")
        results_by_path[path].append(r)

    all_findings: List[Finding] = []
    n_unmatched = 0
    for fp, file_samples in by_file.items():
        for result in results_by_path.get(fp, []):
            file_line = result.get("line")
            if file_line is None:
                continue
            matched_any = False
            for sample in file_samples:
                ranges = [tuple(r) for r in sample.extra.get("line_ranges", [])]
                snippet_line = _snippet_line_for(file_line, ranges)
                if snippet_line is not None:
                    idx = len(all_findings)
                    finding = cppcheck_runner.result_to_finding(
                        result, sample.sample_id, idx, snippet_line=snippet_line)
                    all_findings.append(finding)
                    matched_any = True
            if not matched_any:
                n_unmatched += 1

    if n_unmatched:
        logger.info("%d cppcheck finding(s) fell outside any sample's line range.", n_unmatched)
    return all_findings


def run_cppcheck_on_snippets(samples: List[Sample], timeout: int = 3600) -> List[Finding]:
    """Same fix as run_static_analysis_on_snippets, for cppcheck."""
    relevant = [s for s in samples if not s.extra.get("file_path")]
    if not relevant:
        return []
    return cppcheck_runner.run_on_snippets_batch(relevant, timeout=timeout)
