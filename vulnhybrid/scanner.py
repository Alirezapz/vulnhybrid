"""
The scan orchestrator: point the thesis's Phase 2-4 logic at real C/C++ code.

Flow (this is the "thin CLI layer" the thesis's future-work section calls for,
made concrete):

    discover .c/.cpp files
      -> run the SAME semgrep rules / cppcheck runner the thesis measured
      -> attribute every file:line finding to its enclosing function
      -> (optional) send each flagged function to the SAME LLM verifier,
         with the static finding shown as an "unconfirmed lead"
      -> combine with the SAME calibrated hybrid_score()
      -> rank functions by vulnerability probability

Nothing here re-implements detection logic. It imports the validated pipeline
modules (``semgrep_runner``, ``cppcheck_runner``, ``get_verdict``,
``hybrid_score``) and feeds them arbitrary code instead of a labeled dataset.
The one new step is mapping findings to functions, which ``function_extractor``
handles.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import _bootstrap
_bootstrap.activate()

# --- reused, unchanged pipeline logic (flat imports, see _bootstrap) ---------
from findings_schema import Finding                                  # noqa: E402
from llm_stage.llm_schema import Verdict                             # noqa: E402
from llm_stage.llm_client import get_verdict, LLMConfig              # noqa: E402
from hybrid_scoring import hybrid_score, SECURITY_RELEVANT_RULE_IDS  # noqa: E402
from static_analysis import semgrep_runner, cppcheck_runner          # noqa: E402

from .function_extractor import (                                    # noqa: E402
    Function, extract_functions, enclosing_function, window_unit,
    is_source_file, detect_language,
)

logger = logging.getLogger("vulnhybrid.scanner")

DEFAULT_RULES_DIR = Path(semgrep_runner.RULES_DIR)

# Directories never worth scanning; keeps big vendored/build trees out.
_DEFAULT_EXCLUDES = {
    ".git", ".hg", ".svn", ".bzr", "node_modules", "build", "_build", "dist",
    "out", "bin", "obj", ".venv", "venv", "env", "__pycache__", ".idea",
    ".vscode", ".mypy_cache", ".pytest_cache", "third_party", "third-party",
    "vendor", "cmake-build-debug", "cmake-build-release", "CMakeFiles",
}

_SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1,
                   "none": 0, "info": 0}

# Ranking of the unit statuses, most-important first.
_STATUS_RANK = {"confirmed": 4, "uncertain": 3, "review": 2, "cleared": 1, "clear": 0}

# A prediction whose final probability sits this close to the decision
# threshold is a coin-flip, not a confident call -- the thesis measured model
# accuracy at exactly 0.500 for low-confidence verdicts ("when the model hedges
# it knows nothing"). Such units are labelled 'uncertain' rather than
# 'confirmed', so a hedged verdict never reads as a firm confirmation. This is
# a presentation refinement only: `predicted_vulnerable` (and therefore any
# metric computed from it) still uses the plain score >= threshold rule.
UNCERTAIN_MARGIN = 0.05


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class UnitResult:
    """One analyzed function (or context window), with everything the report
    layer needs. ``score`` is a heuristic hybrid risk score in [0, 1].
    Target-domain calibration is required before treating it as a probability.
    """

    unit_id: str
    file: str
    function: str
    start_line: int
    end_line: int
    language: str
    kind: str
    findings: List[Finding]
    verdict: Optional[Verdict]
    score: float
    static_p: float
    llm_p: Optional[float]
    predicted_vulnerable: bool
    status: str                 # "confirmed" | "review" | "cleared" | "clear"
    severity: str
    cwe: Optional[str]
    explanation: List[str]
    code: str = ""

    def to_dict(self) -> dict:
        v = self.verdict
        return {
            "unit_id": self.unit_id,
            "file": self.file,
            "function": self.function,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "kind": self.kind,
            "score": round(self.score, 4),
            "predicted_vulnerable": self.predicted_vulnerable,
            "status": self.status,
            "severity": self.severity,
            "cwe": self.cwe,
            "static_p": round(self.static_p, 4),
            "llm_p": round(self.llm_p, 4) if self.llm_p is not None else None,
            "static_findings": [
                {"tool": f.tool, "rule_id": f.rule_id, "cwe": f.cwe,
                 "severity": f.severity, "line": f.file_line,
                 "snippet_line": f.snippet_line, "message": f.message.strip()}
                for f in self.findings
            ],
            "llm_verdict": None if v is None else {
                "model": v.model,
                "is_vulnerable": v.is_vulnerable,
                "confidence": v.confidence,
                "cwe": v.cwe,
                "severity": v.severity,
                "taint_source": v.taint_source,
                "explanation": v.explanation,
                "suggested_fix": v.suggested_fix,
                "error": v.error,
            },
            "explanation": self.explanation,
        }


@dataclass
class ScanReport:
    target: str
    root: str
    files_scanned: int
    functions_total: int
    units_analyzed: int
    llm_calls: int
    llm_errors: int
    input_tokens: int
    output_tokens: int
    provider: Optional[str]
    model: Optional[str]
    mode: str
    tools: List[str]
    w_llm: float
    prior: float
    threshold: float
    duration_s: float
    units: List[UnitResult] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def flagged(self) -> List[UnitResult]:
        """Units the pipeline judged vulnerable (score >= threshold)."""
        return [u for u in self.units if u.predicted_vulnerable]

    def by_status(self, status: str) -> List[UnitResult]:
        return [u for u in self.units if u.status == status]

    def candidates(self) -> List[UnitResult]:
        """Unconfirmed static candidates awaiting (or lacking) LLM review."""
        return self.by_status("review")

    def uncertain(self) -> List[UnitResult]:
        """Units above threshold but on the decision boundary (model hedged)."""
        return self.by_status("uncertain")

    def status_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for u in self.units:
            counts[u.status] = counts.get(u.status, 0) + 1
        return counts

    def severity_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for u in self.flagged():
            counts[u.severity] = counts.get(u.severity, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "tool": "vulnhybrid",
            "target": self.target,
            "summary": {
                "files_scanned": self.files_scanned,
                "functions_total": self.functions_total,
                "units_analyzed": self.units_analyzed,
                "flagged": len(self.flagged()),
                "status_counts": self.status_counts(),
                "severity_counts": self.severity_counts(),
                "llm_calls": self.llm_calls,
                "llm_errors": self.llm_errors,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "duration_s": round(self.duration_s, 2),
            },
            "config": {
                "mode": self.mode,
                "tools": self.tools,
                "provider": self.provider,
                "model": self.model,
                "w_llm": self.w_llm,
                "prior": self.prior,
                "threshold": self.threshold,
            },
            "notes": self.notes,
            "results": [u.to_dict() for u in self.units],
        }


# ---------------------------------------------------------------------------
# Availability probes
# ---------------------------------------------------------------------------


def semgrep_available() -> bool:
    import shutil
    return shutil.which(semgrep_runner._semgrep_binary()) is not None


def cppcheck_available() -> bool:
    import shutil
    return shutil.which(cppcheck_runner._cppcheck_binary()) is not None


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _discover_files(root: str, single_file: Optional[str],
                    excludes: Sequence[str]) -> List[str]:
    """Return source-file paths relative to ``root``."""
    if single_file is not None:
        return [os.path.relpath(single_file, root)]

    found: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _DEFAULT_EXCLUDES]
        for fn in filenames:
            if not is_source_file(fn):
                continue
            abs_path = os.path.join(dirpath, fn)
            rel = os.path.relpath(abs_path, root)
            if any(fnmatch(rel, pat) or fnmatch(fn, pat) for pat in excludes):
                continue
            found.append(rel)
    found.sort()
    return found


def _norm_path(p: str, root: str) -> str:
    if not p:
        return p
    if os.path.isabs(p):
        try:
            return os.path.relpath(p, root)
        except ValueError:
            return p
    return p[2:] if p.startswith("./") else p


# ---------------------------------------------------------------------------
# Static analysis -> findings attributed to functions
# ---------------------------------------------------------------------------


class _Unit:
    """Mutable accumulator during a scan; frozen into a UnitResult at the end."""

    __slots__ = ("fn", "findings")

    def __init__(self, fn: Function):
        self.fn = fn
        self.findings: List[Finding] = []


def _unit_id(fn: Function) -> str:
    return f"{fn.file}::{fn.name}@L{fn.start_line}"


def _attribute(raw_results: List[dict], tool: str, files_rel: List[str],
               functions_by_file: Dict[str, List[Function]],
               source_by_file: Dict[str, str], root: str,
               units: Dict[str, _Unit]) -> int:
    """Fold one tool's raw results into ``units`` (keyed by unit_id). Returns
    the number of findings attributed."""
    known = set(files_rel)
    n = 0
    for r in raw_results:
        if tool == "semgrep":
            path = _norm_path(r.get("path", ""), root)
            line = (r.get("start") or {}).get("line")
        else:  # cppcheck
            path = _norm_path(r.get("path", ""), root)
            line = r.get("line")
        if path not in known or line is None:
            continue

        fns = functions_by_file.get(path, [])
        fn = enclosing_function(fns, line)
        if fn is None:
            fn = window_unit(source_by_file.get(path, ""), path, line,
                             language=detect_language(path))
        uid = _unit_id(fn)
        unit = units.get(uid)
        if unit is None:
            unit = units[uid] = _Unit(fn)
        idx = len(unit.findings)
        if tool == "semgrep":
            finding = semgrep_runner.result_to_finding(
                r, uid, idx, snippet_line=fn.snippet_line(line))
        else:
            finding = cppcheck_runner.result_to_finding(
                r, uid, idx, snippet_line=fn.snippet_line(line))
        unit.findings.append(finding)
        n += 1
    return n


# ---------------------------------------------------------------------------
# Severity / CWE derivation for the report
# ---------------------------------------------------------------------------


def _static_severity(findings: List[Finding]) -> str:
    """Map the strongest security-relevant static finding's own severity to a
    bucket -- used to rank *unconfirmed candidates* (semgrep ERROR/WARNING,
    cppcheck error/warning). Deliberately caps at the tool's stated severity;
    it is a candidate, not a confirmation."""
    rank = {"error": 3, "warning": 2, "style": 1, "performance": 1,
            "portability": 1, "info": 1, "information": 0}
    best = 0
    for f in findings:
        if f.rule_id in SECURITY_RELEVANT_RULE_IDS:
            best = max(best, rank.get((f.severity or "").lower(), 1))
    return {3: "high", 2: "medium", 1: "low", 0: "low"}[best]


def _derive_severity(status: str, score: float, verdict: Optional[Verdict],
                     findings: List[Finding]) -> str:
    if status in ("cleared", "clear"):
        return "none"
    if status == "review":
        return _static_severity(findings)
    # confirmed: prefer the model's self-reported severity, else bucket by the
    # calibrated probability.
    if verdict is not None and verdict.error is None and verdict.severity:
        sev = verdict.severity.lower()
        if sev in _SEVERITY_ORDER and sev != "none":
            return sev
    if score >= 0.85:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def _derive_cwe(verdict: Optional[Verdict], findings: List[Finding]) -> Optional[str]:
    if verdict is not None and verdict.error is None and verdict.cwe:
        return verdict.cwe
    sec = [f for f in findings if f.rule_id in SECURITY_RELEVANT_RULE_IDS and f.cwe]
    if sec:
        sec.sort(key=lambda f: _SEVERITY_ORDER.get((f.severity or "").lower(), 0),
                 reverse=True)
        return sec[0].cwe
    for f in findings:
        if f.cwe:
            return f.cwe
    return None


def _security_findings(findings: List[Finding], include_nonsecurity: bool) -> List[Finding]:
    if include_nonsecurity:
        return list(findings)
    return [f for f in findings if f.rule_id in SECURITY_RELEVANT_RULE_IDS]


def _rule_messages(findings: List[Finding]) -> List[Tuple[str, int, str]]:
    return [(f.rule_id, f.snippet_line or f.file_line or 0, f.message) for f in findings]


# ---------------------------------------------------------------------------
# The main entry point
# ---------------------------------------------------------------------------


def scan(target: str,
         *,
         tools: Sequence[str] = ("semgrep",),
         mode: str = "triage",
         llm: Optional[LLMConfig] = None,
         rules_dir: Optional[str] = None,
         w_llm: float = 1.0,
         prior: float = 0.4,
         threshold: float = 0.5,
         concurrency: int = 4,
         max_units: Optional[int] = None,
         excludes: Sequence[str] = (),
         include_nonsecurity: bool = False,
         cppcheck_include_dir: Optional[str] = None,
         timeout: int = 3600,
         progress: Optional[Callable[[int, int], None]] = None) -> ScanReport:
    """Scan ``target`` (a file or directory of C/C++) and return a ScanReport.

    ``mode``:
      * ``triage``  -- LLM verifies only functions with a security-relevant
        static finding (the thesis's cost-bounded "verify-only" default).
      * ``deep``    -- LLM runs on every function, findings or not (higher
        recall ceiling, proportionally more model calls).
      * ``static``  -- no LLM; rank by static evidence alone. Works with only
        semgrep installed, zero model setup.

    If ``llm`` is ``None``, the scan runs in ``static`` mode regardless of the
    requested mode (with a note), so the tool always produces *something*.
    ``w_llm`` defaults to 1.0 -- the value the thesis fitted on validation data
    (see docs/RESULTS.md); the static rules then act as the candidate filter,
    not as scoring weight.
    """
    t0 = time.monotonic()
    target = os.path.abspath(target)
    notes: List[str] = []
    # Validate scoring arguments even when there are no candidate functions.
    hybrid_score('config-validation', [], None, prior=prior, w_llm=w_llm, threshold=threshold)
    if concurrency < 1 or (max_units is not None and max_units < 0):
        raise ValueError('concurrency must be positive and max_units nonnegative')

    # Offline-first, matching the pipeline's own design (it already passes
    # `--metrics off` and loads local rules, never the Semgrep Registry). The
    # version check is one more silent call to semgrep.dev; on a locked-down
    # network it *hangs* until timeout (~1.5 min for a 4-file scan observed),
    # so disable it unless the user has deliberately set it.
    os.environ.setdefault("SEMGREP_ENABLE_VERSION_CHECK", "0")

    if os.path.isdir(target):
        root, single_file = target, None
    elif os.path.isfile(target):
        root, single_file = os.path.dirname(target) or ".", target
    else:
        raise FileNotFoundError(f"Target does not exist: {target}")

    rules_path = Path(rules_dir) if rules_dir else DEFAULT_RULES_DIR
    tools = list(tools)

    # -- resolve LLM / mode --------------------------------------------------
    if llm is None and mode != "static":
        notes.append(f"No LLM configured; running static-only instead of '{mode}'.")
        mode = "static"
    if mode not in ("triage", "deep", "static"):
        raise ValueError(f"Unknown mode: {mode!r}")

    # -- discover files & extract functions ----------------------------------
    files_rel = _discover_files(root, single_file, excludes)
    functions_by_file: Dict[str, List[Function]] = {}
    source_by_file: Dict[str, str] = {}
    functions_total = 0
    for rel in files_rel:
        abs_path = os.path.join(root, rel)
        try:
            with open(abs_path, encoding="utf-8", errors="replace") as f:
                src = f.read()
        except OSError as e:
            logger.warning("Could not read %s: %s", rel, e)
            continue
        source_by_file[rel] = src
        fns = extract_functions(src, path=rel, language=detect_language(rel))
        functions_by_file[rel] = fns
        functions_total += len(fns)

    if not files_rel:
        notes.append("No C/C++ source files found under the target.")

    # -- static analysis -----------------------------------------------------
    units: Dict[str, _Unit] = {}
    ran_tools: List[str] = []

    if "semgrep" in tools:
        if not semgrep_available():
            notes.append("semgrep not found on PATH; skipped (pip install semgrep).")
        elif files_rel:
            if single_file is not None:
                raw = semgrep_runner.run_on_file(single_file, rules_path)
            else:
                raw = semgrep_runner.run_on_directory(root, rules_path, timeout=timeout)
            _attribute(raw, "semgrep", files_rel, functions_by_file,
                       source_by_file, root, units)
            ran_tools.append("semgrep")

    if "cppcheck" in tools:
        if not cppcheck_available():
            notes.append("cppcheck not found on PATH; skipped (apt-get install cppcheck).")
        elif files_rel:
            if single_file is not None:
                raw = cppcheck_runner._run_cppcheck(
                    single_file, timeout=timeout,
                    include_dir=cppcheck_include_dir, cwd=None)
            else:
                raw = cppcheck_runner.run_on_directory(
                    root, include_dir=cppcheck_include_dir, timeout=timeout)
            _attribute(raw, "cppcheck", files_rel, functions_by_file,
                       source_by_file, root, units)
            ran_tools.append("cppcheck")

    if files_rel and not ran_tools and mode != 'deep':
        raise RuntimeError('No requested static analyzer ran; scan coverage is unavailable')

    # -- assemble the set of units to score ----------------------------------
    # deep mode also scores functions that had zero findings.
    if mode == "deep":
        for rel, fns in functions_by_file.items():
            for fn in fns:
                uid = _unit_id(fn)
                if uid not in units:
                    units[uid] = _Unit(fn)

    # Which units does the LLM see?
    if mode == "static":
        llm_units: List[_Unit] = []
    elif mode == "deep":
        llm_units = list(units.values())
    else:  # triage
        llm_units = [u for u in units.values()
                     if _security_findings(u.findings, include_nonsecurity)]

    # Deterministic ordering, then optional cap on model calls.
    llm_units.sort(key=lambda u: (u.fn.file, u.fn.start_line))
    if max_units is not None:
        llm_units = llm_units[:max_units]

    # -- LLM verification stage ---------------------------------------------
    verdicts: Dict[str, Verdict] = {}
    in_tok = out_tok = llm_errors = 0
    provider = model = None

    if llm is not None and llm_units:
        provider, model = llm.provider, llm.model
        total = len(llm_units)
        done = 0

        def _one(unit: _Unit) -> Tuple[str, Verdict, int, int]:
            sec = _security_findings(unit.findings, include_nonsecurity)
            rm = _rule_messages(sec)
            try:
                v, i, o = get_verdict(_unit_id(unit.fn), unit.fn.code,
                                      unit.fn.language, rm, llm)
                return _unit_id(unit.fn), v, i, o
            except Exception as e:  # last-resort net; mirror run_llm_stage
                return (_unit_id(unit.fn),
                        Verdict(sample_id=_unit_id(unit.fn), model=llm.model,
                                is_vulnerable=None, confidence=None,
                                static_findings_shown=[r for r, _, _ in rm],
                                error=str(e)), 0, 0)

        if concurrency <= 1:
            for unit in llm_units:
                uid, v, i, o = _one(unit)
                verdicts[uid] = v
                in_tok += i; out_tok += o
                llm_errors += 1 if v.error else 0
                done += 1
                if progress:
                    progress(done, total)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(_one, u) for u in llm_units]
                for fut in as_completed(futures):
                    uid, v, i, o = fut.result()
                    verdicts[uid] = v
                    in_tok += i; out_tok += o
                    llm_errors += 1 if v.error else 0
                    done += 1
                    if progress:
                        progress(done, total)

    # -- score every candidate unit -----------------------------------------
    # In triage/static, only units with >=1 finding are candidates. In deep,
    # every function is a candidate.
    if mode == "deep":
        scored_units = list(units.values())
    else:
        scored_units = [u for u in units.values() if u.findings]

    results: List[UnitResult] = []
    for unit in scored_units:
        fn = unit.fn
        uid = _unit_id(fn)
        sec = _security_findings(unit.findings, include_nonsecurity)
        verdict = verdicts.get(uid)
        hr = hybrid_score(uid, sec, verdict, prior=prior, w_llm=w_llm,
                          threshold=threshold)
        predicted = bool(hr.label)

        if predicted:
            # crossed the threshold, but if it's sitting right on the decision
            # boundary the model was essentially guessing -> 'uncertain', not a
            # firm 'confirmed' (avoids "CONFIRMED ... model: not vulnerable").
            if abs(hr.score - threshold) <= UNCERTAIN_MARGIN:
                status = "uncertain"
            else:
                status = "confirmed"
        elif (verdict is not None and verdict.error is None
              and verdict.is_vulnerable is False):
            # the model actively judged this safe -> a real "cleared" signal,
            # distinct from a static candidate no model has looked at yet
            status = "cleared"
        elif sec:
            status = "review"
        else:
            status = "clear"

        severity = _derive_severity(status, hr.score, verdict, unit.findings)
        cwe = _derive_cwe(verdict, unit.findings)
        results.append(UnitResult(
            unit_id=uid, file=fn.file, function=fn.name,
            start_line=fn.start_line, end_line=fn.end_line,
            language=fn.language, kind=fn.kind,
            # report the security-relevant findings by default (a security
            # report shouldn't surface cppcheck's unusedFunction/style noise);
            # --include-nonsecurity keeps everything.
            findings=sec, verdict=verdict,
            score=hr.score, static_p=hr.static_p, llm_p=hr.llm_p,
            predicted_vulnerable=predicted, status=status,
            severity=severity, cwe=cwe,
            explanation=hr.explanation, code=fn.code,
        ))

    # Rank: confirmed > review > cleared > clear, then by probability, then by
    # severity.
    results.sort(key=lambda u: (_STATUS_RANK.get(u.status, 0), u.score,
                                _SEVERITY_ORDER.get(u.severity, 0)), reverse=True)

    if llm is not None and verdicts and llm_errors == len(verdicts):
        notes.append(
            f"Every LLM call failed ({llm.provider}:{llm.model}) — check the "
            "server/key/endpoint. Scores fell back to static evidence only.")

    return ScanReport(
        target=target, root=root, files_scanned=len(files_rel),
        functions_total=functions_total, units_analyzed=len(results),
        llm_calls=len(verdicts), llm_errors=llm_errors,
        input_tokens=in_tok, output_tokens=out_tok,
        provider=provider, model=model, mode=mode, tools=ran_tools,
        w_llm=w_llm, prior=prior, threshold=threshold,
        duration_s=time.monotonic() - t0, units=results, notes=notes,
    )
