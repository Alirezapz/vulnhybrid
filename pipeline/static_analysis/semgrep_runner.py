"""
Wraps the semgrep CLI and turns its JSON output into Finding objects.

Deliberately uses LOCAL rule files (static_analysis/rules/*.yml) rather than
`--config=auto` / `--config=p/c`, which pull from the Semgrep Registry over
the network (semgrep.dev). That's a real risk for this project specifically
-- Hugging Face access from this network already needed a fallback path
once; no reason to introduce a second silent dependency on a US-based
service being reachable when a local ruleset works just as well and is
fully auditable besides.

Two entry points, matching the two situations Phase 1 leaves us in:
  - run_on_file(): for Juliet, where the ORIGINAL file is on disk and
    multiple samples (bad/good) may share one file -- callers should run
    this ONCE per unique file and attribute results via line ranges
    (see attribute_findings.py), not once per sample.
  - run_on_snippet(): for Devign (and any sample with no original file on
    disk), where the sample's own `code` text is all there is. Writes it to
    a temp file since semgrep needs a file path.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from findings_schema import Finding
from schema import Sample
from .errors import AnalysisError

logger = logging.getLogger("semgrep_runner")

RULES_DIR = Path(__file__).parent / "rules"

# semgrep's own severity strings, normalized to one of these three
_SEVERITY_MAP = {"ERROR": "ERROR", "WARNING": "WARNING", "INFO": "INFO"}

_EXT_BY_LANGUAGE = {"c": ".c", "cpp": ".cpp", "java": ".java"}


def _semgrep_binary() -> str:
    """Resolve the semgrep executable. Defaults to whatever 'semgrep' on
    PATH resolves to; set SEMGREP_BIN to override (e.g. if it's only
    installed inside a venv that isn't activated)."""
    import os
    return os.environ.get("SEMGREP_BIN", "semgrep")


def _run_semgrep(target_path: str, rules_dir: Path = RULES_DIR, timeout: int = 60,
                  cwd: Optional[str] = None) -> dict:
    cmd = [
        _semgrep_binary(), "--config", str(rules_dir),
        "--json", "--quiet", "--no-git-ignore",
        "--metrics", "off",  # no telemetry call home
        target_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, cwd=cwd,
                               encoding="utf-8", errors="replace")
    except FileNotFoundError:
        raise RuntimeError(
            "semgrep executable not found. Install with: pip install semgrep "
            "(use --break-system-packages on Debian/Ubuntu, or a venv)."
        )
    except subprocess.TimeoutExpired:
        raise AnalysisError(f"semgrep timed out on {target_path} after {timeout}s")

    if proc.returncode not in (0, 1):  # 1 = findings present, still success
        raise AnalysisError(f"semgrep exited {proc.returncode}: {proc.stderr[-500:]}")

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise AnalysisError(f"semgrep produced non-JSON output for {target_path}")
    if not isinstance(data, dict) or not isinstance(data.get('results'), list):
        raise AnalysisError("semgrep response has no results array")
    if data.get('errors'):
        raise AnalysisError(f"semgrep reported {len(data['errors'])} scan error(s); "
                            "inspect the raw output before interpreting coverage")
    return data


def run_on_directory(root_dir: str, rules_dir: Path = RULES_DIR, timeout: int = 3600) -> List[dict]:
    """ONE semgrep invocation across the entire tree under root_dir, with
    cwd set to root_dir so every result's `path` field comes back relative
    to root_dir -- the same convention Sample.extra['file_path'] uses, so
    the two line up for attribution with no path-rewriting needed.

    This matters for performance, not just style: one subprocess per file
    does NOT scale -- verified directly, ~15k separate invocations timed
    out where one invocation over the same ~20k files took under 6 minutes.
    Default timeout is generous (1 hour) since a full Juliet checkout is
    ~105k files; override if your environment is faster or slower."""
    data = _run_semgrep(".", rules_dir, timeout=timeout, cwd=root_dir)
    return data.get("results", [])


def result_to_finding(result: dict, sample_id: str, finding_idx: int,
                       snippet_line: Optional[int] = None) -> Finding:
    check_id = result.get("check_id", "")
    rule_id = check_id.split(".")[-1] if check_id else "unknown"
    metadata = result.get("extra", {}).get("metadata", {})
    severity_raw = result.get("extra", {}).get("severity", "INFO")
    return Finding(
        finding_id=f"semgrep_{sample_id}_{finding_idx}",
        sample_id=sample_id,
        tool="semgrep",
        rule_id=rule_id,
        cwe=metadata.get("cwe"),
        severity=_SEVERITY_MAP.get(severity_raw, severity_raw),
        message=result.get("extra", {}).get("message", "").strip(),
        file_line=result.get("start", {}).get("line"),
        snippet_line=snippet_line,
        extra={},
    )


def run_on_file(file_path: str, rules_dir: Path = RULES_DIR) -> List[dict]:
    """Runs semgrep on a real file ON DISK and returns the raw `results`
    list (file:line still refers to THIS file -- attribution to a specific
    sample happens one layer up, in attribute_findings.py, since one file
    commonly contains more than one sample (a bad() and a good() pair)."""
    data = _run_semgrep(file_path, rules_dir)
    return data.get("results", [])


def run_on_snippet(code: str, sample_id: str, language: str = "c",
                    rules_dir: Path = RULES_DIR) -> List[Finding]:
    """Runs semgrep on a SINGLE isolated code snippet. Kept for one-off use
    and testing -- for anything more than a handful of samples, use
    run_on_snippets_batch instead. One subprocess per snippet does not
    scale: confirmed directly, this is what made the Devign path take
    "2 hours and counting" on a real run, with individual 60-second
    timeouts on some files despite each being a single small function.
    Whatever the exact per-process overhead cause on a given machine
    (antivirus scanning every new temp file and every new process launch is
    the leading suspect on Windows), batching into one process sidesteps it
    entirely rather than chasing the cause."""
    ext = _EXT_BY_LANGUAGE.get(language, ".c")
    with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False, encoding="utf-8") as tmp:
        tmp.write(code)
        tmp_path = tmp.name
    try:
        results = run_on_file(tmp_path, rules_dir)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    findings = []
    for i, r in enumerate(results):
        # snippet_line: since the temp file IS the snippet, file line ==
        # snippet line directly, no offset needed.
        f = result_to_finding(r, sample_id, i, snippet_line=r.get("start", {}).get("line"))
        findings.append(f)
    return findings


def run_on_snippets_batch(samples: List[Sample], rules_dir: Path = RULES_DIR,
                           timeout: int = 3600) -> List[Finding]:
    """Writes EVERY sample's code to its own file in ONE temp directory
    (named by sample_id, so results map back with no ambiguity), then runs
    semgrep ONCE against that whole directory -- the same one-invocation
    shape as run_on_directory, applied to synthetic per-sample files
    instead of an existing real tree. This is the function
    attribute_findings.run_static_analysis_on_snippets should call; do not
    loop run_on_snippet() per sample at any real scale.
    """
    import shutil

    tmp_dir = Path(tempfile.mkdtemp(prefix="semgrep_snippets_"))
    fname_to_sample_id = {}
    try:
        for s in samples:
            ext = _EXT_BY_LANGUAGE.get(s.language, ".c")
            fname = f"{s.sample_id}{ext}"
            (tmp_dir / fname).write_text(s.code, encoding="utf-8")
            fname_to_sample_id[fname] = s.sample_id

        logger.info("Scanning %d snippet(s) in one batch invocation...", len(samples))
        data = _run_semgrep(".", rules_dir, timeout=timeout, cwd=str(tmp_dir))
        raw_results = data.get("results", [])

        findings = []
        unmatched = 0
        for i, r in enumerate(raw_results):
            fname = Path(r.get("path", "")).name
            sample_id = fname_to_sample_id.get(fname)
            if sample_id is None:
                unmatched += 1
                continue
            findings.append(result_to_finding(
                r, sample_id, i, snippet_line=r.get("start", {}).get("line")))
        if unmatched:
            logger.warning("%d result(s) didn't map to a known sample filename", unmatched)
        return findings
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
