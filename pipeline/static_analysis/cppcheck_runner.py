"""
Wraps the cppcheck CLI and turns its XML output into Finding objects.

Why cppcheck was added despite the negative result documented in the
README: it's a genuinely different analysis technique from semgrep
(value-flow/dataflow vs syntax pattern matching), so even where it doesn't
help, that's worth measuring rather than assuming. Verified directly
against the two real gap cases semgrep's dangerous-copy rule couldn't
cover (array-index-OOB-via-incomplete-bounds-check, and
memcpy-with-wrong-sizeof-argument) -- even with --check-level=exhaustive,
--inconclusive, and the real std_testcase.h on the include path, cppcheck
reported NOTHING for either. That's not a config mistake: CheckBufferOverrun
and CheckSizeof checkers were both confirmed ACTIVE via --checkers-report.
Documented in the README rather than silently shipping cppcheck as "the
fix" for a gap it doesn't actually close for these patterns.

Like semgrep_runner.py, one invocation per file does not scale -- use
run_on_directory() for a whole tree.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

from findings_schema import Finding
from schema import Sample
from .errors import AnalysisError

logger = logging.getLogger("cppcheck_runner")

_EXT_BY_LANGUAGE = {"c": ".c", "cpp": ".cpp"}


def _cppcheck_binary() -> str:
    import os
    return os.environ.get("CPPCHECK_BIN", "cppcheck")


def _run_cppcheck(target_path: str, timeout: int, include_dir: Optional[str],
                   cwd: Optional[str], _use_exhaustive: bool = True) -> List[dict]:
    cmd = [_cppcheck_binary(), "--enable=all", "--inconclusive"]
    if _use_exhaustive:
        cmd.append("--check-level=exhaustive")
    cmd += ["--xml", "--quiet"]
    if include_dir:
        cmd += ["-I", include_dir]
    cmd.append(target_path)

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, cwd=cwd,
                               encoding="utf-8", errors="replace")
    except FileNotFoundError:
        raise RuntimeError(
            "cppcheck executable not found. Install with: apt-get install cppcheck "
            "(Debian/Ubuntu) or see cppcheck.sourceforge.io for other platforms."
        )
    except subprocess.TimeoutExpired:
        raise AnalysisError(f"cppcheck timed out on {target_path} after {timeout}s")

    # cppcheck writes its XML to stderr, not stdout
    try:
        root = ET.fromstring(proc.stderr)
    except ET.ParseError:
        # --check-level=exhaustive was added in a later cppcheck release than
        # some distros ship (e.g. Ubuntu 22.04's apt package is 2.7) -- an
        # older binary rejects the flag and prints a plain-text error instead
        # of XML, which silently looked like "0 findings" instead of a
        # version mismatch. Retry once without it before giving up.
        if _use_exhaustive and 'check-level' in (proc.stdout + proc.stderr) \
                and ('unrecognized' in (proc.stdout + proc.stderr).lower()
                     or 'unknown' in (proc.stdout + proc.stderr).lower()):
            logger.warning(
                "cppcheck produced non-XML output for %s -- retrying without "
                "--check-level=exhaustive in case this is an older cppcheck "
                "that doesn't support it (stderr started with: %r)",
                target_path, proc.stderr[:200],
            )
            return _run_cppcheck(target_path, timeout, include_dir, cwd, _use_exhaustive=False)
        raise AnalysisError(f"cppcheck produced non-XML output: {proc.stderr[:300]}")

    if proc.returncode != 0:
        raise AnalysisError(f"cppcheck exited {proc.returncode}")

    errors_el = root.find("errors")
    if errors_el is None:
        return []

    results = []
    for err in errors_el.findall("error"):
        if err.get('id') in ('syntaxError', 'internalError', 'cppcheckError'):
            raise AnalysisError(f"cppcheck could not analyze input: {err.get('msg')}")
        if err.get("severity") == "information":
            continue  # missingInclude / checkersReport noise, never a real finding
        locations = err.findall("location")
        if not locations:
            continue
        primary = locations[0]
        results.append({
            "id": err.get("id"),
            "severity": err.get("severity"),
            "cwe": err.get("cwe"),
            "message": err.get("msg", ""),
            "path": primary.get("file"),
            "line": int(primary.get("line", 0)) or None,
        })
    return results


def run_on_directory(root_dir: str, include_dir: Optional[str] = None,
                      timeout: int = 3600) -> List[dict]:
    """ONE cppcheck invocation across the whole tree under root_dir, cwd
    set to root_dir so result paths come back relative to it -- same
    convention as semgrep_runner.run_on_directory, for the same reason
    (Sample.extra['file_path'] is root-relative)."""
    return _run_cppcheck(".", timeout=timeout, include_dir=include_dir, cwd=root_dir)


def run_on_snippet(code: str, sample_id: str, language: str = "c",
                    timeout: int = 60) -> List[Finding]:
    """For a SINGLE sample with no original file on disk. Kept for one-off
    use -- for anything more than a handful of samples, use
    run_on_snippets_batch (same one-process-per-sample scaling problem as
    semgrep_runner.run_on_snippet, fixed the same way)."""
    ext = _EXT_BY_LANGUAGE.get(language, ".c")
    with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False, encoding="utf-8") as tmp:
        tmp.write(code)
        tmp_path = tmp.name
    try:
        raw_results = _run_cppcheck(tmp_path, timeout=timeout, include_dir=None, cwd=None)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    findings = []
    for i, r in enumerate(raw_results):
        findings.append(result_to_finding(r, sample_id, i, snippet_line=r.get("line")))
    return findings


def run_on_snippets_batch(samples: List[Sample], timeout: int = 3600) -> List[Finding]:
    """Writes every sample's code to its own file in ONE temp directory,
    then runs cppcheck ONCE against that whole directory. Mirrors
    semgrep_runner.run_on_snippets_batch."""
    import shutil

    tmp_dir = Path(tempfile.mkdtemp(prefix="cppcheck_snippets_"))
    fname_to_sample_id = {}
    try:
        for s in samples:
            ext = _EXT_BY_LANGUAGE.get(s.language, ".c")
            fname = f"{s.sample_id}{ext}"
            (tmp_dir / fname).write_text(s.code, encoding="utf-8")
            fname_to_sample_id[fname] = s.sample_id

        logger.info("cppcheck scanning %d snippet(s) in one batch invocation...", len(samples))
        raw_results = _run_cppcheck(".", timeout=timeout, include_dir=None, cwd=str(tmp_dir))

        findings = []
        unmatched = 0
        for i, r in enumerate(raw_results):
            fname = Path(r.get("path", "")).name
            sample_id = fname_to_sample_id.get(fname)
            if sample_id is None:
                unmatched += 1
                continue
            findings.append(result_to_finding(r, sample_id, i, snippet_line=r.get("line")))
        if unmatched:
            logger.warning("%d result(s) didn't map to a known sample filename", unmatched)
        return findings
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def result_to_finding(result: dict, sample_id: str, finding_idx: int,
                       snippet_line: Optional[int] = None) -> Finding:
    return Finding(
        finding_id=f"cppcheck_{sample_id}_{finding_idx}",
        sample_id=sample_id,
        tool="cppcheck",
        rule_id=result.get("id", "unknown"),
        cwe=f"CWE-{result['cwe']}" if result.get("cwe") else None,
        severity=result.get("severity"),
        message=result.get("message", ""),
        file_line=result.get("line"),
        snippet_line=snippet_line,
        extra={},
    )
