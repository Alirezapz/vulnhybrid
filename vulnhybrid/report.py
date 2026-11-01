"""
Render a ScanReport in the format the situation calls for:

  * ``terminal`` -- a coloured, human-readable summary for the console.
  * ``json``     -- the full structured report (machine-readable, CI-friendly).
  * ``markdown`` -- a self-contained report document (attach to a PR, paste
    into a thesis appendix).
  * ``sarif``    -- SARIF 2.1.0 for GitHub Code Scanning / any SARIF viewer.

The honesty the thesis insists on is carried through here: a unit's *status*
(confirmed / review / cleared) is shown as prominently as its severity, so a
static-only run reads as "here are N candidates to review", never as a false
"N confirmed vulnerabilities".
"""

from __future__ import annotations

import json
import os
import textwrap
from typing import List

from .scanner import ScanReport, UnitResult

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_STATUS_LABEL = {
    "confirmed": "CONFIRMED",
    "uncertain": "UNCERTAIN",
    "review": "REVIEW",
    "cleared": "CLEARED",
    "clear": "CLEAR",
}

_ANSI = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "red": "\033[31m", "yellow": "\033[33m", "green": "\033[32m",
    "cyan": "\033[36m", "magenta": "\033[35m", "grey": "\033[90m",
    "bred": "\033[91m", "byellow": "\033[93m",
}

_SEVERITY_COLOR = {
    "critical": "bred", "high": "red", "medium": "yellow", "low": "cyan",
    "none": "grey",
}
_STATUS_COLOR = {
    "confirmed": "red", "uncertain": "byellow", "review": "yellow",
    "cleared": "green", "clear": "grey",
}


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return True


def _c(text: str, color: str, use_color: bool) -> str:
    if not use_color or color not in _ANSI:
        return text
    return f"{_ANSI[color]}{text}{_ANSI['reset']}"


def _wrap(text: str, width: int, indent: str) -> str:
    text = " ".join((text or "").split())
    if not text:
        return ""
    return "\n".join(textwrap.wrap(text, width=width,
                                   initial_indent=indent, subsequent_indent=indent))


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------


def render_terminal(report: ScanReport, show_all: bool = False,
                    use_color: bool = True, width: int = 88) -> str:
    use_color = use_color and _supports_color()
    out: List[str] = []

    def line(s: str = ""):
        out.append(s)

    title = "vulnhybrid — hybrid C/C++ vulnerability scan"
    line(_c(title, "bold", use_color))
    line(_c("─" * min(width, len(title) + 4), "grey", use_color))
    line(f"target        : {report.target}")
    line(f"mode          : {report.mode}   tools: {', '.join(report.tools) or 'none'}")
    if report.provider:
        line(f"model         : {report.provider}:{report.model}   "
             f"(w_llm={report.w_llm}, threshold={report.threshold})")
    line(f"scanned       : {report.files_scanned} files, "
         f"{report.functions_total} functions, {report.units_analyzed} analyzed")
    if report.llm_calls:
        line(f"llm           : {report.llm_calls} calls, {report.llm_errors} errors, "
             f"{report.input_tokens} in / {report.output_tokens} out tokens")
    line(f"time          : {report.duration_s:.2f}s")

    sc = report.status_counts()
    summary_bits = []
    for st in ("confirmed", "uncertain", "review", "cleared"):
        if sc.get(st):
            summary_bits.append(_c(f"{sc[st]} {_STATUS_LABEL[st].lower()}",
                                   _STATUS_COLOR[st], use_color))
    line()
    line("result        : " + ("  ".join(summary_bits) if summary_bits
                               else _c("nothing to report", "green", use_color)))

    for note in report.notes:
        line(_c(f"  note: {note}", "grey", use_color))

    shown = report.units if show_all else [
        u for u in report.units if u.status in ("confirmed", "uncertain", "review")]

    if shown:
        line()
        line(_c("─" * width, "grey", use_color))
    for u in shown:
        _render_unit_terminal(u, out, use_color, width)

    if not shown:
        line()
        line(_c("No confirmed vulnerabilities or candidates to review.",
                "green", use_color))

    line()
    return "\n".join(out)


def _render_unit_terminal(u: UnitResult, out: List[str], use_color: bool, width: int):
    status_col = _STATUS_COLOR.get(u.status, "grey")
    sev_col = _SEVERITY_COLOR.get(u.severity, "grey")
    bullet = {"confirmed": "●", "uncertain": "◐", "review": "○",
              "cleared": "✓", "clear": "·"}.get(u.status, "·")

    head = (f"{_c(bullet, status_col, use_color)} "
            f"{_c(_STATUS_LABEL.get(u.status, u.status), status_col, use_color)}  "
            f"{_c(u.severity.upper(), sev_col, use_color)}  "
            f"score {u.score:.2f}")
    if u.cwe:
        head += f"  {_c(u.cwe, 'magenta', use_color)}"
    out.append("")
    out.append(head)

    loc = f"  {u.file}:{u.start_line}  {_c(u.function + '()', 'cyan', use_color)}"
    if u.kind == "window":
        loc += _c("  [context window — no enclosing function recovered]", "grey", use_color)
    out.append(loc)

    # static evidence
    for f in u.findings:
        ln = f.file_line or f.snippet_line
        out.append(_c(f"  static : {f.tool} {f.rule_id}"
                      f"{(' ' + f.cwe) if f.cwe else ''} @ line {ln}", "grey", use_color))

    # model reasoning
    v = u.verdict
    if v is not None and v.error is None:
        verdict_word = "vulnerable" if v.is_vulnerable else "not vulnerable"
        conf = f" (confidence {v.confidence:.2f})" if v.confidence is not None else ""
        out.append(f"  model  : {verdict_word}{conf}"
                   + (f" — taint: {v.taint_source}" if v.taint_source else ""))
        if v.explanation:
            out.append(_wrap(v.explanation, width, "           "))
        if u.predicted_vulnerable and v.suggested_fix:
            out.append(_c("  fix    :", "green", use_color))
            out.append(_wrap(v.suggested_fix, width, "           "))
    elif v is not None and v.error:
        out.append(_c(f"  model  : call failed ({v.error}); using static evidence only",
                      "grey", use_color))
    elif u.status == "review":
        out.append(_c("  model  : not verified — run with --provider to confirm/clear",
                      "grey", use_color))

    if u.status == "uncertain":
        out.append(_c("  note   : on the decision boundary — the model was ~50/50; "
                      "treat as needs-review, not a firm finding", "grey", use_color))

    # Confirmed even though the model's own binary verdict was "not vulnerable":
    # the calibrated static evidence (measured ~0% FP) outweighed a low-confidence
    # model hedge. Make that hybrid reasoning explicit rather than leave a
    # "CONFIRMED ... model: not vulnerable" line looking self-contradictory.
    if (u.status == "confirmed" and v is not None and v.error is None
            and v.is_vulnerable is False):
        out.append(_c("  note   : confirmed by calibrated static evidence — the model "
                      "was low-confidence here (lower --w-llm gives it more weight)",
                      "grey", use_color))


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def render_json(report: ScanReport, indent: int = 2) -> str:
    return json.dumps(report.to_dict(), indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def render_markdown(report: ScanReport) -> str:
    d = report.to_dict()
    s = d["summary"]
    lines: List[str] = []
    lines.append("# vulnhybrid scan report")
    lines.append("")
    lines.append(f"**Target:** `{report.target}`  ")
    lines.append(f"**Mode:** {report.mode} · **Tools:** {', '.join(report.tools) or 'none'}"
                 + (f" · **Model:** {report.provider}:{report.model}" if report.provider else ""))
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Files scanned | {s['files_scanned']} |")
    lines.append(f"| Functions | {s['functions_total']} |")
    lines.append(f"| Units analyzed | {s['units_analyzed']} |")
    lines.append(f"| Confirmed vulnerable | {s['status_counts'].get('confirmed', 0)} |")
    lines.append(f"| Uncertain (model ~50/50) | {s['status_counts'].get('uncertain', 0)} |")
    lines.append(f"| To review (candidates) | {s['status_counts'].get('review', 0)} |")
    lines.append(f"| Cleared by model | {s['status_counts'].get('cleared', 0)} |")
    if report.llm_calls:
        lines.append(f"| LLM calls / errors | {s['llm_calls']} / {s['llm_errors']} |")
        lines.append(f"| Tokens (in/out) | {s['input_tokens']} / {s['output_tokens']} |")
    lines.append(f"| Duration | {s['duration_s']}s |")
    lines.append("")

    for note in report.notes:
        lines.append(f"> _Note: {note}_")
    if report.notes:
        lines.append("")

    confirmed = report.by_status("confirmed")
    uncertain = report.uncertain()
    review = report.candidates()

    if confirmed:
        lines.append("## Confirmed vulnerabilities")
        lines.append("")
        for u in confirmed:
            _md_unit(u, lines)
    if uncertain:
        lines.append("## Uncertain (on the decision boundary)")
        lines.append("")
        lines.append("_Crossed the threshold, but the model was ~50/50 (its "
                     "self-reported confidence was measured to be no better than "
                     "chance here). Treat as needs-review, not a firm finding._")
        lines.append("")
        for u in uncertain:
            _md_unit(u, lines)
    if review:
        lines.append("## Candidates to review")
        lines.append("")
        lines.append("_Static analysis flagged a dangerous sink here but no model "
                     "confirmed a taint path (or none was run). Review manually or "
                     "re-run with `--provider`._")
        lines.append("")
        for u in review:
            _md_unit(u, lines)

    if not confirmed and not uncertain and not review:
        lines.append("_No confirmed vulnerabilities or candidates to review._")
        lines.append("")

    return "\n".join(lines)


def _md_unit(u: UnitResult, lines: List[str]):
    header = f"### `{u.file}:{u.start_line}` — `{u.function}()`"
    lines.append(header)
    lines.append("")
    meta = [f"**Status:** {_STATUS_LABEL.get(u.status, u.status)}",
            f"**Severity:** {u.severity}", f"**Score:** {u.score:.2f}"]
    if u.cwe:
        meta.append(f"**CWE:** {u.cwe}")
    lines.append(" · ".join(meta))
    lines.append("")
    if u.findings:
        lines.append("Static findings:")
        for f in u.findings:
            ln = f.file_line or f.snippet_line
            lines.append(f"- `{f.tool}:{f.rule_id}`"
                         + (f" ({f.cwe})" if f.cwe else "") + f" at line {ln}")
        lines.append("")
    v = u.verdict
    if v is not None and v.error is None:
        verdict_word = "vulnerable" if v.is_vulnerable else "not vulnerable"
        conf = f" (confidence {v.confidence:.2f})" if v.confidence is not None else ""
        lines.append(f"**Model verdict:** {verdict_word}{conf}"
                     + (f" — taint source: _{v.taint_source}_" if v.taint_source else ""))
        lines.append("")
        if v.explanation:
            lines.append(v.explanation)
            lines.append("")
        if u.predicted_vulnerable and v.suggested_fix:
            lines.append(f"**Suggested fix:** {v.suggested_fix}")
            lines.append("")


# ---------------------------------------------------------------------------
# SARIF 2.1.0
# ---------------------------------------------------------------------------

_SEVERITY_TO_SARIF_LEVEL = {
    "critical": "error", "high": "error", "medium": "warning",
    "low": "note", "none": "note",
}


def render_sarif(report: ScanReport, include_review: bool = True) -> str:
    """Minimal valid SARIF 2.1.0. Confirmed units map to error/warning by
    severity; review candidates map to ``note``."""
    rules: dict = {}
    results: list = []

    units = [u for u in report.units
             if u.status in ("confirmed", "uncertain")
             or (include_review and u.status == "review")]

    for u in units:
        rule_id = u.cwe or (u.findings[0].rule_id if u.findings else "vulnhybrid")
        if rule_id not in rules:
            rules[rule_id] = {
                "id": rule_id,
                "name": rule_id.replace("-", "_"),
                "shortDescription": {"text": f"Potential vulnerability ({rule_id})"},
                "helpUri": (f"https://cwe.mitre.org/data/definitions/{rule_id.split('-')[-1]}.html"
                            if rule_id.startswith("CWE-") else "https://semgrep.dev"),
            }
        level = _SEVERITY_TO_SARIF_LEVEL.get(u.severity, "note")
        if u.status == "uncertain":
            level = "warning"
        elif u.status == "review":
            level = "note"
        v = u.verdict
        text = (v.explanation if (v is not None and v.error is None and v.explanation)
                else f"{u.status}: {rule_id} in {u.function}()")
        results.append({
            "ruleId": rule_id,
            "level": level,
            "message": {"text": f"[{_STATUS_LABEL.get(u.status, u.status)}] {text}"},
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": {"uri": u.file.replace(os.sep, "/")},
                    "region": {"startLine": u.start_line, "endLine": u.end_line},
                },
            }],
            "properties": {
                "score": round(u.score, 4),
                "status": u.status,
                "severity": u.severity,
                "vulnhybrid_predicted_vulnerable": u.predicted_vulnerable,
            },
        })

    sarif = {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {
                "name": "vulnhybrid",
                "informationUri": "https://github.com/alirezapz/proj",
                "version": _version(),
                "rules": list(rules.values()),
            }},
            "results": results,
        }],
    }
    return json.dumps(sarif, indent=2, ensure_ascii=False)


def _version() -> str:
    try:
        from . import __version__
        return __version__
    except Exception:
        return "1.0.0"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def render(report: ScanReport, fmt: str, *, show_all: bool = False,
           use_color: bool = True) -> str:
    fmt = fmt.lower()
    if fmt == "terminal":
        return render_terminal(report, show_all=show_all, use_color=use_color)
    if fmt == "json":
        return render_json(report)
    if fmt in ("markdown", "md"):
        return render_markdown(report)
    if fmt == "sarif":
        return render_sarif(report)
    raise ValueError(f"Unknown format: {fmt}")
