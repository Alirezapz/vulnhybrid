"""
The ``vulnhybrid`` command-line tool.

    vulnhybrid scan path/to/code                 # static triage (no model)
    vulnhybrid scan src/ --provider ollama       # + local LLM verification
    vulnhybrid scan src/ --provider anthropic --format sarif -o out.sarif
    vulnhybrid demo                              # scan the bundled examples
    vulnhybrid version

Design: the CLI is a thin front-end over ``vulnhybrid.scanner.scan`` and
``vulnhybrid.report``. All the detection logic is the validated thesis
pipeline; this module only parses arguments, wires an ``LLMConfig``, and prints.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from . import __version__, _bootstrap

_bootstrap.activate()
from llm_stage.llm_client import LLMConfig  # noqa: E402

from . import report as report_mod          # noqa: E402
from . import scanner as scanner_mod        # noqa: E402

_DEFAULT_MODELS = {
    "ollama": "qwen2.5-coder:7b",       # the model the thesis actually evaluated
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o-mini",
}

_FAIL_ON_CHOICES = ("none", "confirmed", "review", "any")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vulnhybrid",
        description="Hybrid static-analysis + LLM vulnerability scanner for C/C++.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--version", action="version", version=f"vulnhybrid {__version__}")
    sub = p.add_subparsers(dest="command")

    scan = sub.add_parser("scan", help="Scan a file or directory of C/C++ code.",
                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    scan.add_argument("path", help="File or directory to scan.")

    grp = scan.add_argument_group("analysis")
    grp.add_argument("--mode", choices=["triage", "deep", "static"], default="triage",
                     help="triage: LLM verifies static candidates. deep: LLM on every "
                          "function. static: no LLM.")
    grp.add_argument("--tools", nargs="+", choices=["semgrep", "cppcheck"],
                     default=["semgrep"], help="Static analyzers to run.")
    grp.add_argument("--cppcheck-include-dir", default=None,
                     help="Extra include dir for cppcheck (avoids missingInclude noise).")
    grp.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                     help="Glob of paths to skip (repeatable).")
    grp.add_argument("--include-nonsecurity", action="store_true",
                     help="Also treat style/quality findings as evidence.")
    grp.add_argument("--timeout", type=int, default=3600,
                     help="Max seconds for a whole-tree static scan, per tool.")

    grp = scan.add_argument_group("model (LLM verification)")
    grp.add_argument("--provider", choices=["none", "ollama", "anthropic", "openai"],
                     default="none",
                     help="LLM backend. 'none' = static only. 'ollama' = free/local.")
    grp.add_argument("--model", default=None, help="Model id (provider-specific default).")
    grp.add_argument("--base-url", default=None,
                     help="Override endpoint (LM Studio / vLLM / remote Ollama).")
    grp.add_argument("--api-key", default=None, help="Override API key.")
    grp.add_argument("--concurrency", type=int, default=4,
                     help="Parallel model calls.")
    grp.add_argument("--max-units", type=int, default=None,
                     help="Cap functions sent to the model (cost/smoke-test guard).")

    grp = scan.add_argument_group("scoring")
    grp.add_argument("--w-llm", type=float, default=1.0,
                     help="Weight of the LLM vs static evidence (thesis-fitted = 1.0).")
    grp.add_argument("--prior", type=float, default=0.4, help="Base vulnerability rate.")
    grp.add_argument("--threshold", type=float, default=0.5,
                     help="Probability at/above which a unit is 'confirmed'.")

    grp = scan.add_argument_group("output")
    grp.add_argument("--format", choices=["terminal", "json", "markdown", "sarif"],
                     default="terminal", help="Report format.")
    grp.add_argument("-o", "--output", default=None, help="Write report to a file.")
    grp.add_argument("--all", action="store_true",
                     help="Terminal: also show cleared/clear units.")
    grp.add_argument("--no-color", action="store_true", help="Disable ANSI colour.")
    grp.add_argument("-q", "--quiet", action="store_true", help="Suppress progress.")
    grp.add_argument("--fail-on", choices=_FAIL_ON_CHOICES, default="none",
                     help="Exit non-zero if results at/above this level exist "
                          "(for CI gating).")

    sub.add_parser("demo", help="Scan the bundled example vulnerable code.")
    sub.add_parser("version", help="Print version.")
    return p


def _make_llm(args) -> Optional[LLMConfig]:
    if args.provider == "none":
        return None
    model = args.model or _DEFAULT_MODELS.get(args.provider)
    return LLMConfig(provider=args.provider, model=model,
                     base_url=args.base_url, api_key=args.api_key)


def _progress_printer(quiet: bool):
    if quiet or not sys.stderr.isatty():
        return None

    def cb(done: int, total: int):
        print(f"\r  verifying {done}/{total} …", end="", file=sys.stderr, flush=True)
        if done == total:
            print("\r" + " " * 40 + "\r", end="", file=sys.stderr, flush=True)

    return cb


def _run_scan(args) -> int:
    llm = _make_llm(args)
    use_color = not args.no_color and not os.environ.get("NO_COLOR")

    try:
        report = scanner_mod.scan(
            args.path,
            tools=tuple(args.tools),
            mode=args.mode,
            llm=llm,
            w_llm=args.w_llm,
            prior=args.prior,
            threshold=args.threshold,
            concurrency=args.concurrency,
            max_units=args.max_units,
            excludes=tuple(args.exclude),
            include_nonsecurity=args.include_nonsecurity,
            cppcheck_include_dir=args.cppcheck_include_dir,
            timeout=args.timeout,
            progress=_progress_printer(args.quiet),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    text = report_mod.render(report, args.format, show_all=args.all, use_color=use_color)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + ("\n" if not text.endswith("\n") else ""))
        if not args.quiet:
            print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(text)

    return _exit_code(report, args.fail_on)


def _exit_code(report: scanner_mod.ScanReport, fail_on: str) -> int:
    if fail_on == "none":
        return 0
    n_confirmed = len(report.by_status("confirmed"))
    n_uncertain = len(report.uncertain())
    n_review = len(report.candidates())
    if fail_on == "confirmed":
        # firm confirmations only -- a hedged 'uncertain' boundary case does
        # not fail the build here
        return 1 if n_confirmed else 0
    if fail_on in ("review", "any"):
        # anything needing a human: confirmed, uncertain, or a static candidate
        return 1 if (n_confirmed or n_uncertain or n_review) else 0
    return 0


def _run_demo(args) -> int:
    examples = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "examples", "vulnerable-code")
    if not os.path.isdir(examples):
        print("error: bundled examples not found (expected examples/vulnerable-code/).",
              file=sys.stderr)
        return 2
    print(f"Scanning bundled examples at {examples}\n"
          f"(static triage — add --provider ollama to a real 'scan' to verify)\n",
          file=sys.stderr)
    report = scanner_mod.scan(examples, mode="static")
    print(report_mod.render_terminal(report, use_color=not os.environ.get("NO_COLOR")))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "scan":
        return _run_scan(args)
    if args.command == "demo":
        return _run_demo(args)
    if args.command == "version":
        print(f"vulnhybrid {__version__}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
