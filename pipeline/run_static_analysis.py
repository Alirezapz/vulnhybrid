"""
Phase 2 entry point: runs semgrep (via static_analysis/) against a Phase 1
dataset (juliet.jsonl / devign.jsonl / combined.jsonl) and writes a
findings.jsonl, joinable back to the samples by sample_id.

Usage:
    python run_static_analysis.py --samples data/combined.jsonl \\
        --juliet-root juliet-test-suite-c --tools semgrep cppcheck \\
        --out data/findings.jsonl

Juliet samples are matched against their original file on disk (via the
file_path/line_ranges metadata Phase 1 now records) so semgrep gets full
file context instead of an isolated snippet. Devign samples (no original
file exists) are analyzed directly as snippets. --juliet-root is only
needed if your samples file contains Juliet samples.
"""

from __future__ import annotations

import argparse
import logging

from schema import read_jsonl
from findings_schema import write_jsonl as write_findings_jsonl
from static_analysis.attribute_findings import (
    run_static_analysis_on_juliet,
    run_static_analysis_on_snippets,
    run_cppcheck_on_juliet,
    run_cppcheck_on_snippets,
)

logger = logging.getLogger("run_static_analysis")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True, help="Phase 1 output jsonl")
    ap.add_argument("--juliet-root", default=None,
                     help="Root of the Juliet checkout (required if --samples "
                          "contains Juliet samples)")
    ap.add_argument("--tools", nargs="+", choices=["semgrep", "cppcheck"],
                     default=["semgrep"],
                     help="Which tool(s) to run. cppcheck is a real dataflow "
                          "analyzer but, per the README, measured to NOT catch "
                          "the array-bounds/sizeof-confusion gaps semgrep misses "
                          "-- run it for what it DOES catch, not as 'the fix'.")
    ap.add_argument("--cppcheck-include-dir", default=None,
                     help="Path to Juliet's testcasesupport/ (defines "
                          "std_testcase.h) -- optional, avoids include noise")
    ap.add_argument("--out", default="data/findings.jsonl")
    ap.add_argument("--timeout", type=int, default=3600,
                     help="Max seconds for the whole-tree scan, per tool (default 1hr)")
    args = ap.parse_args()

    samples = read_jsonl(args.samples)
    logger.info("Loaded %d samples from %s", len(samples), args.samples)

    juliet_samples = [s for s in samples if s.dataset == "juliet"]
    other_samples = [s for s in samples if s.dataset != "juliet"]

    all_findings = []
    if juliet_samples and not args.juliet_root:
        ap.error("--juliet-root is required: samples file contains Juliet samples")

    def checkpoint(stage_name: str):
        write_findings_jsonl(all_findings, args.out)
        logger.info("Checkpoint after %s: %d findings written to %s so far",
                     stage_name, len(all_findings), args.out)

    if "semgrep" in args.tools:
        if juliet_samples:
            all_findings.extend(run_static_analysis_on_juliet(
                juliet_samples, args.juliet_root, timeout=args.timeout))
            checkpoint("semgrep/juliet")
        if other_samples:
            all_findings.extend(run_static_analysis_on_snippets(
                other_samples, timeout=args.timeout))
            checkpoint("semgrep/snippets")

    if "cppcheck" in args.tools:
        if juliet_samples:
            all_findings.extend(run_cppcheck_on_juliet(
                juliet_samples, args.juliet_root,
                include_dir=args.cppcheck_include_dir, timeout=args.timeout))
            checkpoint("cppcheck/juliet")
        if other_samples:
            all_findings.extend(run_cppcheck_on_snippets(other_samples, timeout=args.timeout))
            checkpoint("cppcheck/snippets")

    write_findings_jsonl(all_findings, args.out)
    logger.info("Wrote %d findings to %s", len(all_findings), args.out)

    by_rule = {}
    for f in all_findings:
        by_rule[f.rule_id] = by_rule.get(f.rule_id, 0) + 1
    print(f"\nFindings by rule: {by_rule}")


if __name__ == "__main__":
    main()
