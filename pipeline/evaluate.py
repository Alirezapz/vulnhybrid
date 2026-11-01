"""
Phase 5: the evaluation harness. Computes the proposal's exact defined
metrics (Accuracy, Precision, Recall, F1) for three configurations on the
same held-out test set:

  1. static-only:  did ANY security-relevant static finding fire?
  2. llm-only:     what did the LLM verdict say, ignoring static evidence?
  3. hybrid:       Phase 4's combined score, thresholded.

This three-way comparison is the actual evidence for the proposal's
central claim ("hybrid outperforms either approach alone") -- the proposal
asserts this; this script is what lets you measure whether it's true on
your data, rather than asserting it.

Usage:
    python evaluate.py --samples data/combined.jsonl --findings data/findings.jsonl \\
        --verdicts data/verdicts.jsonl --split test

If --verdicts is omitted (Phase 3 hasn't been run yet, or only ran on a
subset), llm-only and hybrid metrics are computed only over whatever
samples DO have a verdict, with the sample count reported explicitly --
never silently averaged over a different denominator than static-only.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from schema import Sample, read_jsonl as read_samples
from findings_schema import Finding, read_jsonl as read_findings, group_by_sample
from llm_stage.llm_schema import Verdict, read_jsonl as read_verdicts, by_sample as verdicts_by_sample
from hybrid_scoring import hybrid_score, SECURITY_RELEVANT_RULE_IDS


@dataclass
class Metrics:
    n: int
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total else float("nan")

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else float("nan")

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else float("nan")

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denom if denom else 0.0

    def __str__(self) -> str:
        return (f"n={self.n:6d}  TP={self.tp:5d} FP={self.fp:5d} TN={self.tn:5d} FN={self.fn:5d}  "
                f"Accuracy={self.accuracy:.3f}  Precision={self.precision:.3f}  "
                f"Recall={self.recall:.3f}  F1={self.f1:.3f}")


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Metrics:
    if len(y_true) != len(y_pred):
        raise ValueError("Ground truth and predictions must have the same length")
    if any(type(v) not in (int, bool) or v not in (0, 1) for v in [*y_true, *y_pred]):
        raise ValueError("Metrics require binary labels and predictions")
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return Metrics(n=len(y_true), tp=tp, fp=fp, tn=tn, fn=fn)


def static_only_predict(findings: List[Finding]) -> int:
    return int(any(f.rule_id in SECURITY_RELEVANT_RULE_IDS for f in findings))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--findings", default=None)
    ap.add_argument("--verdicts", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test", ""])
    ap.add_argument("--prior", type=float, default=0.4,
                     help="Base vulnerability rate used by hybrid scoring (default matches "
                          "this project's combined Juliet+Devign rate, ~40%%)")
    ap.add_argument("--w-llm", type=float, default=0.7)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    samples = read_samples(args.samples)
    if args.split:
        samples = [s for s in samples if s.split == args.split]
    print(f"Evaluating on {len(samples)} samples (split={args.split or 'all'})\n")

    findings_by_sample: Dict[str, List[Finding]] = defaultdict(list)
    if args.findings:
        for f in read_findings(args.findings):
            findings_by_sample[f.sample_id].append(f)

    verdict_map: Dict[str, Verdict] = {}
    if args.verdicts:
        verdict_map = verdicts_by_sample(read_verdicts(args.verdicts))

    # --- Static-only: every sample in the split has a defined prediction ---
    y_true_static, y_pred_static = [], []
    for s in samples:
        y_true_static.append(s.label)
        y_pred_static.append(static_only_predict(findings_by_sample.get(s.sample_id, [])))
    m_static = compute_metrics(y_true_static, y_pred_static)

    print("=== Static-only ===")
    print(m_static)

    # --- LLM-only and hybrid: restricted to samples that actually have a verdict ---
    with_verdict = [s for s in samples if s.sample_id in verdict_map
                     and verdict_map[s.sample_id].error is None]
    if not with_verdict:
        print("\n=== LLM-only / Hybrid ===")
        print("No usable verdicts found for this split -- run run_llm_stage.py first, "
              "or check --verdicts/--split match what you ran it on.")
        return

    y_true_llm, y_pred_llm = [], []
    y_true_hybrid, y_pred_hybrid = [], []
    for s in with_verdict:
        v = verdict_map[s.sample_id]
        y_true_llm.append(s.label)
        y_pred_llm.append(int(bool(v.is_vulnerable)))

        result = hybrid_score(s.sample_id, findings_by_sample.get(s.sample_id, []), v,
                               prior=args.prior, w_llm=args.w_llm, threshold=args.threshold)
        y_true_hybrid.append(s.label)
        y_pred_hybrid.append(result.label)

    m_llm = compute_metrics(y_true_llm, y_pred_llm)
    m_hybrid = compute_metrics(y_true_hybrid, y_pred_hybrid)
    m_static_matched = compute_metrics(
        y_true_llm,
        [static_only_predict(findings_by_sample.get(s.sample_id, [])) for s in with_verdict],
    )

    coverage_note = (f"  ({len(with_verdict)}/{len(samples)} samples have a verdict -- "
                      f"compare cautiously against static-only's full-split numbers above "
                      f"if this is a small fraction)")
    print(f"\n=== LLM-only ==={coverage_note}")
    print(m_llm)
    print(f"\n=== Hybrid (w_llm={args.w_llm}, prior={args.prior}, threshold={args.threshold}) ===")
    print(m_hybrid)

    print("\n=== Matched-subset comparison (conditional on usable verdict) ===")
    print(f"{'Config':<14} {'n':>7} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    for name, m in (("static-only", m_static_matched), ("llm-only", m_llm), ("hybrid", m_hybrid)):
        print(f"{name:<14} {m.n:>7} {m.accuracy:>9.3f} {m.precision:>10.3f} {m.recall:>8.3f} {m.f1:>8.3f}")
    print(f"Coverage: {len(with_verdict)}/{len(samples)}. Missing/failed verdicts: "
          f"{len(samples) - len(with_verdict)}. This table does not measure full-pipeline recall.")


if __name__ == "__main__":
    main()
