"""
Fits hybrid_scoring.py's w_llm on the VALIDATION split against real ground
truth, instead of trusting the w_llm=0.7 default forever (see hybrid_scoring.py's
own docstring and the README's Phase 4 "What's NOT done" section, which both
call this out as the next step once real -- not simulated -- verdicts exist).

Deliberately a plain grid search over a metric on held-out data rather than
a fitted logistic regression: with only one free parameter (w_llm, since
static_p and llm_p are both already scaled 0-1 probabilities, not raw
features), grid search is simpler, has no extra dependency (no sklearn/numpy
needed), and is exactly as correct here as gradient-based fitting would be --
a smooth 1-D grid over [0, 1] finds the same optimum, and printing every
point additionally lets you SEE the whole curve, not just the argmax.

CRITICAL: this must run on the VAL split, never the test split -- test is
held out specifically so the final Phase 5 report is an honest, unbiased
number. If you fit w_llm on test and then report Phase 5 metrics on the
same test split, that's leakage of exactly the kind splitter.py's grouped
split was built to prevent at the data level; doing it here at the
hyperparameter level would quietly undo that protection.

Usage:
    python fit_hybrid_weight.py --samples data/combined.jsonl \\
        --findings data/findings.jsonl --verdicts data/verdicts.jsonl --split val
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))
from schema import read_jsonl as read_samples
from findings_schema import Finding, read_jsonl as read_findings
from llm_stage.llm_schema import read_jsonl as read_verdicts, by_sample as verdicts_by_sample
from hybrid_scoring import hybrid_score
from evaluate import compute_metrics, Metrics  # reuse the exact same metric definitions as Phase 5


def grid_search_w_llm(samples, findings_by_sample, verdict_map, prior: float,
                       threshold: float, metric: str, step: float = 0.05):
    import math
    if not math.isfinite(step) or not 0 < step <= 1:
        raise ValueError('step must be finite and in (0, 1]')
    with_verdict = [s for s in samples if s.sample_id in verdict_map
                     and verdict_map[s.sample_id].error is None]
    if not with_verdict:
        raise SystemExit("No usable verdicts on this split -- run run_llm_stage.py "
                          "against the val split first.")

    y_true = [s.label for s in with_verdict]
    rows = []
    w = 0.0
    while w <= 1.0 + 1e-9:
        w_clamped = min(w, 1.0)
        y_pred = []
        for s in with_verdict:
            v = verdict_map[s.sample_id]
            result = hybrid_score(s.sample_id, findings_by_sample.get(s.sample_id, []), v,
                                   prior=prior, w_llm=w_clamped, threshold=threshold)
            y_pred.append(result.label)
        m = compute_metrics(y_true, y_pred)
        rows.append((round(w_clamped, 2), m))
        w += step

    key_fn = {"accuracy": lambda m: m.accuracy, "f1": lambda m: m.f1,
              "precision": lambda m: m.precision, "recall": lambda m: m.recall}[metric]
    # Exclude nan rows (e.g. w_llm=0 with zero predicted positives -> undefined
    # precision/F1) from the argmax itself: NaN poisons Python's max() silently
    # if it's the first element seen, since `x > nan` is always False -- caught
    # by testing, not by inspection. Still shown in the printed table below.
    scored_rows = [(w, m) for w, m in rows if key_fn(m) == key_fn(m)]  # drops NaNs
    if not scored_rows:
        raise SystemExit(f"Every w_llm in the grid produced an undefined {metric} "
                          f"(likely too few positive predictions anywhere) -- try "
                          f"--metric accuracy instead, or check your data.")
    best_score = max(key_fn(m) for _, m in scored_rows)
    # Ties happen often near the top of the curve (thresholding is a step
    # function, so once the LLM dominates the sign of every sample's score,
    # more w_llm changes nothing). Prefer the SMALLEST tied w_llm: it keeps
    # some contribution from static evidence rather than a hybrid score that
    # has quietly become "ignore Phase 2 entirely."
    best_w, best_m = min((r for r in scored_rows if key_fn(r[1]) == best_score),
                          key=lambda r: r[0])
    return rows, best_w, best_m, len(with_verdict)


def _fmt(x: float) -> str:
    return "   n/a" if x != x else f"{x:6.3f}"  # x != x is the nan check


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--findings", default=None)
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test", ""],
                     help="Fit on val (default). Never fit on test -- that's what "
                          "Phase 5 reports on, and fitting there would be leakage.")
    ap.add_argument("--prior", type=float, default=0.4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--metric", choices=["accuracy", "f1", "precision", "recall"], default="f1",
                     help="Which metric to maximize (default f1, a reasonable balance "
                          "unless your thesis has a specific reason to prefer another).")
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()

    if args.split != 'val':
        ap.error('Calibration requires --split val; test and mixed partitions are forbidden')

    samples = read_samples(args.samples)
    if args.split:
        samples = [s for s in samples if s.split == args.split]

    findings_by_sample: Dict[str, List[Finding]] = defaultdict(list)
    if args.findings:
        for f in read_findings(args.findings):
            findings_by_sample[f.sample_id].append(f)

    verdict_map = verdicts_by_sample(read_verdicts(args.verdicts))

    rows, best_w, best_m, n = grid_search_w_llm(
        samples, findings_by_sample, verdict_map,
        prior=args.prior, threshold=args.threshold, metric=args.metric, step=args.step,
    )

    print(f"Fitting w_llm on {n} {args.split}-split samples with a usable verdict "
          f"(maximizing {args.metric})\n")
    print(f"{'w_llm':>6} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    for w, m in rows:
        marker = "  <-- best" if w == best_w else ""
        print(f"{w:>6.2f} {m.accuracy:>9.3f} {_fmt(m.precision):>10} {_fmt(m.recall):>8} {_fmt(m.f1):>8}{marker}")

    print(f"\nBest w_llm = {best_w} ({args.metric}={getattr(best_m, args.metric):.3f} on "
          f"{n} val samples)")
    print(f"Next step: re-run evaluate.py on --split test with --w-llm {best_w} "
          f"to get your final, honestly-held-out Phase 5 numbers.")


if __name__ == "__main__":
    main()
