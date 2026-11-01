"""
Publication-grade statistics for the three-way comparison (static / LLM / hybrid).

`evaluate.py` reports point estimates. A journal reviewer will (rightly) ask:
is the hybrid's gain over LLM-only statistically significant, and what are the
confidence intervals? This script answers both, on the SAME paired subset (the
samples that have a usable verdict), so the comparison is apples-to-apples:

  * bootstrap 95% CIs for Accuracy / Precision / Recall / F1, per configuration;
  * McNemar's exact test on the paired predictions for hybrid-vs-LLM-only and
    hybrid-vs-static-only (the correct test for comparing two classifiers on the
    same instances).

Usage:
    python scripts/statistical_analysis.py --samples data/combined.jsonl \\
        --findings data/findings.jsonl --verdicts data/verdicts.jsonl --split test

The `stat` functions at the top are pure and unit-tested
(tests/test_statistics.py); everything below `main` just wires in the data via
the same loaders `evaluate.py` uses.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from math import comb
from typing import Callable, List, Tuple

_PIPELINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline")
if os.path.isdir(_PIPELINE) and _PIPELINE not in sys.path:
    sys.path.insert(0, _PIPELINE)


# ---------------------------------------------------------------------------
# Pure statistics (no I/O) -- unit-tested
# ---------------------------------------------------------------------------


def bootstrap_ci(y_true: List[int], y_pred: List[int],
                 metric_fn: Callable[[List[int], List[int]], float],
                 n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> Tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high) for a metric via the
    percentile bootstrap. NaN resamples (e.g. undefined precision when a
    resample has no predicted positives) are dropped."""
    n = len(y_true)
    point = metric_fn(y_true, y_pred)
    if n == 0:
        return point, float("nan"), float("nan")
    rng = random.Random(seed)
    vals: List[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        v = metric_fn(yt, yp)
        if v == v:  # not NaN
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    vals.sort()
    lo = vals[max(0, int((alpha / 2) * len(vals)))]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return point, lo, hi


def mcnemar_exact(y_true: List[int], pred_a: List[int],
                  pred_b: List[int]) -> Tuple[int, int, float]:
    """Exact (binomial) two-sided McNemar test comparing classifier A vs B on
    the same instances. Returns (n_a_only_correct, n_b_only_correct, p_value).

    Counts the discordant pairs -- cases where exactly one of A/B is correct --
    and tests them against a fair coin. The exact binomial is used (not the
    chi-square approximation), which is correct even for the small discordant
    counts typical of a few-hundred-sample eval."""
    n10 = n01 = 0  # n10: A correct & B wrong ; n01: B correct & A wrong
    for t, a, b in zip(y_true, pred_a, pred_b):
        a_ok, b_ok = (a == t), (b == t)
        if a_ok and not b_ok:
            n10 += 1
        elif b_ok and not a_ok:
            n01 += 1
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    # two-sided exact p = 2 * P(X <= k) under X ~ Binomial(n, 0.5), capped at 1
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return n10, n01, min(1.0, 2.0 * tail)


def label_from_sample_id(sample_id: str):
    """Ground-truth label from a Juliet sample_id's suffix (``..._bad`` -> 1,
    ``..._good`` -> 0), or ``None`` if it can't be determined (e.g. Devign ids,
    which carry no such suffix). This is the exact "recompute from raw verdicts
    without touching combined.jsonl" check the thesis used for verification."""
    sid = sample_id.rstrip()
    if sid.endswith("bad"):
        return 1
    if sid.endswith("good"):
        return 0
    return None


# ---------------------------------------------------------------------------
# Data wiring (mirrors evaluate.py) + reporting
# ---------------------------------------------------------------------------


def _load_paired_predictions(args):
    from collections import defaultdict
    from schema import read_jsonl as read_samples
    from findings_schema import read_jsonl as read_findings
    from llm_stage.llm_schema import read_jsonl as read_verdicts, by_sample as verdicts_by_sample
    from hybrid_scoring import hybrid_score
    from evaluate import static_only_predict

    samples = read_samples(args.samples)
    if args.split:
        samples = [s for s in samples if s.split == args.split]

    findings_by_sample = defaultdict(list)
    if args.findings:
        for f in read_findings(args.findings):
            findings_by_sample[f.sample_id].append(f)
    verdict_map = verdicts_by_sample(read_verdicts(args.verdicts))

    # Paired subset: only samples with a usable verdict, so all three configs
    # predict on the SAME instances (fixes evaluate.py's mismatched denominators).
    subset = [s for s in samples if s.sample_id in verdict_map
              and verdict_map[s.sample_id].error is None]

    y_true, y_static, y_llm, y_hybrid = [], [], [], []
    for s in subset:
        v = verdict_map[s.sample_id]
        fnds = findings_by_sample.get(s.sample_id, [])
        y_true.append(s.label)
        y_static.append(static_only_predict(fnds))
        y_llm.append(int(bool(v.is_vulnerable)))
        hr = hybrid_score(s.sample_id, fnds, v, prior=args.prior,
                          w_llm=args.w_llm, threshold=args.threshold)
        y_hybrid.append(hr.label)
    return y_true, {"static-only": y_static, "llm-only": y_llm, "hybrid": y_hybrid}


def _load_from_verdicts_only(args):
    """Lightweight path: reconstruct the LLM-only vs hybrid comparison from JUST
    the verdicts file, deriving labels from the Juliet _bad/_good id suffix. No
    combined.jsonl / findings.jsonl needed (handy when those are large/remote).

    The hybrid here is the confidence-thresholded prediction (p = confidence if
    flagged else 1 - confidence, thresholded) -- i.e. hybrid at w_llm = 1.0,
    which is exactly the value the thesis fitted, so this reproduces the
    headline hybrid-vs-LLM comparison. static-only is omitted (it needs the full
    findings, and on the verify-only subset it is trivially all-positive)."""
    from llm_stage.llm_schema import read_jsonl as read_verdicts

    y_true, y_llm, y_hybrid = [], [], []
    n_unlabeled = 0
    for v in read_verdicts(args.verdicts):
        if v.error is not None or v.is_vulnerable is None:
            continue
        label = label_from_sample_id(v.sample_id)
        if label is None:
            n_unlabeled += 1
            continue
        conf = v.confidence if v.confidence is not None else 0.5
        llm_p = conf if v.is_vulnerable else (1.0 - conf)
        y_true.append(label)
        y_llm.append(int(bool(v.is_vulnerable)))
        y_hybrid.append(int(llm_p >= args.threshold))
    if n_unlabeled:
        print(f"(skipped {n_unlabeled} verdict(s) whose label couldn't be derived "
              f"from the id -- e.g. Devign samples)\n")
    return y_true, {"llm-only": y_llm, "hybrid": y_hybrid}


def main():
    from evaluate import compute_metrics  # reuse the exact metric definitions

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default=None,
                    help="Phase-1 combined.jsonl. Omit for the lightweight path "
                         "that reconstructs labels from the verdict ids and needs "
                         "ONLY --verdicts (no big combined/findings download).")
    ap.add_argument("--findings", default=None)
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test", ""])
    ap.add_argument("--prior", type=float, default=0.4)
    ap.add_argument("--w-llm", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.samples:
        y_true, preds = _load_paired_predictions(args)
    else:
        print("No --samples given: reconstructing labels from verdict ids "
              "(Juliet _bad/_good). Comparing llm-only vs hybrid only.\n")
        y_true, preds = _load_from_verdicts_only(args)

    n = len(y_true)
    if n == 0:
        print("No paired samples with a usable verdict -- run the LLM stage first "
              "(or, in the lightweight path, the verdicts have no derivable labels).")
        return

    metrics = {
        "Accuracy": lambda yt, yp: compute_metrics(yt, yp).accuracy,
        "Precision": lambda yt, yp: compute_metrics(yt, yp).precision,
        "Recall": lambda yt, yp: compute_metrics(yt, yp).recall,
        "F1": lambda yt, yp: compute_metrics(yt, yp).f1,
    }

    print(f"Paired statistical analysis on {n} samples with a usable verdict "
          f"(split={args.split or 'all'}, {args.n_boot} bootstrap resamples)\n")
    print(f"{'Config':<12} " + "  ".join(f"{m:>22}" for m in metrics))
    print("-" * 12 + "  " + "  ".join("-" * 22 for _ in metrics))
    for name, y_pred in preds.items():
        cells = []
        for _, fn in metrics.items():
            pt, lo, hi = bootstrap_ci(y_true, y_pred, fn, n_boot=args.n_boot, seed=args.seed)
            cells.append(f"{pt:0.3f} [{lo:0.3f},{hi:0.3f}]")
        print(f"{name:<12} " + "  ".join(f"{c:>22}" for c in cells))

    print("\nMcNemar's exact test (paired predictions):")
    for a, b in (("hybrid", "llm-only"), ("hybrid", "static-only")):
        if a not in preds or b not in preds:
            continue
        n10, n01, p = mcnemar_exact(y_true, preds[a], preds[b])
        verdict = "significant" if p < 0.05 else "NOT significant"
        print(f"  {a} vs {b}: {a} uniquely-correct={n10}, {b} uniquely-correct={n01}, "
              f"p={p:.4f}  ({verdict} at α=0.05)")

    print("\nInterpretation: if the hybrid-vs-llm-only CIs overlap heavily AND the "
          "McNemar p-value is > 0.05, the headline gain is not statistically "
          "established at this sample size -- raise --max-per-cwe in the LLM stage "
          "and/or add more models before claiming superiority.")


if __name__ == "__main__":
    main()
