"""
Descriptive analysis of an LLM verdicts file -- the evidence for the paper's
*solid* contributions (which don't depend on the fragile hybrid-vs-LLM
significance): the per-CWE capability map and the confidence calibration of the
small code model. Needs ONLY the verdicts file (labels + CWE are read from the
Juliet sample_id), so it runs in seconds on the small file you already have.

Produces:
  1. Per-CWE capability map  -> reproduces & extends thesis Table 4-3.
  2. Confidence calibration   -> reproduces thesis Table 4-2, AND adds the
     Expected Calibration Error (ECE), a standard, publishable calibration
     metric the thesis does not report, plus reliability-diagram data (and a
     PNG if matplotlib is installed).

Usage:
    python scripts/analyze_verdicts.py --verdicts data/verdicts_eval_v2.jsonl \\
        [--min-n 6] [--out-dir figures]

The pure functions (cwe_from_sample_id, expected_calibration_error) are
unit-tested in tests/test_statistics.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from statistical_analysis import label_from_sample_id  # noqa: E402

_CWE_RE = re.compile(r"CWE[-_]?(\d+)")


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def cwe_from_sample_id(sample_id: str):
    """Extract a CWE id from a Juliet sample_id (e.g. juliet_CWE415_... -> CWE-415)."""
    m = _CWE_RE.search(sample_id or "")
    return f"CWE-{m.group(1)}" if m else None


def expected_calibration_error(confidences: List[float], correct: List[int],
                               n_bins: int = 10) -> Tuple[float, list]:
    """ECE = sum over confidence bins of (|bin| / N) * |accuracy - mean_confidence|.
    Returns (ece, rows) where each row is (bin_lo, bin_hi, n, mean_conf, accuracy)."""
    n = len(confidences)
    if n == 0:
        return float("nan"), []
    bins: List[list] = [[] for _ in range(n_bins)]
    for c, ok in zip(confidences, correct):
        b = min(n_bins - 1, max(0, int(c * n_bins)))
        bins[b].append((c, ok))
    ece = 0.0
    rows = []
    for i, bucket in enumerate(bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if not bucket:
            rows.append((lo, hi, 0, float("nan"), float("nan")))
            continue
        conf_mean = sum(c for c, _ in bucket) / len(bucket)
        acc = sum(ok for _, ok in bucket) / len(bucket)
        ece += (len(bucket) / n) * abs(acc - conf_mean)
        rows.append((lo, hi, len(bucket), conf_mean, acc))
    return ece, rows


# ---------------------------------------------------------------------------
# Data loading + analyses
# ---------------------------------------------------------------------------


def load_label_map(samples_path: str) -> dict:
    """sample_id -> label from a Phase-1 samples jsonl. Needed for datasets
    whose ground truth is NOT encoded in the sample_id (Devign: sample_id
    'devign_test_123' with the label carried in the record), where
    label_from_sample_id has nothing to work with."""
    m = {}
    for line in open(samples_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("sample_id") is not None and d.get("label") is not None:
            m[d["sample_id"]] = int(d["label"])
    return m


def load_records(path: str, label_map: dict = None):
    recs = []
    n_skipped = 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("error") is not None or d.get("is_vulnerable") is None:
            continue
        label = label_from_sample_id(d["sample_id"])
        if label is None and label_map:
            label = label_map.get(d["sample_id"])
        if label is None:
            n_skipped += 1
            continue
        conf = d.get("confidence")
        recs.append({
            "sid": d["sample_id"],
            "cwe": cwe_from_sample_id(d["sample_id"]),
            "label": label,
            "pred": int(bool(d["is_vulnerable"])),
            "conf": 0.5 if conf is None else float(conf),
        })
    return recs, n_skipped


def per_cwe_capability(recs, min_n: int):
    by = defaultdict(list)
    for r in recs:
        by[r["cwe"] or "unknown"].append(r)
    rows = []
    for cwe, g in by.items():
        n = len(g)
        acc = sum(1 for r in g if r["pred"] == r["label"]) / n
        pos = [r for r in g if r["label"] == 1]
        rec = (sum(1 for r in pos if r["pred"] == 1) / len(pos)) if pos else float("nan")
        rows.append((cwe, n, acc, rec))
    rows.sort(key=lambda x: (-x[2], -x[1]))
    return [r for r in rows if r[1] >= min_n], [r for r in rows if r[1] < min_n]


def calibration_three_bins(recs):
    buckets = {"high (>=0.90)": [], "mid (0.70-0.90)": [], "low (<0.70)": []}
    for r in recs:
        ok = int(r["pred"] == r["label"])
        c = r["conf"]
        key = "high (>=0.90)" if c >= 0.9 else "mid (0.70-0.90)" if c >= 0.7 else "low (<0.70)"
        buckets[key].append(ok)
    return [(name, len(v), (sum(v) / len(v) if v else float("nan")))
            for name, v in buckets.items()]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(x):
    return " n/a" if x != x else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--samples", default=None,
                    help="Phase-1 samples jsonl to read labels from, for datasets "
                         "whose sample_id does not encode the label (e.g. Devign).")
    ap.add_argument("--min-n", type=int, default=6,
                    help="Only show per-CWE rows with at least this many samples.")
    ap.add_argument("--out-dir", default=None, help="Write CSVs (+ PNG if matplotlib).")
    args = ap.parse_args()

    label_map = load_label_map(args.samples) if args.samples else None
    recs, n_skipped = load_records(args.verdicts, label_map)
    if not recs:
        if n_skipped:
            print(f"No usable verdicts with a DERIVABLE label "
                  f"({n_skipped} verdicts had no Juliet-style _bad/_good id).\n"
                  "By default this script reads the ground truth from the sample_id,\n"
                  "which only works for Juliet/SARD. For a dataset whose labels live\n"
                  "in the samples file instead (e.g. Devign), pass that file too:\n\n"
                  "    python3 scripts/analyze_verdicts.py \\\n"
                  "        --verdicts <your_verdicts>.jsonl --samples data/devign.jsonl\n\n"
                  "(or score the full static/llm/hybrid table with pipeline/evaluate.py).")
        else:
            print("No usable, labelled verdicts found (file empty or all had errors).")
        return
    overall = sum(1 for r in recs if r["pred"] == r["label"]) / len(recs)
    print(f"Analyzed {len(recs)} usable, labelled verdicts "
          f"(skipped {n_skipped} without a derivable label). "
          f"Overall LLM accuracy: {overall:.3f}\n")

    # --- 1. Per-CWE capability map ---
    shown, small = per_cwe_capability(recs, args.min_n)
    print(f"=== Per-CWE capability map (n >= {args.min_n}), strongest first ===")
    print(f"{'CWE':<10} {'n':>4} {'accuracy':>9} {'recall':>8}")
    for cwe, n, acc, rec in shown:
        print(f"{cwe:<10} {n:>4} {acc:>9.3f} {_fmt(rec):>8}")
    if small:
        print(f"\n({len(small)} more CWEs with n < {args.min_n}, omitted as too small "
              f"to read: {', '.join(c for c, *_ in small[:12])}{'...' if len(small) > 12 else ''})")

    # --- 2. Confidence calibration ---
    print("\n=== Confidence calibration (thesis Table 4-2 style) ===")
    print(f"{'confidence band':<18} {'n':>5} {'accuracy':>9}")
    for name, n, acc in calibration_three_bins(recs):
        print(f"{name:<18} {n:>5} {_fmt(acc):>9}")

    confs = [r["conf"] for r in recs]
    correct = [int(r["pred"] == r["label"]) for r in recs]
    ece, rel_rows = expected_calibration_error(confs, correct, n_bins=10)
    print(f"\nExpected Calibration Error (ECE, 10 bins): {ece:.3f}")
    print("  (0 = perfectly calibrated; lower is better. This is the standard "
          "metric to report alongside the 3-band table above.)")

    # --- optional CSV / figure ---
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "per_cwe.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["cwe", "n", "accuracy", "recall"])
            w.writerows(shown + small)
        with open(os.path.join(args.out_dir, "reliability.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["bin_lo", "bin_hi", "n", "mean_conf", "accuracy"])
            w.writerows(rel_rows)
        print(f"\nWrote per_cwe.csv and reliability.csv to {args.out_dir}/")
        _try_plot(rel_rows, ece, args.out_dir)


def _try_plot(rel_rows, ece, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  (install matplotlib to also get reliability.png: pip install matplotlib)")
        return
    xs = [(lo + hi) / 2 for lo, hi, n, *_ in rel_rows if n]
    accs = [acc for *_, n, cm, acc in rel_rows if n]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    ax.plot(xs, accs, "o-", label=f"model (ECE={ece:.3f})")
    ax.set_xlabel("model confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_title("Reliability diagram"); ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reliability.png"), dpi=150)
    print(f"  Wrote reliability.png to {out_dir}/")


if __name__ == "__main__":
    main()
