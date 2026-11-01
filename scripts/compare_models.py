"""
Side-by-side comparison of two or more models' verdict files -- the multi-model
table a paper needs to turn "one model's behaviour" into "a study". Produces,
from ONLY the (small) verdicts files:

  * a summary row per model (n, overall accuracy, ECE, and the 3-band
    high/mid/low confidence accuracies), and
  * a per-CWE accuracy table with one column per model,

so you can see at a glance whether the capability map and the
calibrated-but-overconfident pattern hold ACROSS models (generalisation) or are
an artefact of one.

Usage:
    python scripts/compare_models.py \\
        qwen7b=data/verdicts_eval_v2.jsonl \\
        llama8b=data/verdicts_eval_llama.jsonl \\
        [--min-n 6] [--samples data/devign.jsonl]

--samples is required for datasets whose sample_id does not encode the label
(Devign); labels are then read from that Phase-1 samples file.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_verdicts import (  # noqa: E402
    load_records, load_label_map, expected_calibration_error, calibration_three_bins,
)
from statistical_analysis import mcnemar_exact  # noqa: E402


def _model_stats(recs):
    n = len(recs)
    overall = sum(1 for r in recs if r["pred"] == r["label"]) / n if n else float("nan")
    ece, _ = expected_calibration_error(
        [r["conf"] for r in recs], [int(r["pred"] == r["label"]) for r in recs])
    bands = {name: acc for name, _, acc in calibration_three_bins(recs)}
    by_cwe = defaultdict(list)
    for r in recs:
        by_cwe[r["cwe"] or "unknown"].append(r)
    cwe_acc = {c: (sum(1 for r in g if r["pred"] == r["label"]) / len(g), len(g))
               for c, g in by_cwe.items()}
    return {"n": n, "overall": overall, "ece": ece, "bands": bands, "cwe_acc": cwe_acc}


def _fmt(x):
    return "  n/a" if x != x else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", metavar="NAME=verdicts.jsonl",
                    help="One or more NAME=path pairs.")
    ap.add_argument("--samples", default=None,
                    help="Phase-1 samples jsonl to read labels from, for datasets "
                         "whose sample_id does not encode the label (e.g. Devign).")
    ap.add_argument("--min-n", type=int, default=6)
    args = ap.parse_args()

    label_map = load_label_map(args.samples) if args.samples else None
    stats, records = {}, {}
    for spec in args.models:
        if "=" not in spec:
            raise SystemExit(f"Expected NAME=path, got: {spec!r}")
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            raise SystemExit(
                f"Verdicts file not found for model {name!r}: {path}\n"
                "  Verdicts (.jsonl) are run outputs and are git-ignored, so they are\n"
                "  NOT shipped in the source archive -- you generate them with\n"
                "  pipeline/run_llm_stage.py (or the Colab notebook). To locate an\n"
                "  existing one:  find ~ -name '*.jsonl' | grep -i verdict\n"
                "  Then pass its real path, or drop the model to compare fewer.")
        recs, _ = load_records(path, label_map)
        if not recs:
            raise SystemExit(
                f"No usable, labelled verdicts in {path}"
                + ("" if label_map else
                   " (for Devign-style files whose sample_id does not encode the"
                   " label, add --samples data/devign.jsonl)"))
        stats[name] = _model_stats(recs)
        records[name] = recs
    names = list(stats)

    # --- summary ---
    print("=== Model summary ===")
    print(f"{'model':<12} {'n':>5} {'accuracy':>9} {'ECE':>6} "
          f"{'conf>=.9':>9} {'.7-.9':>7} {'<.7':>7}")
    for nm in names:
        s = stats[nm]
        print(f"{nm:<12} {s['n']:>5} {s['overall']:>9.3f} {s['ece']:>6.3f} "
              f"{_fmt(s['bands'].get('high (>=0.90)', float('nan'))):>9} "
              f"{_fmt(s['bands'].get('mid (0.70-0.90)', float('nan'))):>7} "
              f"{_fmt(s['bands'].get('low (<0.70)', float('nan'))):>7}")
    print("\n(ECE: 0 = perfectly calibrated; a high ECE with monotone bands = "
          "rank-informative but overconfident.)")

    # --- per-CWE, side by side ---
    all_cwes = set()
    for nm in names:
        all_cwes |= {c for c, (_, n) in stats[nm]["cwe_acc"].items() if n >= args.min_n}
    order = sorted(all_cwes, key=lambda c: -stats[names[0]]["cwe_acc"].get(c, (0, 0))[0])

    print(f"\n=== Per-CWE accuracy (n >= {args.min_n} in some model), "
          f"sorted by {names[0]} ===")
    print(f"{'CWE':<10} " + "  ".join(f"{nm:>14}" for nm in names))
    for c in order:
        cells = []
        for nm in names:
            acc, n = stats[nm]["cwe_acc"].get(c, (float("nan"), 0))
            cells.append(f"{_fmt(acc)} (n={n})" if n else f"{'--':>14}")
        print(f"{c:<10} " + "  ".join(f"{cell:>14}" for cell in cells))

    # --- paired significance + scaling analysis (first two models) ---
    if len(names) >= 2:
        a, b = names[0], names[1]
        ma = {r["sid"]: r for r in records[a]}
        mb = {r["sid"]: r for r in records[b]}
        shared = sorted(set(ma) & set(mb))
        if shared:
            y_true = [ma[s]["label"] for s in shared]
            pa = [ma[s]["pred"] for s in shared]
            pb = [mb[s]["pred"] for s in shared]
            n10, n01, p = mcnemar_exact(y_true, pa, pb)
            sig = "significant" if p < 0.05 else "NOT significant"
            print(f"\n=== Paired McNemar: {a} vs {b} ({len(shared)} shared samples) ===")
            print(f"  {a} uniquely-correct={n10}, {b} uniquely-correct={n01}, "
                  f"p={p:.4g}  ({sig} at α=0.05)")

        # which CWEs did the second model recover, and which stayed weak in BOTH?
        gains, persistent = [], []
        for c in all_cwes:
            aa, na = stats[a]["cwe_acc"].get(c, (float("nan"), 0))
            bb, nb = stats[b]["cwe_acc"].get(c, (float("nan"), 0))
            if na and nb:
                gains.append((bb - aa, c, aa, bb))
                if aa < 0.5 and bb < 0.5:
                    persistent.append((c, aa, bb))
        gains.sort(reverse=True)
        print(f"\n=== Scaling: largest accuracy gains ({b} - {a}) ===")
        for d, c, aa, bb in gains[:5]:
            print(f"  {c:<10} {aa:.3f} -> {bb:.3f}   (+{d:.3f})")
        if persistent:
            print(f"\n=== Persistent weaknesses (accuracy < 0.5 in BOTH models) ===")
            for c, aa, bb in persistent:
                print(f"  {c:<10} {a}={aa:.3f}  {b}={bb:.3f}   <- capacity does NOT fix this")
            print("  ^ These are the model's FUNDAMENTAL limits (conceptual, not size):"
                  " the paper's strongest capability finding.")

    print("\nRead: gains that appear at 14B but not 7B are capacity-limited; "
          "weaknesses that persist at BOTH sizes are fundamental to the approach "
          "-- that separation is the paper's key scaling result.")


if __name__ == "__main__":
    main()
