"""
Turns Phase 5's numbers into figures worth putting in the actual thesis
document, instead of just a terminal table. Reuses evaluate.py's own
compute_metrics/static_only_predict and hybrid_scoring.py's own
hybrid_score -- a presentation layer on top of already-tested Phase 5
logic, not a second implementation that could quietly drift out of sync
with it.

Produces three figures, each saved as both .png (300dpi, for slides/
drafts) and .pdf (vector art, for the final printed thesis):

  1. method_comparison   -- grouped bars: Accuracy/Precision/Recall/F1
                             for static-only vs LLM-only vs hybrid
  2. confusion_matrices   -- one annotated 2x2 heatmap per method
  3. w_llm_sensitivity    -- the fit_hybrid_weight.py grid curve, so a
                             reader can SEE why a given w_llm was chosen
                             instead of taking the argmax on faith
                             (conceptually this should be computed on the
                             VAL split, same caution as fit_hybrid_weight.py)

Usage:
    python visualize_results.py --samples data/combined.jsonl \\
        --findings data/findings.jsonl --verdicts data/verdicts_test.jsonl \\
        --split test --w-llm 0.7 --out-dir figures
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # headless -- no display in a server/sandbox environment
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from schema import read_jsonl as read_samples
from findings_schema import Finding, read_jsonl as read_findings
from llm_stage.llm_schema import read_jsonl as read_verdicts, by_sample as verdicts_by_sample
from hybrid_scoring import hybrid_score
from evaluate import compute_metrics, static_only_predict, Metrics
from fit_hybrid_weight import grid_search_w_llm

# A small, consistent palette used across all three figures so the same
# method always gets the same color if a reader flips between figures.
COLORS = {"static-only": "#8c8c8c", "llm-only": "#4C72B0", "hybrid": "#DD8452"}


def load(samples_path, findings_path, verdicts_path, split):
    samples = read_samples(samples_path)
    if split:
        samples = [s for s in samples if s.split == split]
    findings_by_sample: Dict[str, List[Finding]] = defaultdict(list)
    if findings_path:
        for f in read_findings(findings_path):
            findings_by_sample[f.sample_id].append(f)
    verdict_map = verdicts_by_sample(read_verdicts(verdicts_path)) if verdicts_path else {}
    return samples, findings_by_sample, verdict_map


def compute_all_metrics(samples, findings_by_sample, verdict_map, prior, w_llm, threshold):
    y_true_all = [s.label for s in samples]
    y_pred_static = [static_only_predict(findings_by_sample.get(s.sample_id, [])) for s in samples]
    m_static = compute_metrics(y_true_all, y_pred_static)

    with_verdict = [s for s in samples if s.sample_id in verdict_map
                     and verdict_map[s.sample_id].error is None]
    m_llm = m_hybrid = None
    if with_verdict:
        y_true_v = [s.label for s in with_verdict]
        y_pred_llm = [int(bool(verdict_map[s.sample_id].is_vulnerable)) for s in with_verdict]
        m_llm = compute_metrics(y_true_v, y_pred_llm)

        y_pred_hybrid = []
        for s in with_verdict:
            r = hybrid_score(s.sample_id, findings_by_sample.get(s.sample_id, []),
                              verdict_map[s.sample_id], prior=prior, w_llm=w_llm, threshold=threshold)
            y_pred_hybrid.append(r.label)
        m_hybrid = compute_metrics(y_true_v, y_pred_hybrid)

    return m_static, m_llm, m_hybrid, len(with_verdict)


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_method_comparison(results: Dict[str, Metrics], out_dir: Path):
    metric_names = ["accuracy", "precision", "recall", "f1"]
    methods = [m for m in ("static-only", "llm-only", "hybrid") if results.get(m) is not None]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(metric_names))
    width = 0.8 / len(methods)
    for i, method in enumerate(methods):
        m = results[method]
        values = [getattr(m, mn) for mn in metric_names]
        values = [0 if v != v else v for v in values]  # nan -> 0 for plotting only
        bars = ax.bar(x + i * width, values, width, label=f"{method} (n={m.n})",
                       color=COLORS[method], edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([m.capitalize() for m in metric_names])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Static-only vs LLM-only vs Hybrid")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out_dir, "method_comparison")


def plot_confusion_matrices(results: Dict[str, Metrics], out_dir: Path):
    methods = [m for m in ("static-only", "llm-only", "hybrid") if results.get(m) is not None]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 4))
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        m = results[method]
        mat = np.array([[m.tn, m.fp], [m.fn, m.tp]])
        im = ax.imshow(mat, cmap="Blues", vmin=0)
        for (i, j), v in np.ndenumerate(mat):
            text_color = "white" if v > mat.max() * 0.6 else "black"
            ax.text(j, i, str(v), ha="center", va="center", fontsize=13, color=text_color)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: Safe", "Pred: Vulnerable"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True: Safe", "True: Vulnerable"])
        ax.set_title(f"{method}\n(n={m.n}, acc={m.accuracy:.2f})")

    fig.tight_layout()
    _save(fig, out_dir, "confusion_matrices")


def plot_w_llm_sensitivity(samples, findings_by_sample, verdict_map, prior, threshold, out_dir: Path,
                            metric: str = "f1"):
    rows, best_w, best_m, n = grid_search_w_llm(samples, findings_by_sample, verdict_map,
                                                 prior=prior, threshold=threshold, metric=metric)
    ws = [w for w, _ in rows]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for mname, color in [("accuracy", "#4C72B0"), ("f1", "#DD8452"),
                          ("precision", "#55A868"), ("recall", "#C44E52")]:
        vals = [getattr(m, mname) for _, m in rows]
        ax.plot(ws, vals, marker="o", markersize=3, label=mname, color=color,
                linewidth=2 if mname == metric else 1, alpha=1.0 if mname == metric else 0.55)
    ax.axvline(best_w, color="black", linestyle="--", linewidth=1,
               label=f"chosen w_llm={best_w} (max {metric})")
    ax.set_xlabel("w_llm  (0 = static evidence only, 1 = LLM verdict only)")
    ax.set_ylabel("Score")
    ax.set_title(f"Hybrid weight sensitivity (n={n} samples with a verdict)")
    ax.legend(loc="lower center", ncol=3, frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "w_llm_sensitivity")
    return best_w


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--findings", default=None)
    ap.add_argument("--verdicts", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--prior", type=float, default=0.4)
    ap.add_argument("--w-llm", type=float, default=0.7)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--skip-sensitivity", action="store_true",
                     help="Skip the w_llm sensitivity plot (e.g. if --verdicts is a test-split "
                          "file and you don't want a val-only diagnostic mixed into test figures)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    samples, findings_by_sample, verdict_map = load(args.samples, args.findings, args.verdicts, args.split)
    m_static, m_llm, m_hybrid, n_v = compute_all_metrics(
        samples, findings_by_sample, verdict_map, args.prior, args.w_llm, args.threshold)

    results = {"static-only": m_static, "llm-only": m_llm, "hybrid": m_hybrid}
    plot_method_comparison(results, out_dir)
    plot_confusion_matrices(results, out_dir)
    print(f"Wrote {out_dir}/method_comparison.{{png,pdf}} and confusion_matrices.{{png,pdf}}")

    if not args.skip_sensitivity and n_v > 0:
        best_w = plot_w_llm_sensitivity(samples, findings_by_sample, verdict_map,
                                         args.prior, args.threshold, out_dir)
        print(f"Wrote {out_dir}/w_llm_sensitivity.{{png,pdf}} (argmax at w_llm={best_w})")
    elif n_v == 0:
        print("Skipped w_llm_sensitivity: no samples with a usable verdict on this split.")


if __name__ == "__main__":
    main()
