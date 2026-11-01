# Frozen experiment protocol — 1 November 2026

This protocol was written before inspecting the new test predictions. It fixes
four evaluation regimes, all methods, and the selection rule. The new experiments
are CPU replications of standard classifier families and project-specific fusion
prototypes. They are **not reproductions of Devign's GNN, LineVul, IRIS, or any
other named paper's implementation**. No live LLM is used in this experiment set.

## Data and provenance

- Devign/CodeXGLUE: all 27,318 functions from a pinned upstream revision.
- SVEN: the 846 C/C++ functions (423 before/after pairs) in the original training
  and validation release, used here **only as an external transfer probe**. It
  is not an official SVEN detection test split. Exclude whole pairs if either
  version duplicates any retained Devign function or both versions have the
  same normalized code. No external labels influence training or calibration.
- Preserve download SHA-256, exact retained IDs, excluded IDs, and per-sample
  scores. Do not redistribute upstream source code under the project's MIT
  license; download it separately using the supplied script.

## Partitions

1. **official_dedup**: keep original CodeXGLUE membership, then token-identical
   code deduplication. This is a conventional comparison regime, with residual
   same-commit leakage explicitly measured and reported.
2. **commit_disjoint** (primary): SHA-256(seed:project:commit) assigns entire
   commits to train/val/test at 80/10/10 hash intervals. Apply code deduplication
   afterward. This controls commit overlap, not chronological or near-clone
   overlap, and does not imply pretraining decontamination.
3. **FFmpeg_to_QEMU**: train/val contain only FFmpeg from their original splits;
   test contains only QEMU from its original test split.
4. **QEMU_to_FFmpeg**: reverse direction.

Deduplication is label-blind across partitions: train takes precedence over
val, then test. Within a partition, drop contradictory-label clone groups;
otherwise keep the lexicographically first ID. Removing a test label cannot
change training membership. Full original conflicts remain in the audit.
Comments are removed from all model inputs while preserving literals. Juliet
hint removal is supported by the pipeline but no new Juliet corpus is claimed.

Split validation into commit-disjoint halves using GroupShuffleSplit(seed=42):
**val_fit** selects regularization and fits score calibration/fusion;
**val_select** chooses thresholds and the final method. Test and external labels
are used only for the final comparison. Alternative methods are exploratory;
select the deployment candidate using validation F1, never the maximum test F1.

## Methods and fixed search space

- Always-safe, always-vulnerable, and train prevalence.
- Bundled Semgrep security-pattern pack, Cppcheck warning checks (normal level),
  Flawfinder (level >=1), and an experimental explicit-source Semgrep taint rule.
  The unvalidated arithmetic draft is excluded from scoring/features.
- Token TF-IDF logistic regression; character TF-IDF logistic regression.
  C in {0.5, 2, 8}, selected by val_fit average precision. Both use train-only
  vocabularies (40k token / 50k character features, min_df=3, max_df=0.98).
- Token Multinomial NB (alpha=1); token LinearSVC (C=1) with val_fit Platt
  calibration; token random forest (100 trees, depth 24, min leaf 2).
- Static-presence logistic regression and a shallow code-shape logistic control.
- Mean token/character probability; lexical/static log-odds mixture with lexical
  weight in {0, .25, .5, .75, 1}, selected on val_select.
- **Held-out evidence stack**: logistic regression on token/character/static
  logits, C=.1, fit only on val_fit.
- **Gated residual**: fixed coefficient 1 on the mean lexical logit plus an
  intercept and binary tool-rule corrections, with L2 penalty 10; fit only on
  val_fit. This prevents duplicate warning counts from acting as independent
  evidence. It is an experimental combination of established ideas, without
  a claim of world-first novelty.
- Intercept-only recalibration ablation (L2=10), separating base-rate adaptation
  from added static information.

Use the same val_select F1 threshold search for all learned scores: a 0.01 grid
plus an all-negative endpoint; break F1 ties closest to .5, then higher threshold.
Static and constant baselines retain threshold .5. Report default-threshold F1
alongside tuned F1 so operating-point changes are visible.

## Analysis and failure accounting

Report n, confusion counts, accuracy, balanced accuracy, precision/recall/F1,
MCC, ROC-AUC, average precision, FPR, Brier score, and 10-bin probability ECE.
Use 1,000 commit-cluster bootstrap resamples for F1 intervals and paired F1
*difference* intervals. McNemar is a secondary sample-level accuracy test;
Holm-adjust the family of comparisons against token LR. It is not a test of F1,
and correlated functions can invalidate its independence assumption; prioritize
clustered intervals. Intervals do not include model-selection/training variance.

External SVEN: frozen thresholds, both-versions-correct fraction, strict
vulnerable>fixed ranking fraction, ties, and score margin; do not compare these
detection scores with SVEN's code-generation paper metrics.

An empty finding list and a successful parse are different observations. Save
all analyzer diagnostics and scan counts. On the complete cohort, static scores
measure operational **alert/no-alert** behavior; no-alert on a parse failure
is not evidence of safety. Analyze an error-free common subset separately.
No negative static experiment is removed from the comparison.

## Scope of publication claims

These runs assess lightweight detection and calibrated fusion; they do not
establish SOTA, end-to-end LLM superiority, exploitability, or publication
readiness. A paper additionally needs named learned baselines on the same
splits, a clean chronological/paired benchmark such as PrimeVul, modern LLMs
with recorded model digests and costs, and manually checked real-project cases.
