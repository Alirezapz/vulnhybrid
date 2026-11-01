# VulnHybrid research upgrade: audit, experiments and limitations

**Run date: 1 November 2026. Author: Alireza Pazouki.**

## Findings that matter

The code and experimental infrastructure are stronger, but these experiments
**do not demonstrate a reliable new detection improvement**. On the primary
commit-disjoint cohort, the validation-selected method is **gated_residual**:
accuracy **51.16%**, F1 **0.6688**, MCC **0.0597**.
An always-vulnerable classifier already reaches F1 **0.6684**.
The selected method's F1 advantage over that trivial baseline is
**0.00039**, with a commit-cluster 95% interval
**[-0.00992, 0.00918]**.
This interval does not establish a positive gain.

The frozen external probe is even more diagnostic: **0.00%**
of **415** vulnerable/fixed pairs have both labels correct for
that method. Its external ROC-AUC is **0.5044** and FPR is
**100.00%**. Strong recall/F1 here is mostly an operating-point
artifact, not evidence of understanding the repair. No SOTA claim is justified.

## What was actually executed

Downloaded all **27,318** Devign/CodeXGLUE functions and **846** C/C++ SVEN
functions from pinned revisions. Evaluated **19 methods
in each of four regimes** (76 matched test
comparisons), plus frozen transfer evaluations for every method. The prototype
methods combine lightweight lexical classifiers with measured static evidence;
they are not new LLM runs and not named neural-paper reproductions.

Core regressions and original end-to-end tests were run locally, including the
real mock HTTP transport. Static engines and taint fixtures were executed.
The taint fixture contains an explicit expected miss for an array update.
No paid API, GPU, exploit generation, or real-project vulnerability discovery
was used. Published-looking historical thesis tables were not treated as
new measurements because their supporting full outputs are absent from the ZIP.

## Data audit and leakage controls

The original data has **97**
normalized duplicate groups, **91**
conflicting-label groups, **23**
clone groups crossing official splits, and
**1902** commit IDs in multiple
splits. Official-split deduplication removes token-identical clones but leaves
**1891** overlapping commit groups.

The primary protocol puts entire commits in one split, then deduplicates code.
It retains **22078** train,
**2572** validation and
**2500** test functions, with zero
normalized clone or commit overlap. Of validation functions,
**1254** fit models/calibrators and
**1318** select thresholds and the method. These
halves are also commit-disjoint. The split is randomized by stable hash,
not chronological; near-clone and label-noise threats remain.

SVEN is used only for transfer, not training. Eight whole C/C++ pairs are
excluded because of Devign overlap or equal normalized before/after code,
leaving **415 pairs / 830 functions**. This repurposes its upstream training
and validation release as an external detection probe; it is not an official
SVEN test leaderboard. Scores/thresholds are frozen before this evaluation.

## Matched comparisons across regimes

Each row below is the method selected by that regime's own validation set.
Cohorts differ across regimes, so their absolute scores do not constitute a
paired causal estimate of leakage inflation. The official-to-commit decline
is diagnostic evidence consistent with dataset shortcuts, not proof of its
unique cause.

| Protocol | Validation-selected method | Test n | Accuracy | F1 | MCC | External pair success |
|---|---|---|---|---|---|---|
| official_dedup | logodds_lexical_static | 2713 | 0.6181 | 0.6675 | 0.2900 | 0.0048 |
| commit_disjoint | gated_residual | 2500 | 0.5116 | 0.6688 | 0.0597 | 0.0000 |
| ffmpeg_to_qemu | intercept_only | 1754 | 0.4230 | 0.5893 | -0.0212 | 0.0024 |
| qemu_to_ffmpeg | char_lr | 974 | 0.5370 | 0.6324 | 0.0576 | 0.0048 |

![Matched accuracy and MCC](../results/2026-11-01/method_comparison.png)

## All primary test results

All methods below share the same **2500** test functions. F1 zero division
uses zero. AP means average precision. Static scores are binary alert/no-alert
scores; their Brier/ECE should not be read as calibrated uncertainty.

| Method | Accuracy | Precision | Recall | F1 | MCC | AP | FPR |
|---|---|---|---|---|---|---|---|
| always_safe | 0.4980 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5020 | 0.0000 |
| always_vulnerable | 0.5020 | 0.5020 | 1.0000 | 0.6684 | 0.0000 | 0.5020 | 1.0000 |
| train_prevalence | 0.4980 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5020 | 0.0000 |
| semgrep_any | 0.5016 | 0.6552 | 0.0151 | 0.0296 | 0.0332 | 0.5043 | 0.0080 |
| cppcheck_any | 0.5072 | 0.6949 | 0.0327 | 0.0624 | 0.0600 | 0.5083 | 0.0145 |
| flawfinder_any | 0.5116 | 0.5515 | 0.1450 | 0.2297 | 0.0386 | 0.5092 | 0.1189 |
| taint_any | 0.4980 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5020 | 0.0000 |
| token_lr | 0.5096 | 0.5059 | 0.9888 | 0.6694 | 0.0565 | 0.5484 | 0.9735 |
| token_nb | 0.5184 | 0.5105 | 0.9849 | 0.6725 | 0.0945 | 0.5747 | 0.9518 |
| token_svm_platt | 0.5020 | 0.5020 | 0.9976 | 0.6679 | 0.0002 | 0.5097 | 0.9976 |
| token_rf | 0.5084 | 0.5053 | 0.9857 | 0.6681 | 0.0454 | 0.5704 | 0.9727 |
| char_lr | 0.5084 | 0.5053 | 0.9912 | 0.6694 | 0.0528 | 0.5624 | 0.9783 |
| static_lr | 0.5020 | 0.5020 | 1.0000 | 0.6684 | 0.0000 | 0.5102 | 1.0000 |
| shape_lr | 0.5040 | 0.5030 | 0.9976 | 0.6688 | 0.0305 | 0.5666 | 0.9936 |
| mean_token_char | 0.5112 | 0.5068 | 0.9865 | 0.6696 | 0.0622 | 0.5523 | 0.9679 |
| logodds_lexical_static | 0.5112 | 0.5068 | 0.9865 | 0.6696 | 0.0622 | 0.5523 | 0.9679 |
| heldout_evidence_stack | 0.5116 | 0.5070 | 0.9849 | 0.6694 | 0.0624 | 0.5591 | 0.9655 |
| gated_residual | 0.5116 | 0.5070 | 0.9825 | 0.6688 | 0.0597 | 0.5526 | 0.9631 |
| intercept_only | 0.5112 | 0.5068 | 0.9865 | 0.6696 | 0.0622 | 0.5523 | 0.9679 |

Thresholds are selected on validation only. The maximum test row is descriptive;
it does not override the validation-selected method. Full confusion counts,
thresholds, default-threshold F1, ROC-AUC, Brier/ECE and commit-cluster F1 intervals
are in each metrics.json and leaderboard.csv.

## New combinations and ablations

**Held-out evidence stack:** a regularized logistic combination of token,
character and static logits fit on val_fit, separate from training and selection.
**Gated residual:** anchor the mean lexical logit, then fit an intercept plus
shrunken binary tool-rule corrections. Multiple warning lines do not multiply
the same evidence. **Intercept-only** isolates base-rate recalibration; **mean**
and **log-odds fusion** are established controls.

These are project-specific experimental combinations of existing statistical
ideas. They are not claimed as world-first methods. On the primary cohort, the
gated residual minus token-LR F1 difference is **-0.00053**,
cluster interval **[-0.00411, 0.00299]**.
The static evidence has not earned a demonstrated generalization improvement.
The narrow explicit-source taint rule produced zero corpus alerts; negative
results remain in the comparison.

McNemar tests measure accuracy differences, not F1. Their Holm-adjusted p-values
are included as secondary analyses; same-commit correlation makes the clustered
F1 difference intervals more appropriate for the central F1 question. Intervals
are conditional on this run and exclude training/selection variance.

## Paired repair sensitivity

![F1 compared with repair-pair success](../results/2026-11-01/paired_evaluation.png)

| Method | External accuracy | F1 | Both versions correct | Vulnerable ranks higher | Tied |
|---|---|---|---|---|---|
| always_safe | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| always_vulnerable | 0.5000 | 0.6667 | 0.0000 | 0.0000 | 1.0000 |
| train_prevalence | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| semgrep_any | 0.5012 | 0.0841 | 0.0024 | 0.0024 | 0.9976 |
| cppcheck_any | 0.5012 | 0.0961 | 0.0072 | 0.0072 | 0.9880 |
| flawfinder_any | 0.4964 | 0.3907 | 0.0000 | 0.0000 | 0.9928 |
| taint_any | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| token_lr | 0.5000 | 0.6667 | 0.0000 | 0.4410 | 0.0795 |
| token_nb | 0.5000 | 0.6667 | 0.0000 | 0.4602 | 0.0795 |
| token_svm_platt | 0.5000 | 0.6667 | 0.0000 | 0.4627 | 0.0795 |
| token_rf | 0.5000 | 0.6667 | 0.0000 | 0.1711 | 0.5157 |
| char_lr | 0.5000 | 0.6667 | 0.0000 | 0.5566 | 0.0024 |
| static_lr | 0.5000 | 0.6661 | 0.0000 | 0.0120 | 0.9663 |
| shape_lr | 0.5000 | 0.6661 | 0.0000 | 0.3470 | 0.0434 |
| mean_token_char | 0.5000 | 0.6667 | 0.0000 | 0.5253 | 0.0000 |
| logodds_lexical_static | 0.5000 | 0.6667 | 0.0000 | 0.5253 | 0.0000 |
| heldout_evidence_stack | 0.5000 | 0.6667 | 0.0000 | 0.5470 | 0.0000 |
| gated_residual | 0.5000 | 0.6667 | 0.0000 | 0.5277 | 0.0000 |
| intercept_only | 0.5000 | 0.6667 | 0.0000 | 0.5253 | 0.0000 |

A high F1 with zero pair success is an unhelpful classifier that labels both
versions vulnerable. Ranking the vulnerable version above the fix is a separate
criterion; it does not guarantee a useful classification threshold.

## Analyzer coverage and diagnostic-free sensitivity analysis

| Engine | Inputs | Diagnostic records | Affected inputs | Seconds | Version |
|---|---|---|---|---|---|
| semgrep | 28164 | 4921 | 4921 | 55.5614 | 1.183.0 |
| taint | 28164 | 46 | 46 | 8.2137 | 1.183.0 |
| cppcheck | 28164 | 2601 | 2355 | 177.1686 | Cppcheck 2.22.0 |
| flawfinder | 28164 | 0 | 0 | 9.7272 | 2.0.20 |

Diagnostics include partial parses and unknown macros; counts are not all
complete failures, and a partially parsed function may still yield a warning.
The full-cohort static comparison measures operational alerts: no alert on
unparsed code is **not evidence of safety**. Error records and exact affected
IDs are retained. A second, common diagnostic-free primary subset contains
**2086** functions:

| Method | n | Accuracy | F1 | MCC |
|---|---|---|---|---|
| semgrep_any | 2086 | 0.5149 | 0.0269 | 0.0415 |
| cppcheck_any | 2086 | 0.5240 | 0.0694 | 0.0906 |
| flawfinder_any | 2086 | 0.5259 | 0.2292 | 0.0532 |
| token_lr | 2086 | 0.5014 | 0.6610 | 0.0902 |
| char_lr | 2086 | 0.4990 | 0.6599 | 0.0779 |
| gated_residual | 2086 | 0.5034 | 0.6603 | 0.0853 |

This subset conditions on analyzer success and may be easier/different; it is a
sensitivity analysis, not a replacement for full-cohort results. Review-budget
precision/recall at 1%, 5%, 10% and 20% is separately saved in review_budget.csv,
with the expected random-review baseline. It is an offline ranking analysis,
not simulated LLM performance or measured human review time.

## Engineering changes

Strict model-output validation; duplicate-verdict rejection; rule-event
deduplication and stable log odds; explicit analyzer errors; matched-denominator
evaluation; validation-only weight fitting; seeded sample selection; benchmark
hint removal; and model/prompt/input-aware checkpoint hashes. Details and
remaining limitations are in CODE_AUDIT.md. None of these fixes retroactively
validates the old paper's missing experiments.

## Publication assessment

This is a reproducible research artifact with useful negative evidence, not yet
a submission-ready claim of a better vulnerability detector. A stronger paper
needs actual named neural/modern-LLM baselines on the same cohort, chronological
PrimeVul and paired testing, multiple split/training seeds, manual real-project
alert verification, and a mechanism that improves those outcomes. The literature
review separates reviewed work from implementations actually executed.

## Reproduction and evidence

Use research/fetch_data.py, research/collect_static.py, research/benchmark.py and
research/summarize.py. README contains the exact commands. The protocol and its
hash were recorded before inspecting these test predictions. Data revisions,
byte hashes, cohort memberships, exclusions, validation partitions, per-sample
scores, structured analyzer findings, diagnostics, environment versions and comparisons
are included. Underlying upstream source datasets are downloaded separately.

All four run durations (seconds):
official_dedup: 162.9, commit_disjoint: 158.1, ffmpeg_to_qemu: 84.5, qemu_to_ffmpeg: 85.1.
They overlapped in pairs and are not an additive wall-clock estimate. These are
end-to-end CPU benchmark times, not model-only inference latency. No dollar or
LLM token savings are inferred.
