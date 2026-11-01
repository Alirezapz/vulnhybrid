# VulnHybrid graph study and all-family fusion

**1 November 2026 — version 1.2.0. Research artifact; not peer reviewed.**

## Main result

The primary validation-MCC-selected configuration is **x_ggnn_sum_port**. On the
2,500-function commit-disjoint test cohort it reaches accuracy
**52.84%**, F1 **0.6158**, MCC **0.0616** and
average precision **0.5649**. Token LR under the same
validation-MCC policy reaches accuracy **50.96%** and MCC
**0.0565**. The selected model's MCC difference is
**0.0051**, with 95% commit-cluster interval
**[-0.0916, 0.1161]**. The conditional clustered interval does not establish a positive MCC gain.

This does not establish a universal best detector or world-first algorithm.
It is evidence about these configurations, representations, labels and fixed
cohorts. The selected method was chosen by validation, not by the largest test
row. Historical v1 F1-selected numbers are retained separately; changing the
threshold objective is not a new model improvement.

## Executed scope and complete evidence

- **54 baseline neural training runs**: nine configurations on
  four cohorts, with three initialization/shuffle seeds on the primary cohort.
- **24 additional neural runs**: edge/payload/size controls,
  evidence-focused and operator/field-aware representations, relation gating,
  perturbation consistency, and architectural ports of typed GGNN/readout.
- **32 selected classical graph models**, including node
  histograms, AST/typed/CFG/last-identifier WL features and structural boosting.
- **18 learned or fixed fusion recipes per cohort**, including all-family
  stacking, convex mixtures, rank stacking, boosting, anchored residuals and
  confidence/quality gating. Every available informative executed expert can
  enter the all-family recipes; no unexecuted LLM is simulated.
- **267 scored configuration/cohort
  comparisons** across the four matched test sets. This count includes earlier
  baselines, seed repeats, ensemble configurations and ablations; it is not a
  count of that many distinct algorithms. F1/MCC operating points are reported
  separately, not counted as independent models.
- Every method is also scored on frozen SVEN and verified PrimeVul external
  pairs. Raw scores, IDs, fit/selection membership, seeds, best epochs,
  histories, weights, timings, diagnostics and repaired implementation failures
  are retained. All scientific neural runs use the full training cohort.

The code includes original PyG GCN, GraphSAGE, GIN, GAT, R-GCN and GGNN
operators, trained from scratch. Devign-style and typed architecture ports are
adaptations on a shared frontend. **Full end-to-end Devign, ReVeal, IVDetect,
DeepWukong and AMPLE paper reproductions are not claimed.** Their original
repositories and exactness requirements are listed in GRAPH_REPLICATION_LEDGER.
No finite registry can contain every graph method published in the field.
The supplemental 2025–2026 review also covers HAGNN, LLMxCPG, VulGNN and
Vul-CTG, with explicit reading depth and unexecuted status.

## Input and protocol integrity

All four old cohort memberships and validation halves are unchanged. Old
text/static predictions were independently reproduced within 1e-7 before
extending them to the new external code. Train-only TF-IDF, validation-only
calibration, and separate validation selection remain enforced. Graph features
contain no IDs, labels, CWE/CVE metadata, filenames or commit identity.

Primary neural budgets are shared: 48 hidden units, two conventional message
layers or three ordinary gated steps; eight typed gated steps for the inspected
Devign architectural port; batch 128; AdamW .001, weight decay .0001; up to
eight epochs with patience two and minimum three epochs. Checkpoints maximize
val_fit AP. Prototype and cross-project runs use seed 42. This is a bounded
CPU budget, not exhaustive architecture/hyperparameter optimization.

The extension reuses previously seen v1 test cohorts. The graph protocol was
frozen before full graph fitting; a representation addendum was driven by input
coverage before inspecting graph test metrics. The entire extension remains
exploratory, not an externally preregistered independent confirmation.

## Representation audit

The base graph contains tree-sitter named AST nodes and leaf payload, next-leaf
links, last-identical-identifier links and a structured statement CFG. The latter
two are approximate: they are not sound pointer-aware reaching definitions or
a compiler CPG. Switch, goto and exception semantics are diagnosed, not solved.
The base named-node input omits unnamed operators and AST field roles; a matched
richer control preserves those. Edge-only ablations share the node set.

Across **28164** Devign/SVEN inputs, tree-sitter reports error nodes in
**5288** functions and the 256-node representation truncates
**9458**. Errors and truncation stay in the full comparison.
A common error-free/non-truncated sensitivity subset is separately recorded.

| Representation | Node cap | Input-identical SVEN pairs | Loose pair-success upper bound | Truncated functions |
|---|---|---|---|---|
| graph_cache | 256 | 78 | 0.8120 | 9458 |
| graph_cache_evidence | 256 | 80 | 0.8072 | 9458 |
| graph_cache_semantic | 256 | 70 | 0.8313 | 9458 |
| graph_cache_semantic_bfs | 256 | 71 | 0.8289 | 9458 |
| graph_cache_512 | 512 | 43 | 0.8964 | 4304 |

Input-identical before/after graphs cannot both be classified correctly by a
deterministic classifier with one threshold. The bound is loose: distinct inputs
can still be isomorphic to a permutation-invariant model or lack sufficient
semantic context. A sensitive-API selection can also discard repair evidence;
it is not assumed beneficial. The evidence/semantic/512 controls isolate these
representation changes from a change of GNN operator.

## Validation-selected results across cohorts

| Cohort | Selected by validation MCC | Test n | Accuracy | F1 | MCC | AP |
|---|---|---|---|---|---|---|
| commit_disjoint | x_ggnn_sum_port | 2500 | 0.5284 | 0.6158 | 0.0616 | 0.5649 |
| official_dedup | fusion_all_residual | 2713 | 0.6513 | 0.5612 | 0.2928 | 0.7019 |
| ffmpeg_to_qemu | fusion_graph_residual | 1754 | 0.5804 | 0.1111 | 0.0514 | 0.4732 |
| qemu_to_ffmpeg | fusion_graph_text_stack | 974 | 0.5195 | 0.3217 | 0.0907 | 0.6068 |

Each cohort has its own training/selection population. Absolute differences
between cohort rows are not a paired causal estimate of leakage or project
transfer. All methods within one cohort have identical denominators.

![Graph and hybrid comparison](../results/2026-11-01-graphs/VulnHybrid_Graph_Comparison.png)

## All primary graph baselines (MCC operating point)

| Configuration | Accuracy | Precision | Recall | F1 | MCC | AP | FPR |
|---|---|---|---|---|---|---|---|
| g_deepsets_hetero_s42 | 0.5184 | 0.5113 | 0.9211 | 0.6576 | 0.0571 | 0.5601 | 0.8876 |
| g_devign_style_hetero_s42 | 0.5208 | 1.0000 | 0.0454 | 0.0869 | 0.1521 | 0.5931 | 0.0000 |
| g_gat_hetero_s42 | 0.5304 | 0.5231 | 0.7299 | 0.6094 | 0.0646 | 0.5591 | 0.6707 |
| g_gcn_hetero_s42 | 0.5188 | 1.0000 | 0.0414 | 0.0796 | 0.1452 | 0.5854 | 0.0000 |
| g_ggnn_hetero_s42 | 0.5320 | 0.5265 | 0.6733 | 0.5909 | 0.0656 | 0.5725 | 0.6104 |
| g_ggnn_supcon_hetero_s42 | 0.5212 | 0.5199 | 0.6048 | 0.5591 | 0.0423 | 0.5687 | 0.5631 |
| g_gin_hetero_s42 | 0.5368 | 0.5237 | 0.8526 | 0.6489 | 0.0919 | 0.5644 | 0.7815 |
| g_rgcn_hetero_s42 | 0.5064 | 0.5064 | 0.6598 | 0.5730 | 0.0122 | 0.5683 | 0.6482 |
| g_sage_hetero_s42 | 0.5208 | 1.0000 | 0.0454 | 0.0869 | 0.1521 | 0.5857 | 0.0000 |
| k_ast_wl2 | 0.5176 | 1.0000 | 0.0390 | 0.0752 | 0.1408 | 0.5807 | 0.0000 |
| k_ast_wl3 | 0.5136 | 0.5085 | 0.9339 | 0.6584 | 0.0444 | 0.5936 | 0.9100 |
| k_ast_wl3_svm | 0.5104 | 1.0000 | 0.0247 | 0.0482 | 0.1116 | 0.5431 | 0.0000 |
| k_cfg_wl2 | 0.5340 | 0.5289 | 0.6566 | 0.5859 | 0.0691 | 0.5576 | 0.5896 |
| k_graph_structure_hgb | 0.5160 | 0.7778 | 0.0502 | 0.0943 | 0.1009 | 0.5463 | 0.0145 |
| k_hetero_wl2 | 0.5140 | 0.5084 | 0.9602 | 0.6648 | 0.0550 | 0.6168 | 0.9357 |
| k_last_ident_wl2 | 0.5388 | 0.5271 | 0.7904 | 0.6325 | 0.0876 | 0.5852 | 0.7149 |
| k_node_hist | 0.5284 | 0.5201 | 0.7825 | 0.6249 | 0.0637 | 0.5792 | 0.7277 |

## All primary representation/architecture prototypes and controls

| Configuration | Accuracy | Precision | Recall | F1 | MCC | AP |
|---|---|---|---|---|---|---|
| x_deepsets_anonymous | 0.5212 | 0.5142 | 0.8390 | 0.6376 | 0.0518 | 0.5655 |
| x_devign_relational_port | 0.5384 | 0.5805 | 0.2900 | 0.3868 | 0.0909 | 0.5800 |
| x_ggnn_invariant | 0.5316 | 0.5278 | 0.6359 | 0.5768 | 0.0638 | 0.5688 |
| x_ggnn_sum_port | 0.5284 | 0.5209 | 0.7530 | 0.6158 | 0.0616 | 0.5649 |
| x_relation_gate | 0.5260 | 0.9861 | 0.0566 | 0.1070 | 0.1667 | 0.6051 |
| x_relation_uniform | 0.5200 | 0.9661 | 0.0454 | 0.0868 | 0.1443 | 0.5905 |
| x_sage_anonymous | 0.5312 | 0.5340 | 0.5195 | 0.5267 | 0.0625 | 0.5595 |
| x_sage_ast | 0.5160 | 0.8814 | 0.0414 | 0.0791 | 0.1180 | 0.5695 |
| x_sage_cap512 | 0.5284 | 0.5734 | 0.2367 | 0.3350 | 0.0730 | 0.5662 |
| x_sage_cfg | 0.5152 | 0.5091 | 0.9546 | 0.6641 | 0.0571 | 0.5580 |
| x_sage_evidence | 0.5164 | 0.5118 | 0.7976 | 0.6235 | 0.0370 | 0.5720 |
| x_sage_last_ident | 0.5156 | 0.5097 | 0.9179 | 0.6555 | 0.0475 | 0.5554 |
| x_sage_semantic_bfs | 0.5152 | 0.9388 | 0.0367 | 0.0706 | 0.1235 | 0.5957 |
| x_sage_semantic_evidence | 0.5176 | 0.9016 | 0.0438 | 0.0836 | 0.1264 | 0.5981 |
| x_sage_sequence | 0.5336 | 0.5221 | 0.8367 | 0.6430 | 0.0816 | 0.5587 |

The semantic payload corrects a representation omission. Evidence-focused node
retention is related to established slicing. Relation gating and consistency
regularization adapt known techniques. These names describe implemented project
prototypes; they do not establish literature-wide novelty.

## All-family and narrower fusion recipes

| Configuration | Accuracy | Precision | Recall | F1 | MCC | AP | FPR |
|---|---|---|---|---|---|---|---|
| fusion_neural_mean | 0.5192 | 1.0000 | 0.0422 | 0.0810 | 0.1466 | 0.5956 | 0.0000 |
| fusion_all_graph_mean | 0.5368 | 0.5266 | 0.7641 | 0.6235 | 0.0807 | 0.5914 | 0.6924 |
| fusion_all_expert_mean | 0.5396 | 0.5260 | 0.8382 | 0.6464 | 0.0960 | 0.5998 | 0.7614 |
| fusion_family_balanced_mean | 0.5112 | 0.5068 | 0.9865 | 0.6696 | 0.0622 | 0.5806 | 0.9679 |
| fusion_graph_stack | 0.5196 | 1.0000 | 0.0430 | 0.0825 | 0.1480 | 0.5975 | 0.0000 |
| fusion_extended_graph_stack | 0.5196 | 1.0000 | 0.0430 | 0.0825 | 0.1480 | 0.5928 | 0.0000 |
| fusion_graph_text_stack | 0.5236 | 0.9706 | 0.0526 | 0.0998 | 0.1567 | 0.6016 | 0.0016 |
| fusion_graph_static_stack | 0.5208 | 0.9831 | 0.0462 | 0.0883 | 0.1496 | 0.5944 | 0.0008 |
| fusion_all_logit_stack | 0.5220 | 0.9688 | 0.0494 | 0.0940 | 0.1513 | 0.6115 | 0.0016 |
| fusion_all_probability_stack | 0.5220 | 0.9839 | 0.0486 | 0.0926 | 0.1537 | 0.6110 | 0.0008 |
| fusion_all_rank_stack | 0.5412 | 0.6244 | 0.2159 | 0.3209 | 0.1122 | 0.5952 | 0.1309 |
| fusion_all_boosting | 0.5240 | 0.5154 | 0.8685 | 0.6469 | 0.0627 | 0.6155 | 0.8233 |
| fusion_convex_all | 0.5184 | 1.0000 | 0.0406 | 0.0781 | 0.1437 | 0.6051 | 0.0000 |
| fusion_graph_residual | 0.5192 | 1.0000 | 0.0422 | 0.0810 | 0.1466 | 0.6020 | 0.0000 |
| fusion_all_residual | 0.5224 | 0.9841 | 0.0494 | 0.0941 | 0.1550 | 0.6300 | 0.0008 |
| fusion_intercept_control | 0.5112 | 0.5068 | 0.9865 | 0.6696 | 0.0622 | 0.5523 | 0.9679 |
| fusion_confidence_quality_gate | 0.5144 | 0.9556 | 0.0343 | 0.0662 | 0.1228 | 0.6002 | 0.0016 |
| fusion_quality_fallback | 0.5172 | 0.5102 | 0.9578 | 0.6657 | 0.0662 | 0.5415 | 0.9269 |

Means and convex mixtures are controls for learned stacking. Separate graph-only,
graph/text, graph/static and all-family stacks test complementarity. Probability,
logit and validation-reference rank transforms are separate recipes. The anchored
residual has an intercept-only control. The relation gate has a uniform control.
Quality fallback and a fitted confidence/quality gate test whether frontend
limitations should change expert weighting. All meta fitting uses val_fit; the
anti-leakage regression test changes the entire test distribution and verifies
that learned meta parameters and validation outputs do not change.

## Training seed sensitivity

| Architecture | Mean accuracy | SD accuracy | Mean MCC | SD MCC |
|---|---|---|---|---|
| deepsets | 0.5315 | 0.0156 | 0.0728 | 0.0221 |
| gcn | 0.5223 | 0.0032 | 0.1176 | 0.0535 |
| sage | 0.5297 | 0.0165 | 0.1329 | 0.0298 |
| gin | 0.5257 | 0.0096 | 0.1033 | 0.0431 |
| gat | 0.5211 | 0.0083 | 0.0764 | 0.0547 |
| rgcn | 0.5197 | 0.0117 | 0.0495 | 0.0326 |
| ggnn | 0.5312 | 0.0144 | 0.0994 | 0.0355 |
| devign_style | 0.5208 | 0.0000 | 0.1505 | 0.0029 |
| ggnn_supcon | 0.5303 | 0.0164 | 0.0970 | 0.0520 |

These are three initialization/shuffle seeds on the same split, with each
threshold selected using validation. They do not estimate dataset/split variance.
Single-seed prototypes and external probes require replication before general
claims. Raw per-seed rows and three-seed probability ensembles are retained.

## Frozen external repair sensitivity

For the validation-MCC-selected primary configuration:

| External probe | Pairs | Both labels correct | Strict vulnerable>fixed ranking | Accuracy | FPR |
|---|---:|---:|---:|---:|---:|
| SVEN retained pairs | 415 | 0.0024 | 0.2916 | 0.4892 | 0.9036 |
| PrimeVul mirror verified subset | 348 | 0.0057 | 0.2845 | 0.4971 | 0.9310 |

In counts, the selected configuration gets both labels right in
**1/415 SVEN pairs** and
**2/348 PrimeVul pairs**.

![Frozen external pair outcomes](../results/2026-11-01-graphs/VulnHybrid_Graph_Pairs.png)

SVEN uses the previously retained 415 pairs. PrimeVul uses **348 verified pairs**
out of 435 source pairs: 76 incomplete metadata/signature cases, seven overlaps,
two commit mismatches, one filename mismatch and one identical pair are excluded.
The exact exclusions and byte hash are supplied. This conservative subset can be
selection-biased and is not the full official PrimeVul benchmark. The source is
a pinned public mirror; no independent relabeling of its truth was performed.

External thresholds remain frozen; no external label trains/calibrates/selects
a method. Both-correct cluster intervals and nominal independent-pair exact
binomial intervals are stored. Cluster bootstrap intervals can degenerate at
zero observed successes; the binomial interval assumes independent pairs and
does not account for shared-commit correlation. Neither interval is proof of a
population failure rate of exactly zero.

## Perturbation and operating-point analyses

Frozen models are evaluated with edges removed, relation-wise rewiring,
anonymous leaf payload and node permutation. These are diagnostic input
perturbations, not all semantics-preserving program edits. No threshold is
retuned on perturbed test data. DeepSets is a topology-free control; ordinary
GNN pooling is permutation-invariant, whereas convolutional node-axis readout
can depend on order. The complete score changes, flip rates and MCC are in
stability_metrics.csv.

Both F1-selected and MCC-selected operating points are preserved for every
configuration. Low-FPR points are selected at observed val_select caps of .5%,
1% and 5%, with actual test FPR/recall reported. The caps are not deployment
guarantees. Review precision/recall at fixed 1/5/10/20% budgets is also saved,
with the expected random-review baseline. No LLM cost or human-review time is
inferred from these offline rankings.

## Statistics, reproducibility and publication scope

Test intervals and paired differences use 1,000 shared commit-cluster bootstrap
resamples. McNemar tests are secondary accuracy tests with Holm adjustment,
not tests of F1/MCC; sample dependence remains a limitation. Intervals are
conditional on the fitted/selected models and omit split/training uncertainty.

The model archive preserves trained weights and classical estimators. Raw
per-function scores and frozen memberships are in the experiment-data archive;
code, configs and compact tables are tracked in Git. Regenerate graph caches
from the pinned data with the documented extraction commands.
Direct environment versions and input/protocol hashes are recorded. Tests cover
CFG returns, parse diagnostics, edge validity, operator preservation, every
architecture's gradients, topology controls and meta-data leakage. A separate
verifier recomputes scores/selections from exported prediction files.

This provides a substantive graph/fusion research artifact. Publication claims
still require exact named baseline pipelines on their intended representations,
stronger chronological/near-clone controls, more split/training seeds and manual
real-project validation. No unrun method receives an invented score, and no
acceptance/SOTA/world-first claim follows from the experiment count.
