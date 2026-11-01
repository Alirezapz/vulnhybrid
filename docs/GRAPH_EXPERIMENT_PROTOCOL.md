# Graph study protocol — frozen before full model fitting

Date: 1 November 2026. This extends the previous study; its old test outcomes
were already seen. New test metrics are reserved for reporting after recipes
are fixed. This is a targeted, bounded experiment registry, not every graph
algorithm or every paper ever published. Exact paper reproductions and our
operator/representation experiments are explicitly distinguished.

## Cohorts and training boundaries

Reuse all four saved v1 cohorts without changing membership: official_dedup,
commit_disjoint (primary), FFmpeg_to_QEMU, QEMU_to_FFmpeg. Validation fitting
and selection memberships are unchanged and commit-disjoint. Old per-sample
predictions are joined by IDs, never position alone. No test/external label
influences graph construction, node selection, training, calibration or fusion.

All train functions are used in scientific runs. Smoke-test subsets are
excluded from scientific tables. GNNs use full batches of 128 graphs, 48 hidden
units, AdamW lr=.001 / weight_decay=.0001, dropout .2, maximum 8 epochs,
minimum 3 before early stopping, patience 2, gradient clipping 2. Select the
checkpoint by val_fit average precision. Seed 42 for all four protocols;
seeds 43 and 44 additionally for every primary neural baseline. Repeat seeds
change initialization/shuffling, not the fixed data split. Save all histories,
parameter counts, weights, timings, hashes, failures and per-function scores.

## Representations and coverage

Tree-sitter C/C++ named AST nodes, at most 256 nodes including an exit node.
A fixed hash maps type/leaf text to trainable embeddings; no fitted test
vocabulary. Scalar node indicators mark identifiers, statements, security APIs
and literals. Source comments are removed; IDs/labels/commit metadata are never
node features. Standard graph uses breadth-first truncation, recorded per input.

Relations: AST forward/reverse, next leaf forward/reverse, last identical
identifier forward/reverse, and structured statement CFG forward/reverse.
**Last-identical-identifier is a lexical proxy, not sound data dependence.**
Structured CFG omits switch/goto/exception semantics and reports them. This
frontend is not Joern or a full CPG. Parse errors, truncation and unsupported
constructs remain visible in all cohorts; no easy-case-only headline.

## Baseline registry

- Official PyG operators: GCN, GraphSAGE, GIN, GAT, R-GCN and GGNN.
- DeepSets with identical node inputs, no message passing (graph necessity
  control); GGNN plus supervised contrastive loss (weight .1, temperature .2).
- Devign-style GGNN with node-axis convolutional readout. It changes the original
  input frontend, features, capacity and readout details: **not an exact Devign
  paper replication**. The epicosy implementation is itself marked AST-only and
  under development. No copied end-to-end paper result is claimed.
- Classical graph representations: node histogram, AST WL (2/3 iterations),
  typed multi-relation WL, CFG WL, last-identifier WL, plus structural features.
  Fit train-only TF-IDF/scalers, logistic C in {.1,1,10} by val_fit AP; calibrated
  linear SVM as an additional kernel-feature baseline.
- Primary ablations: GraphSAGE on AST / CFG / last-identifier / sequence only;
  GraphSAGE and DeepSets with anonymized identifier payload; topology-free
  DeepSets; node-feature/topology perturbations on frozen predictions.

## Proposed combinations (after baseline training)

1. Evidence-focused graph selection: prioritize sensitive-API nodes and their
   AST ancestors, then fill breadth-first up to the same cap. Fit the same
   GraphSAGE encoder as its control. This is a project prototype related to
   established slicing ideas, not a claim of unprecedented novelty.
2. Relation-gated aggregation: learn weights for separate relation messages,
   compared with the existing relational GNN and fixed all-edge aggregation.
3. Learned graph/text/static fusion: mean, validation-selected convex weights,
   regularized logit stacking, shrinkage residual correction, confidence-aware
   expert gating, and an all-family stack. Each has a corresponding simpler
   ablation. All learned meta models fit val_fit, all selections use val_select.
   All-family uses every available old score and newly trained graph score;
   it does not pretend unexecuted LLMs were available.
4. Multi-view graph stack: combine independent AST/CFG/identifier/heterogeneous
   encoders, then add lexical and static experts. Keep representation and
   decision-threshold gains separate.

## Scoring and independent external probes

Report default .5 and val_select-F1 thresholds, and a validation-selected MCC
operating point. Include accuracy/balanced accuracy, F1, MCC, ROC-AUC, AP, FPR,
calibration, uncertainty and rank-based review budgets. Report sensitivity at a
val_select FPR cap (.5%, 1%, 5%) with actual test FPR; do not imply a guaranteed
deployment cap. Clustered uncertainty is by commit for function metrics and by
commit for external pair outcomes. Repeated training seeds do not estimate
variance across independent datasets.

Continue frozen SVEN repair evaluation and add the pinned PrimeVul paired-test
mirror if its pair identities and Devign-overlap controls can be verified.
Keep it an external transfer probe; it is not in-distribution PrimeVul training.
Perform exact before/after sensitivity and an error-free/non-truncated subset
analysis, without removing difficult cases from the full cohort table.

## Named-paper replication ledger

Devign, ReVeal, IVDetect, DeepWukong and AMPLE need their actual frontend,
input features, splits, training code and dependencies for exact reproduction.
Review original repositories and record what is runnable, adapted or blocked.
A GCN trained on our AST is called GCN-on-our-AST, never IVDetect or DeepWukong.
Do not claim all named papers were run if those requirements were not met.
