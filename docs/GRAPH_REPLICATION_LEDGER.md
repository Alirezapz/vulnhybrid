# Graph research: exactness and execution ledger

This ledger prevents an operator experiment from being mislabeled as a full
paper reproduction. The machine-readable run registries are authoritative for
completed, repaired, failed and pending executions. All code/results are English.

| Method/source | This study executes | Exactness limits |
|---|---|---|
| PyG GCN, GraphSAGE, GIN, GAT, R-GCN, GGNN | Actual maintained operator implementations, trained on all functions in each saved training cohort | Fixed shared 48-dimensional budget and our bounded frontend; not the papers' original benchmark results |
| DeepSets | Identical node features/readout, no inter-node messages | Tests whether topology adds value beyond node-feature bags |
| Weisfeiler-Lehman | Iterative neighborhood refinement with hashed histograms and train-only TF-IDF | Directed/typed adaptations are explicitly named; these are our implementations of the feature construction |
| Devign-style readout | GGNN plus dual convolution heads | An adapted architectural control, not original Devign |
| [ReVeal authors' Devign implementation](https://github.com/saikat107/Devign) | A PyTorch port of typed eight-step gated messages and dual convolution/product/mean readout; also the sum-readout control | Same mathematical components, shared new node features, 48-dimensional capacity, fixed padding and frontend. Not original 2019 authors' exact end-to-end experiment. Inspected commit: be2f831123efbf85a3780dfce12205dc7be23e94 |
| [epicosy/devign](https://github.com/epicosy/devign) | Source inspection informed a separate adapted readout control | The repository itself states that it is AST-only and under development; do not call it a faithful complete CPG reproduction |
| [ReVeal](https://github.com/VulDetProject/ReVeal) | Repository/processing inspection; GGNN+supervised contrastive is a separately named baseline | Full Joern/representation-learning pipeline and its triplet training are not reproduced. Commit ca31b783384b4cdb09b69950e48f79fa0748ef1d |
| [IVDetect](https://github.com/vulnerabilitydetection/VulnerabilityDetectionResearch) | Original model/dependencies inspected | Tree embeddings, dependency graphs and interpretation/localization pipeline are not executed as IVDetect. Commit 9c2fe0e442f9a95a912c8e92550ca04988749422 |
| [DeepWukong](https://github.com/jumormt/DeepWukong) | Maintained repository, requirements and preprocessing inspected | Our GCN is not its sensitive-operation slicing/PDG pipeline or released training. Commit 28449e57262dd24c0938108687e1019f0fc6ddd5 |
| [AMPLE](https://github.com/Xin-Cheng-Wen/AMPLE) | Original graph-transformer/kernel code inspected | Full simplification/Joern/embedding pipeline is not executed. The inspected forward path also selects CUDA explicitly; our relation gate is not AMPLE. Commit 688776e4d4012efa77665d9618ba6d5d2f67232b |
| [Joern CPG specification](https://cpg.joern.io/) | Used to distinguish a CPG's reaching-definition/control-dependence edges from lexical proxies | Joern was not used for this extraction. AST + structured CFG + last-identifier links is not a sound CPG/PDG |

## Registered new combinations

Evidence-focused node retention; operator/child-role preservation with matched
breadth-first and 512-node controls; learned relation gating with a uniform
control; graph/identifier perturbation consistency; graph/text/static means,
convex mixtures, logit/probability/rank stacking, boosted stacking, anchored
residual correction, confidence/quality gating and quality fallback. The
all-family stack includes every available informative executed expert, including
the old text/static recipes. Constant scores are excluded as redundant with an
intercept. No unexecuted LLM is included or simulated.

These are adaptations and combinations of established ideas. Positive results
would support this project/configuration, not establish world-first novelty.
The registry cannot cover every published graph algorithm. Unexecuted named
pipelines remain explicitly unexecuted rather than being assigned fake scores.
The additional [2025–2026 review](GRAPH_RECENT_WORK.md) covers HAGNN, LLMxCPG,
VulGNN and Vul-CTG, with reading depth, source pins and unexecuted status.

## Data provenance and representation caveats

All v1 cohorts and val_fit/val_select memberships are preserved. Three training
seeds repeat the primary neural baselines; the extra prototypes use seed 42.
Edge-only ablations retain the shared node set. Anonymous payload ablations zero
all leaf-text IDs, while syntax and security-API indicators remain. No label,
ID, CWE/CVE metadata, filename or commit identity is a graph feature.

The base named-node frontend omits unnamed operator/child-role information;
its richer control adds that information. Structured CFG does not model
switch/goto/exception semantics; unresolved pointers/aliases and scope-sensitive
def-use are not reconstructed. Diagnostics, truncation and pair-input collisions
are therefore part of the result, not hidden preprocessing failures.

The additional PrimeVul paired-test file comes from the pinned public mirror
starsofchance/PrimeVul at a30276e1bbec449c1c932f6a664b4c06942154f1, with raw
SHA-256 384758c25142b640f1768bb0d334c43f315c580022ae94973c66e77128fb3821.
Ordered pairs are checked for opposite labels and shared project/commit; missing
filenames require matching signatures. Whole overlapping/identical/ambiguous
pairs are excluded and listed. This is a Devign-trained transfer probe, not an
in-distribution official PrimeVul leaderboard result.
