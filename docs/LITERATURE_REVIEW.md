# Related work and reproducibility ledger

Review date: **1 November 2026**. This is a targeted primary-source review,
not an exhaustive systematic literature review. Strong published results and
negative replication evidence are both included. Paper metrics from different
languages, populations, label schemes, and granularities cannot rank a universal
"best" or "worst" detector. Our measured rankings are in EXPERIMENT_REPORT.md.
Reading depth is explicit: full-method HTML sections for PrimeVul, IRIS, LLMDFA,
and the 2026 project-scale study; official abstracts, publication pages and/or
replication repositories for the other entries. No unread paper is presented as
a completed replication.

Version 1.2 adds a separate [graph execution ledger](GRAPH_REPLICATION_LEDGER.md)
and [2025–2026 graph/hybrid source review](GRAPH_RECENT_WORK.md). The table below
describes the original v1 review; its Devign/PrimeVul status is superseded by
those documents: architectural graph ports and a verified external PrimeVul
subset now have measured results, without claiming full named-paper replication.

## Scientific approaches

| Work | Core idea and useful contribution | Limitation or failure mode relevant here | Status in this project |
|---|---|---|---|
| [VulDeePecker, NDSS 2018](https://www.ndss-symposium.org/wp-content/uploads/2018/02/ndss2018_03A-2_Li_paper.pdf) | Learn from code gadgets associated with sensitive operations. An early learned vulnerability detector. | API/slice selection determines coverage; gadget-level metrics are not whole-function metrics. | Reviewed official paper/repository description; neural implementation **not run**. |
| [SySeVR, TDSC 2021](https://arxiv.org/abs/1807.06756) | Combine syntax-triggered candidates with semantic dependency representations. | Dependency extraction and candidate selection introduce missed cases and preprocessing cost. | Reviewed; **not run**. Motivates separating candidate coverage from verifier quality. |
| [Devign, NeurIPS 2019](https://arxiv.org/abs/1909.03496) | Graph neural classification over multiple program relations. | Requires graph extraction; performance depends on dataset and split. The public CodeXGLUE subset is not the full four-project experiment. | Full public two-project data acquired; **GNN not reproduced**. |
| [ReVeal / Are We There Yet?, TSE](https://arxiv.org/abs/2009.07235) | Reassesses learning under realistic imbalance, duplication and irrelevant features. | Reported high benchmark accuracy can collapse when data/protocol artifacts change. | Reviewed paper abstract/official repository; simple lexical baselines are separate implementations, **not ReVeal**. |
| [LineVul, MSR 2022](https://github.com/awsm-research/LineVul) | Transformer-based function prediction and line prioritization. Its artifact also includes classical baselines. | Function detection and line localization require different ground truth; attention is not causal proof. | Reviewed original replication repository; token RF is a conventional baseline, **not a LineVul replication**. |
| [Data Quality for Software Vulnerability Datasets, ICSE 2023](https://arxiv.org/abs/2301.05456) | Checks label correctness, consistency, duplication and other data-quality attributes. | The studied corpora have substantial quality problems; proportions are dataset-specific. | Applied explicit hash/conflict/commit auditing to the actual downloaded data. |
| [SVEN, CCS 2023](https://www.sri.inf.ethz.ch/publications/ccs23-llmsec) | Security-controlled code generation using property-specific learned vectors and curated fixes. | This is a generation method. Its generation-security rate cannot be compared with detection F1. | Original C/C++ before/after data used as a frozen external detection probe; **SVEN model not run**. |
| [PrimeVul, ICSE 2025](https://arxiv.org/html/2403.18624v2) | Improved labels/deduplication, chronological same-commit splits, low-FPR and paired evaluation. | Its results show older benchmark scores can greatly overstate practical detection. Function context and imbalance remain hard. | Full evaluation-method sections reviewed; commit and paired checks implemented. **PrimeVul corpus/model runs pending**. |
| [IRIS, ICLR 2025](https://arxiv.org/html/2405.17238v3) | Infer source/sink specifications with an LLM, execute CodeQL dataflow, then contextual triage. | Evaluation is Java; missing side-effect edges, implicit flows and specification errors still cause misses. | Method/error-analysis sections reviewed; **not ported or run on C/C++**. The small C taint experiment is not IRIS. |
| [LLMDFA, NeurIPS 2024](https://arxiv.org/html/2402.10754v2) | Separate extraction, per-function flow summaries and solver-backed path checks. | Long prompts, large functions, complex pointers and incorrectly encoded conditions limit reliability/scalability. | Method/limitations reviewed; no theorem-prover or model execution claimed. |
| [RepoAudit, ICML 2025](https://proceedings.mlr.press/v267/guo25n.html) | On-demand repository exploration with memory and dataflow/path validators. | Its real-project bug-discovery evaluation has a different denominator and context from isolated-function classification. | Official paper reviewed at abstract level; **not run**. Repository context/validation is a future extension. |
| [ReVD, Findings ACL 2025](https://aclanthology.org/2025.findings-acl.467/) | Synthetic reasoning and curriculum preference optimization for detection. | Training data construction and optimization are part of the method; a generic prompt cannot reproduce it. | Reviewed official abstract; **not trained or run**. |
| [VulnLLM-R, 2025 preprint](https://arxiv.org/abs/2512.07533) | Specialized reasoning data/training and a repository agent scaffold. | Claims require the actual weights, training recipe and correct benchmark setting. | Reviewed official abstract/repository; **not run**, no model weights or local GPU server available in this run. |
| [LLM-based Vulnerability Detection at Project Scale, 2026 v2](https://arxiv.org/html/2601.19239v2) | Broad project-scale comparison and manually validated warning/error analysis. | Incomplete API models, truncated context and prompt/reasoning errors cause false alarms; cost can be large. | Study design and error analysis reviewed; motivates explicit cost/coverage and future manual case studies. |

## Executed tool families

| Tool/source | What was actually run | Interpretation |
|---|---|---|
| [Semgrep](https://docs.semgrep.dev/writing-rules/data-flow/taint-mode/overview) | Version recorded in coverage.json; original local C/C++ rule pack and a separate explicit-source taint rule. | Intraprocedural pattern/taint analysis, with partial parse diagnostics retained. Not whole-program proof. |
| [Cppcheck](https://github.com/cppcheck-opensource/cppcheck) | Tag 2.22.0 built locally; warning checks, normal check level, inconclusive reports, four workers. | Different settings/version from the historical thesis; rerun results are independent measurements. |
| [Flawfinder](https://dwheeler.com/flawfinder/) | Version recorded in coverage.json; level >=1 warning baseline. | Broad suspicious-API alert baseline, not semantic vulnerability confirmation. |

## Lessons translated into implementation

1. Compare methods on identical sample IDs; list failures and coverage separately.
2. Train-only TF-IDF dictionaries; commit-disjoint validation fitting/selection.
3. Evaluate token, character, shape and tool-presence representations separately.
4. Test fusion against simple mean/log-odds combinations and intercept-only
   recalibration. A named prototype must earn an advantage in ablations.
5. Count each tool-rule event once. Treat correlated engines and distribution
   shift as threats, rather than claiming independent Bayesian evidence.
6. Compare patched/vulnerable pairs with frozen scores/thresholds. Detecting a
   project's coding style is insufficient evidence of vulnerability reasoning.
7. Retain the taint rule's zero-yield result and its known array-update false
   negative. A plausible dataflow idea is not assumed to improve performance.

## What a defensible next replication needs

Use the existing frozen cohorts to run actual CodeBERT/LineVul and Devign-family
artifacts with pinned tokenizer/graph-tool versions; then use a chronological
PrimeVul paired split. Compare independent-prompt LLM, static-primed LLM,
verify-only pipeline, deep pipeline, and critic/solver variants with the same
inputs, failure policy, token budgets and costs. Record model weight digests,
prompt/schema hashes, and every verdict, including errors. Refit calibration
only on validation. Validate real-project alerts against fixes or maintainers.
No universal novelty, SOTA, or acceptance claim follows from this review.
