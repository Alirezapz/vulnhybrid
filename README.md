# VulnHybrid

**C/C++ vulnerability triage with static analysis, optional LLM verification,
and reproducible research experiments.** Bachelor project by Alireza Pazouki.

Version 1.2.0 adds measured graph and all-family fusion experiments:
**78 neural training runs, 32 classical graph models and 267 matched
configuration/cohort comparisons**. The count includes seed repeats and
ablations, not 267 distinct algorithms. All scores and negative results are
retained. **The validation-selected graph model has not demonstrated a reliable
primary MCC gain over token LR.** This repository does not claim state-of-the-art
detection or full reproduction of every named graph paper.

- [Graph study and complete comparisons](docs/GRAPH_STUDY_REPORT.md)
- [Graph research report](docs/GRAPH_STUDY_REPORT.md)
- [Graph reproduction instructions](docs/GRAPH_REPRODUCTION.md)
- [Graph exactness/execution ledger](docs/GRAPH_REPLICATION_LEDGER.md)
- [Recent graph and graph/LLM papers](docs/GRAPH_RECENT_WORK.md)

- [Experiment report and all comparisons](docs/EXPERIMENT_REPORT.md)
- [Earlier research report](docs/EXPERIMENT_REPORT.md)
- [Code/scientific audit](docs/CODE_AUDIT.md)
- [Related work and replication ledger](docs/LITERATURE_REVIEW.md)
- [Frozen evaluation protocol](docs/EXPERIMENT_PROTOCOL.md)
- [Original pipeline design](docs/PIPELINE.md)

## Research artifact availability

This checkout contains the maintained scanner, research implementations,
technical reports and tests. Raw numerical experiment records and fitted models
are packaged separately; the research release has not yet been published.
See [DOWNLOADS](docs/DOWNLOADS.md) for the prepared artifact layout.

Academic manuscripts, thesis files, obsolete notebooks and temporary sample
verdicts are excluded from this public source distribution.

## Scanner

From this source checkout, using Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,openai]'
python -m vulnhybrid scan examples/vulnerable-code --format json -o scan.json
```

Optional Cppcheck: install it through your system package manager. Scanning
with both engines:

```bash
python -m vulnhybrid scan src --tools semgrep cppcheck --format sarif -o scan.sarif
```

For a running local Ollama model:

```bash
python -m vulnhybrid scan src --mode triage --provider ollama --model qwen2.5-coder:7b
python -m vulnhybrid scan src --mode deep --provider ollama --model qwen2.5-coder:7b
```

Triage verifies only static candidates; deep mode verifies every extracted
function and can cost more. Hosted providers are optional. Source code is sent
to the selected provider only when configured. Mock tests do not measure an
LLM's vulnerability-detection ability.

Output formats: terminal, JSON, Markdown, SARIF. Statuses include confirmed,
uncertain, review, cleared and clear. **Confirmed means the score crossed a
threshold; it is a detector prediction, not a proven vulnerability.** Scores
reuse historical Juliet-derived evidence and are not calibrated deployment
probabilities. Analyzer failure is an error, rather than an empty clean scan.

## Graph measurements (v1.2)

Actual PyG GCN, GraphSAGE, GIN, GAT, R-GCN and GGNN operators are compared with
DeepSets, WL features, structural boosting, representation/edge ablations and
18 fusion recipes per cohort. Three primary training seeds, perturbation
diagnostics and independent score verification accompany the results.

The primary commit-disjoint test has 2,500 functions. Each method's threshold
below maximizes MCC on val_select; the final configuration is also chosen there.

| Configuration | Accuracy | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|
| Token LR | 0.5096 | 0.5059 | 0.9888 | 0.6694 | 0.0565 |
| Typed GGNN + sum readout (validation-selected) | 0.5284 | 0.5209 | 0.7530 | 0.6158 | 0.0616 |
| Relation-gated prototype | 0.5260 | 0.9861 | 0.0566 | 0.1070 | 0.1667 |
| All-family logit stack | 0.5220 | 0.9688 | 0.0494 | 0.0940 | 0.1513 |

The selected model's MCC difference from token LR is 0.0051, with a 95%
commit-cluster interval of **[-0.0916, 0.1161]**. The relation gate/stack have
higher test MCC at very low recall; they were not the final validation winners.
Showing those rows does not retrospectively select a better test result.
The selected model correctly classifies both versions in only **1/415 SVEN
pairs and 2/348 verified PrimeVul-mirror pairs**. External generalization remains
poor. The report includes every configuration under both F1/MCC threshold
policies, uncertainty, calibration and low-FPR/review-budget analyses.

Our tree-sitter AST/structured-CFG/last-identifier frontend is approximate;
it is not a sound CPG/PDG. Named-paper architecture ports are explicitly marked.
The input audit and matched richer-representation controls expose lost operators,
child roles, parse errors and truncation. See the exactness ledger before
describing any experiment as Devign, ReVeal, IVDetect, DeepWukong or AMPLE.

For the complete graph run, use [GRAPH_REPRODUCTION](docs/GRAPH_REPRODUCTION.md)
and [requirements-graph-lock.txt](requirements-graph-lock.txt). Graph dependencies
are optional and do not change the static scanner's default installation.

## Earlier measurements (v1.1, F1 selection)

The primary protocol groups entire commits, removes token-identical clones,
fits on train, fits calibration on one validation half, and chooses thresholds
on a different validation half. The test partition has 2,500 functions.

| Primary method | Accuracy | F1 | MCC |
|---|---:|---:|---:|
| Always vulnerable | 0.5020 | 0.6684 | 0.0000 |
| Token logistic regression | 0.5096 | 0.6694 | 0.0565 |
| Mean token/character scores | 0.5112 | 0.6696 | 0.0622 |
| Gated residual (validation-selected) | 0.5116 | 0.6688 | 0.0597 |

The tiny apparent F1 gains are not reliable improvements. On the retained
415 external SVEN repair pairs, the validation-selected method correctly
classifies both versions in **0 pairs**. Its high recall/F1 reflects an
all-positive external operating point. Read the report's intervals, MCC,
false-positive rates and paired outcomes alongside F1.

The original official split has 1,902 commit IDs spanning partitions; after
code deduplication it still has 1,891. The stronger primary split has zero
commit or normalized-code overlap. This does not eliminate near-clones,
chronological leakage or incorrect labels.

## Reproduce the research

The complete run used Python 3.12.14 and pinned versions in
[requirements-research-lock.txt](requirements-research-lock.txt). Upstream
C/C++ datasets retain their original rights and are downloaded separately.

```bash
python -m pip install -e '.[research,dev,openai]'
python -m pip install -r requirements-research-lock.txt
python -m research.fetch_data --out data/research
# Cppcheck 2.22.0 must be on PATH; CPPCHECK_BIN can point to another installation.
python -m research.collect_static \
  --samples data/research/devign/samples.jsonl data/research/sven/samples.jsonl \
  --out results/reproduction/static
```

Run each regime with the same source data and findings:

```bash
python -m research.benchmark --samples data/research/devign/samples.jsonl \
  --external data/research/sven/samples.jsonl \
  --findings results/reproduction/static/findings.jsonl \
  --out results/reproduction/commit_disjoint --commit-disjoint

python -m research.benchmark --samples data/research/devign/samples.jsonl \
  --external data/research/sven/samples.jsonl \
  --findings results/reproduction/static/findings.jsonl \
  --out results/reproduction/official_dedup

python -m research.benchmark --samples data/research/devign/samples.jsonl \
  --external data/research/sven/samples.jsonl \
  --findings results/reproduction/static/findings.jsonl \
  --out results/reproduction/ffmpeg_to_qemu --source-project FFmpeg

python -m research.benchmark --samples data/research/devign/samples.jsonl \
  --external data/research/sven/samples.jsonl \
  --findings results/reproduction/static/findings.jsonl \
  --out results/reproduction/qemu_to_ffmpeg --source-project qemu

python -m research.summarize --root results/reproduction \
  --samples data/research/devign/samples.jsonl data/research/sven/samples.jsonl
```

Set OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1 to match the recorded run.
The collector preserves diagnostics and coverage. It does not silently treat
partial parsing as proof of safety. The report includes a common diagnostic-free
sensitivity analysis. No GPU or model API is required for these CPU experiments.

## Resume and old results

Research LLM runs now remove explicit Juliet benchmark hints by default and
fingerprint actual model/prompt/input/rules/schema. Changed or legacy checkpoints
require a new output path; they cannot silently mix with a new run. Failed
verdicts are retried on resume. Weight fitting requires validation.

Earlier academic manuscripts are excluded from this distribution. Their historical
performance claims were not reproduced by this upgrade. Version 1.2 runs graph
operators and explicit architectural ports as described above; it does not
reproduce complete named-paper pipelines. LineVul, IRIS, RepoAudit and large
language models remain unexecuted. The literature ledgers list exactly what was
reviewed, implemented, executed, and remains open.

## Tests and layout

```bash
python -m pytest -q
PYTHONPATH=pipeline python -m llm_stage.test_llm_client
SEMGREP_ENABLE_VERSION_CHECK=0 semgrep --test \
  --config research/rules/taint.yml research/rules/taint_test.c
```

`vulnhybrid/` contains the CLI scanner; `pipeline/` contains dataset, analysis,
LLM and scoring stages; `research/` contains pinned-data acquisition, experiments
and report generation; `tests/` contains regression tests; `results/2026-11-01/`
contains the earlier measured evidence; `results/2026-11-01-graphs/` contains
the graph/fusion study; `docs/` contains the audit, review and protocols.

All **90 tests pass** in the recorded CPU environment, and a separate verifier
recomputes exported metrics/selections and checks full training membership for
all 78 neural runs. Trained models and raw numerical records are separate release assets. Graph
caches are regenerated with the pinned extraction commands.

MIT for the project code, with the original copyright preserved. See
[LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
