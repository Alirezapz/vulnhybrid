# Source and scientific audit — 1 November 2026

The supplied ZIP has 96 entries, including code, example programs, notebooks,
English paper drafts, and a Persian thesis. The architecture is a functional
five-stage pipeline plus a CLI; it is not a multi-LLM analyst/critic/judge system.
This audit covers loaders, splitting, attribution, static rules/runners, prompts,
provider parsing/retries, scoring, evaluation, statistics, CLI reports, tests and
notebook experiment setup. Historical quantitative claims are distinguished
from newly reproduced measurements.

## Architecture understood

| Component | Actual behavior | Consequence |
|---|---|---|
| schema / loaders | Common JSONL; Juliet guarded bad/good blocks; Devign function snippets | A Juliet sample can contain several functions, while Devign is isolated-function data. |
| splitter | Group Juliet flow variants using filename stems | Does not prevent label hints inside code, pretraining memorization, Devign clones or same-commit overlap. |
| static analysis | Batch Semgrep patterns and Cppcheck; map original-file lines to snippet offsets | C/C++ parse/context failures must be counted separately from no findings. |
| LLM | One provider call per selected sample, structured verdict, retry and JSON fallback | Mock-server tests validate transport/parsing, not model security expertise. |
| scoring | Imported per-rule likelihood ratios, prior, log-odds mixture | Correlation and dataset shift prevent interpreting the score as a universally calibrated posterior. |
| evaluation | Historical full-static vs verdict-subset comparison | Mismatched populations cannot establish hybrid superiority. |
| CLI | Function extraction, triage/deep/static modes, terminal/JSON/Markdown/SARIF | Triage recall is bounded by candidate recall; a status named confirmed is a model decision, not proof. |

## Findings and changes

| Priority | Finding | Evidence / change |
|---|---|---|
| High | A textual false became True through Python bool conversion | llm_client.args_to_verdict now requires a JSON boolean. Regression covers safe verdict reversal. |
| High | Nonfinite/out-of-range confidence accepted | Reject null, strings, booleans, NaN, infinity and values outside [0,1]. |
| High | Multiple matches from one rule multiplied the same binary LR repeatedly | Count a tool-rule event once; accumulate in stable log odds. Cross-tool correlation still remains. |
| High | Analyzer timeout, malformed output or bad configuration returned an empty success | Production runners raise AnalysisError; CLI scan returns error exit code. Research collector saves diagnostics and measures alert coverage separately. |
| High | No installed analyzer could produce a clean-looking report | Static/triage scans of source files now fail when no requested analyzer ran. |
| High | Main evaluation table compared different denominators | Main comparison now includes a matched static row and explicit missing/failed-verdict coverage. Full-static metrics are separate. |
| High | Duplicate verdict IDs silently overwrote previous results | Ambiguous verdict joins fail; choose one model/run explicitly. |
| High | Resume reused IDs regardless of model, prompt or changed inputs | Persist request hashes over actual prompt/code/rules/provider/model/schema and check before reuse; retry failed records; reject legacy/mismatched checkpoints. |
| High | Juliet good/bad, CWE names and comments can reveal labels in model inputs | New label-blind preprocessing is the default for research LLM runs on Juliet; preserve source lines and literals. A legacy-hints flag supports ablation. |
| High | CodeXGLUE official membership was asserted to be leakage-free | Removed this assertion; measured clone/conflict and same-commit overlaps. Added commit-disjoint protocol. |
| Medium | F1 became NaN with no correct positive predictions | Use direct 2TP/(2TP+FP+FN), with explicit zero convention; reject invalid metric inputs. |
| Medium | The limit sampled the first source-ordered cases | Seeded shuffle independent of input order, including uncapped data. |
| Medium | Test/mixed partitions were permitted for weight fitting | Calibration CLI now requires validation; invalid grid steps fail instead of looping indefinitely. |
| Medium | Strong calibration/independence claims exceed evidence | README now describes heuristic risk scores; old results and drafts are marked historical/unverified in this ZIP. |
| Medium | Unvalidated arithmetic rule is loaded but excluded from security scoring | It remains a historical draft; new benchmark excludes it from features and alerts. |

## What was and was not recoverable

The ZIP has three files of ten LLM verdicts each. It does not include the
192,201-row combined dataset, full findings, full validation/test model
verdicts, fitted configurations or complete model digests supporting the main
paper. Notebook outputs were largely cleared. The original numbers, including
model scaling, calibration, priming, and significance, are therefore
**historical claims that cannot be independently reproduced from this ZIP**.
They are not necessarily false. New runs use newly acquired pinned upstream
data and publish their own raw prediction evidence.

The README's historical claim of calibrated confidence conflicts with the
later draft's own overconfidence discussion. Two tested model sizes cannot
establish a fundamental, scale-independent capability limit. First-ever
calibration/novelty claims require a stronger literature search.

## Remaining engineering and scientific limitations

- A function can lack macros, type definitions, call context and whole-program
  dependencies. Research alert/no-alert metrics include these misses and report
  diagnostics; production parsing failures are explicit errors.
- The historical LR table still uses Juliet measurements and a hand-specified
  prior. It should not set deployment confidence without target-domain fitting.
- Removing explicit benchmark hints does not eliminate latent template artifacts
  or establish that a public corpus was absent from LLM pretraining.
- Token-identical clone removal does not eliminate near-clones. Commit-disjoint
  splitting is not chronological evaluation.
- The array-update taint fixture is a documented expected miss: assigning a null
  terminator after read can lose buffer taint in this engine. It is not hidden
  by deleting the example or counted as successful detection.
- The scanner does not validate exploitability or execute a proof of concept.
  No newly discovered vulnerability is claimed.
- LLM transport tests are real local mock HTTP tests; no new hosted/local LLM
  capability result is claimed. No API key, inference server or GPU was used.
- The new linear methods are research prototypes. Weak transfer results are a
  reason to keep them out of the default scanner until stronger validation.
