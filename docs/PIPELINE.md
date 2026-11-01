# Dataset pipeline -- SARD / Juliet / Devign

Phase 1 of the hybrid vulnerability-detection project: turns the three
datasets named in the approved proposal into one common, function-level,
binary-labeled format that the static-analysis stage, the LLM stage, and
the evaluation harness can all consume the same way.

## Files

| File | Purpose |
|---|---|
| `schema.py` | The shared `Sample` dataclass + JSONL read/write/summarize helpers |
| `sard_manifest_parser.py` | Generic parser for any SARD-format test suite (manifest.xml + source tree). Juliet is one such suite, so this is shared infrastructure, not Juliet-specific code. **Tested against the real Juliet 1.3 corpus**, not just the documented format. |
| `juliet_loader.py` | Thin CLI wrapper pointing the generic parser at a Juliet checkout |
| `devign_loader.py` | Devign/CodeXGLUE loader, Hugging Face primary path + local-jsonl fallback |
| `splitter.py` | CWE-stratified, leakage-safe train/val/test split for Juliet (see below) |
| `build_dataset.py` | Orchestrator: runs whichever loaders you point it at, writes `juliet.jsonl` / `devign.jsonl` / `combined.jsonl`, prints a summary table |

## Why there's a dedicated splitter

Juliet generates many "flow variant" files per underlying vulnerability --
the same flaw wrapped in dozens of superficially different control/data-flow
patterns (`..._01.c`, `..._02.c`, ... `..._54c.c`). A random per-sample
train/val/test split scatters near-duplicate variants of the same case
across the boundary, which inflates whatever you measure later -- the model
(or a learned hybrid-scoring weight) ends up partially evaluated on code
it's effectively already seen. `splitter.py` groups every flow variant of a
base case together and assigns the whole group to one split, stratified by
CWE so rare categories aren't wiped out of val/test by chance. It also
self-checks the result and raises if any group ends up split across more
than one partition -- this was verified by deliberately injecting a leakage
case and confirming the check fires.

Devign doesn't need this: it already ships official splits, and each
function is an independent real-world sample rather than a templated
variant of another sample in the set.

Turn it on with `--assign-splits` (off by default, so existing commands
behave the same as before):

```bash
python build_dataset.py --juliet-root juliet-test-suite-c --assign-splits --out-dir data
```
Ratios default to 80/10/10 (`--train-frac` / `--val-frac` / `--test-frac`),
matching Devign's existing split proportions. Small CWE categories get a
guaranteed minimum: any CWE with >=3 base-case groups gets at least 1 group
in val AND 1 in test even if its raw share would round to 0; with exactly 2
groups it gets 1 train + 1 test; with exactly 1, it goes to train (nothing
to hold out). 61 of Juliet's 118 CWEs have fewer than 10 base-case groups
(verified against the real manifest), so without this floor, roughly half
of all CWE categories would have ended up with zero validation coverage.

## A fully-explained, harmless discrepancy you may notice

The split log reports "118 CWE strata" while the overall summary reports
"Distinct CWEs: 119" -- these measure two different things and are *not*
expected to match. The 119-count comes from the CWE the manifest's
`<flaw name="...">` attributes, while grouping for splitting purposes uses
the CWE that LEADS each file's own name (it has to, since that's what
exists on disk). The one CWE that's flaw-attributed but never leads
anyone's filename is **CWE-135**: every test case it's cited in is
physically packaged inside the CWE-121 (stack buffer overflow) family's
files as a secondary/compound tag, e.g.
`CWE121_Stack_Based_Buffer_Overflow__CWE135_01.c`. Those files are
correctly grouped under CWE-121 for splitting -- that's the right behavior,
since that's their actual file family -- they just don't generate a
separate CWE-135 stratum. Verified directly against the manifest, not
inferred.

(Tangential, separately verified, and not a bug: a handful of low CWE
numbers -- 15, 23, 36, 78, 90 -- are written without zero-padding when they
lead a filename (`CWE23_...`) but WITH zero-padding when cited inside
another file's flaw name (`CWE-023: ...`). Each representation is
internally consistent on its own, so this doesn't fragment or double-count
anything -- it's just a cosmetic quirk of how Juliet itself names things.)

## Output schema

Every sample, regardless of source, looks like this:

```json
{
  "sample_id": "juliet_CWE78_..._bad",
  "dataset": "juliet",
  "language": "c",
  "code": "void CWE78_..._bad() { ... }",
  "label": 1,
  "cwe": "CWE-78",
  "cwe_name": "CWE-78: OS Command Injection",
  "source_file": "CWE78_.../CWE78_..._01.c",
  "flaw_line": 47,
  "split": null,
  "extra": {}
}
```

`label`: `1` = vulnerable, `0` = not vulnerable. `flaw_line` refers to the
line number in the *original* source file, not the extracted snippet.

## A design decision worth knowing about (ties back to the proposal review)

The proposal's evaluation section never states what one "instance" is for
TP/FP/TN/FN purposes. Devign only ever existed at function granularity, so
this pipeline standardizes on **function-level** samples everywhere: rather
than treating a whole multi-function Juliet file as one sample, each
file's `bad()` and `good()`/`goodG2B()`/etc. functions are sliced out
separately via their `#ifndef OMITBAD` / `#ifndef OMITGOOD` guards. That
puts Juliet and Devign on the same footing in the evaluation harness later.
If you'd rather evaluate at file or line level for some reason, say so --
it changes how Phase 5 (evaluation) has to be written.

## Getting the data

**Juliet C/C++** -- recommended source is the GitHub mirror, which is
already laid out exactly how `sard_manifest_parser.py` expects (manifest.xml
at the root, `testcases/<CWE_DIR>/*.c`):

```bash
git clone --depth 1 https://github.com/arichardson/juliet-test-suite-c.git
```

The authoritative source is NIST's own download
(<https://samate.nist.gov/SARD/test-suites/112>, Juliet C/C++ 1.3,
public-domain / CC0) if you want to cite the original rather than a mirror
-- same format, just point `--root`/`--manifest` at wherever you unzip it.

**Devign** -- via Hugging Face (`pip install datasets`), dataset id
`google/code_x_glue_cc_defect_detection`, fields `func`/`target`/`idx`,
pre-split 80/10/10. **Caveat:** Hugging Face Hub is a US-based service; if
it's unreachable from your network, use the `jsonl-dir` fallback documented
at the top of `devign_loader.py`, which only needs Google Drive via `gdown`
and the original CodeXGLUE repo's preprocessing script.

**SARD (the broader collection)** -- there isn't a single bulk "SARD"
archive separate from its individual test suites; Juliet *is* one of those
suites (the proposal says as much). `sard_manifest_parser.py` is written
generically against the manifest.xml schema so it should work unmodified
against other SARD suites that use the same schema -- but I've only verified
that schema directly against Juliet 1.3, so spot-check the first run
against any additional suite before trusting it at scale.

## Quick start

```bash
pip install -r requirements.txt   # only needed for the Devign HF path

python build_dataset.py \
    --juliet-root juliet-test-suite-c \
    --devign-source huggingface \
    --out-dir data
```

Smoke-test on a slice first (Juliet's manifest has ~64k entries; this caps
how many get processed):

```bash
python build_dataset.py --juliet-root juliet-test-suite-c --limit 200
```

Each loader is also runnable standalone (`python juliet_loader.py --help`,
`python devign_loader.py --help`) if you only want one dataset at a time.

## What's NOT in this phase

The LLM verification stage, hybrid scoring, and the full evaluation harness
are Phases 3-5 -- Phase 1 gets clean labeled data into `data/combined.jsonl`,
Phase 2 (below) adds static-analysis findings on top of it.

---

# Phase 2 -- static analysis integration

## Tool choice: semgrep, with local custom rules

Semgrep was chosen over Cppcheck/Flawfinder for one concrete reason found
by inspecting real Juliet source: Cppcheck needs a fairly complete
translation unit to do its dataflow checks, and most of what Phase 1
extracts is an isolated function (no surrounding `#include`s, no
`std_testcase.h` macros). Semgrep's pattern matching tolerates that fine.
Devign makes this non-negotiable in one direction: there is no original
file for Devign samples at all (CodeXGLUE only ever released the extracted
function), so whatever tool is used MUST tolerate snippet-only input for
that dataset regardless of what's chosen for Juliet.

Rules live in `static_analysis/rules/*.yml` and are loaded from disk, not
from `--config=auto`/`p/c` (the Semgrep Registry, semgrep.dev). Same
reasoning as the Hugging Face caveat in Phase 1: a second silent
dependency on a US-based service being reachable isn't worth it when local
rules work as well and are fully auditable.

## Files

| File | Purpose |
|---|---|
| `findings_schema.py` | `Finding` dataclass + JSONL I/O, deliberately separate from `Sample` -- a Finding is a tool's opinion, a Sample is ground truth |
| `static_analysis/rules/*.yml` | The actual semgrep rules (see below) |
| `static_analysis/semgrep_runner.py` | Wraps the semgrep CLI; `run_on_directory()` for Juliet, `run_on_snippet()` for Devign |
| `static_analysis/attribute_findings.py` | Maps file:line findings back to the correct sample via `line_ranges` |
| `run_static_analysis.py` | CLI entry point |

## A real performance lesson, not a style choice

The first version of `attribute_findings.py` ran semgrep once PER FILE
(grouping samples by their shared original file first). Across the
~20,000-file test corpus this timed out -- subprocess startup overhead
dominates at that scale. The fix was a single semgrep invocation across the
WHOLE tree (`run_on_directory`, `cwd` set to the checkout root so result
paths come back matching `Sample.extra['file_path']` exactly), which
scanned the same ~20,000 files in under 5 minutes. If you extend this to
more rules/files, keep the one-invocation-per-tree shape; don't go back to
per-file subprocesses.

## Phase 1 had to grow two new fields to make this possible

`sard_manifest_parser.py` now records, per sample: `extra['file_path']`
(the original file, root-relative) and `extra['line_ranges']` (1-indexed
`[start, end]` pairs locating that sample's extracted code inside the
original file). Static analysis runs against the REAL file (proper
`#include`s, macros, full context) and findings are mapped back to
whichever sample's range contains the flagged line. **Re-run
`build_dataset.py` if your `data/*.jsonl` predates this change** --
`run_static_analysis.py` will skip (with a warning, not silently) any
Juliet sample missing this metadata.

## The rules, and what they're honestly scoped to catch

- **`dangerous-copy`** (CWE-120/121/122): unbounded string/memory functions
  with no destination-size parameter at all (`strcpy`, `strcat`, `gets`,
  `sprintf`). Deliberately does NOT attempt the two other CWE-121/122
  mechanisms found by inspecting real Juliet source: raw array-index writes
  with an incomplete bounds check (`buffer[data] = 1` checked only for
  `data >= 0`), and `memcpy`/`snprintf` calls where the SIZE ARGUMENT
  itself is wrong (e.g. `sizeof(*wholeStruct)` instead of
  `sizeof(oneField)`). Both require real bounds/size reasoning, not syntax
  matching -- documented as a known gap, not silently missed.
- **`command-injection`** (CWE-78): `system`/`popen`/`execl`-family calls
  -- AND Juliet's own uppercase macro aliases (`EXECL`, `POPEN`, etc.),
  found by inspecting real source; a rule matching only the bare libc names
  would have silently missed every Juliet CWE-78 case, since Juliet's
  actual call sites go through these macros.
- **`format-string-non-literal`** (CWE-134): printf-family calls whose
  format argument is a bare variable rather than a string literal.

## Measured against real ground truth (not assumed)

Ran the full pipeline against every CWE-78 and CWE-134 sample, plus a
representative slice of CWE-121/122 (3 of 9/11 sink-variant subfolders
each), and checked each rule's hits against the dataset's own labels:

| Rule | CWE | Recall (label=1) | False positive rate (label=0) |
|---|---|---|---|
| `command-injection` | 78 | 60.0% | 60.0% |
| `format-string-non-literal` | 134 | 41.7% | 41.7% |
| `dangerous-copy` | 121 | 9.6% | 9.7% |
| `dangerous-copy` | 122 | 5.7% | 5.6% |

**Recall and false-positive rate come out essentially equal for every
rule.** Traced this to its actual cause rather than just reporting the
number: Juliet's "good" samples for these CWEs are heavily
"goodG2B"/"goodB2G" flow variants, where the sink call is preserved
byte-for-byte between the safe and vulnerable version -- only the upstream
data source differs. One real example, found by reading the actual source
(`CWE78_OS_Command_Injection__char_connect_socket_execl_01.c`):

```c
/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B() {
    char * data;
    char dataBuffer[100] = COMMAND_ARG2;
    data = dataBuffer;
    strcat(data, "*.*");   /* FIX is here: fixed string, not external input */
    EXECL(COMMAND_INT_PATH, COMMAND_INT_PATH, COMMAND_ARG1, COMMAND_ARG3, NULL);
}
```

The comment inside Juliet's own "safe" function literally says
`POTENTIAL FLAW: Execute command without validating input`. A rule that
only inspects the sink call cannot do better than chance on cases like
this -- which is the precise, measured justification for the LLM semantic
stage (Phase 3): it needs to trace `data` back to its source, not just
recognize the sink.

## Quick start

```bash
pip install semgrep
python run_static_analysis.py --samples data/combined.jsonl \
    --juliet-root juliet-test-suite-c --out data/findings.jsonl
```
Set `SEMGREP_BIN` if semgrep is only installed inside a venv that isn't on
PATH. `findings.jsonl` joins back to `combined.jsonl` by `sample_id`.

## What's NOT in this phase (yet)

More rules for the other ~115 CWEs in the corpus; the hybrid scoring stage
that combines this output with the LLM stage (Phase 4).

---

# Cppcheck addition -- closing (part of) the array-bounds gap

## Don't assume a "real" static analyzer automatically fixes a syntactic
## rule's gap -- measure it

Before writing any integration code, tested cppcheck directly against the
exact two real cases that defeated `dangerous-copy`:
the array-index-with-incomplete-bounds-check case
(`CWE121_..._connect_socket_01.c`) and the
memcpy-with-wrong-sizeof-argument case
(`CWE122_..._memcpy_01.c`). Tried `--enable=all`,
`--check-level=exhaustive`, `--inconclusive`, and the real
`testcasesupport/std_testcase.h` on the include path. **Cppcheck reported
nothing for either case** -- confirmed via `--checkers-report` that the
relevant checkers (`CheckBufferOverrun::bufferOverflow`,
`CheckBufferOverrun::arrayIndex`) were genuinely ACTIVE, not disabled by
configuration. So for these two specific patterns, cppcheck is not "the
fix" -- it's a different tool with different blind spots, not a strictly
better one.

## What cppcheck DOES add: measured against the broader real corpus

Ran cppcheck against the full CWE-121/122 subset on disk (3 of 9/11
sink-variant subfolders each, 5,542 files) and checked its hits against
ground truth the same way as the semgrep rules:

| Check | CWE | Recall (label=1) | False positive rate (label=0) |
|---|---|---|---|
| `arrayIndexOutOfBounds` | 121 | 1.0% | **0.0%** |
| `arrayIndexOutOfBounds` | 122 | 10.6% | **0.0%** |
| `nullPointer` | 121 | 1.1% | 1.1% |
| `nullPointer` | 122 | 10.2% | 11.2% |
| `uninitvar` | 121 | 13.0% | 13.8% |

This is a genuinely different shape of result from every semgrep rule:
**`arrayIndexOutOfBounds` has low recall but a measured 0% false positive
rate** -- it misses most of the array-bounds cases, but when it does fire,
it has not been wrong once in this corpus. That's valuable specifically
*because* it's asymmetric: in Phase 4's hybrid scoring, a sparse,
high-precision signal like this is worth weighting heavily when present,
unlike `command-injection` or `nullPointer`, which (per the recall ≈ FP
rate pattern) carry close to zero information on their own.

## Usage

```bash
apt-get install cppcheck   # or see cppcheck.sourceforge.io for other platforms
python run_static_analysis.py --samples data/combined.jsonl \
    --juliet-root juliet-test-suite-c --tools semgrep cppcheck \
    --cppcheck-include-dir juliet-test-suite-c/testcasesupport \
    --out data/findings.jsonl
```
`--cppcheck-include-dir` points at Juliet's `testcasesupport/` (defines
`std_testcase.h`) -- optional, just avoids include-not-found noise.
**Cppcheck is much slower than semgrep per file** (~29 files/sec vs ~58 in
this environment, both single-core) -- budget accordingly for a full
~105k-file run, and raise `--timeout` if needed. One real limitation found
while testing: cppcheck does **not** follow symlinked directories (errors
with "could not find or open any of the paths given") -- if you ever want
to scope a run to a subset via symlinks the way this was tested, copy the
files instead.

A real Windows-specific bug, found and fixed during actual use: Python's
`subprocess.run(..., text=True)` decodes subprocess output using the
system's default codepage (cp1252 on a typical Windows install), not
UTF-8. Real source code -- comments with non-ASCII characters, author
names, etc. -- broke this silently: the JSON output got corrupted, parsing
failed, and the failure was swallowed by the JSON-error handler, so the
whole stage just looked like "0 findings" with no visible error. Both
runners now explicitly pass `encoding="utf-8", errors="replace"`. Invisible
on Linux/Mac (UTF-8 is already the default there), which is exactly why it
only showed up once this was run for real on Windows.

Another real bug from actual use: the Devign/snippet path originally ran
one subprocess PER SAMPLE (27,318 of them) -- the same scaling mistake
already fixed once for the Juliet directory scan, reintroduced by not
applying the same fix everywhere. Fixed the same way: write every sample
to one temp directory, scan it in a single invocation
(`run_on_snippets_batch`). Verified at the real scale (27,318 synthetic
samples, 17.6 seconds, 100% accuracy) before considering it done.

---

# Phase 3 -- LLM semantic verification

## What this is for, concretely

Phase 2 measured that most static rules here are close to non-discriminative
on their own (recall ≈ false-positive rate for `command-injection`,
`format-string-non-literal`, `nullPointer`, `uninitvar`) because Juliet's
goodG2B/goodB2G variants preserve the exact sink call between safe and
vulnerable code -- the only difference is where the data came from. Stage 2
exists specifically to answer that question. The system prompt
(`llm_stage/prompts.py`) is built directly around this: trace each
relevant variable back to its source before deciding, don't just recognize
the operation. A static finding, when one exists, is shown to the model as
an unconfirmed lead (worded that way explicitly), not a fact -- the model
is told the tool that produced it has a measured false-positive rate.

## What was and wasn't validated

**No API credentials were available in the environment this was built in**
(checked directly -- a bare request to `api.anthropic.com` returned 401,
no key in the environment). So the actual live model call could not be
tested here. Everything around that call boundary WAS tested, using a
mocked provider function standing in for the real API (`llm_stage/
test_llm_client.py`):
- retry behavior (3 attempts with exponential backoff, confirmed it
  retries exactly as many times as configured and gives up cleanly rather
  than crashing the whole batch on one bad sample)
- response-to-Verdict conversion
- prompt construction, including that the static-evidence block correctly
  appears/disappears and is worded as "unconfirmed"
- the full orchestrator (`run_llm_stage.py`) -- sample selection, the
  findings join, checkpointing -- using the same mocking approach

## Running this for free with a local model instead of a paid API

`llm_client.py` supports `--provider ollama`, which talks to a model
running on your own machine via [Ollama](https://ollama.com) -- no API
key, no cost, at the cost of running on your own CPU/GPU instead of a
provider's. This works because Ollama (like LM Studio and vLLM) exposes
the same OpenAI-compatible `/v1/chat/completions` endpoint that `--provider
openai` already targets -- `ollama` is really just a thin preset with
sensible local defaults (`http://localhost:11434/v1`, a placeholder API
key Ollama ignores).

**What was and wasn't validated here:** the actual Ollama binary couldn't
be installed in the sandbox this was built in (ollama.com isn't reachable
from there). Instead, the real HTTP/JSON request-response path was
validated against a local mock server speaking the identical protocol
(`llm_stage/mock_openai_server.py`) -- including the fallback path for
models that don't reliably honor `tool_choice` (smaller/quantized local
models are less consistent at this than Claude or GPT-4; if a model
ignores the tool-calling instruction and just answers in plain text with
embedded JSON, the client now tries to recover that JSON before giving up).
That confirms the networking and parsing logic is correct. It does **not**
confirm how well any specific small model actually reasons about the
taint-tracing task -- that depends on the model, and you should check it
yourself with a small `--limit` run before trusting it.

### Setup (Ubuntu)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5-coder:7b     # ~4.7GB download, code-specialized model
ollama serve                      # leave running in its own terminal
```

In another terminal:
```bash
python3 run_llm_stage.py --samples data/combined.jsonl --findings data/findings.jsonl \
    --mode verify-only --split test --limit 10 \
    --provider ollama --model qwen2.5-coder:7b --out data/verdicts.jsonl
```

No `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` needed for this path.

**Hardware reality check:** without a discrete GPU, expect roughly
5-15 tokens/sec on a modern laptop CPU -- workable for smoke-testing and a
stratified sample of a few hundred (`--max-per-cwe`), but not for running
across the full ~192k-sample corpus, which would take many hours to days
at that rate. Use the local model for free iteration on the prompt and a
bounded evaluation sample; use a paid API (or a rented GPU) if you need
the full-scale run.

If `qwen2.5-coder:7b` seems to frequently produce `error` entries in
`verdicts.jsonl` (check with `grep -c '"error": null' data/verdicts.jsonl`
vs total line count), try `ollama pull llama3.1:8b` instead -- its
tool-calling template is one of the more reliable ones in Ollama's catalog
-- via `--model llama3.1:8b`.

## Usage

```bash
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY with --provider openai
python run_llm_stage.py --samples data/combined.jsonl --findings data/findings.jsonl \
    --mode verify-only --split test --limit 10 --out data/verdicts.jsonl
```

**Cost warning, read before scaling up:** this calls a paid API once per
sample. `--limit` and `--max-per-cwe` (stratified cap) exist specifically
so you don't accidentally point this at all 192,201 samples. The script
prints total input/output token counts at the end -- check those against
your provider's CURRENT pricing page (not reproduced here, since it
changes and a stale number would be actively misleading).

`--mode verify-only` (default) only runs on samples that already have a
static finding -- cheaper, but the system's recall ceiling becomes static
analysis's recall ceiling, which Phase 2 measured to be genuinely low for
several CWEs. `--mode all-samples` runs on everything in scope regardless,
costing proportionally more but not inheriting that ceiling.

---

# Phase 4 -- hybrid scoring

## The weights are measured, not invented

Phase 2 measured an actual (recall, false-positive-rate) pair for every
rule against real ground truth. A rule's **likelihood ratio**
(recall / false-positive-rate) is exactly how much that rule firing should
multiply your odds of "this is vulnerable" -- basic Bayesian reasoning, not
a tuning knob. `hybrid_scoring.py`'s `RULE_STATS` table is those Phase 2
numbers directly:

- `dangerous-copy` on CWE-121/122: LR ≈ 1.0 (recall ≈ FP rate) -- firing
  barely moves the prior at all. Verified: feeding it through the actual
  formula moves a 0.4 prior to 0.398 -- the math confirms what Phase 2
  already showed empirically, not a coincidence.
- `arrayIndexOutOfBounds`: 0 measured false positives (Laplace-smoothed
  rather than treated as literally impossible) -> a large LR -> firing
  pushes the same 0.4 prior up to 0.97.

Multiple static findings on one sample combine via sequential odds
multiplication (each rule's LR is independent evidence against the same
prior). The LLM verdict combines with the static evidence via a weighted
average in **log-odds space** (`w_llm=0.7` by default) rather than another
LR multiplication -- the LLM's confidence is already its own probability
estimate (a posterior), so treating it as a likelihood ratio would
double-count the prior.

**`w_llm=0.7` is a starting point, not a fitted value.** Once Phase 5 has
real held-out results (not the simulated verdicts used for testing here),
fit it properly -- a small grid search or 1-variable logistic regression
against the validation split, rather than trusting the default forever.

Validated against real Phase 2 data (no LLM verdict available, so static-
only fallback): ranked the top/bottom 100 samples by score on a 2,000-
sample real slice -- the top 100 were 100% genuinely vulnerable, the
bottom 100 were 100% genuinely safe, against an overall base rate of 38%.
Also validated the disagreement case explicitly: a near-neutral static
signal (`dangerous-copy`) combined with a confident LLM "safe" verdict
correctly lands on "safe" -- the LLM's semantic judgment dominates a weak
static signal, which is the entire point of the architecture.

---

# Phase 5 -- evaluation harness

Computes the proposal's exact defined metrics (Accuracy, Precision,
Recall, F1) for three configurations on the same held-out test set:
static-only, LLM-only, and hybrid. This three-way comparison is the actual
evidence for "hybrid beats either approach alone" -- the proposal asserts
it, this is what lets you measure whether it's true on your data.

```bash
python evaluate.py --samples data/combined.jsonl --findings data/findings.jsonl \
    --verdicts data/verdicts.jsonl --split test
```

Degrades gracefully and explicitly: if `--verdicts` is omitted (Phase 3
hasn't run yet) or only covers part of the split, it reports static-only
metrics over the full split and llm-only/hybrid metrics over however many
samples actually have a verdict -- with that count printed, never silently
treated as the same denominator as static-only.

**Validated end-to-end** with real Juliet samples/findings and a
*simulated* LLM (85% accuracy by construction, confidence correlated with
correctness -- not a strawman perfect oracle) standing in for a real one:

| Config | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| static-only | 0.452 | 0.344 | 0.498 | 0.407 |
| llm-only (simulated) | 0.859 | 0.785 | 0.863 | 0.822 |
| hybrid | **0.874** | **0.814** | 0.863 | **0.838** |

Hybrid beating the simulated LLM alone on every metric except recall (tied)
confirms the scoring mechanism extracts real value from the static
evidence even though that evidence is individually weak -- exactly the
behavior the architecture is supposed to produce. **These specific numbers
are from a simulated LLM and will not match your real results** -- rerun
this once `run_llm_stage.py` has produced real verdicts.

## What's NOT done

The actual live LLM call needs your own API key to validate (see Phase 3).
`w_llm` needs fitting against real (not simulated) held-out results. More
static rules/CWEs would narrow the static-only gap further. The "Distinct
CWEs: 119 vs 118 strata" and similar Phase 1 footnotes still apply to
whatever subset of the corpus you actually evaluate on.

---

# Addendum — memory-lifecycle rules (post-thesis extension)

Phase 5's per-CWE breakdown found the code-LLM strong on tainted-data-flow
CWEs but weak on **memory-lifecycle / state-tracking** flaws (CWE-401 memory
leak recall 0.00, CWE-415 double-free accuracy ~0.31, CWE-457 uninitialized
0.31). That is exactly the regime where syntactic/value-flow static analysis
is comparatively strong, so three Semgrep rules were added in
`static_analysis/rules/memory_safety.yml` to *complement* the model there
rather than duplicate its strengths:

| Rule | CWE | What it matches | Deliberate scope limit |
|---|---|---|---|
| `double-free` | CWE-415 | `free(p); ... free(p);` | reassignment-aware (`free(p); p = new; free(p);` is NOT flagged); intra-procedural only |
| `use-after-free` | CWE-416 | `free(p)` then a direct deref `*p` / `p->f` / `p[i]` | reassignment-aware; passing the freed pointer to a callee is not flagged (too noisy without taint) |
| `return-stack-address` | CWE-562 | `return &local;` / `return local_array;` | a `static` local would be a false positive (rare; not special-cased) |

**Validation status — now MEASURED on Juliet** (see
[`RESULTS_memory_lifecycle.md`](RESULTS_memory_lifecycle.md) for the full
write-up). Beyond the crafted good/bad pairs
(`examples/vulnerable-code/memory_lifecycle.c`), all three CWEs were measured
against the real Juliet 1.3 subsets (3,588 labelled samples). Headline: every
detector had a **measured 0% false-positive rate**, and Cppcheck's dataflow
checkers far outperform the syntactic Semgrep rules on recall (double-free
29.1% vs 1.2%; `returnDanglingLifetime` catches all 3 CWE-562 cases). Those
numbers now populate `RULE_STATS`, so `double-free`, `use-after-free`,
`doubleFree`, `deallocuse`, and `returnDanglingLifetime` carry real, non-zero
likelihood ratios in the hybrid score (a single such finding is now enough to
*confirm* in static-only mode) — the concrete completion of Chapter 5's
future-work item #4. Only Semgrep's `return-stack-address` stays on
`DEFAULT_STATS` (0/3 on Juliet — its cases route the address through a helper
function the syntactic rule cannot follow).

CWE-401 (leak), CWE-457 (uninitialized) and CWE-476 (null deref) are left to
Cppcheck's value-flow checkers (`memleak`, `uninitvar`, `nullPointer`, already
wired in and in `SECURITY_RELEVANT_RULE_IDS`), because soundly proving "this
allocation is never freed on any path" is a dataflow question a syntactic
Semgrep rule cannot answer without producing exactly the guess-dressed-as-a-fix
this project avoids elsewhere.
