# Reproduce the graph study

Use Python 3.12, the fixed data revisions from the original reproduction, and
CPU PyTorch. Restore the experiment-data release asset first, as described in
[DOWNLOADS](DOWNLOADS.md); it supplies frozen cohort memberships and per-function
scores. The optional model archive restores all fitted weights. Graph caches
are rebuilt from pinned source data with the commands below. Upstream data
retains its original rights.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[research,graph,dev,openai]'
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-graph-lock.txt
python -m research.fetch_data --out data/research
python -m research.fetch_primevul --out data/research/primevul
python -m research.prepare_primevul \
  --raw data/research/primevul/primevul_test_paired.jsonl \
  --reference data/research/devign/samples.jsonl data/research/sven/samples.jsonl \
  --out data/research/primevul/samples.jsonl
```

The experiment-data archive includes the v1 cohort CSVs and validation memberships. Graph experiments
reuse them exactly, rather than creating a new test split in every model.

```bash
python -m research.graph_data \
  --samples data/research/devign/samples.jsonl data/research/sven/samples.jsonl \
  --out graph_artifacts/graph_cache
python -m research.graph_data \
  --samples data/research/devign/samples.jsonl data/research/sven/samples.jsonl \
  --out graph_artifacts/graph_cache_evidence --selection evidence
python -m research.graph_data \
  --samples data/research/devign/samples.jsonl data/research/sven/samples.jsonl \
  --out graph_artifacts/graph_cache_semantic --selection evidence --semantic
python -m research.graph_data \
  --samples data/research/devign/samples.jsonl data/research/sven/samples.jsonl \
  --out graph_artifacts/graph_cache_semantic_bfs --semantic
python -m research.graph_data \
  --samples data/research/devign/samples.jsonl data/research/sven/samples.jsonl \
  --out graph_artifacts/graph_cache_512 --max-nodes 512

python -m research.run_graph_matrix --root results/2026-11-01-graphs/neural \
  --graphs graph_artifacts/graph_cache \
  --samples data/research/devign/samples.jsonl data/research/sven/samples.jsonl
python -m research.graph_kernels --graphs graph_artifacts/graph_cache \
  --samples data/research/devign/samples.jsonl data/research/sven/samples.jsonl \
  --out results/2026-11-01-graphs/classical
python -m research.run_graph_extras --root results/2026-11-01-graphs/experimental \
  --cache-root graph_artifacts \
  --samples data/research/devign/samples.jsonl data/research/sven/samples.jsonl
```

The extras scheduler waits for all 54 baseline runs. Registries and per-run
run.json files distinguish complete scientific training from smoke tests. A
successful cached run is reused. Use a fresh output directory/configured source
checkout when deliberately changing code/data/config; the completed-run reuse
mechanism is not a general-purpose provenance-aware training cache.

Build corresponding PrimeVul graphs using the same five graph_data calls, with
only the PrimeVul sample file and output names respectively:
primevul_graph_cache, primevul_graph_cache_evidence,
primevul_graph_cache_evidence_semantic, primevul_graph_cache_semantic,
primevul_graph_cache_512. Keep selections/caps/payload identical to their training
counterparts. The cache summary hashes select the compatible frontend.

```bash
python -m research.collect_static --samples data/research/primevul/samples.jsonl \
  --out results/2026-11-01-graphs/static_primevul
python -m research.extend_baselines --data-root data/research
python -m research.graph_kernel_predict --graphs graph_artifacts/primevul_graph_cache \
  --samples data/research/primevul/samples.jsonl \
  --runs results/2026-11-01-graphs/classical
python -m research.graph_stability --graphs graph_artifacts/graph_cache
python -m research.finish_graph_study --root results/2026-11-01-graphs \
  --cache-root graph_artifacts --prime data/research/primevul/samples.jsonl
python -m research.verify_graph_study
python -m research.render_graph_report --cache-root graph_artifacts
```

Set OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1 for the classical/meta stages;
neural children explicitly use two intra-op threads. The recorded scheduler
used three concurrent training workers on eight CPU cores with 8 GiB memory.
Full training is not a few-second smoke test. The repository tests are separate
from the scientific matrix and do not retrain the full corpus.
