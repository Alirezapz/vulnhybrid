# Third-party material

The project's MIT license covers its own code and does not relicense datasets,
models, upstream project sources, tool binaries or research papers.

- Devign/CodeXGLUE data is downloaded from the Google-hosted CodeXGLUE mirror.
  Its metadata identifies FFmpeg/QEMU commits. The source code retains its
  upstream project licenses. Dataset revision and byte hashes are recorded.
- SVEN data is obtained from eth-sri/sven at a pinned commit. C/C++ snippets
  originate in multiple upstream repositories and retain their applicable
  rights. Its original model task is security-controlled generation.
- The PrimeVul paired-test mirror is pinned with a byte hash and conservative
  pair exclusions in the graph-study provenance files. It retains its original
  dataset/upstream code rights. The verified subset is not relabeled as a new
  project-authored benchmark.
- Semgrep, Cppcheck and Flawfinder remain separate dependencies, not vendored
  binaries. Their versions and exact invocations are recorded with results.
- Raw analyzer diagnostics may quote limited excerpts of upstream source.
  They are evidence from those tools and are not claimed as project-authored
  source code. Complete raw datasets are omitted from the source archive.
- PyTorch and PyTorch Geometric are separate installed dependencies. Named
  architecture ports are credited in the graph replication ledger and do not
  redistribute the inspected third-party repositories.
- The optional checkpoint archive includes this study's fitted estimators,
  trained weights and run metadata. Their availability does not relicense
  underlying data or make the caches project-authored MIT source code.
- Linked papers are credited to their authors. The review is a short original
  synthesis; paper texts are not redistributed.

Academic manuscripts and the thesis are excluded from this public distribution.
The code audit records the supplied input's provenance and the limits of
historical performance claims.
