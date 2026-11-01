# Release downloads and restoration

Use the [v1.2.0 release](https://github.com/Alirezapz/vulnhybrid/releases/tag/v1.2.0).
The repository tracks the maintained code and self-contained technical reports.
Raw numerical records, fitted models and matching run metadata are separate
downloads:

| Asset | Contents |
|---|---|
| VulnHybrid_Experiment_Data_v1.2.0.zip | Frozen cohort/validation memberships, per-function predictions, structured static-analysis findings and coverage, and perturbation scores needed to reproduce and independently check the measured comparisons. |
| VulnHybrid_Models_v1.2.0.zip | All 78 neural checkpoints, 32 fitted classical estimators and matching run metadata. |
| VulnHybrid_Research_Report.html | Self-contained technical experiment report with complete matched comparison tables and embedded figures. |
| VulnHybrid_Method_Comparison.png | Primary graph/fusion comparison with commit-cluster MCC intervals. |
| SHA256SUMS | Exact hashes of the four assets above. |

GitHub's generated source archives contain the tagged source checkout. Thesis
files, article manuscripts, legacy notebooks/sample verdicts, duplicate
diagnostic summaries, raw analyzer stdout/stderr and regenerable graph caches
are excluded. The experiment report is retained as technical documentation.

```bash
git clone https://github.com/Alirezapz/vulnhybrid.git
cd vulnhybrid
gh release download v1.2.0 --repo Alirezapz/vulnhybrid --dir release-assets
(cd release-assets && sha256sum --check SHA256SUMS)
unzip -o release-assets/VulnHybrid_Experiment_Data_v1.2.0.zip -d .
unzip -o release-assets/VulnHybrid_Models_v1.2.0.zip -d .
```

The model download is optional for using the static scanner. The experiment-data
download is required for checks that read frozen memberships or raw predictions.
Extracting restores paths directly beneath `results/`; no manual renaming is
needed. Rebuild graph caches with [GRAPH_REPRODUCTION](GRAPH_REPRODUCTION.md).
The model archive supplies per-run graph configurations and training histories.
The self-contained HTML reports include the measured comparison tables. Compact
intermediate JSON/CSV summaries are not included in this distribution; regenerate
them with the documented experiment commands before running verifiers that need
those files in addition to raw predictions.

Only load trusted fitted estimators: joblib is a Python object serialization
format. Upstream dataset/code rights remain separate from the project's MIT
license; see [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md).

To rebuild these release assets from a complete experiment checkout:

```bash
python -m research.package_graph_artifacts --project-root . \
  --checkpoints release-assets/VulnHybrid_Models_v1.2.0.zip --out release-assets
```

The packager generates `PUBLICATION_MANIFEST.json` to record its source/evidence
selection and hashes. The packager excludes manuscripts even if they are present
locally. Its full-checkout selection can include intermediate result summaries
that are absent from this published source checkout.
