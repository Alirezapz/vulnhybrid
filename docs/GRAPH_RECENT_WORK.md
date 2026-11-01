# Recent graph and graph/LLM work

Checked 1 November 2026. This supplements the earlier literature review and
the exactness ledger. It is a targeted source review, not an exhaustive search
or a ranking of papers. Published scores from different corpora/splits cannot
be inserted as matched baselines in our tables. This update occurred after the
training registry was frozen and did not add test-selected configurations.

| Work and primary source | Contribution and relevance | What this study can and cannot establish |
|---|---|---|
| [HAGNN / Detecting Code Vulnerabilities with Heterogeneous GNN Training, 2025](https://arxiv.org/html/2502.16835v1) | Builds inter-procedural abstract graphs, compresses syntactic structures and combines six separately learned relation subgraphs with attention. Read representation and heterogeneous-training sections. | Our relation gate tests relation weighting on a different intra-function frontend. It does not reproduce IPAG construction, callee expansion, compression or HAGNN. Its published aggregate accuracy is not directly comparable with our commit-disjoint Devign test. **Not executed as HAGNN.** |
| [LLMxCPG, USENIX Security 2025](https://www.usenix.org/conference/usenixsecurity25/presentation/lekssays), [authors' artifact](https://github.com/qcri/llmxcpg) | Uses an LLM to generate CPG slice queries, then a second LLM for detection. The repository documents Joern 4.0.408 and two specialized 32B models. Read conference abstract and original artifact README. | A sensitive-API graph selector is not LLM-generated slicing. No model API, model weights, query execution or costs are invented here. The actual two-model/Joern pipeline requires a separate replication. **Not executed.** |
| [VulGNN, 2026 preprint](https://arxiv.org/html/2603.29216v1), [authors' source](https://github.com/CIVA-Lab/VulGNN) | Uses Joern CPGs, token/type inputs and attentional GeneralConv/relational variants; studies cross-project and cross-dataset behavior. Read architecture/generalization sections and inspect network/config source. | Our GAT/R-GCN runs are different configurations. The inspected README and paper differ in some default depth/width/dropout descriptions, so an exact reproduction must pin source/config as well as the paper. The released environment targets CUDA; adapting to CPU is possible but would still require its actual frontend and recipe. **Not executed as VulGNN.** |
| [Vul-CTG, TIFS 2026](https://doi.org/10.1109/TIFS.2026.3695431), [authors' artifact](https://github.com/ryxFry/Vul-CTG) | Aligns text and program-graph representations with a cross-modal architecture and contrastive training. Read official publication abstract and original README; full publisher text was not available in this workflow. | Our late score stacking cannot reproduce fine-grained cross-modal alignment or its pretraining. The artifact documents Joern 1.0.105, additional environment/data downloads and an 80GB A800 experimental system. That is a reported environment, not proof of a minimum hardware requirement. **Not executed.** |

Pinned inspection commits and inspected-file hashes are recorded in
`results/2026-11-01-graphs/provenance/recent_source_inspection.json`. These files
are inspection records, not execution receipts. No third-party model source is
relicensed or copied into this project's implementation.

The practical lesson is specific: adding a relation gate, contrastive loss or
text/graph combination does not establish a new research direction by itself.
A defensible contribution needs a precise difference from these prior methods,
matched controls and independently repeated evidence. Our current artifact
contributes measured representation/fusion controls and transparent failure
analysis; it does not claim that these established building blocks are new.
