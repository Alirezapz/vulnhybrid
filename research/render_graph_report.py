"""Create exact tables/figures and a self-contained English research report."""
from pathlib import Path
import argparse,base64,html,json,statistics
import numpy as np
import pandas as pd
from markdown_it import MarkdownIt
from research.summarize import table


def render(root,cache_root):
    runs={p:json.loads((root/'comparisons'/p/'metrics.json').read_text()) for p in ['commit_disjoint','official_dedup','ffmpeg_to_qemu','qemu_to_ffmpeg']}
    primary=runs['commit_disjoint'];winner=primary['selected_on_validation']['mcc']
    def point(d,name,policy='mcc'):return next(r for r in d['metrics'] if r['method']==name and r['policy']==policy)
    def pair(d,name,phase,policy='mcc'):return next(r for r in d['external_metrics'] if r['method']==name and r['phase']==phase and r['policy']==policy)
    best=point(primary,winner);anchor=point(primary,'v1_token_lr');sven=pair(primary,winner,'external');prime=pair(primary,winner,'primevul')
    delta=next(r for r in primary['comparisons'] if r['method']==winner and r['policy']=='mcc')
    audit=json.loads((root/'representation_audit.json').read_text());coverage=json.loads((cache_root/'graph_cache/graph_summary.json').read_text())
    manifest=json.loads((root/'provenance/primevul_manifest.json').read_text())
    source_names=[p.parent.name for p in (root/'classical').glob('*/*/run.json')]
    neural_runs=list((root/'neural').glob('*/*/run.json'));extra_runs=list((root/'experimental').glob('*/*/run.json'))
    selected=[]
    for protocol,d in runs.items():
        for policy in ['mcc','f1']:
            name=d['selected_on_validation'][policy];m=point(d,name,policy)
            selected.append(dict(protocol=protocol,policy=policy,method=name,n=m['n'],accuracy=m['accuracy'],f1=m['f1'],mcc=m['mcc'],ap=m['average_precision']))
    selected_mcc=[r for r in selected if r['policy']=='mcc']
    mcc=[r for r in primary['metrics'] if r['policy']=='mcc']
    graph_rows=[r for r in mcc if r['method'].startswith(('g_','k_'))]
    experimental_rows=[r for r in mcc if r['method'].startswith('x_')]
    fusion_rows=[r for r in mcc if r['method'].startswith('fusion_')]
    seed_rows=[]
    for arch in ['deepsets','gcn','sage','gin','gat','rgcn','ggnn','devign_style','ggnn_supcon']:
        names=['g_'+arch+'_hetero_s42']+['repeat_'+arch+'_hetero_s'+str(s) for s in [43,44]]
        rows=[point(primary,n) for n in names]
        seed_rows.append(dict(architecture=arch,accuracy_mean=np.mean([r['accuracy'] for r in rows]),accuracy_sd=np.std([r['accuracy'] for r in rows],ddof=1),mcc_mean=np.mean([r['mcc'] for r in rows]),mcc_sd=np.std([r['mcc'] for r in rows],ddof=1)))
    pd.DataFrame(seed_rows).to_csv(root/'training_seed_summary.csv',index=False)
    stability=[]
    for file in (root/'stability').glob('*_scores.csv'):
        arch=file.name.removesuffix('_scores.csv');df=pd.read_csv(file);row=point(primary,'g_'+arch+'_hetero_s42');threshold=row['threshold']
        y=pd.read_csv('results/2026-11-01/commit_disjoint/test_predictions.csv').set_index('sample_id').loc[df.sample_id,'label'].to_numpy()
        from sklearn.metrics import matthews_corrcoef
        for variant in ['no_edges','anonymous_payload','rewired_relations','node_permutation']:
            stability.append(dict(architecture=arch,variant=variant,mean_score_change=float(np.mean(np.abs(df[variant]-df.original))),prediction_flip_rate=float(np.mean((df[variant]>=threshold)!=(df.original>=threshold))),mcc=float(matthews_corrcoef(y,df[variant]>=threshold))))
    pd.DataFrame(stability).to_csv(root/'stability_metrics.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    show=['v1_token_lr','v1_char_lr','k_hetero_wl2','g_deepsets_hetero_s42','g_gcn_hetero_s42','g_sage_hetero_s42','g_gin_hetero_s42','g_gat_hetero_s42','g_rgcn_hetero_s42','g_ggnn_hetero_s42','x_sage_semantic_evidence','x_relation_gate','x_devign_relational_port','fusion_all_logit_stack','fusion_all_boosting',winner]
    show=list(dict.fromkeys(n for n in show if any(r['method']==n for r in mcc)))
    rows=[point(primary,n) for n in show];fig,axes=plt.subplots(1,3,figsize=(16,9),layout='constrained');y=np.arange(len(rows))
    for ax,key,title in zip(axes,['accuracy','mcc','average_precision'],['Accuracy','MCC (95% commit-cluster interval)','Average precision']):
        values=np.array([r[key] for r in rows]);ax.barh(y,values,color=['#d48b3d' if r['method']==winner else '#397c88' for r in rows])
        if key=='mcc':
            limits=np.array([r['mcc_cluster_ci'] for r in rows]);ax.hlines(y,limits[:,0],limits[:,1],color='#1b3548',lw=1.4);ax.axvline(0,color='#777',lw=.6)
        ax.set_yticks(y,show if ax is axes[0] else []);ax.invert_yaxis();ax.set_title(title);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_xlim(0,1);axes[2].set_xlim(0,1);fig.suptitle('VulnHybrid graph study: primary commit-disjoint test (validation-MCC thresholds)',fontsize=14)
    fig.savefig(root/'VulnHybrid_Graph_Comparison.png',dpi=170);plt.close(fig)
    show_pair=['v1_token_lr','k_hetero_wl2','g_ggnn_hetero_s42','x_sage_semantic_evidence','fusion_all_logit_stack',winner]
    show_pair=list(dict.fromkeys(show_pair));fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,phase,title in zip(axes,['external','primevul'],['SVEN transfer','PrimeVul mirror transfer']):
        values=[pair(primary,n,phase)['both_correct'] for n in show_pair]
        bars=ax.barh(np.arange(len(show_pair)),values,color='#b46f4e');ax.set_yticks(np.arange(len(show_pair)),show_pair);ax.invert_yaxis()
        ax.bar_label(bars,labels=[f'{v:.2%}' for v in values],padding=4,fontsize=9)
        ax.set_xlim(0,max(.02,max(values)*1.4));ax.set_xlabel('Fraction of pairs with both labels correct (zoomed)');ax.set_title(title)
    fig.savefig(root/'VulnHybrid_Graph_Pairs.png',dpi=170);plt.close(fig)
    lo,hi=delta['delta_mcc_cluster_ci'];verdict=('The conditional clustered interval is positive on this held-out cohort.' if lo>0 else 'The conditional clustered interval does not establish a positive MCC gain.' if hi>=0 else 'The conditional clustered interval favors the token baseline.')
    summary=f'''# VulnHybrid graph study and all-family fusion

**1 November 2026 — version 1.2.0. Research artifact; not peer reviewed.**

## Main result

The primary validation-MCC-selected configuration is **{winner}**. On the
2,500-function commit-disjoint test cohort it reaches accuracy
**{best['accuracy']:.2%}**, F1 **{best['f1']:.4f}**, MCC **{best['mcc']:.4f}** and
average precision **{best['average_precision']:.4f}**. Token LR under the same
validation-MCC policy reaches accuracy **{anchor['accuracy']:.2%}** and MCC
**{anchor['mcc']:.4f}**. The selected model's MCC difference is
**{delta['delta_mcc']:.4f}**, with 95% commit-cluster interval
**[{lo:.4f}, {hi:.4f}]**. {verdict}

This does not establish a universal best detector or world-first algorithm.
It is evidence about these configurations, representations, labels and fixed
cohorts. The selected method was chosen by validation, not by the largest test
row. Historical v1 F1-selected numbers are retained separately; changing the
threshold objective is not a new model improvement.

## Executed scope and complete evidence

- **{len(neural_runs)} baseline neural training runs**: nine configurations on
  four cohorts, with three initialization/shuffle seeds on the primary cohort.
- **{len(extra_runs)} additional neural runs**: edge/payload/size controls,
  evidence-focused and operator/field-aware representations, relation gating,
  perturbation consistency, and architectural ports of typed GGNN/readout.
- **{len(source_names)} selected classical graph models**, including node
  histograms, AST/typed/CFG/last-identifier WL features and structural boosting.
- **18 learned or fixed fusion recipes per cohort**, including all-family
  stacking, convex mixtures, rank stacking, boosting, anchored residuals and
  confidence/quality gating. Every available informative executed expert can
  enter the all-family recipes; no unexecuted LLM is simulated.
- **{sum(d['n_methods'] for d in runs.values())} scored configuration/cohort
  comparisons** across the four matched test sets. This count includes earlier
  baselines, seed repeats, ensemble configurations and ablations; it is not a
  count of that many distinct algorithms. F1/MCC operating points are reported
  separately, not counted as independent models.
- Every method is also scored on frozen SVEN and verified PrimeVul external
  pairs. Raw scores, IDs, fit/selection membership, seeds, best epochs,
  histories, weights, timings, diagnostics and repaired implementation failures
  are retained. All scientific neural runs use the full training cohort.

The code includes original PyG GCN, GraphSAGE, GIN, GAT, R-GCN and GGNN
operators, trained from scratch. Devign-style and typed architecture ports are
adaptations on a shared frontend. **Full end-to-end Devign, ReVeal, IVDetect,
DeepWukong and AMPLE paper reproductions are not claimed.** Their original
repositories and exactness requirements are listed in GRAPH_REPLICATION_LEDGER.
No finite registry can contain every graph method published in the field.
The supplemental 2025–2026 review also covers HAGNN, LLMxCPG, VulGNN and
Vul-CTG, with explicit reading depth and unexecuted status.

## Input and protocol integrity

All four old cohort memberships and validation halves are unchanged. Old
text/static predictions were independently reproduced within 1e-7 before
extending them to the new external code. Train-only TF-IDF, validation-only
calibration, and separate validation selection remain enforced. Graph features
contain no IDs, labels, CWE/CVE metadata, filenames or commit identity.

Primary neural budgets are shared: 48 hidden units, two conventional message
layers or three ordinary gated steps; eight typed gated steps for the inspected
Devign architectural port; batch 128; AdamW .001, weight decay .0001; up to
eight epochs with patience two and minimum three epochs. Checkpoints maximize
val_fit AP. Prototype and cross-project runs use seed 42. This is a bounded
CPU budget, not exhaustive architecture/hyperparameter optimization.

The extension reuses previously seen v1 test cohorts. The graph protocol was
frozen before full graph fitting; a representation addendum was driven by input
coverage before inspecting graph test metrics. The entire extension remains
exploratory, not an externally preregistered independent confirmation.

## Representation audit

The base graph contains tree-sitter named AST nodes and leaf payload, next-leaf
links, last-identical-identifier links and a structured statement CFG. The latter
two are approximate: they are not sound pointer-aware reaching definitions or
a compiler CPG. Switch, goto and exception semantics are diagnosed, not solved.
The base named-node input omits unnamed operators and AST field roles; a matched
richer control preserves those. Edge-only ablations share the node set.

Across **{coverage['n']}** Devign/SVEN inputs, tree-sitter reports error nodes in
**{coverage['parse_errors']}** functions and the 256-node representation truncates
**{coverage['truncated']}**. Errors and truncation stay in the full comparison.
A common error-free/non-truncated sensitivity subset is separately recorded.

{table(audit,[('cache','Representation'),('max_nodes','Node cap'),('identical_pairs','Input-identical SVEN pairs'),('deterministic_pair_success_upper_bound','Loose pair-success upper bound'),('truncated','Truncated functions')])}

Input-identical before/after graphs cannot both be classified correctly by a
deterministic classifier with one threshold. The bound is loose: distinct inputs
can still be isomorphic to a permutation-invariant model or lack sufficient
semantic context. A sensitive-API selection can also discard repair evidence;
it is not assumed beneficial. The evidence/semantic/512 controls isolate these
representation changes from a change of GNN operator.

## Validation-selected results across cohorts

{table(selected_mcc,[('protocol','Cohort'),('method','Selected by validation MCC'),('n','Test n'),('accuracy','Accuracy'),('f1','F1'),('mcc','MCC'),('ap','AP')])}

Each cohort has its own training/selection population. Absolute differences
between cohort rows are not a paired causal estimate of leakage or project
transfer. All methods within one cohort have identical denominators.

![Graph and hybrid comparison](VulnHybrid_Graph_Comparison.png)

## All primary graph baselines (MCC operating point)

{table(graph_rows,[('method','Configuration'),('accuracy','Accuracy'),('precision','Precision'),('recall','Recall'),('f1','F1'),('mcc','MCC'),('average_precision','AP'),('fpr','FPR')])}

## All primary representation/architecture prototypes and controls

{table(experimental_rows,[('method','Configuration'),('accuracy','Accuracy'),('precision','Precision'),('recall','Recall'),('f1','F1'),('mcc','MCC'),('average_precision','AP')])}

The semantic payload corrects a representation omission. Evidence-focused node
retention is related to established slicing. Relation gating and consistency
regularization adapt known techniques. These names describe implemented project
prototypes; they do not establish literature-wide novelty.

## All-family and narrower fusion recipes

{table(fusion_rows,[('method','Configuration'),('accuracy','Accuracy'),('precision','Precision'),('recall','Recall'),('f1','F1'),('mcc','MCC'),('average_precision','AP'),('fpr','FPR')])}

Means and convex mixtures are controls for learned stacking. Separate graph-only,
graph/text, graph/static and all-family stacks test complementarity. Probability,
logit and validation-reference rank transforms are separate recipes. The anchored
residual has an intercept-only control. The relation gate has a uniform control.
Quality fallback and a fitted confidence/quality gate test whether frontend
limitations should change expert weighting. All meta fitting uses val_fit; the
anti-leakage regression test changes the entire test distribution and verifies
that learned meta parameters and validation outputs do not change.

## Training seed sensitivity

{table(seed_rows,[('architecture','Architecture'),('accuracy_mean','Mean accuracy'),('accuracy_sd','SD accuracy'),('mcc_mean','Mean MCC'),('mcc_sd','SD MCC')])}

These are three initialization/shuffle seeds on the same split, with each
threshold selected using validation. They do not estimate dataset/split variance.
Single-seed prototypes and external probes require replication before general
claims. Raw per-seed rows and three-seed probability ensembles are retained.

## Frozen external repair sensitivity

For the validation-MCC-selected primary configuration:

| External probe | Pairs | Both labels correct | Strict vulnerable>fixed ranking | Accuracy | FPR |
|---|---:|---:|---:|---:|---:|
| SVEN retained pairs | {sven['n_pairs']} | {sven['both_correct']:.4f} | {sven['vulnerable_ranks_higher']:.4f} | {sven['accuracy']:.4f} | {sven['fpr']:.4f} |
| PrimeVul mirror verified subset | {prime['n_pairs']} | {prime['both_correct']:.4f} | {prime['vulnerable_ranks_higher']:.4f} | {prime['accuracy']:.4f} | {prime['fpr']:.4f} |

In counts, the selected configuration gets both labels right in
**{round(sven['both_correct']*sven['n_pairs'])}/{sven['n_pairs']} SVEN pairs** and
**{round(prime['both_correct']*prime['n_pairs'])}/{prime['n_pairs']} PrimeVul pairs**.

![Frozen external pair outcomes](VulnHybrid_Graph_Pairs.png)

SVEN uses the previously retained 415 pairs. PrimeVul uses **348 verified pairs**
out of 435 source pairs: 76 incomplete metadata/signature cases, seven overlaps,
two commit mismatches, one filename mismatch and one identical pair are excluded.
The exact exclusions and byte hash are supplied. This conservative subset can be
selection-biased and is not the full official PrimeVul benchmark. The source is
a pinned public mirror; no independent relabeling of its truth was performed.

External thresholds remain frozen; no external label trains/calibrates/selects
a method. Both-correct cluster intervals and nominal independent-pair exact
binomial intervals are stored. Cluster bootstrap intervals can degenerate at
zero observed successes; the binomial interval assumes independent pairs and
does not account for shared-commit correlation. Neither interval is proof of a
population failure rate of exactly zero.

## Perturbation and operating-point analyses

Frozen models are evaluated with edges removed, relation-wise rewiring,
anonymous leaf payload and node permutation. These are diagnostic input
perturbations, not all semantics-preserving program edits. No threshold is
retuned on perturbed test data. DeepSets is a topology-free control; ordinary
GNN pooling is permutation-invariant, whereas convolutional node-axis readout
can depend on order. The complete score changes, flip rates and MCC are in
stability_metrics.csv.

Both F1-selected and MCC-selected operating points are preserved for every
configuration. Low-FPR points are selected at observed val_select caps of .5%,
1% and 5%, with actual test FPR/recall reported. The caps are not deployment
guarantees. Review precision/recall at fixed 1/5/10/20% budgets is also saved,
with the expected random-review baseline. No LLM cost or human-review time is
inferred from these offline rankings.

## Statistics, reproducibility and publication scope

Test intervals and paired differences use 1,000 shared commit-cluster bootstrap
resamples. McNemar tests are secondary accuracy tests with Holm adjustment,
not tests of F1/MCC; sample dependence remains a limitation. Intervals are
conditional on the fitted/selected models and omit split/training uncertainty.

The model archive preserves trained weights and classical estimators. Raw
per-function scores and frozen memberships are in the experiment-data archive;
code, configs and compact tables are tracked in Git. Regenerate graph caches
from the pinned data with the documented extraction commands.
Direct environment versions and input/protocol hashes are recorded. Tests cover
CFG returns, parse diagnostics, edge validity, operator preservation, every
architecture's gradients, topology controls and meta-data leakage. A separate
verifier recomputes scores/selections from exported prediction files.

This provides a substantive graph/fusion research artifact. Publication claims
still require exact named baseline pipelines on their intended representations,
stronger chronological/near-clone controls, more split/training seeds and manual
real-project validation. No unrun method receives an invented score, and no
acceptance/SOTA/world-first claim follows from the experiment count.
'''
    (root/'GRAPH_STUDY_REPORT.md').write_text(summary)
    Path('docs/GRAPH_STUDY_REPORT.md').write_text(summary.replace('](VulnHybrid_Graph_Comparison.png)','](../results/2026-11-01-graphs/VulnHybrid_Graph_Comparison.png)').replace('](VulnHybrid_Graph_Pairs.png)','](../results/2026-11-01-graphs/VulnHybrid_Graph_Pairs.png)'))
    body=MarkdownIt('commonmark').enable('table').render(summary)
    body+='<h2>Complete matched tables: every configuration and threshold policy</h2><p>Expand a cohort to inspect all rows, including lexical/static controls, seed repeats and ensembles. Rows follow the fixed registry order. Test maxima do not select the reported final model.</p>'
    columns=['method','policy','threshold','accuracy','precision','recall','f1','mcc','average_precision','fpr','brier','ece']
    for protocol,result in runs.items():
        full=pd.DataFrame(result['metrics'])[columns]
        body+='<details><summary>'+html.escape(protocol)+' — '+str(result['n_methods'])+' configurations, two policies</summary>'+full.to_html(index=False,float_format=lambda x:f'{x:.5f}',border=0)+'</details>'
    for name in ['VulnHybrid_Graph_Comparison.png','VulnHybrid_Graph_Pairs.png']:
        body=body.replace('src="'+name+'"','src="data:image/png;base64,'+base64.b64encode((root/name).read_bytes()).decode()+'"')
    css='body{font:16px/1.65 system-ui,sans-serif;background:#eef2f5;color:#1e303e;margin:0}main{max-width:1450px;background:white;padding:42px;margin:auto}h1{font-size:2.2em;line-height:1.2;color:#17465a}h2{margin-top:2em;color:#1f6473}table{border-collapse:collapse;display:block;overflow:auto;font-size:12px;width:100%}th,td{border-bottom:1px solid #d9e4e8;padding:7px;text-align:left}th{background:#edf5f5}img{width:100%;height:auto}a{color:#176b86}@media(max-width:650px){main{padding:16px}h1{font-size:1.6em}}'
    (root/'VulnHybrid_Graph_Study.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VulnHybrid Graph Study</title><style>'+css+'</style><main>'+body+'</main></html>')
    print(json.dumps(dict(methods=sum(d['n_methods'] for d in runs.values()),selected=winner,accuracy=best['accuracy'],f1=best['f1'],mcc=best['mcc'],mcc_delta_ci=delta['delta_mcc_cluster_ci'],sven_pair=sven['both_correct'],prime_pair=prime['both_correct']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,default=Path('results/2026-11-01-graphs'));p.add_argument('--cache-root',type=Path,required=True);a=p.parse_args();render(a.root,a.cache_root)
