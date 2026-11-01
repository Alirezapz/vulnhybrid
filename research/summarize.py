"""Recompute diagnostics, tables and standalone report from saved predictions."""
from pathlib import Path
import argparse,base64,collections,csv,hashlib,html,json,re
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from research.benchmark import load,metrics,bootstrap


def table(rows,cols):
    lines=['| '+' | '.join(label for key,label in cols)+' |','|'+'|'.join('---' for _ in cols)+'|']
    for row in rows:
        cells=[]
        for key,label in cols:
            value=row.get(key,'—')
            if isinstance(value,(float,np.floating)):value=f'{value:.4f}'
            cells.append(str(value))
        lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)


def diagnostics(root,sample_files):
    rows=[r for path in sample_files for r in load(path)]
    mapping={f'u{i:06d}'+('.cpp' if r['language']=='cpp' else '.c'):r['sample_id'] for i,r in enumerate(rows)}
    pd.DataFrame(mapping.items(),columns=['filename','sample_id']).to_csv(root/'static/sample_index.csv',index=False)
    coverage=json.loads((root/'static/coverage.json').read_text());failed=set();summary=[]
    for tool,c in coverage.items():
        files=set()
        types=collections.Counter()
        for error in c['errors']:
            text=json.dumps(error) if isinstance(error,dict) else error
            files.update(re.findall(r'\bu\d{6}\.(?:c|cpp)\b',text))
            if isinstance(error,dict):
                kind=error.get('type','unknown');kind=kind[0] if isinstance(kind,list) else kind
            else:
                m=re.search('id="([^"]+)"',text);kind=m.group(1) if m else 'unknown'
            types[kind]+=1
        ids={mapping[f] for f in files if f in mapping};failed.update(ids)
        c['affected_sample_ids']=sorted(ids);c['diagnostic_counts']=dict(types)
        summary.append(dict(tool=tool,requested=c['requested'],diagnostics=len(c['errors']),
            affected_samples=len(ids),seconds=c['duration_s'],version=c['version'].split('\n')[0]))
    (root/'static/diagnostic_summary.json').write_text(json.dumps(coverage,indent=2))
    return coverage,summary,failed


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,default=Path('results/2026-11-01'))
    p.add_argument('--samples',type=Path,nargs='+',required=True);args=p.parse_args();root=args.root
    regimes=['official_dedup','commit_disjoint','ffmpeg_to_qemu','qemu_to_ffmpeg']
    runs={name:json.loads((root/name/'metrics.json').read_text()) for name in regimes}
    coverage,coverrows,failed=diagnostics(root,args.samples)
    common=[];routing=[]
    for regime,d in runs.items():
        predictions=pd.read_csv(root/regime/'test_predictions.csv')
        clean=predictions[~predictions.sample_id.isin(failed)]
        cols=[m['method'] for m in d['metrics']]
        for name in cols:
            r=metrics(clean.label.to_numpy(),clean[name].to_numpy(),d['thresholds'][name]);r.update(protocol=regime,method=name);common.append(r)
        # Offline ranking: human-review budget, not simulated LLM correctness.
        for name in ['token_lr','mean_token_char','heldout_evidence_stack','gated_residual']:
            scores=predictions[name].to_numpy();y=predictions.label.to_numpy()
            order=np.argsort(-scores,kind='stable')
            for budget in [.01,.05,.10,.20]:
                n=max(1,int(np.ceil(len(y)*budget)));ix=order[:n]
                routing.append(dict(protocol=regime,method=name,budget=budget,n_reviewed=n,
                    precision_in_budget=float(y[ix].mean()),vulnerabilities_recovered=float(y[ix].sum()/y.sum()),
                    random_expected_precision=float(y.mean()),random_expected_recall=n/len(y)))
        (root/regime/'error_free_metrics.json').write_text(json.dumps([r for r in common if r['protocol']==regime],indent=2))
    pd.DataFrame(routing).to_csv(root/'review_budget.csv',index=False)
    primary=runs['commit_disjoint'];original=runs['official_dedup'];selected=primary['selected_on_validation']
    main=next(r for r in primary['metrics'] if r['method']==selected)
    trivial=next(r for r in primary['metrics'] if r['method']=='always_vulnerable')
    external=next(r for r in primary['external_metrics'] if r['method']==selected)
    anchor=next(r for r in primary['comparisons'] if r['method']==selected)
    # Paired clustered delta vs always-positive, also baseline not just token LR.
    from research.benchmark import paired_comparisons
    pred=pd.read_csv(root/'commit_disjoint/test_predictions.csv')
    comparisons=paired_comparisons(pred.label.to_numpy(),{
        'always_vulnerable':np.ones(len(pred),bool),selected:pred[selected].to_numpy()>=primary['thresholds'][selected]},
        pred.commit_id.fillna(pred.sample_id).tolist(),anchor='always_vulnerable')
    (root/'selected_vs_trivial.json').write_text(json.dumps(comparisons,indent=2))
    plot_methods=['always_vulnerable','semgrep_any','cppcheck_any','flawfinder_any','token_lr','char_lr','mean_token_char','heldout_evidence_stack','gated_residual']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#566b85','#008080','#de9248','#935f9a']
    fig,axes=plt.subplots(1,2,figsize=(13,5.8),layout='constrained')
    y=np.arange(len(plot_methods));height=.18
    for i,(regime,d) in enumerate(runs.items()):
        ms={r['method']:r for r in d['metrics']}
        for ax,key in zip(axes,['accuracy','mcc']):
            ax.barh(y+(i-1.5)*height,[ms[n][key] for n in plot_methods],height,label=regime,color=colors[i])
            ax.set_yticks(y,plot_methods);ax.invert_yaxis();ax.set_xlabel(key.upper());ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_xlim(0,1);axes[1].set_xlim(-.12,.4);axes[1].axvline(0,color='#777',lw=.7)
    axes[0].set_title('Accuracy on matched test cohorts');axes[1].set_title('MCC exposes trivial all-positive behavior')
    axes[0].legend(loc='lower right',fontsize=8);fig.suptitle('VulnHybrid: 19 methods, four held-out evaluation regimes',fontsize=14)
    fig.savefig(root/'method_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    ms={r['method']:r for r in primary['metrics']}
    names=['always_vulnerable','token_lr','char_lr','mean_token_char','gated_residual']
    axes[0].bar(names,[ms[n]['f1'] for n in names],color='#357f91')
    axes[0].set_ylim(0,1);axes[0].set_ylabel('F1');axes[0].set_title('Similar F1 can hide weak detection');axes[0].tick_params(axis='x',labelrotation=25)
    em={r['method']:r for r in primary['external_metrics']}
    axes[1].bar(names,[em[n]['both_correct'] for n in names],color='#b9654b')
    axes[1].set_ylim(0,1);axes[1].set_ylabel('Fraction of pairs');axes[1].set_title(f'SVEN: both versions classified correctly ({external["n_pairs"]} pairs)');axes[1].tick_params(axis='x',labelrotation=25)
    if all(em[n]['both_correct']==0 for n in names):
        axes[1].text(.5,.55,f'0 of {external["n_pairs"]} pairs\nfor all five methods',
                     transform=axes[1].transAxes,ha='center',va='center',fontsize=16,color='#a24e3a')
    fig.savefig(root/'paired_evaluation.png',dpi=180);plt.close(fig)
    regime_rows=[]
    for name,d in runs.items():
        r=next(m for m in d['metrics'] if m['method']==d['selected_on_validation']);e=next(m for m in d['external_metrics'] if m['method']==d['selected_on_validation'])
        regime_rows.append(dict(protocol=name,selected=d['selected_on_validation'],n=r['n'],accuracy=r['accuracy'],f1=r['f1'],mcc=r['mcc'],external_pair_correct=e['both_correct']))
    common_rows=[r for r in common if r['protocol']=='commit_disjoint']
    baseline_diff=comparisons[0]
    text=f'''# VulnHybrid research upgrade: audit, experiments and limitations

**Run date: 1 November 2026. Author: Alireza Pazouki.**

## Findings that matter

The code and experimental infrastructure are stronger, but these experiments
**do not demonstrate a reliable new detection improvement**. On the primary
commit-disjoint cohort, the validation-selected method is **{selected}**:
accuracy **{main['accuracy']:.2%}**, F1 **{main['f1']:.4f}**, MCC **{main['mcc']:.4f}**.
An always-vulnerable classifier already reaches F1 **{trivial['f1']:.4f}**.
The selected method's F1 advantage over that trivial baseline is
**{baseline_diff['delta_f1']:.5f}**, with a commit-cluster 95% interval
**[{baseline_diff['delta_f1_cluster_ci'][0]:.5f}, {baseline_diff['delta_f1_cluster_ci'][1]:.5f}]**.
This interval does not establish a positive gain.

The frozen external probe is even more diagnostic: **{external['both_correct']:.2%}**
of **{external['n_pairs']}** vulnerable/fixed pairs have both labels correct for
that method. Its external ROC-AUC is **{external['roc_auc']:.4f}** and FPR is
**{external['fpr']:.2%}**. Strong recall/F1 here is mostly an operating-point
artifact, not evidence of understanding the repair. No SOTA claim is justified.

## What was actually executed

Downloaded all **27,318** Devign/CodeXGLUE functions and **846** C/C++ SVEN
functions from pinned revisions. Evaluated **{len(primary['metrics'])} methods
in each of four regimes** ({len(primary['metrics'])*len(runs)} matched test
comparisons), plus frozen transfer evaluations for every method. The prototype
methods combine lightweight lexical classifiers with measured static evidence;
they are not new LLM runs and not named neural-paper reproductions.

Core regressions and original end-to-end tests were run locally, including the
real mock HTTP transport. Static engines and taint fixtures were executed.
The taint fixture contains an explicit expected miss for an array update.
No paid API, GPU, exploit generation, or real-project vulnerability discovery
was used. Published-looking historical thesis tables were not treated as
new measurements because their supporting full outputs are absent from the ZIP.

## Data audit and leakage controls

The original data has **{original['original_audit']['duplicate_groups']}**
normalized duplicate groups, **{original['original_audit']['conflicting_label_groups']}**
conflicting-label groups, **{original['original_audit']['cross_split_clone_groups']}**
clone groups crossing official splits, and
**{original['original_audit']['cross_split_commit_groups']}** commit IDs in multiple
splits. Official-split deduplication removes token-identical clones but leaves
**{original['clean_audit']['cross_split_commit_groups']}** overlapping commit groups.

The primary protocol puts entire commits in one split, then deduplicates code.
It retains **{primary['clean_audit']['partitions']['train']}** train,
**{primary['clean_audit']['partitions']['val']}** validation and
**{primary['clean_audit']['partitions']['test']}** test functions, with zero
normalized clone or commit overlap. Of validation functions,
**{primary['validation_fit_n']}** fit models/calibrators and
**{primary['validation_select_n']}** select thresholds and the method. These
halves are also commit-disjoint. The split is randomized by stable hash,
not chronological; near-clone and label-noise threats remain.

SVEN is used only for transfer, not training. Eight whole C/C++ pairs are
excluded because of Devign overlap or equal normalized before/after code,
leaving **415 pairs / 830 functions**. This repurposes its upstream training
and validation release as an external detection probe; it is not an official
SVEN test leaderboard. Scores/thresholds are frozen before this evaluation.

## Matched comparisons across regimes

Each row below is the method selected by that regime's own validation set.
Cohorts differ across regimes, so their absolute scores do not constitute a
paired causal estimate of leakage inflation. The official-to-commit decline
is diagnostic evidence consistent with dataset shortcuts, not proof of its
unique cause.

{table(regime_rows,[('protocol','Protocol'),('selected','Validation-selected method'),('n','Test n'),('accuracy','Accuracy'),('f1','F1'),('mcc','MCC'),('external_pair_correct','External pair success')])}

![Matched accuracy and MCC](method_comparison.png)

## All primary test results

All methods below share the same **{main['n']}** test functions. F1 zero division
uses zero. AP means average precision. Static scores are binary alert/no-alert
scores; their Brier/ECE should not be read as calibrated uncertainty.

{table(primary['metrics'],[('method','Method'),('accuracy','Accuracy'),('precision','Precision'),('recall','Recall'),('f1','F1'),('mcc','MCC'),('average_precision','AP'),('fpr','FPR')])}

Thresholds are selected on validation only. The maximum test row is descriptive;
it does not override the validation-selected method. Full confusion counts,
thresholds, default-threshold F1, ROC-AUC, Brier/ECE and commit-cluster F1 intervals
are in each metrics.json and leaderboard.csv.

## New combinations and ablations

**Held-out evidence stack:** a regularized logistic combination of token,
character and static logits fit on val_fit, separate from training and selection.
**Gated residual:** anchor the mean lexical logit, then fit an intercept plus
shrunken binary tool-rule corrections. Multiple warning lines do not multiply
the same evidence. **Intercept-only** isolates base-rate recalibration; **mean**
and **log-odds fusion** are established controls.

These are project-specific experimental combinations of existing statistical
ideas. They are not claimed as world-first methods. On the primary cohort, the
gated residual minus token-LR F1 difference is **{anchor['delta_f1']:.5f}**,
cluster interval **[{anchor['delta_f1_cluster_ci'][0]:.5f}, {anchor['delta_f1_cluster_ci'][1]:.5f}]**.
The static evidence has not earned a demonstrated generalization improvement.
The narrow explicit-source taint rule produced zero corpus alerts; negative
results remain in the comparison.

McNemar tests measure accuracy differences, not F1. Their Holm-adjusted p-values
are included as secondary analyses; same-commit correlation makes the clustered
F1 difference intervals more appropriate for the central F1 question. Intervals
are conditional on this run and exclude training/selection variance.

## Paired repair sensitivity

![F1 compared with repair-pair success](paired_evaluation.png)

{table(primary['external_metrics'],[('method','Method'),('accuracy','External accuracy'),('f1','F1'),('both_correct','Both versions correct'),('vulnerable_ranks_higher','Vulnerable ranks higher'),('tied','Tied')])}

A high F1 with zero pair success is an unhelpful classifier that labels both
versions vulnerable. Ranking the vulnerable version above the fix is a separate
criterion; it does not guarantee a useful classification threshold.

## Analyzer coverage and diagnostic-free sensitivity analysis

{table(coverrows,[('tool','Engine'),('requested','Inputs'),('diagnostics','Diagnostic records'),('affected_samples','Affected inputs'),('seconds','Seconds'),('version','Version')])}

Diagnostics include partial parses and unknown macros; counts are not all
complete failures, and a partially parsed function may still yield a warning.
The full-cohort static comparison measures operational alerts: no alert on
unparsed code is **not evidence of safety**. Error records and exact affected
IDs are retained. A second, common diagnostic-free primary subset contains
**{common_rows[0]['n']}** functions:

{table([r for r in common_rows if r['method'] in ['semgrep_any','cppcheck_any','flawfinder_any','token_lr','char_lr','gated_residual']],[('method','Method'),('n','n'),('accuracy','Accuracy'),('f1','F1'),('mcc','MCC')])}

This subset conditions on analyzer success and may be easier/different; it is a
sensitivity analysis, not a replacement for full-cohort results. Review-budget
precision/recall at 1%, 5%, 10% and 20% is separately saved in review_budget.csv,
with the expected random-review baseline. It is an offline ranking analysis,
not simulated LLM performance or measured human review time.

## Engineering changes

Strict model-output validation; duplicate-verdict rejection; rule-event
deduplication and stable log odds; explicit analyzer errors; matched-denominator
evaluation; validation-only weight fitting; seeded sample selection; benchmark
hint removal; and model/prompt/input-aware checkpoint hashes. Details and
remaining limitations are in CODE_AUDIT.md. None of these fixes retroactively
validates the old paper's missing experiments.

## Publication assessment

This is a reproducible research artifact with useful negative evidence, not yet
a submission-ready claim of a better vulnerability detector. A stronger paper
needs actual named neural/modern-LLM baselines on the same cohort, chronological
PrimeVul and paired testing, multiple split/training seeds, manual real-project
alert verification, and a mechanism that improves those outcomes. The literature
review separates reviewed work from implementations actually executed.

## Reproduction and evidence

Use research/fetch_data.py, research/collect_static.py, research/benchmark.py and
research/summarize.py. README contains the exact commands. The protocol and its
hash were recorded before inspecting these test predictions. Data revisions,
byte hashes, cohort memberships, exclusions, validation partitions, per-sample
scores, structured analyzer findings, diagnostics, environment versions and comparisons
are included. Underlying upstream source datasets are downloaded separately.

All four run durations (seconds):
{', '.join(name+': '+format(d['wall_seconds'],'.1f') for name,d in runs.items())}.
They overlapped in pairs and are not an additive wall-clock estimate. These are
end-to-end CPU benchmark times, not model-only inference latency. No dollar or
LLM token savings are inferred.
'''
    (root/'EXPERIMENT_REPORT.md').write_text(text)
    Path('docs/EXPERIMENT_REPORT.md').write_text(text.replace('](method_comparison.png)','](../results/2026-11-01/method_comparison.png)').replace('](paired_evaluation.png)','](../results/2026-11-01/paired_evaluation.png)'))
    from markdown_it import MarkdownIt
    rich=MarkdownIt('commonmark').enable('table').render(text)
    for name in ['method_comparison.png','paired_evaluation.png']:
        url='data:image/png;base64,'+base64.b64encode((root/name).read_bytes()).decode()
        rich=rich.replace('src="'+name+'"','src="'+url+'"')
    document='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VulnHybrid Research Report</title><style>body{font:16px/1.65 system-ui,sans-serif;color:#202e3b;background:#eef2f5;margin:0}main{max-width:1180px;margin:auto;background:white;padding:45px}h1{font-size:2.25em;line-height:1.2;color:#173d50}h2{margin-top:2em;color:#1d6070}table{border-collapse:collapse;font-size:12px;display:block;overflow:auto;width:100%}th,td{border-bottom:1px solid #dce4e9;padding:8px;text-align:left}th{background:#edf5f5}img{width:100%;height:auto}code{font-size:.9em}a{color:#15667e}@media(max-width:650px){main{padding:18px}h1{font-size:1.7em}}</style><main>'+rich+'</main></html>'
    (root/'VulnHybrid_Research_Report.html').write_text(document)
    print('Generated report, two figures, sensitivity tables and review-budget analysis.')


if __name__=='__main__':main()
