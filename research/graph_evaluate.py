"""Matched final evaluation of graph baselines and frozen fusion recipes."""
from pathlib import Path
import argparse,collections,json,time
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score,matthews_corrcoef
from research.benchmark import load,metrics,pair_metrics
from research.graph_fusion import compose


def aligned(path,ids):
    df=pd.read_csv(path).set_index('sample_id')
    if not df.index.is_unique:raise ValueError('Duplicate predictions: '+str(path))
    result=df.loc[ids,'probability'].to_numpy()
    if not np.isfinite(result).all() or not ((result>=0)&(result<=1)).all():raise ValueError('Invalid score file: '+str(path))
    return result


def threshold(y,p,objective='mcc'):
    grid=np.r_[np.linspace(0,1,101),1.000001]
    positive=p[:,None]>=grid
    tp=((y[:,None]==1)&positive).sum(axis=0);fp=((y[:,None]==0)&positive).sum(axis=0)
    fn=(y==1).sum()-tp;tn=(y==0).sum()-fp
    if objective=='f1':value=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(len(grid)),where=(2*tp+fp+fn)>0)
    else:
        den=np.sqrt((tp+fp).astype(float)*(tp+fn)*(tn+fp)*(tn+fn))
        value=np.divide(tp*tn-fp*fn,den,out=np.zeros(len(grid)),where=den>0)
    best=value.max();ix=min(np.flatnonzero(np.isclose(value,best,atol=1e-12,rtol=0)),key=lambda i:(abs(grid[i]-.5),-grid[i]))
    return float(grid[ix]),float(best)


def clustered(y,predictions,groups,n=1000,seed=42):
    names=list(predictions);pred=np.column_stack([predictions[k] for k in names]);groups=np.asarray(groups)
    unique=sorted(set(groups));counts=[]
    for group in unique:
        ix=groups==group;t=y[ix,None];p=pred[ix]
        counts.append(np.stack([((t==1)&p).sum(axis=0),((t==0)&p).sum(axis=0),((t==1)&~p).sum(axis=0),((t==0)&~p).sum(axis=0)],axis=-1))
    counts=np.asarray(counts);rng=np.random.default_rng(seed);boot=[]
    for _ in range(n):
        tp,fp,fn,tn=counts[rng.integers(len(unique),size=len(unique))].sum(axis=0).T
        fden=2*tp+fp+fn;mden=np.sqrt((tp+fp).astype(float)*(tp+fn)*(tn+fp)*(tn+fn))
        f=np.divide(2*tp,fden,out=np.zeros(len(names)),where=fden>0)
        m=np.divide(tp*tn-fp*fn,mden,out=np.zeros(len(names)),where=mden>0)
        acc=(tp+tn)/(tp+fp+fn+tn);boot.append(np.stack([f,m,acc],axis=-1))
    return names,np.asarray(boot)


def pair_intervals(rows,scores,thresholds,n_boot=1000):
    names=list(scores);pairs={}
    for i,r in enumerate(rows.itertuples()):pairs.setdefault(r.pair_id,{})[int(r.label)]=(i,str(r.commit_id))
    groups=[];correct=[]
    for pair,d in pairs.items():
        if set(d)!={0,1}:raise ValueError('Incomplete external pair')
        groups.append(d[1][1]);correct.append([scores[name][d[1][0]]>=thresholds[name] and scores[name][d[0][0]]<thresholds[name] for name in names])
    correct=np.asarray(correct);groups=np.asarray(groups);unique=sorted(set(groups))
    counts=np.array([correct[groups==g].sum(axis=0) for g in unique]);sizes=np.array([(groups==g).sum() for g in unique])
    rng=np.random.default_rng(42);values=[]
    for _ in range(n_boot):
        ix=rng.integers(len(unique),size=len(unique));values.append(counts[ix].sum(axis=0)/sizes[ix].sum())
    values=np.asarray(values)
    return {name:dict(pair_cluster_ci=np.quantile(values[:,j],[.025,.975]).tolist(),
         pair_binomial_ci=list(binomtest(int(correct[:,j].sum()),len(correct)).proportion_ci(method='exact')))
         for j,name in enumerate(names)}


def build_frames(root,regime,prime_rows,cohorts=Path('results/2026-11-01')):
    oldroot=cohorts/regime
    original=json.loads((oldroot/'metrics.json').read_text());oldnames=[m['method'] for m in original['metrics']]
    extended=root/'extended_baselines'/regime
    frames={};metadata={}
    for phase in ['validation','test','external','primevul']:
        if phase=='primevul':df=pd.DataFrame([dict(sample_id=r['sample_id'],label=r['label'],commit_id=r['extra']['commit_id'],pair_id=r['extra']['pair_id']) for r in prime_rows])
        else:df=pd.read_csv(oldroot/(phase+'_predictions.csv'))
        ids=df.sample_id.tolist();metadata[phase]=df
        source=pd.read_csv(extended/('external_predictions.csv' if phase in ['external','primevul'] else phase+'_predictions.csv')).set_index('sample_id')
        frames[phase]=pd.DataFrame({'v1_'+name:source.loc[ids,name].to_numpy() for name in oldnames},index=ids)
    for folder in sorted((root/'neural'/regime).glob('*')):
        if not (folder/'run.json').exists():continue
        name='g_'+folder.name if folder.name.endswith('_s42') else 'repeat_'+folder.name
        for phase,df in metadata.items():frames[phase][name]=aligned(folder/(phase+'_scores.csv'),df.sample_id.tolist())
    for folder in sorted((root/'classical'/regime).glob('*')):
        if not (folder/'run.json').exists():continue
        for phase,df in metadata.items():frames[phase]['k_'+folder.name]=aligned(folder/(phase+'_scores.csv'),df.sample_id.tolist())
    for folder in sorted((root/'experimental'/regime).glob('*')):
        if not (folder/'run.json').exists():continue
        for phase,df in metadata.items():frames[phase]['x_'+folder.name]=aligned(folder/(phase+'_scores.csv'),df.sample_id.tolist())
    # Three-seed means are explicitly different from a single model's result.
    if regime=='commit_disjoint':
        for arch in ['deepsets','gcn','sage','gin','gat','rgcn','ggnn','devign_style','ggnn_supcon']:
            names=['g_'+arch+'_hetero_s42']+['repeat_'+arch+'_hetero_s'+str(s) for s in [43,44]]
            for phase in frames:frames[phase]['seed_mean_'+arch]=frames[phase][names].mean(axis=1)
    return frames,metadata


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,default=Path('results/2026-11-01-graphs'))
    p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--prime',type=Path,required=True);a=p.parse_args()
    root=a.root;prime=load(a.prime);allresults=[];selected_rows=[]
    quality_map={r['sample_id']:[float(r['parse_error']),float(r['truncated'])] for r in load(a.cache_root/'graph_cache/graph_metadata.jsonl')}
    quality_map.update({r['sample_id']:[float(r['parse_error']),float(r['truncated'])] for r in load(a.cache_root/'primevul_graph_cache/graph_metadata.jsonl')})
    for regime in ['commit_disjoint','official_dedup','ffmpeg_to_qemu','qemu_to_ffmpeg']:
        start=time.monotonic();out=root/'comparisons'/regime;out.mkdir(parents=True,exist_ok=True)
        frames,meta=build_frames(root,regime,prime)
        partition=pd.read_csv(Path('results/2026-11-01')/regime/'validation_partitions.csv').set_index('sample_id')
        fit=np.array([partition.loc[s,'partition']=='val_fit' for s in meta['validation'].sample_id]);sel=~fit
        yv=meta['validation'].label.to_numpy()
        quality={phase:np.array([quality_map[s] for s in df.sample_id]) for phase,df in meta.items()}
        output,recipes=compose(frames,fit,yv,quality)
        for phase,scores in output.items():
            df=meta[phase][[c for c in ['sample_id','label','commit_id','pair_id'] if c in meta[phase]]].copy()
            for name,values in scores.items():df[name]=values
            df.to_csv(out/(phase+'_predictions.csv'),index=False)
        (out/'fusion_recipes.json').write_text(json.dumps(recipes,indent=2))
        thresholds={};validation=[]
        for name,pv in output['validation'].items():
            for policy in ['f1','mcc']:
                th,score=threshold(yv[sel],pv[sel],policy);thresholds[(name,policy)]=th
                validation.append(dict(method=name,policy=policy,threshold=th,validation_score=score))
        # Final model selection uses validation only, before reading test labels.
        winner={policy:max([r for r in validation if r['policy']==policy],key=lambda r:(r['validation_score'],r['method']))['method'] for policy in ['f1','mcc']}
        (out/'validation_selection.json').write_text(json.dumps(dict(selected=winner,rows=validation),indent=2))
        table=[];comparisons=[]
        yt=meta['test'].label.to_numpy();groups=meta['test'].commit_id.fillna(meta['test'].sample_id).tolist()
        for policy in ['f1','mcc']:
            preds={name:values>=thresholds[(name,policy)] for name,values in output['test'].items()}
            names,boot=clustered(yt,preds,groups)
            anchor_idx=names.index('v1_token_lr')
            for j,name in enumerate(names):
                row=metrics(yt,output['test'][name],thresholds[(name,policy)])
                row.update(method=name,policy=policy,protocol=regime,
                    f1_cluster_ci=np.quantile(boot[:,j,0],[.025,.975]).tolist(),
                    mcc_cluster_ci=np.quantile(boot[:,j,1],[.025,.975]).tolist(),
                    accuracy_cluster_ci=np.quantile(boot[:,j,2],[.025,.975]).tolist(),
                    default_f1=float(f1_score(yt,output['test'][name]>=.5,zero_division=0)),
                    default_mcc=float(matthews_corrcoef(yt,output['test'][name]>=.5)))
                table.append(row)
                ac=preds['v1_token_lr']==yt;bc=preds[name]==yt;n10=int(np.sum(ac&~bc));n01=int(np.sum(~ac&bc));nn=n10+n01
                comparisons.append(dict(method=name,policy=policy,anchor='v1_token_lr',
                    delta_f1=row['f1']-f1_score(yt,preds['v1_token_lr']),
                    delta_mcc=row['mcc']-matthews_corrcoef(yt,preds['v1_token_lr']),
                    delta_f1_cluster_ci=np.quantile(boot[:,j,0]-boot[:,anchor_idx,0],[.025,.975]).tolist(),
                    delta_mcc_cluster_ci=np.quantile(boot[:,j,1]-boot[:,anchor_idx,1],[.025,.975]).tolist(),
                    mcnemar_p=float(binomtest(n10,nn,.5).pvalue) if nn else 1.))
            running=0.
            for i,row in enumerate(sorted([r for r in comparisons if r['policy']==policy],key=lambda x:x['mcnemar_p'])):
                running=max(running,min(1.,row['mcnemar_p']*(len(names)-i)));row['holm_p']=running
            for phase in ['external','primevul']:
                rows=meta[phase];ye=rows.label.to_numpy()
                paired=[dict(label=int(r.label),extra={'pair_id':r.pair_id}) for r in rows.itertuples()]
                intervals=pair_intervals(rows,output[phase],{name:thresholds[(name,policy)] for name in output[phase]})
                for name,pv in output[phase].items():
                    row=metrics(ye,pv,thresholds[(name,policy)]);row.update(method=name,policy=policy,protocol=regime,phase=phase,
                        **pair_metrics(paired,pv,thresholds[(name,policy)]),**intervals[name]);allresults.append(row)
            # Val-observed FPR thresholds are frozen before corresponding test metrics.
        operating=[]
        for name,pv in output['validation'].items():
            for cap in [.005,.01,.05]:
                negative=pv[sel][yv[sel]==0];grid=np.r_[np.unique(negative),1.000001]
                eligible=[t for t in grid if np.mean(negative>=t)<=cap]
                t=min(eligible);pp=output['test'][name];mm=metrics(yt,pp,float(t));operating.append(dict(method=name,val_fpr_cap=cap,**mm))
        # Retain a common diagnostic-free/non-truncated sensitivity subset.
        clean=np.array([not any(quality_map[s]) for s in meta['test'].sample_id])
        sensitivity=[]
        for name,pv in output['test'].items():
            if clean.sum():sensitivity.append(dict(method=name,**metrics(yt[clean],pv[clean],thresholds[(name,'mcc')])))
        # Offline review ranking and coverage, using no test-label selection.
        review=[]
        for name in output['test']:
            pv=output['test'][name];order=np.argsort(-pv,kind='stable')
            for budget in [.01,.05,.1,.2]:
                n=max(1,int(np.ceil(len(pv)*budget)));ix=order[:n]
                review.append(dict(method=name,budget=budget,n=n,precision=float(yt[ix].mean()),recall=float(yt[ix].sum()/yt.sum()),random_expected_precision=float(yt.mean())))
        result=dict(protocol=regime,n_methods=len(output['test']),selected_on_validation=winner,metrics=table,
                    comparisons=comparisons,external_metrics=[r for r in allresults if r['protocol']==regime],
                    operating_points=operating,diagnostic_free=sensitivity,review_budget=review,seconds=time.monotonic()-start)
        (out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,list)} for r in table]).to_csv(out/'leaderboard.csv',index=False)
        for policy,name in winner.items():selected_rows.append(next(r for r in table if r['method']==name and r['policy']==policy))
        print('EVALUATED',regime,len(output['test']),'methods','selected',winner,'seconds',round(result['seconds'],1),flush=True)
    (root/'selected_results.json').write_text(json.dumps(selected_rows,indent=2))


if __name__=='__main__':main()
