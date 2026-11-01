"""Locked, matched-cohort CPU benchmark. No published model is impersonated.

Train base models on train; use commit-disjoint halves of validation for
model/calibration fitting and threshold/method selection; evaluate test once.
Run: python -m research.benchmark --samples data/research/devign/samples.jsonl \
     --external data/research/sven/samples.jsonl --findings ... --out ...
"""
from __future__ import annotations
import argparse, collections, hashlib, importlib.metadata, json, os, platform, re, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from pipeline.benchmark_hygiene import code_hash, sanitize_code


def load(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def audit(rows):
    by_hash = collections.defaultdict(list); commits = collections.defaultdict(set)
    for r in rows:
        by_hash[code_hash(r['code'])].append(r)
        commit = r.get('extra', {}).get('commit_id')
        if commit: commits[commit].add(r['split'])
    return dict(n=len(rows), partitions=dict(collections.Counter(r['split'] for r in rows)),
        normalized_unique=len(by_hash), duplicate_groups=sum(len(g)>1 for g in by_hash.values()),
        conflicting_label_groups=sum(len({r['label'] for r in g})>1 for g in by_hash.values()),
        cross_split_clone_groups=sum(len({r['split'] for r in g})>1 for g in by_hash.values()),
        cross_split_commit_groups=sum(len(s)>1 for s in commits.values()))


def deduplicate(rows):
    """Earliest partition wins (train, val, test); labels never cross splits.
    Conflicting-label groups within a partition are excluded there. All of a
    partition's hashes are reserved, even if contradictory. No test label can
    change training data. This is exact token identity, not near-clone removal.
    """
    kept = []; seen = set(); exclusions = []
    for part in ('train', 'val', 'test'):
        groups = collections.defaultdict(list)
        for r in rows:
            if r['split'] == part: groups[code_hash(r['code'])].append(r)
        for h, g in groups.items():
            g.sort(key=lambda r: r['sample_id'])
            reason = ('earlier_partition_clone' if h in seen else
                      'within_partition_label_conflict' if len({r['label'] for r in g})>1 else None)
            if reason:
                exclusions.extend(dict(sample_id=r['sample_id'], reason=reason) for r in g)
            else:
                kept.append(g[0])
                exclusions.extend(dict(sample_id=r['sample_id'], reason='within_partition_clone') for r in g[1:])
        seen.update(groups)
    return sorted(kept, key=lambda r:r['sample_id']), exclusions


def select_threshold(y, p):
    """Same validation-F1 rule for every learned score, deterministic ties.
    Include all-positive/all-negative endpoints. Selection never sees test.
    """
    candidates = np.r_[np.linspace(0, 1, 101), 1.000001]
    scores = [f1_score(y, p >= t, zero_division=0) for t in candidates]
    best = max(scores)
    tied = [i for i,s in enumerate(scores) if abs(s-best) < 1e-12]
    idx = min(tied, key=lambda i:(abs(candidates[i]-.5), -candidates[i]))
    return float(candidates[idx]), float(best)


def ece(y, p):
    bins = np.minimum((np.asarray(p)*10).astype(int),9)
    return float(sum(np.mean(bins==b) * abs(np.mean(y[bins==b])-np.mean(p[bins==b]))
                     for b in range(10) if np.any(bins==b)))


def metrics(y, p, threshold):
    pred = p >= threshold; tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(n=len(y), positives=int(sum(y)), accuracy=accuracy_score(y,pred),
        balanced_accuracy=balanced_accuracy_score(y,pred), precision=precision_score(y,pred,zero_division=0),
        recall=recall_score(y,pred,zero_division=0), f1=f1_score(y,pred,zero_division=0),
        mcc=matthews_corrcoef(y,pred), roc_auc=roc_auc_score(y,p) if len(set(y))==2 else None,
        average_precision=average_precision_score(y,p), brier=brier_score_loss(y,np.clip(p,0,1)),
        ece=ece(y,p), fpr=float(fp/(fp+tn)) if fp+tn else None,
        tp=int(tp),fp=int(fp),tn=int(tn),fn=int(fn),threshold=threshold)


def grouped_indices(rows, seed):
    groups = [r.get('extra',{}).get('commit_id',r['sample_id']) for r in rows]
    return list(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed).split(rows,groups=groups))[0]


def bootstrap(y, pred, groups, seed=42, n=1000):
    unique = sorted(set(groups)); members = [np.flatnonzero(np.asarray(groups)==g) for g in unique]
    # Aggregate confusion counts per commit for fast clustered resampling.
    table = np.array([[sum((y[ix]==1)&pred[ix]),sum((y[ix]==0)&pred[ix]),
                       sum((y[ix]==1)&~pred[ix])] for ix in members])
    rng=np.random.default_rng(seed); vals=[]
    for _ in range(n):
        tp,fp,fn=table[rng.integers(len(members),size=len(members))].sum(axis=0)
        vals.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return [float(x) for x in np.quantile(vals,[.025,.975])]


def paired_comparisons(y, preds, groups, anchor='token_lr', n_boot=1000):
    out=[]; rng=np.random.default_rng(123); unique=sorted(set(groups))
    members=[np.flatnonzero(np.asarray(groups)==g) for g in unique]
    def counts(pred):
        return np.array([[sum((y[ix]==1)&pred[ix]),sum((y[ix]==0)&pred[ix]),
                          sum((y[ix]==1)&~pred[ix])] for ix in members])
    ac=counts(preds[anchor])
    for name,p in preds.items():
        if name==anchor: continue
        a=preds[anchor]==y;b=p==y;n10=int(sum(a&~b));n01=int(sum(~a&b));nc=n10+n01
        pval=float(binomtest(n10,nc,.5).pvalue) if nc else 1.
        bc=counts(p);deltas=[]
        for _ in range(n_boot):
            ix=rng.integers(len(members),size=len(members))
            def f(t):
                tp,fp,fn=t[ix].sum(axis=0);d=2*tp+fp+fn
                return 2*tp/d if d else 0.
            deltas.append(f(bc)-f(ac))
        out.append(dict(anchor=anchor,method=name,anchor_only_correct=n10,method_only_correct=n01,
            mcnemar_p=pval,delta_f1=float(f1_score(y,p)-f1_score(y,preds[anchor])),
            delta_f1_cluster_ci=[float(x) for x in np.quantile(deltas,[.025,.975])]))
    running=0
    for i,r in enumerate(sorted(out,key=lambda r:r['mcnemar_p'])):
        running=max(running,min(1.,r['mcnemar_p']*(len(out)-i)));r['holm_p']=running
    return out


def pair_metrics(rows, p, threshold):
    pairs=collections.defaultdict(dict)
    for r,prob in zip(rows,p):pairs[r['extra']['pair_id']][r['label']]=float(prob)
    pairs=[d for d in pairs.values() if set(d)=={0,1}]
    if not pairs:return dict(n_pairs=0)
    return dict(n_pairs=len(pairs), both_correct=float(np.mean([d[1]>=threshold and d[0]<threshold for d in pairs])),
        vulnerable_ranks_higher=float(np.mean([d[1]>d[0] for d in pairs])),
        tied=float(np.mean([d[1]==d[0] for d in pairs])),
        mean_score_difference=float(np.mean([d[1]-d[0] for d in pairs])))


def static_features(rows, findings):
    by=collections.defaultdict(list)
    for f in findings:by[f['sample_id']].append(f)
    outputs=[]
    for r in rows:
        fs=by[r['sample_id']];features={}
        for f in fs:
            if f['tool']=='semgrep' and f['rule_id']=='unchecked-arithmetic-in-allocation':continue
            features[f'{f["tool"]}:{f["rule_id"]}']=1.
        # Presence features avoid treating correlated duplicate lines as evidence.
        outputs.append(features)
    return outputs


def shape_features(code):
    return [np.log1p(len(code)),np.log1p(code.count('\n')),np.log1p(len(re.findall(r'\bif\b',code))),
            np.log1p(len(re.findall(r'\b(?:for|while)\b',code))),np.log1p(code.count('*')),
            np.log1p(code.count('->')),np.log1p(code.count('[')),np.log1p(code.count('return'))]


def run(args):
    start=time.monotonic();args.out.mkdir(parents=True,exist_ok=True)
    rows=load(args.samples)
    if args.commit_disjoint:
        for r in rows:
            group = r.get('extra',{}).get('project','') + ':' + r['extra']['commit_id']
            bucket = int(hashlib.sha256(f'{args.seed}:{group}'.encode()).hexdigest()[:12],16) / 16**12
            r['split'] = 'train' if bucket < .8 else 'val' if bucket < .9 else 'test'
    if args.source_project:
        rows=[r for r in rows if (r['split']!='test' and r['extra']['project']==args.source_project)
              or (r['split']=='test' and r['extra']['project']!=args.source_project)]
    original_audit=audit(rows);rows,excluded=deduplicate(rows)
    ids=[r['sample_id'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate sample IDs')
    parts={p:[r for r in rows if r['split']==p] for p in ('train','val','test')}
    tr,va,te=[parts[p] for p in ('train','val','test')]
    fit,select=grouped_indices(va,args.seed)
    external=load(args.external) if args.external else []
    seen={code_hash(r['code']) for r in rows}; bypair=collections.defaultdict(list)
    for r in external:bypair[r['extra']['pair_id']].append(r)
    retained=[];external_excluded=[]
    for pair,group in bypair.items():
        if (len(group)!=2 or {r['label'] for r in group}!={0,1}
                or any(code_hash(r['code']) in seen for r in group)
                or code_hash(group[0]['code'])==code_hash(group[1]['code'])):
            external_excluded.append(pair)
        else:retained.extend(group)
    external=retained
    allrows=tr+va+te+external
    ytr=np.array([r['label'] for r in tr]);yv=np.array([r['label'] for r in va]);yt=np.array([r['label'] for r in te])
    yy=np.r_[yv,yt,[r['label'] for r in external]].astype(int)
    code=[sanitize_code(r['code'],r['dataset']=='juliet') for r in allrows]
    ntr=len(tr);nv=len(va);nt=len(te);ev=slice(0,nv);test=slice(nv,nv+nt);ext=slice(nv+nt,None)
    features=static_features(allrows,load(args.findings))
    dv=DictVectorizer();xf=dv.fit_transform(features[:ntr]);xrest=dv.transform(features[ntr:])
    shape=np.array([shape_features(c) for c in code]);shape_tr=shape[:ntr];shape_rest=shape[ntr:]
    scores={};details={};train_times={}
    def register(name,p,detail=None):
        scores[name]=np.asarray(p,dtype=float);details[name]=detail or {}
        print('SCORED',name,flush=True)
    register('always_safe',np.zeros(len(yy)));register('always_vulnerable',np.ones(len(yy)))
    register('train_prevalence',np.full(len(yy),ytr.mean()))
    for tool in ('semgrep','cppcheck','flawfinder','taint'):
        if not any(f['tool']==tool for f in load(args.findings)):
            # Missing engines are not silently interpreted as all-clear.
            if not (Path(args.findings).parent/'coverage.json').exists():raise RuntimeError('Missing tool coverage')
            cov=json.loads((Path(args.findings).parent/'coverage.json').read_text())
            if tool not in cov:raise RuntimeError(f'{tool} was not run')
        values=np.array([any(k.startswith(tool+':') for k in f) for f in features[ntr:]],float)
        register(tool+'_any',values)
    for view in ('token','char'):
        t=time.monotonic()
        vectorizer=TfidfVectorizer(analyzer='word' if view=='token' else 'char',
            token_pattern=r'(?u)\b\w+\b' if view=='token' else r'(?u)\b\w\w+\b',
            ngram_range=(1,2) if view=='token' else (3,5),min_df=3,max_df=.98,
            max_features=40000 if view=='token' else 50000,sublinear_tf=True,dtype=np.float32)
        x=vectorizer.fit_transform(code[:ntr]);z=vectorizer.transform(code[ntr:])
        candidates=[]
        for c in (.5,2.,8.):
            model=LogisticRegression(C=c,solver='liblinear',max_iter=300,random_state=args.seed)
            model.fit(x,ytr);p=model.predict_proba(z)[:,1]
            candidates.append((average_precision_score(yv[fit],p[fit]),c,model,p))
        _,c,model,p=max(candidates,key=lambda d:(d[0],-d[1]));register(view+'_lr',p,dict(C=c,vocabulary=len(vectorizer.vocabulary_)))
        train_times[view+'_lr']=time.monotonic()-t
        if view=='token':
            m=MultinomialNB(alpha=1.);m.fit(x,ytr);register('token_nb',m.predict_proba(z)[:,1],dict(alpha=1.))
            m=LinearSVC(C=1.,random_state=args.seed,dual='auto');m.fit(x,ytr)
            margin=m.decision_function(z)
            cal=LogisticRegression(C=1.,solver='liblinear',random_state=args.seed)
            cal.fit(margin[fit,None],yv[fit]);register('token_svm_platt',cal.predict_proba(margin[:,None])[:,1])
            m=RandomForestClassifier(n_estimators=100,max_depth=24,min_samples_leaf=2,n_jobs=2,random_state=args.seed)
            m.fit(x,ytr);register('token_rf',m.predict_proba(z)[:,1],dict(trees=100,max_depth=24))
    for name,x,z in [('static_lr',xf,xrest),('shape_lr',shape_tr,shape_rest)]:
        m=LogisticRegression(C=1.,solver='liblinear',max_iter=500,random_state=args.seed)
        m.fit(x,ytr);register(name,m.predict_proba(z)[:,1])
    register('mean_token_char',(scores['token_lr']+scores['char_lr'])/2)
    base=logit(np.clip(scores['mean_token_char'],1e-5,1-1e-5))
    static=logit(np.clip(scores['static_lr'],1e-5,1-1e-5))
    # Conventional log-odds mixture ablation: no LLM confidence is fabricated.
    options=[]
    for w in (0.,.25,.5,.75,1.):
        p=expit(w*base+(1-w)*static);th,f=select_threshold(yv[select],p[select]);options.append((f,w,p))
    _,weight,p=max(options,key=lambda x:(x[0],x[1]));register('logodds_lexical_static',p,dict(lexical_weight=weight))
    # Prototype 1: fit a regularized stack on held-out predictions, not training predictions.
    stack=np.column_stack([logit(np.clip(scores[n],1e-5,1-1e-5)) for n in ('token_lr','char_lr','static_lr')])
    m=LogisticRegression(C=.1,solver='liblinear',random_state=args.seed);m.fit(stack[fit],yv[fit])
    register('heldout_evidence_stack',m.predict_proba(stack)[:,1],dict(coefficients=m.coef_.tolist(),intercept=m.intercept_.tolist()))
    # Prototype 2: evidence-gated residual update, anchored to lexical logit.
    # Sparse tool indicators earn their correction on val-fit, with L2 shrinkage.
    # Optimize logistic loss with a fixed coefficient of one on the base score.
    from scipy.optimize import minimize
    gate=np.asarray(xrest.toarray(),float)
    gate=np.column_stack([np.ones(len(gate)),gate])
    lam=10.
    def objective(beta):
        eta=base[fit]+gate[fit]@beta;p=expit(eta)
        loss=np.logaddexp(0,eta).sum()-yv[fit]@eta + .5*lam*np.dot(beta,beta)
        gradient=gate[fit].T@(p-yv[fit])+lam*beta
        return loss,gradient
    result=minimize(objective,np.zeros(gate.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':300})
    if not result.success:raise RuntimeError('Residual calibration did not converge')
    register('gated_residual',expit(base+gate@result.x),dict(l2=lam,converged=bool(result.success),
        intercept=float(result.x[0]),corrections=dict(zip(dv.feature_names_,result.x[1:].tolist()))))
    # Ablation: intercept-only recalibration separates base-rate from static gains.
    def offset_loss(b):
        eta=base[fit]+b[0]
        return float(np.logaddexp(0,eta).sum()-yv[fit]@eta+.5*lam*b[0]**2)
    off=minimize(offset_loss,np.zeros(1),method='BFGS')
    register('intercept_only',expit(base+off.x[0]),dict(intercept=float(off.x[0])))
    thresholds={};table=[];predictions={}
    fixed=set(['always_safe','always_vulnerable','train_prevalence','semgrep_any','cppcheck_any','flawfinder_any','taint_any'])
    val_scores={}
    for name,p in scores.items():
        th,vf=(.5,float(f1_score(yv[select],p[select]>=.5,zero_division=0))) if name in fixed else select_threshold(yv[select],p[select])
        thresholds[name]=th;val_scores[name]=vf;predictions[name]=p[test]>=th
        met=metrics(yt,p[test],th);met.update(method=name,validation_f1=vf,
            default_threshold_f1=float(f1_score(yt,p[test]>=.5,zero_division=0)))
        groups=[r.get('extra',{}).get('commit_id',r['sample_id']) for r in te]
        met['f1_cluster_ci']=bootstrap(yt,predictions[name],groups,args.seed,args.bootstrap)
        table.append(met)
    winner=max(scores,key=lambda n:(val_scores[n],n))
    comparisons=paired_comparisons(yt,predictions,groups,n_boot=args.bootstrap)
    external_results=[]
    if external:
        ye=yy[ext]
        for name,p in scores.items():
            met=metrics(ye,p[ext],thresholds[name]);met.update(method=name,**pair_metrics(external,p[ext],thresholds[name]));external_results.append(met)
    select_frame=pd.DataFrame([dict(sample_id=r['sample_id'],partition='val_fit' if i in set(fit) else 'val_select',
        commit_id=r.get('extra',{}).get('commit_id')) for i,r in enumerate(va)])
    select_frame.to_csv(args.out/'validation_partitions.csv',index=False)
    for scope,scope_rows,sl in [('validation',va,ev),('test',te,test),('external',external,ext)]:
        if not scope_rows:continue
        df=pd.DataFrame([dict(sample_id=r['sample_id'],label=r['label'],
            code_sha256=code_hash(r['code']),project=r.get('extra',{}).get('project'),
            commit_id=r.get('extra',{}).get('commit_id'),pair_id=r.get('extra',{}).get('pair_id')) for r in scope_rows])
        for name,p in scores.items():df[name]=p[sl]
        df.to_csv(args.out/f'{scope}_predictions.csv',index=False)
    # Membership file permits exact reconstruction without redistributing code.
    pd.DataFrame([dict(sample_id=r['sample_id'],split=r['split'],code_sha256=code_hash(r['code'])) for r in rows]).to_csv(args.out/'cohort.csv',index=False)
    result=dict(protocol='v1-frozen-2026-11-01',seed=args.seed,source_project=args.source_project,
        commit_disjoint=args.commit_disjoint,
        original_audit=original_audit,clean_audit=audit(rows),exclusions=excluded,
        external_n=len(external),external_pairs_excluded=external_excluded,
        validation_fit_n=len(fit),validation_select_n=len(select),selected_on_validation=winner,
        thresholds=thresholds,details=details,train_times=train_times,metrics=table,
        comparisons=comparisons,external_metrics=external_results,
        wall_seconds=time.monotonic()-start,environment=dict(python=platform.python_version(),
        versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','pandas']},
        cpu=platform.processor(),threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']}))
    # Strict JSON prevents NaN silently entering an artifact.
    (args.out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,list)} for r in table]).to_csv(args.out/'leaderboard.csv',index=False)
    print(pd.DataFrame(table)[['method','n','accuracy','precision','recall','f1','mcc','validation_f1']].to_string(index=False))
    print('VALIDATION-SELECTED:',winner,'seconds',result['wall_seconds'],flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--external',type=Path);p.add_argument('--findings',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--source-project')
    p.add_argument('--commit-disjoint',action='store_true',help='Hash whole commits into 80/10/10 groups before code deduplication')
    args=p.parse_args();run(args)


if __name__=='__main__':main()
