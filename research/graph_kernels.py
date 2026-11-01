"""Hashed Weisfeiler-Lehman feature baselines with train-only TF-IDF fitting."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
import joblib
from research.graph_data import GraphStore

FEATURES=[('node_hist','ast',0,False),('ast_wl2','ast',2,False),('ast_wl3','ast',3,False),
          ('hetero_wl2','hetero',2,True),('cfg_wl2','cfg',2,False),('last_ident_wl2','ddg',2,False)]


def wl(g,iterations=2,typed=False):
    colors=[str(v) for v in g['types']];neighbors=[set() for _ in colors]
    for a,b,rel in g['edges']:
        neighbors[int(b)].add((int(a),int(rel) if typed else 0))
    features=Counter('0:'+c for c in colors)
    for it in range(1,iterations+1):
        nxt=[]
        for i,c in enumerate(colors):
            message=c+'|'+','.join(sorted(str(r)+':'+colors[a] for a,r in neighbors[i]))
            nxt.append(hashlib.blake2b(message.encode(),digest_size=8).hexdigest())
        colors=nxt;features.update(str(it)+':'+c for c in colors)
    return dict(features)


def structure(g,meta):
    n=len(g['types']);e=g['edges'];degree=np.bincount(e[:,1],minlength=n) if len(e) else np.zeros(n)
    vals=[np.log1p(n),np.log1p(len(e)),float(meta['parse_error']),float(meta['truncated']),
          np.log1p(meta['full_nodes']),float(degree.mean()),float(degree.max())]
    vals += [float(np.mean(degree==k)) for k in range(8)]
    vals += [float(np.mean(e[:,2]==k)) if len(e) else 0. for k in range(8)]
    vals += g['numeric'].mean(axis=0).tolist()
    return vals


def run(a):
    a.out.mkdir(parents=True,exist_ok=True);store=GraphStore(a.graphs);ids=[r['sample_id'] for r in store.metadata]
    lookup={s:i for i,s in enumerate(ids)};start=time.monotonic();hasher=FeatureHasher(n_features=65536,input_type='dict',alternate_sign=False,dtype=np.float32)
    variants=[]
    for name,view,h,typed in FEATURES:
        cache=a.graphs/(name+'.npz')
        if cache.exists():x=sparse.load_npz(cache)
        else:
            x=hasher.transform(wl(store.get(s,view),h,typed) for s in ids)
            sparse.save_npz(cache,x)
        variants.append((name,x));print('FEATURES',name,x.shape,'nnz',x.nnz,'seconds',round(time.monotonic()-start),flush=True)
    numeric=np.asarray([structure(store.get(s),m) for s,m in zip(ids,store.metadata)],np.float32)
    for regime in ['commit_disjoint','official_dedup','ffmpeg_to_qemu','qemu_to_ffmpeg']:
        root=a.cohorts/regime;cohort=pd.read_csv(root/'cohort.csv');trids=cohort[cohort.split=='train'].sample_id.tolist()
        old={k:pd.read_csv(root/(k+'_predictions.csv')) for k in ['validation','test','external']}
        idx={k:np.asarray([lookup[s] for s in df.sample_id]) for k,df in old.items()};tr=np.asarray([lookup[s] for s in trids])
        partitions=pd.read_csv(root/'validation_partitions.csv').set_index('sample_id')
        fit=np.array([partitions.loc[s,'partition']=='val_fit' for s in old['validation'].sample_id])
        labels={r['sample_id']:r['label'] for file in a.samples for r in __import__('research.benchmark',fromlist=['load']).load(file)}
        y=np.array([labels[s] for s in trids]);yv=old['validation'].label.to_numpy()
        for name,x in variants:
            existing=a.out/regime/name/'run.json'
            reusable=existing.exists() and (existing.parent/'model.joblib').exists()
            svm_root=a.out/regime/'ast_wl3_svm'
            if reusable and (name!='ast_wl3' or ((svm_root/'run.json').exists() and (svm_root/'model.joblib').exists())):
                print('REUSE',regime,name,flush=True);continue
            norm=TfidfTransformer(sublinear_tf=True);train=norm.fit_transform(x[tr]);rest={k:norm.transform(x[ix]) for k,ix in idx.items()}
            best=None
            for c in [.1,1.,10.]:
                model=LogisticRegression(C=c,solver='liblinear',max_iter=500,random_state=42);model.fit(train,y)
                p=model.predict_proba(rest['validation'])[:,1];ap=float(average_precision_score(yv[fit],p[fit]))
                if best is None or ap>best[0]:best=(ap,c,model)
            _,c,model=best;out=a.out/regime/name;out.mkdir(parents=True,exist_ok=True)
            for key,df in old.items():pd.DataFrame({'sample_id':df.sample_id,'probability':model.predict_proba(rest[key])[:,1]}).to_csv(out/(key+'_scores.csv'),index=False)
            joblib.dump(dict(normalizer=norm,model=model,view=name),out/'model.joblib',compress=3)
            (out/'run.json').write_text(json.dumps(dict(status='completed',method=name,C=c,val_fit_ap=best[0],train_n=len(tr),features=65536),indent=2))
            print('CLASSICAL',regime,name,'val_fit_ap',best[0],flush=True)
            if name=='ast_wl3':
                svm=LinearSVC(C=1.,random_state=42,dual='auto');svm.fit(train,y)
                margin={k:svm.decision_function(z) for k,z in rest.items()}
                cal=LogisticRegression(solver='liblinear',random_state=42);cal.fit(margin['validation'][fit,None],yv[fit])
                out=a.out/regime/'ast_wl3_svm';out.mkdir(parents=True,exist_ok=True)
                for key,df in old.items():pd.DataFrame({'sample_id':df.sample_id,'probability':cal.predict_proba(margin[key][:,None])[:,1]}).to_csv(out/(key+'_scores.csv'),index=False)
                joblib.dump(dict(normalizer=norm,model=svm,calibrator=cal),out/'model.joblib',compress=3)
                (out/'run.json').write_text(json.dumps(dict(status='completed',method='ast_wl3_svm',train_n=len(tr)),indent=2))
        # Degree/relation statistics control; fit only on training.
        model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,l2_regularization=10,random_state=42)
        model.fit(numeric[tr],y);out=a.out/regime/'graph_structure_hgb';out.mkdir(parents=True,exist_ok=True)
        for key,df in old.items():pd.DataFrame({'sample_id':df.sample_id,'probability':model.predict_proba(numeric[idx[key]])[:,1]}).to_csv(out/(key+'_scores.csv'),index=False)
        joblib.dump(model,out/'model.joblib',compress=3);(out/'run.json').write_text(json.dumps(dict(status='completed',method='graph_structure_hgb',train_n=len(tr)),indent=2))
    (a.out/'completion.json').write_text(json.dumps(dict(status='completed',seconds=time.monotonic()-start,variants=[v[0] for v in variants]+['ast_wl3_svm','graph_structure_hgb']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--graphs',type=Path,required=True);p.add_argument('--cohorts',type=Path,default=Path('results/2026-11-01'))
    p.add_argument('--samples',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
