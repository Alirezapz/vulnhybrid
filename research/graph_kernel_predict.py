"""Frozen classical-graph prediction on an external cache, no fitting."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction import FeatureHasher
from research.graph_data import GraphStore
from research.graph_kernels import FEATURES,wl,structure
from research.benchmark import load


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--graphs',type=Path,required=True);p.add_argument('--samples',type=Path,required=True);p.add_argument('--runs',type=Path,required=True)
    a=p.parse_args();rows=load(a.samples);ids=[r['sample_id'] for r in rows];store=GraphStore(a.graphs)
    hasher=FeatureHasher(n_features=65536,input_type='dict',alternate_sign=False,dtype=np.float32)
    features={name:hasher.transform(wl(store.get(s,view),h,typed) for s in ids) for name,view,h,typed in FEATURES}
    features['ast_wl3_svm']=features['ast_wl3']
    numeric=np.array([structure(store.get(s),store.metadata[store.ids[s]]) for s in ids],np.float32)
    for folder in sorted(a.runs.glob('*/*')):
        if not (folder/'model.joblib').exists():continue
        trained=joblib.load(folder/'model.joblib')
        if folder.name=='graph_structure_hgb':prob=trained.predict_proba(numeric)[:,1]
        else:
            x=trained['normalizer'].transform(features[folder.name])
            if 'calibrator' in trained:prob=trained['calibrator'].predict_proba(trained['model'].decision_function(x)[:,None])[:,1]
            else:prob=trained['model'].predict_proba(x)[:,1]
        pd.DataFrame({'sample_id':ids,'probability':prob}).to_csv(folder/'primevul_scores.csv',index=False)
        print('KERNEL EXTERNAL',folder,flush=True)


if __name__=='__main__':main()
