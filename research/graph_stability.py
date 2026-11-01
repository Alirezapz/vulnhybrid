"""Frozen input perturbations; never re-tune thresholds on perturbed test data."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from research.graph_data import GraphStore
from research.graph_train import data_for,predict
from research.graph_models import GraphClassifier,ARCHITECTURES


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,default=Path('results/2026-11-01-graphs'));p.add_argument('--graphs',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(42)
    cohort=pd.read_csv('results/2026-11-01/commit_disjoint/test_predictions.csv');store=GraphStore(a.graphs)
    original=[data_for(store,r.sample_id,int(r.label)) for r in cohort.itertuples()];variants={}
    for variant in ['no_edges','anonymous_payload','rewired_relations','node_permutation']:
        arr=[]
        for data in original:
            x=data.clone()
            if variant=='no_edges':x.edge_index=x.edge_index[:,:0];x.edge_type=x.edge_type[:0]
            if variant=='anonymous_payload':x.token_ids.zero_()
            if variant=='rewired_relations':
                for relation in range(8):
                    ix=torch.nonzero(x.edge_type==relation,as_tuple=False).flatten()
                    if len(ix):x.edge_index[1,ix]=x.edge_index[1,ix[torch.randperm(len(ix))]]
            if variant=='node_permutation':
                perm=torch.randperm(x.num_nodes);inverse=torch.empty_like(perm);inverse[perm]=torch.arange(len(perm))
                x.type_ids=x.type_ids[perm];x.token_ids=x.token_ids[perm];x.x_num=x.x_num[perm];x.edge_index=inverse[x.edge_index]
            arr.append(x)
        variants[variant]=arr
    out=a.root/'stability';out.mkdir(exist_ok=True)
    for arch in ARCHITECTURES:
        path=a.root/'neural/commit_disjoint'/(arch+'_hetero_s42')/'model.pt'
        ck=torch.load(path,map_location='cpu',weights_only=True);model=GraphClassifier(ck['architecture'],ck['hidden'],ck['dropout']);model.load_state_dict(ck['state_dict'])
        df=pd.DataFrame({'sample_id':cohort.sample_id})
        for name,data in [('original',original),*variants.items()]:
            scores,_,ids=predict(model,DataLoader(data,batch_size=128,shuffle=False));assert ids==cohort.sample_id.tolist();df[name]=scores
        saved=pd.read_csv(path.parent/'test_scores.csv').set_index('sample_id').loc[cohort.sample_id,'probability'].to_numpy()
        assert np.allclose(saved,df.original,atol=1e-6)
        df.to_csv(out/(arch+'_scores.csv'),index=False);print('STABILITY',arch,flush=True)


if __name__=='__main__':main()
