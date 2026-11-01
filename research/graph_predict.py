"""Export frozen neural scores on a new graph cache, without fitting anything."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from research.graph_data import GraphStore
from research.graph_models import GraphClassifier
from research.graph_train import data_for,predict
from research.benchmark import load


def run(a):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    rows=load(a.samples);labels={r['sample_id']:r['label'] for r in rows}
    # Resolve feature compatibility from the training cache digest, never label quality.
    summaries={}
    for path in a.cache_root.glob('graph_cache*'):
        if (path/'graph_summary.json').exists():
            d=json.loads((path/'graph_summary.json').read_text());summaries[d['cache_sha256']]=d
    stores={};data_cache={}
    for model_path in sorted(a.runs.rglob('model.pt')):
        folder=model_path.parent;meta=json.loads((folder/'run.json').read_text())
        if (folder/'primevul_scores.csv').exists():continue
        config=summaries[meta['graph_cache_sha256']]
        key=(config.get('selection','bfs'),config.get('semantic',False),config['max_nodes'])
        name='primevul_graph_cache' + ('_evidence' if key[0]=='evidence' else '') + ('_semantic' if key[1] else '') + ('_512' if key[2]==512 else '')
        graphpath=a.cache_root/name
        if key not in stores:stores[key]=GraphStore(graphpath)
        dk=key+(meta['view'],meta['anonymous'])
        if dk not in data_cache:data_cache[dk]=[data_for(stores[key],r['sample_id'],r['label'],meta['view'],meta['anonymous']) for r in rows]
        checkpoint=torch.load(model_path,map_location='cpu',weights_only=True)
        model=GraphClassifier(checkpoint['architecture'],checkpoint['hidden'],checkpoint['dropout']);model.load_state_dict(checkpoint['state_dict'])
        p,z,ids=predict(model,DataLoader(data_cache[dk],batch_size=128,shuffle=False))
        assert ids==[r['sample_id'] for r in rows]
        pd.DataFrame({'sample_id':ids,'probability':p}).to_csv(folder/'primevul_scores.csv',index=False)
        np.savez_compressed(folder/'primevul_embeddings.npz',embedding=z,sample_id=np.asarray(ids))
        print('EXTERNAL SCORED',str(folder),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--runs',type=Path,required=True);p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--samples',type=Path,required=True);run(p.parse_args())
