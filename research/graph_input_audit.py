"""Input-collision upper bounds for deterministic repair-pair classification."""
from pathlib import Path
import argparse,hashlib,json,collections
import numpy as np
import pandas as pd
from research.graph_data import GraphStore


def signature(g):
    h=hashlib.sha256()
    for key in ['types','tokens','numeric','edges']:
        h.update(np.ascontiguousarray(g[key]).tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sven-cohort',type=Path,default=Path('results/2026-11-01/commit_disjoint/external_predictions.csv'))
    a=p.parse_args();df=pd.read_csv(a.sven_cohort);rows=[]
    for cache in ['graph_cache','graph_cache_evidence','graph_cache_semantic','graph_cache_semantic_bfs','graph_cache_512']:
        store=GraphStore(a.cache_root/cache);ids=dict(zip(df.sample_id,df.pair_id));pairs=collections.defaultdict(dict)
        for sid,pair in ids.items():pairs[pair][int(df.loc[df.sample_id==sid,'label'].iloc[0])]=signature(store.get(sid))
        identical=[pair for pair,items in pairs.items() if items[0]==items[1]]
        summary=json.loads((a.cache_root/cache/'graph_summary.json').read_text())
        rows.append(dict(cache=cache,n_pairs=len(pairs),identical_pairs=len(identical),
          deterministic_pair_success_upper_bound=(len(pairs)-len(identical))/len(pairs),
          exact_collisions=identical,**{k:summary[k] for k in ['max_nodes','parse_errors','truncated']}))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(rows,indent=2));print(json.dumps([{k:v for k,v in x.items() if k!='exact_collisions'} for x in rows],indent=2))


if __name__=='__main__':main()
