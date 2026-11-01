"""Train/validate graphs on the exact saved cohorts; labels never enter graphs.
Test scores are exported without inspecting test metrics during fitting.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,random,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import average_precision_score,roc_auc_score
from research.graph_data import GraphStore
from research.graph_models import GraphClassifier,ARCHITECTURES,EXPERIMENTAL,supervised_contrastive
from research.benchmark import load


def data_for(store,sid,label,view='hetero',anonymous=False):
    g=store.get(sid,view,anonymous)
    edges=g['edges'].astype(np.int64)
    edge_index=torch.from_numpy(edges[:,:2].T.copy()) if len(edges) else torch.empty((2,0),dtype=torch.long)
    return Data(type_ids=torch.from_numpy(g['types'].astype(np.int64)),
        token_ids=torch.from_numpy(g['tokens'].astype(np.int64)),x_num=torch.from_numpy(g['numeric'].copy()),
        edge_index=edge_index,edge_type=torch.from_numpy(edges[:,2].copy()),
        y=torch.tensor([label],dtype=torch.float32),sid=sid,num_nodes=len(g['types']))


def predict(model,loader):
    model.eval();scores=[];emb=[];ids=[]
    with torch.no_grad():
        for batch in loader:
            logits,z=model(batch);scores.extend(torch.sigmoid(logits).tolist());emb.extend(z.numpy());ids.extend(batch.sid)
    return np.asarray(scores),np.asarray(emb),ids


def train(args):
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    out=args.out;out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    store=GraphStore(args.graphs)
    cohort=pd.read_csv(args.cohort/'cohort.csv');tr=cohort[cohort.split=='train'].sample_id.tolist()
    vdf=pd.read_csv(args.cohort/'validation_predictions.csv');tdf=pd.read_csv(args.cohort/'test_predictions.csv');edf=pd.read_csv(args.cohort/'external_predictions.csv')
    labels={r['sample_id']:r['label'] for f in args.samples for r in load(f)}
    if args.train_limit:tr=sorted(tr,key=lambda s:hashlib.sha256(s.encode()).hexdigest())[:args.train_limit]
    datasets={
        'train':[data_for(store,s,labels[s],args.view,args.anonymous) for s in tr],
        'validation':[data_for(store,s,labels[s],args.view,args.anonymous) for s in vdf.sample_id],
        'test':[data_for(store,s,labels[s],args.view,args.anonymous) for s in tdf.sample_id],
        'external':[data_for(store,s,labels[s],args.view,args.anonymous) for s in edf.sample_id],
    }
    partition=pd.read_csv(args.cohort/'validation_partitions.csv').set_index('sample_id')
    fit_ix=np.array([partition.loc[s,'partition']=='val_fit' for s in vdf.sample_id])
    yv=vdf.label.to_numpy()
    generator=torch.Generator().manual_seed(args.seed)
    loaders={key:DataLoader(value,batch_size=args.batch,shuffle=(key=='train'),num_workers=0,
        generator=generator if key=='train' else None) for key,value in datasets.items()}
    model=GraphClassifier(args.architecture,args.hidden,args.dropout)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    best=-float('inf');best_epoch=0;stale=0;history=[];state=None
    for epoch in range(1,args.epochs+1):
        model.train();total=0.;count=0;t=time.monotonic()
        for batch in loaders['train']:
            optimizer.zero_grad(set_to_none=True);logits,z=model(batch)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,batch.y)
            if args.architecture=='ggnn_supcon':loss=loss+.1*supervised_contrastive(z,batch.y)
            if args.architecture=='ggnn_invariant':
                aug=batch.clone();aug.token_ids=torch.where(torch.rand_like(aug.token_ids,dtype=torch.float32)<.3,0,aug.token_ids)
                keep=torch.rand(len(aug.edge_type))>.2;aug.edge_index=aug.edge_index[:,keep];aug.edge_type=aug.edge_type[keep]
                aug_logits,_=model(aug)
                loss=.5*(loss+torch.nn.functional.binary_cross_entropy_with_logits(aug_logits,batch.y))+.1*torch.nn.functional.mse_loss(torch.sigmoid(logits),torch.sigmoid(aug_logits))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite neural loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step()
            total+=float(loss.detach())*batch.num_graphs;count+=batch.num_graphs
        p,_,_=predict(model,loaders['validation'])
        ap=float(average_precision_score(yv[fit_ix],p[fit_ix]));auc=float(roc_auc_score(yv[fit_ix],p[fit_ix]))
        row=dict(epoch=epoch,loss=total/count,val_fit_ap=ap,val_fit_auc=auc,seconds=time.monotonic()-t)
        history.append(row);print(args.architecture,args.view,args.seed,row,flush=True)
        if ap>best+1e-4:best=ap;best_epoch=epoch;state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        (out/'training_history.json').write_text(json.dumps(history,indent=2))
        if epoch>=3 and stale>=args.patience:break
    model.load_state_dict(state)
    torch.save(dict(state_dict=state,architecture=args.architecture,hidden=args.hidden,dropout=args.dropout,
        view=args.view,anonymous=args.anonymous,seed=args.seed,best_epoch=best_epoch),out/'model.pt')
    for key,df in [('validation',vdf),('test',tdf),('external',edf)]:
        p,emb,ids=predict(model,loaders[key]);assert ids==df.sample_id.tolist()
        pd.DataFrame(dict(sample_id=ids,probability=p)).to_csv(out/f'{key}_scores.csv',index=False)
        # Hidden representations enable learned graph/text fusion without re-training the encoder.
        np.savez_compressed(out/f'{key}_embeddings.npz',embedding=emb,sample_id=np.asarray(ids))
    meta=dict(status='completed',architecture=args.architecture,view=args.view,anonymous=args.anonymous,
        seed=args.seed,train_n=len(tr),hidden=args.hidden,best_epoch=best_epoch,val_fit_ap=best,
        parameters=sum(p.numel() for p in model.parameters()),epochs_run=len(history),seconds=time.monotonic()-start,
        train_id_sha256=hashlib.sha256('\n'.join(tr).encode()).hexdigest(),
        graph_cache_sha256=json.loads((args.graphs/'graph_summary.json').read_text())['cache_sha256'],
        versions=dict(torch=torch.__version__,pyg=__import__('torch_geometric').__version__),history=history)
    (out/'run.json').write_text(json.dumps(meta,indent=2));print('FINISHED',json.dumps({k:v for k,v in meta.items() if k!='history'}),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--graphs',type=Path,required=True);p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--samples',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--architecture',choices=ARCHITECTURES+EXPERIMENTAL,default='gcn');p.add_argument('--view',choices=['hetero','ast','cfg','ddg','sequence','none'],default='hetero')
    p.add_argument('--anonymous',action='store_true');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--hidden',type=int,default=48);p.add_argument('--dropout',type=float,default=.2);p.add_argument('--batch',type=int,default=128)
    p.add_argument('--epochs',type=int,default=8);p.add_argument('--patience',type=int,default=2);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--train-limit',type=int)
    args=p.parse_args()
    try:train(args)
    except Exception as exc:
        args.out.mkdir(parents=True,exist_ok=True)
        (args.out/'failure.json').write_text(json.dumps(dict(status='failed',error=str(exc),traceback=traceback.format_exc()),indent=2));raise


if __name__=='__main__':main()
