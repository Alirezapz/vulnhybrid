"""Bounded scheduler: every run has an immutable registry row and explicit status."""
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from research.graph_models import ARCHITECTURES


def tasks(root,graphs,cohorts,samples,stage='baseline'):
    result=[]
    if stage=='baseline':
        for regime in ['commit_disjoint','official_dedup','ffmpeg_to_qemu','qemu_to_ffmpeg']:
            for arch in ARCHITECTURES:
                for seed in ([42,43,44] if regime=='commit_disjoint' else [42]):
                    result.append(dict(protocol=regime,architecture=arch,seed=seed,view='hetero',anonymous=False,graphs=str(graphs)))
    else:
        for view in ['ast','cfg','ddg','sequence']:
            result.append(dict(protocol='commit_disjoint',architecture='sage',seed=42,view=view,anonymous=False,graphs=str(graphs)))
        for arch in ['sage','deepsets']:
            result.append(dict(protocol='commit_disjoint',architecture=arch,seed=42,view='hetero',anonymous=True,graphs=str(graphs)))
    for r in result:
        name=r['architecture']+'_'+r['view']+('_anonymous' if r['anonymous'] else '')+'_s'+str(r['seed'])
        r['run_id']=r['protocol']+'/'+name;r['out']=str(root/r['protocol']/name)
        r['command']=[sys.executable,'-m','research.graph_train','--graphs',r['graphs'],'--cohort',str(cohorts/r['protocol']),
           '--samples',*[str(p) for p in samples],'--out',r['out'],'--architecture',r['architecture'],
           '--view',r['view'],'--seed',str(r['seed']),'--threads','2']+(['--anonymous'] if r['anonymous'] else [])
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--graphs',type=Path,required=True)
    p.add_argument('--cohorts',type=Path,default=Path('results/2026-11-01'));p.add_argument('--samples',type=Path,nargs='+',required=True)
    p.add_argument('--stage',choices=['baseline','ablation'],default='baseline');p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    registry=tasks(a.root,a.graphs,a.cohorts,a.samples,a.stage)
    path=a.root/(a.stage+'_registry.json');path.write_text(json.dumps(registry,indent=2));start=time.monotonic()
    def one(row):
        out=Path(row['out']);out.mkdir(parents=True,exist_ok=True)
        if (out/'run.json').exists() and (out/'model.pt').exists():return dict(run_id=row['run_id'],status='existing')
        env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
        with (out/'run.log').open('w') as log:
            proc=subprocess.run(row['command'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        return dict(run_id=row['run_id'],status='completed' if proc.returncode==0 else 'failed',exit_code=proc.returncode)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(one,row):row for row in registry}
        for future in concurrent.futures.as_completed(futures):
            try:r=future.result()
            except Exception as exc:r=dict(run_id=futures[future]['run_id'],status='failed',error=str(exc))
            results.append(r);print('MATRIX',len(results),'/',len(registry),r,'seconds',round(time.monotonic()-start),flush=True)
            (a.root/(a.stage+'_status.json')).write_text(json.dumps(results,indent=2))
    if any(r['status']=='failed' for r in results):raise SystemExit(1)


if __name__=='__main__':main()
