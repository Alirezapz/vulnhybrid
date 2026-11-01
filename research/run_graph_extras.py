"""Run the predeclared graph ablations and experimental representations."""
from pathlib import Path
import argparse,json,subprocess,os,sys,concurrent.futures,time


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--cohorts',type=Path,default=Path('results/2026-11-01'));p.add_argument('--samples',type=Path,nargs='+',required=True)
    p.add_argument('--workers',type=int,default=3);a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    baseline=a.root.parent/'neural'
    expected=len(json.loads((baseline/'baseline_registry.json').read_text()))
    deadline=time.monotonic()+7200
    while True:
        status=json.loads((baseline/'baseline_status.json').read_text()) if (baseline/'baseline_status.json').exists() else []
        if any(r['status']=='failed' for r in status):raise RuntimeError('Resolve baseline failures before experimental training')
        if len(status)==expected:break
        if time.monotonic()>deadline:raise TimeoutError('Baseline dependency did not finish')
        time.sleep(5)
    recipes=[('sage_ast','sage','ast','graph_cache',False),('sage_cfg','sage','cfg','graph_cache',False),
      ('sage_last_ident','sage','ddg','graph_cache',False),('sage_sequence','sage','sequence','graph_cache',False),
      ('sage_anonymous','sage','hetero','graph_cache',True),('deepsets_anonymous','deepsets','hetero','graph_cache',True),
      ('sage_evidence','sage','hetero','graph_cache_evidence',False),
      ('sage_semantic_evidence','sage','hetero','graph_cache_semantic',False),
      ('sage_semantic_bfs','sage','hetero','graph_cache_semantic_bfs',False),
      ('sage_cap512','sage','hetero','graph_cache_512',False),
      ('relation_gate','relation_gate','hetero','graph_cache',False),
      ('relation_uniform','relation_uniform','hetero','graph_cache',False),
      ('ggnn_invariant','ggnn_invariant','hetero','graph_cache',False),
      ('devign_relational_port','devign_relational_port','hetero','graph_cache',False),
      ('ggnn_sum_port','ggnn_sum_port','hetero','graph_cache',False)]
    registry=[]
    for protocol in ['commit_disjoint','official_dedup','ffmpeg_to_qemu','qemu_to_ffmpeg']:
        selected=recipes if protocol=='commit_disjoint' else [r for r in recipes if r[0] in ['sage_semantic_evidence','relation_gate','ggnn_invariant']]
        for name,arch,view,cache,anonymous in selected:
            out=a.root/protocol/name
            command=[sys.executable,'-m','research.graph_train','--graphs',str(a.cache_root/cache),'--cohort',str(a.cohorts/protocol),
                '--samples',*[str(x) for x in a.samples],'--out',str(out),'--architecture',arch,'--view',view,'--threads','2']+(['--anonymous'] if anonymous else [])
            registry.append(dict(protocol=protocol,name=name,command=command,out=str(out)))
    (a.root/'registry.json').write_text(json.dumps(registry,indent=2));start=time.monotonic()
    def one(r):
        out=Path(r['out']);out.mkdir(parents=True,exist_ok=True)
        if (out/'run.json').exists() and (out/'model.pt').exists():return dict(protocol=r['protocol'],name=r['name'],status='existing')
        with (out/'run.log').open('w') as f:
            p=subprocess.run(r['command'],stdout=f,stderr=subprocess.STDOUT,timeout=1800,
                env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2'))
        return dict(protocol=r['protocol'],name=r['name'],status='completed' if p.returncode==0 else 'failed',returncode=p.returncode)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(one,r):r for r in registry}
        for future in concurrent.futures.as_completed(futures):
            try:r=future.result()
            except Exception as e:r=dict(name=futures[future]['name'],protocol=futures[future]['protocol'],status='failed',error=str(e))
            results.append(r);(a.root/'status.json').write_text(json.dumps(results,indent=2));print('EXTRA',len(results),'/',len(registry),r,'seconds',round(time.monotonic()-start),flush=True)
    if any(r['status']=='failed' for r in results):raise SystemExit(1)


if __name__=='__main__':main()
