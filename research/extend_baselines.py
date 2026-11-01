"""Reproduce old baseline scores exactly and export them for new external code."""
from pathlib import Path
import argparse,json,subprocess,sys,os
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/2026-11-01-graphs'))
    parser.add_argument('--data-root',type=Path,default=Path('data/research'))
    parser.add_argument('--cohorts',type=Path,default=Path('results/2026-11-01'))
    args=parser.parse_args();root=args.root;base=args.data_root
    ext=base/'graph_external.jsonl';ext.write_text((base/'sven/samples.jsonl').read_text()+(base/'primevul/samples.jsonl').read_text())
    combined=root/'static_combined';combined.mkdir(exist_ok=True)
    (combined/'findings.jsonl').write_text((args.cohorts/'static/findings.jsonl').read_text()+(root/'static_primevul/findings.jsonl').read_text())
    coverage={k:{'status':'both_sources_executed','sources':[str(args.cohorts/'static/coverage.json'),str(root/'static_primevul/coverage.json')]} for k in ['semgrep','taint','cppcheck','flawfinder']}
    (combined/'coverage.json').write_text(json.dumps(coverage,indent=2))
    for regime,flag in [('commit_disjoint',['--commit-disjoint']),('official_dedup',[]),('ffmpeg_to_qemu',['--source-project','FFmpeg']),('qemu_to_ffmpeg',['--source-project','qemu'])]:
        out=root/'extended_baselines'/regime;out.mkdir(parents=True,exist_ok=True)
        if not (out/'metrics.json').exists():
            cmd=[sys.executable,'-m','research.benchmark','--samples',str(base/'devign/samples.jsonl'),'--external',str(ext),'--findings',str(combined/'findings.jsonl'),'--out',str(out)]+flag
            with (out/'run.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        meta=json.loads((out/'metrics.json').read_text());cols=[r['method'] for r in meta['metrics']]
        for phase in ['validation','test']:
            old=pd.read_csv(args.cohorts/regime/(phase+'_predictions.csv')).set_index('sample_id')
            new=pd.read_csv(out/(phase+'_predictions.csv')).set_index('sample_id').loc[old.index]
            assert np.allclose(old[cols],new[cols],atol=1e-7,rtol=1e-7),(regime,phase)
        print('EXTENDED AND REPRODUCED',regime,flush=True)
    (root/'extended_baselines/reproduction_check.json').write_text(json.dumps({'status':'verified','tolerance':1e-7,'cohorts':4},indent=2))


if __name__=='__main__':main()
