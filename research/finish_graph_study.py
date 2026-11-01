"""Finish dependency-ordered prediction and evaluation after all training runs."""
from pathlib import Path
import argparse,json,sys,time,subprocess,os


def wait_status(path,expected):
    deadline=time.monotonic()+7200
    while True:
        rows=json.loads(path.read_text()) if path.exists() else []
        if any(r['status']=='failed' for r in rows):raise RuntimeError('Training failure recorded: '+str(path))
        if len(rows)==expected:return
        if time.monotonic()>deadline:raise TimeoutError(str(path))
        time.sleep(5)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--prime',type=Path,required=True);a=p.parse_args()
    wait_status(a.root/'neural/baseline_status.json',54);wait_status(a.root/'experimental/status.json',24)
    if not (a.root/'extended_baselines/reproduction_check.json').exists():raise RuntimeError('Old baseline extension not verified')
    # Baseline PrimeVul export may already have run concurrently. Reuse only completed score files.
    for category in ['neural','experimental']:
        subprocess.run([sys.executable,'-m','research.graph_predict','--runs',str(a.root/category),'--cache-root',str(a.cache_root),'--samples',str(a.prime)],check=True)
    subprocess.run([sys.executable,'-m','research.graph_evaluate','--root',str(a.root),'--cache-root',str(a.cache_root),'--prime',str(a.prime)],check=True,
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))


if __name__=='__main__':main()
