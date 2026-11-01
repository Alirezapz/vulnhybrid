"""Build a minimal reproducible research release without academic manuscripts."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,zipfile

EXCLUDED_TREES=('paper/','thesis/','docs/legacy/','pipeline/data/',
    'results/2026-11-01-graphs/static_combined/')
EXCLUDED_FILES={'PROJECT_MANIFEST.json','docs/PAPER_DRAFT.md','docs/REVIEWER_CRITIQUE.md',
    'docs/RESULTS.md','docs/RESULTS_memory_lifecycle.md','docs/ROADMAP.md',
    'pipeline/debug_devign_snippets.py','results/2026-11-01/static/diagnostic_summary.json',
    'results/2026-11-01/source_inventory.json',
    'results/2026-11-01-graphs/provenance/source_inventory.json',
    'results/2026-11-01-graphs/baseline_interim_mcc.json'}
IGNORED_DIRS={'.git','build','dist','__pycache__','.pytest_cache','.venv','venv','release-assets'}
KEPT_LOGS={'full-graph-tests.log','graph-verification.log','verified-tests.log','verified-client-tests.log','taint-rule-fixtures.log'}


def include(relative):
    p=PurePosixPath(relative);name=p.name
    if any(x in IGNORED_DIRS or x.endswith('.egg-info') for x in p.parts):return False
    if relative in EXCLUDED_FILES or any(relative.startswith(x) for x in EXCLUDED_TREES):return False
    if p.suffix.lower() in {'.ipynb','.docx','.tex','.bib','.pyc','.pyo','.pt','.joblib','.bak','.tmp','.stdout','.stderr'}:return False
    if name.endswith('_embeddings.npz'):return False
    if p.suffix=='.log' and name not in KEPT_LOGS and 'attempt1' not in name:return False
    return True


def in_git(relative,size):
    if not relative.startswith('results/'):return True
    p=PurePosixPath(relative);name=p.name
    if name in {'cohort.csv','validation_partitions.csv','sample_index.csv','findings.jsonl','coverage.json'}:return False
    if name.endswith(('_predictions.csv','_scores.csv')):return False
    if size>1100000:return False
    return True


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1048576),b''):h.update(data)
    return h.hexdigest()


def manifest_entry(path,name):return {'path':name,'bytes':path.stat().st_size,'sha256':sha(path)}


def verify_zip(path):
    with zipfile.ZipFile(path) as archive:
        bad=archive.testzip()
        if bad:raise RuntimeError('Archive CRC failure: '+bad)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--project-root',type=Path,default=Path('.'));p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoints',type=Path,help='Optional original trusted checkpoint ZIP; only model weights and run metadata are selected.')
    a=p.parse_args();root=a.project_root.resolve();a.out.mkdir(parents=True,exist_ok=True)
    entries=[]
    for file in sorted(root.rglob('*')):
        if not file.is_file() or file.name=='PUBLICATION_MANIFEST.json':continue
        if file.resolve().is_relative_to(a.out.resolve()):continue
        relative=file.relative_to(root).as_posix()
        if include(relative):entries.append(dict(manifest_entry(file,relative),location='git' if in_git(relative,file.stat().st_size) else 'experiment-data-release'))
    evidence=a.out/'VulnHybrid_Experiment_Data_v1.2.0.zip'
    records=[r for r in entries if r['location']=='experiment-data-release']
    with zipfile.ZipFile(evidence,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for row in records:archive.write(root/row['path'],row['path'])
        archive.writestr('EXPERIMENT_DATA_MANIFEST.json',json.dumps({'files':records},indent=2))
    verify_zip(evidence)
    model_count=0
    if a.checkpoints:
        target=a.out/'VulnHybrid_Models_v1.2.0.zip';temporary=target.with_suffix('.zip.tmp');model_records=[]
        with zipfile.ZipFile(a.checkpoints) as original,zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for name in sorted(original.namelist()):
                relative=name.removeprefix('vulnhybrid-project/')
                if not relative.startswith('results/'):continue
                if PurePosixPath(name).name not in {'model.pt','model.joblib','run.json'}:continue
                raw=original.read(name)
                archive.writestr(relative,raw);model_records.append({'path':relative,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
                model_count+=PurePosixPath(name).name in {'model.pt','model.joblib'}
            if model_count!=110:raise RuntimeError('Expected all 78 neural and 32 classical models')
            archive.writestr('MODEL_MANIFEST.json',json.dumps({'models':model_count,'files':model_records,'graph_caches':'Rebuild with the pinned graph_data commands.'},indent=2))
        verify_zip(temporary);temporary.replace(target)
    manifest={'version':'1.2.0','distribution':'Code and compact measured evidence in Git; raw numerical records and reusable models in release assets.',
        'excluded':'Academic manuscripts/thesis, legacy notebooks and sample verdicts, duplicate diagnostic summaries, raw analyzer streams, redundant runtime/install logs and regenerable graph caches.',
        'git_files':sum(r['location']=='git' for r in entries),'experiment_data_files':len(records),'models':model_count,'files':entries}
    (root/'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))


if __name__=='__main__':main()
