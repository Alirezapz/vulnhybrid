"""Fetch pinned upstream datasets and record byte hashes; no model calls.
Usage: python -m research.fetch_data --out data/research
"""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, urllib.request

DEVIGN_REV = '69bd48c03223c2104342acd9a807caf61ac3efb8'
SVEN_REV = 'e8449a0d54a5ae0bbdc6bb236330b8c7ec71df57'


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=90) as response:
        data = response.read()
    path.write_bytes(data)
    return dict(url=url, path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def convert_devign(root):
    import pandas as pd
    rows = []
    for split in ('train', 'validation', 'test'):
        for i, row in pd.read_parquet(root / f'{split}.parquet').iterrows():
            d = row.to_dict(); ours = 'val' if split == 'validation' else split
            rows.append(dict(sample_id=f'devign_{ours}_{d["id"]}', dataset='devign',
                language='c', code=d['func'], label=int(d['target']), split=ours,
                extra={k: str(v) for k, v in d.items() if k not in ('func', 'target')}))
    write_jsonl(rows, root / 'samples.jsonl')
    return rows


def convert_sven(root):
    rows = []
    for p in sorted(root.glob('data_train_val/*/*.jsonl')):
        for i, line in enumerate(p.read_text().splitlines()):
            d = json.loads(line)
            # Use upstream filename, never guess language from function text.
            ext = Path(d['file_name']).suffix.lower()
            if ext not in ('.c', '.h', '.cc', '.cpp', '.cxx', '.hpp'):
                continue
            pair = f'sven_{p.parent.name}_{p.stem}_{i}'
            for label, field in ((1, 'func_src_before'), (0, 'func_src_after')):
                rows.append(dict(sample_id=f'{pair}_{label}', dataset='sven',
                    language='c' if ext in ('.c', '.h') else 'cpp', code=d[field],
                    label=label, split='external', cwe=p.stem.upper(),
                    extra=dict(pair_id=pair, commit_id=d['commit_link'],
                               source_file=d['file_name'], upstream_split=p.parent.name)))
    write_jsonl(rows, root / 'samples.jsonl')
    return rows


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('--out', type=Path, default=Path('data/research'))
    args = p.parse_args(); records = []
    for split in ('train', 'validation', 'test'):
        url = f'https://huggingface.co/datasets/google/code_x_glue_cc_defect_detection/resolve/{DEVIGN_REV}/data/{split}-00000-of-00001.parquet'
        records.append(fetch(url, args.out / 'devign' / f'{split}.parquet'))
    for part in ('train', 'val'):
        for cwe in ('022', '078', '079', '089', '125', '190', '416', '476', '787'):
            rel = f'data_train_val/{part}/cwe-{cwe}.jsonl'
            records.append(fetch(f'https://raw.githubusercontent.com/eth-sri/sven/{SVEN_REV}/{rel}', args.out / 'sven' / rel))
    devign = convert_devign(args.out / 'devign'); sven = convert_sven(args.out / 'sven')
    manifest = dict(devign_revision=DEVIGN_REV, sven_revision=SVEN_REV,
                    devign_n=len(devign), sven_n=len(sven), files=records)
    (args.out / 'download_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k != 'files'}, indent=2))


if __name__ == '__main__': main()
