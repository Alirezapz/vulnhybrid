"""Verify ordered upstream paired-test records and exclude whole overlapping pairs.
The supplied paired file is a pinned public mirror; its provenance is disclosed.
"""
from pathlib import Path
import argparse,collections,hashlib,json
import re
from pipeline.benchmark_hygiene import code_hash,sanitize_code
from research.benchmark import load


def prepare(raw,reference_files,out):
    records=load(raw)
    if len(records)%2:raise ValueError('Odd number of upstream paired records')
    reference={code_hash(r['code']) for p in reference_files for r in load(p)}
    rows=[];excluded=[];seen=set()
    for i in range(0,len(records),2):
        a,b=records[i:i+2]
        if {int(a['target']),int(b['target'])}!={0,1}:raise ValueError(f'Unverified labels in pair {i//2}')
        mismatch=next((key for key in ['project','commit_id'] if a[key]!=b[key]),None)
        if mismatch:
            excluded.append(dict(upstream_pair=i//2,ids=[a['idx'],b['idx']],reason='pair_metadata_mismatch_'+mismatch));continue
        known=[r['file_name'] for r in [a,b] if r.get('file_name') not in [None,'None','']]
        if len(set(known))>1:
            excluded.append(dict(upstream_pair=i//2,ids=[a['idx'],b['idx']],reason='pair_filename_mismatch'));continue
        if len(known)<2:
            sig=lambda r:re.sub(r'\s+','',sanitize_code(r['func']).split('{',1)[0])
            if not known or sig(a)!=sig(b):
                excluded.append(dict(upstream_pair=i//2,ids=[a['idx'],b['idx']],reason='incomplete_pair_metadata'));continue
        hashes=[code_hash(r['func']) for r in [a,b]]
        reason='reference_overlap' if any(h in reference for h in hashes) else 'identical_versions' if hashes[0]==hashes[1] else 'repeated_pair' if any(h in seen for h in hashes) else None
        if reason:excluded.append(dict(upstream_pair=i//2,ids=[a['idx'],b['idx']],reason=reason));continue
        seen.update(hashes);pair=f'primevul_pair_{a["idx"]}_{b["idx"]}'
        for r in [a,b]:
            ext=Path(known[0]).suffix.lower()
            rows.append(dict(sample_id=f'primevul_{r["idx"]}',dataset='primevul',language='cpp' if ext in ['.cc','.cpp','.cxx','.hpp'] else 'c',
               code=r['func'],label=int(r['target']),split='external',cwe=r.get('cwe'),
               extra=dict(pair_id=pair,project=r['project'],commit_id=r['commit_id'],file_name=r['file_name'])))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest=dict(source_file=str(raw),source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),upstream_rows=len(records),
        retained_rows=len(rows),retained_pairs=len(rows)//2,excluded=excluded,
        mirror='starsofchance/PrimeVul',mirror_revision='a30276e1bbec449c1c932f6a664b4c06942154f1',
        pairing='consecutive mirror records; opposite labels and same project/commit; matching known file; if a filename is missing require identical signature',
        reference_files=[str(p) for p in reference_files])
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k!='excluded'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--reference',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();prepare(a.raw,a.reference,a.out)
