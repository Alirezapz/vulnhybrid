"""Download the fixed external mirror file; verify exact bytes before use."""
from pathlib import Path
import argparse,urllib.request,hashlib,json
REVISION='a30276e1bbec449c1c932f6a664b4c06942154f1'
SHA256='384758c25142b640f1768bb0d334c43f315c580022ae94973c66e77128fb3821'


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,default=Path('data/research/primevul'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    url=f'https://huggingface.co/datasets/starsofchance/PrimeVul/resolve/{REVISION}/primevul_test_paired.jsonl'
    data=urllib.request.urlopen(url,timeout=90).read()
    if hashlib.sha256(data).hexdigest()!=SHA256:raise ValueError('External dataset byte hash differs')
    (a.out/'primevul_test_paired.jsonl').write_bytes(data)
    (a.out/'download_manifest.json').write_text(json.dumps(dict(url=url,revision=REVISION,sha256=SHA256,bytes=len(data)),indent=2))


if __name__=='__main__':main()
