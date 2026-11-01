"""Batch measured static engines. Preserve raw diagnostics and coverage separately.
Usage: python -m research.collect_static --samples ... --out results/.../static
"""
import argparse, csv, io, json, os, shutil, subprocess, tempfile, time
import xml.etree.ElementTree as ET
from pathlib import Path
from pipeline.benchmark_hygiene import sanitize_code


def run(samples, out):
    out.mkdir(parents=True, exist_ok=True)
    records = []; coverage = {}
    with tempfile.TemporaryDirectory(prefix='vulnhybrid_eval_') as temp:
        base = Path(temp); file_ids = {}
        for i, row in enumerate(samples):
            filename = f'u{i:06d}' + ('.cpp' if row['language'] == 'cpp' else '.c')
            file_ids[filename] = row['sample_id']
            (base / filename).write_text(sanitize_code(row['code'], row['dataset'] == 'juliet'))
        with (out/'sample_index.csv').open('w') as index_file:
            writer=csv.writer(index_file);writer.writerow(['filename','sample_id'])
            writer.writerows(file_ids.items())
        rules = Path(__file__).resolve().parents[1] / 'pipeline/static_analysis/rules'
        commands = {
            'semgrep': ['semgrep', '--config', str(rules), '--json', '--quiet', '--no-git-ignore',
                        '--metrics', 'off', '--jobs', '4', '--timeout', '10', '.'],
            'taint': ['semgrep', '--config', str(Path(__file__).parent.resolve() / 'rules/taint.yml'),
                      '--json', '--quiet', '--no-git-ignore', '--metrics', 'off', '--jobs', '4', '.'],
            'cppcheck': [os.environ.get('CPPCHECK_BIN', 'cppcheck'), '--enable=warning', '--inconclusive',
                         '--check-level=normal', '--xml', '--quiet', '-j4', '.'],
            'flawfinder': ['flawfinder', '--csv', '--quiet', '--minlevel=1', '.'],
        }
        for tool, cmd in commands.items():
            if not shutil.which(cmd[0]):
                raise RuntimeError(f'Required analyzer unavailable: {tool}')
            start = time.monotonic(); env = dict(os.environ, SEMGREP_ENABLE_VERSION_CHECK='0')
            p = subprocess.run(cmd, cwd=temp, capture_output=True, text=True, timeout=1200, env=env)
            (out / f'{tool}.stdout').write_text(p.stdout); (out / f'{tool}.stderr').write_text(p.stderr)
            diagnostic = []; scanned = None
            if tool in ('semgrep', 'taint'):
                d = json.loads(p.stdout)
                if p.returncode not in (0, 1) or 'results' not in d:
                    raise RuntimeError(f'{tool} failed with exit code {p.returncode}')
                diagnostic = d.get('errors', [])
                scanned = len(d.get('paths', {}).get('scanned', []))
                for r in d['results']:
                    sid = file_ids[Path(r['path']).name]
                    records.append(dict(sample_id=sid, tool=tool, rule_id=r['check_id'].split('.')[-1],
                        line=r['start']['line'], severity=r['extra']['severity'],
                        cwe=r['extra'].get('metadata', {}).get('cwe')))
            elif tool == 'cppcheck':
                if p.returncode: raise RuntimeError('cppcheck failed')
                root = ET.fromstring(p.stderr)
                for e in root.findall('./errors/error'):
                    if e.get('id') in ('syntaxError', 'internalError', 'cppcheckError', 'unknownMacro'):
                        diagnostic.append(ET.tostring(e, encoding='unicode')); continue
                    if e.get('severity') in ('information', 'style', 'performance', 'portability'): continue
                    loc = e.find('location')
                    if loc is None or Path(loc.get('file', '')).name not in file_ids: continue
                    records.append(dict(sample_id=file_ids[Path(loc.get('file')).name], tool=tool,
                        rule_id=e.get('id'), line=int(loc.get('line', '0')), severity=e.get('severity'),
                        cwe=f'CWE-{e.get("cwe")}' if e.get('cwe') else None))
            else:
                if p.returncode: raise RuntimeError('flawfinder failed')
                for r in csv.DictReader(io.StringIO(p.stdout)):
                    if Path(r['File']).name not in file_ids: continue
                    records.append(dict(sample_id=file_ids[Path(r['File']).name], tool=tool,
                        rule_id=r['Name'], line=int(r['Line']), severity=int(r['Level']), cwe=None))
            coverage[tool] = dict(requested=len(samples), explicitly_scanned=scanned, errors=diagnostic,
                duration_s=time.monotonic()-start, command=cmd,
                version=subprocess.run([cmd[0], '--version'], capture_output=True, text=True).stdout.strip())
            print(tool, 'findings', sum(r['tool'] == tool for r in records), 'diagnostics',len(diagnostic), flush=True)
    (out / 'findings.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    (out / 'coverage.json').write_text(json.dumps(coverage, indent=2))


def main():
    p = argparse.ArgumentParser(__doc__);p.add_argument('--samples', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True); args=p.parse_args()
    rows = [json.loads(line) for path in args.samples for line in path.read_text().splitlines() if line.strip()]
    if len({r['sample_id'] for r in rows}) != len(rows): raise ValueError('Duplicate sample IDs')
    run(rows, args.out)


if __name__ == '__main__': main()
