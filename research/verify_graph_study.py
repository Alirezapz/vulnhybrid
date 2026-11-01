"""Independent checks of recorded scores, selections, membership and metrics."""
from pathlib import Path
import hashlib,json,numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef
from research.graph_evaluate import threshold


def main():
    root=Path('results/2026-11-01-graphs');checks=[]
    for status,expected in [(root/'neural/baseline_status.json',54),(root/'experimental/status.json',24)]:
        rows=json.loads(status.read_text());assert len(rows)==expected and all(r['status'] in ['completed','existing'] for r in rows)
    for category,expected in [('neural',54),('experimental',24)]:
        paths=list((root/category).glob('*/*/run.json'));assert len(paths)==expected
        for file in paths:
            run=json.loads(file.read_text());protocol=file.parent.parent.name
            cohort=pd.read_csv(Path('results/2026-11-01')/protocol/'cohort.csv')
            ids=cohort.loc[cohort.split=='train','sample_id'].tolist()
            assert run['train_n']==len(ids)
            assert run['train_id_sha256']==hashlib.sha256('\n'.join(ids).encode()).hexdigest()
            assert run['status']=='completed' and run['epochs_run']>=3
            assert (file.parent/'model.pt').exists()
    for folder in (root/'comparisons').iterdir():
        if not (folder/'metrics.json').exists():continue
        meta=json.loads((folder/'metrics.json').read_text());test=pd.read_csv(folder/'test_predictions.csv');val=pd.read_csv(folder/'validation_predictions.csv').set_index('sample_id')
        part=pd.read_csv(Path('results/2026-11-01')/folder.name/'validation_partitions.csv').set_index('sample_id');sel=part.index[part.partition=='val_select'];val=val.loc[sel]
        assert test.sample_id.is_unique
        for row in meta['metrics']:
            name=row['method'];score=test[name].to_numpy();y=test.label.to_numpy();pred=score>=row['threshold']
            assert len(y)==row['n'] and np.isfinite(score).all()
            for field,fn in [('accuracy',accuracy_score),('f1',f1_score),('mcc',matthews_corrcoef)]:assert abs(row[field]-fn(y,pred))<1e-10,(folder,name,field)
            t,_=threshold(val.label.to_numpy(),val[name].to_numpy(),row['policy']);assert t==row['threshold']
        for phase in ['external','primevul']:
            df=pd.read_csv(folder/(phase+'_predictions.csv'))
            for row in [r for r in meta['external_metrics'] if r['phase']==phase]:
                pred=df[row['method']]>=row['threshold'];correct=(pred==df.label).groupby(df.pair_id).all().mean()
                assert abs(correct-row['both_correct'])<1e-10
        chosen=json.loads((folder/'validation_selection.json').read_text())
        for policy,name in chosen['selected'].items():
            eligible=[r for r in chosen['rows'] if r['policy']==policy]
            assert max(eligible,key=lambda r:(r['validation_score'],r['method']))['method']==name
        checks.append(dict(protocol=folder.name,methods=meta['n_methods'],test_n=len(test),status='verified'))
    assert len(checks)==4
    (root/'independent_verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
