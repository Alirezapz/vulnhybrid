"""Independent recheck of exported predictions, cohorts and headline metrics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


def verify(root=Path('results/2026-11-01')):
    checks=[]
    for folder in sorted(root.iterdir()):
        if not (folder/'metrics.json').exists():continue
        d=json.loads((folder/'metrics.json').read_text())
        cohort=pd.read_csv(folder/'cohort.csv')
        assert cohort.sample_id.is_unique and cohort.code_sha256.is_unique
        validation=pd.read_csv(folder/'validation_predictions.csv')
        partition=pd.read_csv(folder/'validation_partitions.csv').set_index('sample_id')
        fit=partition[partition.partition=='val_fit'].commit_id
        select=partition[partition.partition=='val_select'].commit_id
        assert not (set(fit)&set(select))
        for filename,key in [('test_predictions.csv','metrics'),('external_predictions.csv','external_metrics')]:
            df=pd.read_csv(folder/filename);assert df.sample_id.is_unique
            for m in d[key]:
                name=m['method'];score=df[name].to_numpy();y=df.label.to_numpy()
                assert np.isfinite(score).all() and (score>=0).all() and (score<=1).all()
                pred=score>=d['thresholds'][name]
                assert len(y)==m['n']
                for metric,value in [('accuracy',accuracy_score(y,pred)),('f1',f1_score(y,pred,zero_division=0)),('mcc',matthews_corrcoef(y,pred))]:
                    assert abs(m[metric]-value)<1e-10,(folder,name,metric)
                if key=='metrics':
                    vf=validation.set_index('sample_id').loc[partition[partition.partition=='val_select'].index]
                    assert abs(f1_score(vf.label,vf[name]>=d['thresholds'][name],zero_division=0)-m['validation_f1'])<1e-10
                else:
                    pairs=df[['pair_id','label']].copy();pairs['prediction']=pred
                    correct=pairs.groupby('pair_id').apply(lambda g: bool((g.label==g.prediction).all()),include_groups=False)
                    assert abs(correct.mean()-m['both_correct'])<1e-10
        if d['commit_disjoint']:
            assert d['clean_audit']['cross_split_commit_groups']==0
            assert d['clean_audit']['cross_split_clone_groups']==0
        expected=max(d['metrics'],key=lambda m:(m['validation_f1'],m['method']))['method']
        assert expected==d['selected_on_validation']
        checks.append(dict(protocol=folder.name,methods=len(d['metrics']),test_n=len(pd.read_csv(folder/'test_predictions.csv')),status='verified'))
    print(json.dumps(checks,indent=2))
    return checks


if __name__=='__main__':verify()
