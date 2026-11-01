"""Small adversarial protocol checks, independent of real benchmark outcomes."""
import pytest
pytest.importorskip('sklearn')
from research.benchmark import deduplicate, select_threshold, pair_metrics
import numpy as np


def row(sid,split,code,label):
    return dict(sample_id=sid,split=split,code=code,label=label)


def test_test_labels_do_not_change_training_cohort():
    rows=[row('a','train','int f(){return 0;}',0),row('b','test','int f(){return 0;}',1)]
    kept,excluded=deduplicate(rows)
    assert [r['sample_id'] for r in kept]==['a']
    rows[-1]['label']=0
    assert deduplicate(rows)[0]==kept
    assert excluded[0]['reason']=='earlier_partition_clone'


def test_conflicting_training_duplicates_are_excluded():
    assert deduplicate([row('a','train','x',0),row('b','train','x',1)])[0]==[]


def test_threshold_selection_can_choose_all_positive():
    threshold,score=select_threshold(np.array([1,1,0]),np.array([.1,.1,.1]))
    assert threshold<=.1 and score==.8


def test_pair_success_requires_both_versions():
    rows=[{'label':1,'extra':{'pair_id':'p'}},{'label':0,'extra':{'pair_id':'p'}}]
    assert pair_metrics(rows,np.array([.9,.8]),.5)['both_correct']==0.
    assert pair_metrics(rows,np.array([.9,.1]),.5)['both_correct']==1.
    assert pair_metrics(rows,np.array([.5,.5]),.5)['tied']==1.
