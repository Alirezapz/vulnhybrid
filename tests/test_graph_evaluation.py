import pytest
np=pytest.importorskip('numpy');pytest.importorskip('sklearn')
from research.graph_evaluate import threshold,clustered
from sklearn.metrics import matthews_corrcoef,f1_score


def test_threshold_optimization_matches_independent_metric_search():
    y=np.array([0,1,0,1,1,0]);p=np.array([.1,.4,.8,.9,.6,.3])
    for objective,metric in [('mcc',matthews_corrcoef),('f1',f1_score)]:
        selected,value=threshold(y,p,objective)
        expected=max(metric(y,p>=t) for t in np.r_[np.linspace(0,1,101),1.000001])
        assert abs(value-expected)<1e-12
        assert abs(metric(y,p>=selected)-expected)<1e-12


def test_clustered_bootstrap_uses_shared_resamples_for_identical_models():
    y=np.array([1,0,1,0]);pred=np.array([True,True,False,False])
    names,samples=clustered(y,{'a':pred,'b':pred},['x','x','y','y'],n=50)
    assert samples.shape==(50,2,3)
    assert np.array_equal(samples[:,0],samples[:,1])
    assert np.isfinite(samples).all()
