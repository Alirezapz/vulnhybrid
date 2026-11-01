"""Meta fitting must be independent of all test/external distributions."""
import pytest
np=pytest.importorskip('numpy');pd=pytest.importorskip('pandas');pytest.importorskip('sklearn')
from research.graph_fusion import compose


def test_meta_parameters_do_not_fit_test_distribution():
    rng=np.random.default_rng(4);n=160
    names=['g_gcn_hetero_s42','k_ast_wl2','v1_token_lr','v1_char_lr','v1_token_svm_platt','v1_token_nb','v1_static_lr','v1_mean_token_char']
    validation=pd.DataFrame(rng.uniform(.15,.85,(n,len(names))),columns=names)
    labels=(validation.v1_token_lr.to_numpy()+rng.normal(0,.3,n)>.5).astype(int)
    fit=np.arange(n)<100
    frames={'validation':validation,'test':pd.DataFrame(rng.uniform(.1,.9,(20,len(names))),columns=names)}
    quality={'validation':np.zeros((n,2)),'test':np.zeros((20,2))}
    a,recipes_a=compose(frames,fit,labels,quality)
    altered=dict(frames,test=pd.DataFrame(np.full((20,len(names)),.999),columns=names))
    b,recipes_b=compose(altered,fit,labels,quality)
    assert recipes_a==recipes_b
    for name in a['validation']:
        assert np.allclose(a['validation'][name],b['validation'][name])
        assert np.isfinite(a['test'][name]).all()
        assert ((a['test'][name]>=0)&(a['test'][name]<=1)).all()
