"""Behavioral graph/control-flow/learning invariants; no benchmark target values."""
import pytest
np=pytest.importorskip('numpy')
torch=pytest.importorskip('torch')
pytest.importorskip('torch_geometric')
from torch_geometric.data import Batch
from types import SimpleNamespace
from research.graph_data import Extractor,RELATIONS
from research.graph_train import data_for
from research.graph_models import GraphClassifier,ARCHITECTURES,EXPERIMENTAL
from research.graph_kernels import wl

CODE='int f(int n){int x=0;if(n>0){x=n;}else{x=1;}return x;}'


def batch(code=CODE):
    g=Extractor().extract(code);store=SimpleNamespace(get=lambda *a,**k:g)
    return Batch.from_data_list([data_for(store,'one',0),data_for(store,'two',1)])


def test_graph_has_no_labels_and_valid_typed_edges():
    g=Extractor().extract(CODE)
    assert 'label' not in g and 'sample_id' not in g
    assert g['edges'][:,:2].max()<len(g['types'])
    edges=set(map(tuple,g['edges']))
    for a,b,r in edges:
        assert (b,a,r+1 if r%2==0 else r-1) in edges
    assert g['meta']['parse_error'] is False


def test_return_does_not_flow_into_unreachable_statement():
    g=Extractor().extract('int f(){return 0; int x=1; x=2;}')
    ret=g['node_names'].index('return_statement');assignment=g['node_names'].index('expression_statement')
    cfg={(int(a),int(b)) for a,b,r in g['edges'] if r==6}
    assert (ret,len(g['types'])-1) in cfg
    assert (ret,assignment) not in cfg


def test_unmodeled_goto_and_bad_parse_are_diagnosed():
    g=Extractor().extract('int f(){goto end; end:return 0;}')
    assert g['meta']['unsupported_cfg']['goto_statement']==1
    assert Extractor().extract('int f( {')['meta']['parse_error'] is True


def test_evidence_selection_preserves_late_sensitive_api_under_same_cap():
    code='void f(int *p){'+''.join(f'int variable_{i}={i};' for i in range(100))+'free(p);}'
    ordinary=Extractor(32).extract(code);focused=Extractor(32,'evidence').extract(code)
    assert len(ordinary['types'])==len(focused['types'])==32
    assert ordinary['numeric'][:,2].sum()==0
    assert focused['numeric'][:,2].sum()>0


def test_semantic_graph_preserves_operator_fix_information():
    a='int f(int x){if(x>8)return 0; return 1;}'
    b='int f(int x){if(x>=8)return 0; return 1;}'
    ga=Extractor(256,semantic=True).extract(a);gb=Extractor(256,semantic=True).extract(b)
    assert not np.array_equal(ga['tokens'],gb['tokens'])


@pytest.mark.parametrize('architecture',ARCHITECTURES+EXPERIMENTAL)
def test_all_models_train_with_nonzero_gradient(architecture):
    torch.set_num_threads(1);torch.manual_seed(3)
    data=batch();model=GraphClassifier(architecture,hidden=16,dropout=0)
    logits,embedding=model(data)
    assert logits.shape==(2,) and torch.isfinite(logits).all()
    torch.nn.functional.binary_cross_entropy_with_logits(logits,data.y).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())


def test_deepsets_is_topology_free_but_sage_uses_edges():
    torch.manual_seed(1);a=batch();b=a.clone();b.edge_index=b.edge_index[:,:0];b.edge_type=b.edge_type[:0]
    model=GraphClassifier('deepsets',dropout=0).eval()
    assert torch.allclose(model(a)[0],model(b)[0])
    model=GraphClassifier('sage',dropout=0).eval()
    assert not torch.allclose(model(a)[0],model(b)[0])


def test_wl_features_change_when_topology_changes():
    g=Extractor().extract(CODE);modified=dict(g,edges=g['edges'][:0])
    assert wl(g,iterations=2)!=wl(modified,iterations=2)
    assert wl(g,iterations=0)==wl(modified,iterations=0)
