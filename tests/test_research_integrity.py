"""Regressions for silent evaluation and probability failures discovered in audit."""
import math
import subprocess
from types import SimpleNamespace
import pytest
import vulnhybrid
from benchmark_hygiene import sanitize_code, code_hash
from findings_schema import Finding
from hybrid_scoring import static_probability, sigmoid, hybrid_score
from llm_stage.llm_client import args_to_verdict, LLMConfig
from llm_stage.llm_schema import Verdict, by_sample
from evaluate import compute_metrics
from static_analysis import semgrep_runner, cppcheck_runner
from static_analysis.errors import AnalysisError
from run_llm_stage import request_fingerprint, select_samples
from schema import Sample


def test_boolean_string_cannot_reverse_safe_verdict():
    with pytest.raises(ValueError,match='boolean'):
        args_to_verdict('s','m',{'is_vulnerable':'false','confidence':.9},[])
    assert args_to_verdict('s','m',{'is_vulnerable':False,'confidence':.9},[]).is_vulnerable is False


@pytest.mark.parametrize('confidence',[None,True,'0.9',float('nan'),float('inf'),-0.1,1.1])
def test_invalid_confidence_is_rejected(confidence):
    with pytest.raises(ValueError):
        args_to_verdict('s','m',{'is_vulnerable':False,'confidence':confidence},[])


def test_duplicate_rules_do_not_inflate_evidence():
    f=Finding(finding_id='f',sample_id='s',tool='cppcheck',rule_id='doubleFree',
              cwe='CWE-415',severity='error',message='x',file_line=1)
    assert static_probability([f])[0] == static_probability([f]*1000)[0]


def test_many_independent_rules_remain_finite():
    assert sigmoid(-100000)==0. and sigmoid(100000)==1.


@pytest.mark.parametrize('kwargs',[{'prior':0},{'prior':1},{'w_llm':-1},{'threshold':float('nan')}])
def test_scoring_parameter_validation(kwargs):
    with pytest.raises(ValueError):hybrid_score('s',[],None,**kwargs)


def test_duplicate_verdict_join_fails():
    v=Verdict('s','m',False,.9)
    with pytest.raises(ValueError,match='Duplicate'):by_sample([v,v])


def test_failed_checkpoint_write_preserves_previous_file(tmp_path):
    from llm_stage.llm_schema import write_jsonl
    path=tmp_path/'verdicts.jsonl'
    good=Verdict('s','m',False,.9)
    write_jsonl([good],str(path));original=path.read_bytes()
    bad=Verdict('bad','m',True,float('nan'))
    with pytest.raises(ValueError):write_jsonl([good,bad],str(path))
    assert path.read_bytes()==original
    assert list(tmp_path.glob('*.tmp'))==[]


def test_zero_recall_f1_is_zero_not_nan():
    assert compute_metrics([1,0],[0,0]).f1==0.
    assert compute_metrics([1,0],[0,1]).f1==0.
    with pytest.raises(ValueError):compute_metrics([1],[1,0])
    with pytest.raises(ValueError):compute_metrics([2],[1])


def test_comment_hints_removed_without_corrupting_literals_or_lines():
    code='void CWE78_bad(){/* FLAW\nBUG */ puts("http://good/bad"); //good\n}'
    cleaned=sanitize_code(code,juliet=True)
    assert 'CWE78_bad' not in cleaned and 'FLAW' not in cleaned
    assert '"http://good/bad"' in cleaned
    assert cleaned.count('\n')==code.count('\n')
    raw='const char*x=R"tag(/*literal*/ // not a comment)tag";'
    assert sanitize_code(raw)==raw


def test_clone_hash_preserves_token_boundaries_and_literal_whitespace():
    assert code_hash('int f(){ /*x*/ return 0;}')==code_hash('int f() {return 0; }')
    assert code_hash('a b')!=code_hash('ab')
    assert code_hash('puts("a b");')!=code_hash('puts("ab");')
    assert code_hash('x=1 2;')!=code_hash('x=12;')
    assert code_hash('a + +b')!=code_hash('a++b')


def test_semgrep_failure_cannot_be_clean_scan(monkeypatch):
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=2,stdout='',stderr='bad config'))
    with pytest.raises(AnalysisError):semgrep_runner._run_semgrep('.')


def test_cppcheck_nonxml_cannot_be_clean_scan(monkeypatch):
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=2,stdout='',stderr='fatal error'))
    with pytest.raises(AnalysisError):cppcheck_runner._run_cppcheck('.',1,None,None)


def test_semgrep_partial_parse_is_explicit(monkeypatch):
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='{"results":[],"errors":[{"type":"PartialParsing"}]}',stderr=''))
    with pytest.raises(AnalysisError):semgrep_runner._run_semgrep('.')


def test_resume_fingerprint_covers_prompt_model_and_code():
    s=Sample('s','devign','c','void f(){}',0,split='test')
    a=request_fingerprint(s,{},LLMConfig(model='a'))
    assert a!=request_fingerprint(s,{},LLMConfig(model='b'))
    s.code='void g(){}'
    assert a!=request_fingerprint(s,{},LLMConfig(model='a'))


def test_sampling_cap_does_not_depend_on_source_order():
    samples=[Sample(str(i),'devign','c','x',i%2,split='test') for i in range(100)]
    def pick(s):return [x.sample_id for x in select_samples(s,{},'all-samples','test',None,10,42)]
    assert pick(samples)==pick(list(reversed(samples)))
    assert pick(samples)!=[str(i) for i in range(10)]
