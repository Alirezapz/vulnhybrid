"""Label-blind C/C++ AST and explicitly approximate program-relation graphs.

AST edges come from tree-sitter. CFG is structured statement flow, not a compiler
CFG; switches/gotos/exceptions are diagnosed. Last-identifier-use is a lexical
relation, not a sound reaching-definitions or pointer/alias analysis.
"""
from __future__ import annotations
from collections import Counter,defaultdict,deque
from pathlib import Path
import argparse,hashlib,json,re,time
import numpy as np
from pipeline.benchmark_hygiene import sanitize_code,code_hash

RELATIONS=['AST','AST_REV','NEXT_LEAF','NEXT_LEAF_REV','LAST_IDENT','LAST_IDENT_REV','CFG','CFG_REV']
STATEMENTS={'expression_statement','declaration','return_statement','if_statement','while_statement',
 'for_statement','do_statement','break_statement','continue_statement','switch_statement','case_statement',
 'goto_statement','labeled_statement','try_statement','throw_statement'}
SECURITY={'free','malloc','realloc','calloc','memcpy','memmove','strcpy','strcat','gets','sprintf','snprintf',
 'system','popen','printf','fprintf','read','recv','getenv','fgets','new','delete'}


def stable_bucket(s,n):return int.from_bytes(hashlib.blake2b(s.encode(),digest_size=8).digest(),'little')%n

class Extractor:
    def __init__(self,max_nodes=256,selection='bfs',semantic=False):
        from tree_sitter import Language,Parser
        import tree_sitter_c,tree_sitter_cpp
        self.parsers={'c':Parser(Language(tree_sitter_c.language())),
                      'cpp':Parser(Language(tree_sitter_cpp.language()))}
        self.max_nodes=max_nodes
        self.selection=selection
        self.semantic=semantic

    def extract(self,code,language='c'):
        clean=sanitize_code(code);source=clean.encode();root=self.parsers[language].parse(source).root_node
        nodes=[];parents={};queue=deque([(root,None)]);ast=[]
        # BFS truncation is deterministic and label-blind; its coverage is recorded.
        while queue:
            node,parent=queue.popleft();idx=len(nodes);nodes.append(node)
            if parent is not None:ast.append((parent,idx))
            for child in node.named_children:queue.append((child,idx));parents[child.id]=node.id
        nfull=len(nodes);full=nodes
        chosen=set(range(min(len(full),self.max_nodes-1)))
        if self.selection=='evidence' and nfull>self.max_nodes-1:
            old_index={node.id:i for i,node in enumerate(full)}
            chosen={0}
            critical=[i for i,node in enumerate(full) if not node.named_children
                      and source[node.start_byte:node.end_byte].decode(errors='replace') in SECURITY]
            for i in critical:
                chain=[];node=full[i]
                while node is not None and node.id in old_index:
                    chain.append(old_index[node.id]);node=node.parent
                for k in reversed(chain):
                    if len(chosen)<self.max_nodes-1:chosen.add(k)
            for i in range(nfull):
                if len(chosen)>=self.max_nodes-1:break
                chosen.add(i)
        selected=sorted(chosen);remap={old:new for new,old in enumerate(selected)}
        nodes=[full[i] for i in selected];n=len(nodes);exit_id=n
        index={node.id:i for i,node in enumerate(nodes)}
        labels=[];tokens=[];numeric=[];names=[]
        for node in nodes:
            text=source[node.start_byte:node.end_byte].decode(errors='replace')
            is_leaf=not node.named_children
            token=text[:128] if is_leaf else ''
            kind=node.type
            if self.semantic:
                operator=node.child_by_field_name('operator')
                if operator is not None:
                    token='operator:'+source[operator.start_byte:operator.end_byte].decode(errors='replace')
                parent=node.parent
                if parent is not None:
                    for child_index,child in enumerate(parent.children):
                        if child.id==node.id:
                            role=parent.field_name_for_child(child_index)
                            # Field roles and argument positions distinguish branches
                            # and swapped operands that an unordered AST loses.
                            kind+='@'+(role or ('position'+str(min(child_index,4))))
                            break
            labels.append(stable_bucket(kind,1024)+1)
            tokens.append(stable_bucket(token,8192)+1 if token else 0)
            names.append(node.type)
            numeric.append([float(node.type=='identifier'),float(node.type in STATEMENTS),
                            float(token in SECURITY),float('literal' in node.type)])
        labels.append(0);tokens.append(0);numeric.append([0,0,0,0]);names.append('EXIT')
        edges=set()
        def add(a,b,rel):
            if a is not None and b is not None and a!=b:
                edges.add((a,b,rel));edges.add((b,a,rel+1))
        for a,b in ast:
            if a in remap and b in remap:add(remap[a],remap[b],0)
        leaves=sorted([node for node in nodes if not node.named_children],key=lambda a:a.start_byte)
        for a,b in zip(leaves,leaves[1:]):add(index[a.id],index[b.id],2)
        previous={}
        for node in sorted(nodes,key=lambda a:a.start_byte):
            if node.type=='identifier':
                name=source[node.start_byte:node.end_byte]
                if name in previous:add(previous[name],index[node.id],4)
                previous[name]=index[node.id]
        unsupported=Counter()
        def target(node):return index.get(node.id) if node is not None else None
        def flow(node,succ,break_to=None,continue_to=None):
            if node is None:return succ
            kind=node.type;here=target(node)
            if kind in ('compound_statement','else_clause'):
                current=succ
                for child in reversed(node.named_children):current=flow(child,current,break_to,continue_to)
                return current
            if kind=='if_statement':
                yes=flow(node.child_by_field_name('consequence'),succ,break_to,continue_to)
                no=flow(node.child_by_field_name('alternative'),succ,break_to,continue_to)
                add(here,yes,6);add(here,no,6);return here if here is not None else yes
            if kind in ('while_statement','for_statement','do_statement'):
                body=flow(node.child_by_field_name('body'),here,succ,here)
                add(here,body,6);add(here,succ,6)
                return body if kind=='do_statement' else (here if here is not None else body)
            if kind=='return_statement':add(here,exit_id,6);return here if here is not None else exit_id
            if kind=='break_statement':add(here,break_to,6);return here if here is not None else break_to
            if kind=='continue_statement':add(here,continue_to,6);return here if here is not None else continue_to
            if kind in ('switch_statement','goto_statement','labeled_statement','try_statement','throw_statement'):
                unsupported[kind]+=1
            if kind in STATEMENTS or kind.endswith('_statement'):
                add(here,succ,6);return here if here is not None else succ
            return succ
        functions=[node for node in nodes if node.type=='function_definition']
        for fn in functions:
            first=flow(fn.child_by_field_name('body'),exit_id)
            add(target(fn),first,6)
        ordered=sorted(edges)
        if not ordered:ordered=[(0,0,0)]
        meta=dict(parse_error=bool(root.has_error),full_nodes=nfull,kept_nodes=n+1,truncated=nfull>n,
                  unsupported_cfg=dict(unsupported),functions_seen=len(functions),
                  relation_counts=dict(Counter(RELATIONS[e[2]] for e in ordered)))
        return dict(types=np.asarray(labels,dtype=np.uint16),tokens=np.asarray(tokens,dtype=np.uint16),
                    numeric=np.asarray(numeric,dtype=np.float32),edges=np.asarray(ordered,dtype=np.uint16),
                    node_names=names,meta=meta)


def build(files,out,max_nodes=256,selection='bfs',semantic=False):
    rows=[json.loads(line) for path in files for line in Path(path).read_text().splitlines() if line.strip()]
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Duplicate graph IDs')
    ex=Extractor(max_nodes,selection,semantic);out.mkdir(parents=True,exist_ok=True);graphs=[];metadata=[];start=time.monotonic()
    for i,r in enumerate(rows):
        g=ex.extract(r['code'],r['language']);metadata.append(dict(sample_id=r['sample_id'],
            code_sha256=code_hash(r['code']),dataset=r['dataset'],**g.pop('meta')));graphs.append(g)
        if (i+1)%2000==0:print('EXTRACTED',i+1,'seconds',round(time.monotonic()-start,1),flush=True)
    # CSR-style packed arrays; labels and split membership are deliberately absent.
    nptr=np.r_[0,np.cumsum([len(g['types']) for g in graphs])].astype(np.int64)
    eptr=np.r_[0,np.cumsum([len(g['edges']) for g in graphs])].astype(np.int64)
    np.savez_compressed(out/'graphs.npz',types=np.concatenate([g['types'] for g in graphs]),
        tokens=np.concatenate([g['tokens'] for g in graphs]),numeric=np.concatenate([g['numeric'] for g in graphs]),
        edges=np.concatenate([g['edges'] for g in graphs]),nptr=nptr,eptr=eptr)
    (out/'graph_metadata.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in metadata))
    summary=dict(n=len(rows),max_nodes=max_nodes,selection=selection,semantic=semantic,parse_errors=sum(r['parse_error'] for r in metadata),
        truncated=sum(r['truncated'] for r in metadata),node_quantiles=np.quantile([r['full_nodes'] for r in metadata],[0,.5,.9,.99,1]).tolist(),
        unsupported_cfg=dict(sum((Counter(r['unsupported_cfg']) for r in metadata),Counter())),
        relations=RELATIONS,seconds=time.monotonic()-start,
        input_files=[dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in files],
        cache_sha256=hashlib.sha256((out/'graphs.npz').read_bytes()).hexdigest())
    (out/'graph_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


class GraphStore:
    def __init__(self,path):
        self.path=Path(path);self.a=np.load(self.path/'graphs.npz')
        self.metadata=[json.loads(x) for x in (self.path/'graph_metadata.jsonl').read_text().splitlines()]
        self.ids={r['sample_id']:i for i,r in enumerate(self.metadata)}
        # Load once: repeated access to compressed npz arrays would decompress each time.
        self.a={k:self.a[k] for k in self.a.files}
    def get(self,sid,view='hetero',anonymous=False):
        i=self.ids[sid];a=self.a;n=slice(a['nptr'][i],a['nptr'][i+1]);e=slice(a['eptr'][i],a['eptr'][i+1])
        edges=a['edges'][e]
        if view=='ast':edges=edges[edges[:,2]<2]
        elif view=='cfg':edges=edges[edges[:,2]>=6]
        elif view=='ddg':edges=edges[(edges[:,2]>=4)&(edges[:,2]<6)]
        elif view=='sequence':edges=edges[(edges[:,2]>=2)&(edges[:,2]<4)]
        elif view=='none':edges=edges[:0]
        return dict(types=a['types'][n],tokens=np.zeros_like(a['tokens'][n]) if anonymous else a['tokens'][n],numeric=a['numeric'][n],edges=edges)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--samples',nargs='+',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--max-nodes',type=int,default=256)
    p.add_argument('--selection',choices=['bfs','evidence'],default='bfs')
    p.add_argument('--semantic',action='store_true')
    a=p.parse_args();build(a.samples,a.out,a.max_nodes,a.selection,a.semantic)
