"""Actual PyG graph operators on our frontend; paper-inspired variants are named.
No claim of exact Devign/ReVeal/IVDetect/AMPLE end-to-end reproduction is made.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv,SAGEConv,GINConv,GATConv,RGCNConv,GatedGraphConv,global_mean_pool,global_max_pool
from torch_geometric.utils import to_dense_batch

ARCHITECTURES=['deepsets','gcn','sage','gin','gat','rgcn','ggnn','devign_style','ggnn_supcon']
EXPERIMENTAL=['relation_gate','relation_uniform','ggnn_invariant','devign_relational_port','ggnn_sum_port']


class TypedGated(nn.Module):
    """Relation-specific sum messages and GRU, matching the DGL operator math."""
    def __init__(self,hidden,steps=8):
        super().__init__();self.steps=steps
        self.transforms=nn.ModuleList([nn.Linear(hidden,hidden) for _ in range(8)])
        self.gru=nn.GRUCell(hidden,hidden)
    def forward(self,h,edge_index,edge_type):
        slices=[edge_index[:,edge_type==r] for r in range(8)]
        for _ in range(self.steps):
            message=torch.zeros_like(h)
            for r,(src,dst) in enumerate(slices):
                message.index_add_(0,dst,self.transforms[r](h[src]))
            h=self.gru(message,h)
        return h


class RelationMix(nn.Module):
    def __init__(self,hidden,gated=True):
        super().__init__();self.gated=gated
        self.transforms=nn.ModuleList([nn.Linear(hidden,hidden,bias=False) for _ in range(8)])
        self.gate=nn.Linear(hidden,8) if gated else None
    def forward(self,h,edge_index,edge_type):
        weights=torch.softmax(self.gate(h),dim=-1) if self.gated else h.new_full((len(h),8),1/8)
        output=torch.zeros_like(h)
        for r,linear in enumerate(self.transforms):
            mask=edge_type==r;src,dst=edge_index[:,mask]
            message=torch.zeros_like(h);degree=h.new_zeros(len(h))
            message.index_add_(0,dst,h[src]);degree.index_add_(0,dst,torch.ones_like(dst,dtype=h.dtype))
            output=output+linear(message/degree.clamp_min(1)[:,None])*weights[:,r,None]
        return output


class GraphClassifier(nn.Module):
    def __init__(self,architecture='gcn',hidden=48,dropout=.2):
        super().__init__();self.architecture=architecture;self.dropout=dropout
        self.type_embedding=nn.Embedding(1025,16);self.token_embedding=nn.Embedding(8193,16,padding_idx=0)
        self.input=nn.Linear(36,hidden);self.norms=nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])
        if architecture in ['ggnn','devign_style','ggnn_supcon','ggnn_invariant']:
            self.gated=GatedGraphConv(hidden,num_layers=3)
        elif architecture in ['devign_relational_port','ggnn_sum_port']:
            self.typed_gated=TypedGated(hidden,steps=8)
        else:
            def make():
                if architecture=='gcn':return GCNConv(hidden,hidden)
                if architecture=='sage':return SAGEConv(hidden,hidden)
                if architecture=='gin':return GINConv(nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,hidden)),train_eps=True)
                if architecture=='gat':return GATConv(hidden,hidden,heads=2,concat=False,dropout=dropout)
                if architecture=='rgcn':return RGCNConv(hidden,hidden,num_relations=8,num_bases=4)
                if architecture=='deepsets':return nn.Linear(hidden,hidden)
                if architecture in ['relation_gate','relation_uniform']:return RelationMix(hidden,architecture=='relation_gate')
                raise ValueError(architecture)
            self.convs=nn.ModuleList([make(),make()])
        self.head=nn.Sequential(nn.Linear(hidden*2,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,1))
        if architecture=='devign_style':
            # Independent node-axis convolution heads; differs from the original
            # released AST-only implementation and original CPG/word2vec setup.
            def conv(ch):return nn.Sequential(nn.Conv1d(ch,32,3,padding=1),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(32,16,3,padding=1),nn.ReLU(),nn.AdaptiveMaxPool1d(1),nn.Flatten(),nn.Linear(16,1))
            self.conv_hidden=conv(hidden);self.conv_concat=conv(2*hidden)
        if architecture=='devign_relational_port':
            # Author implementation: dual node-axis convolution/pooling branches,
            # then per-position linear product and mean. Frontend/dimensions differ.
            def branch(ch):return nn.Sequential(nn.Conv1d(ch,ch,3),nn.ReLU(),nn.MaxPool1d(3,stride=2),nn.Conv1d(ch,ch,1),nn.ReLU(),nn.MaxPool1d(2,stride=2))
            self.port_y=branch(hidden);self.port_z=branch(hidden*2)
            self.port_mlp_y=nn.Linear(hidden,1);self.port_mlp_z=nn.Linear(hidden*2,1)
        if architecture=='ggnn_sum_port':self.sum_head=nn.Linear(hidden,1)

    def forward(self,data):
        x=F.relu(self.input(torch.cat([self.type_embedding(data.type_ids),self.token_embedding(data.token_ids),data.x_num],dim=-1)))
        h=x
        if hasattr(self,'gated'):h=self.gated(h,data.edge_index)
        elif hasattr(self,'typed_gated'):h=self.typed_gated(h,data.edge_index,data.edge_type)
        else:
            for i,conv in enumerate(self.convs):
                if self.architecture=='deepsets':z=conv(h)
                elif self.architecture in ['rgcn','relation_gate','relation_uniform']:z=conv(h,data.edge_index,data.edge_type)
                else:z=conv(h,data.edge_index)
                h=F.dropout(F.relu(self.norms[i](z+h)),p=self.dropout,training=self.training)
        embedding=torch.cat([global_mean_pool(h,data.batch),global_max_pool(h,data.batch)],dim=-1)
        if self.architecture=='devign_relational_port':
            hd,_=to_dense_batch(h,data.batch,max_num_nodes=256)
            xd,_=to_dense_batch(x,data.batch,max_num_nodes=256)
            y=self.port_y(hd.transpose(1,2)).transpose(1,2)
            z=self.port_z(torch.cat([hd,xd],dim=-1).transpose(1,2)).transpose(1,2)
            logits=(self.port_mlp_y(y)*self.port_mlp_z(z)).mean(dim=1).flatten()
        elif self.architecture=='ggnn_sum_port':
            from torch_geometric.nn import global_add_pool
            logits=self.sum_head(global_add_pool(h,data.batch)).flatten()
        elif self.architecture=='devign_style':
            hd,mask=to_dense_batch(h,data.batch,max_num_nodes=256)
            xd,_=to_dense_batch(x,data.batch,max_num_nodes=256)
            logits=self.conv_hidden(hd.transpose(1,2))*self.conv_concat(torch.cat([hd,xd],dim=-1).transpose(1,2))
            logits=logits.flatten()
        else:logits=self.head(embedding).flatten()
        return logits,embedding


def supervised_contrastive(embedding,labels,temperature=.2):
    z=F.normalize(embedding,dim=1);similarity=z@z.T/temperature
    eye=torch.eye(len(z),dtype=torch.bool,device=z.device)
    same=(labels[:,None]==labels[None,:])&~eye
    similarity=similarity.masked_fill(eye,-1e9)
    lp=similarity-torch.logsumexp(similarity,dim=1,keepdim=True)
    return -((lp*same).sum(dim=1)/same.sum(dim=1).clamp_min(1)).mean()
