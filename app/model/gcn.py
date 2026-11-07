def dependencies():
 try:
  import torch
  from torch import nn
  import torch.nn.functional as F
  from torch_geometric.nn import GCNConv
  return torch,nn,F,GCNConv
 except ImportError as exc: raise RuntimeError('Faltan torch o torch_geometric. Ejecute: pip install torch torch-geometric') from exc

def create(in_channels=16,dropout=.30):
 torch,nn,F,GCNConv=dependencies()
 class GCN(nn.Module):
  def __init__(self):
   super().__init__(); self.c1=GCNConv(in_channels,32); self.c2=GCNConv(32,16); self.out=nn.Linear(16,2)
  def forward(self,x,edge_index,edge_weight=None):
   x=self.c1(x,edge_index,edge_weight).relu(); x=F.dropout(x,p=dropout,training=self.training); x=self.c2(x,edge_index,edge_weight).relu(); return self.out(x)
 return GCN()
