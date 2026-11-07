import random
DEVICE_TYPES=['Computador','Servidor','Router','Switch','IoT','Impresora','PuntoAcceso']

def edge_stats(g,n):
 e=[g.edges[n,v] for v in g.neighbors(n) if not g.edges[n,v].get('bloqueada',False)]
 return (sum(x['latencia'] for x in e)/max(1,len(e)),sum(x['perdida'] for x in e)/max(1,len(e)))

def label(g,n):
 d=g.nodes[n]; lat,loss=edge_stats(g,n)
 score=sum([d['cpu']>=85,d['memoria']>=90,d['trafico']>=80,d['fallos']>=2,d['temperatura']>=75,lat>150,loss>10])
 return int(score>=3)

def features(g,n):
 d=g.nodes[n]; lat,loss=edge_stats(g,n)
 one=[float(d['tipo']==t) for t in DEVICE_TYPES]
 nums=[d['cpu']/100,d['memoria']/100,d['trafico']/100,d['fallos']/5,(d['temperatura']-20)/70,lat/200,loss/20,g.degree(n)/10,float(d.get('resultado_automata',1))]
 return one+nums

def inject(g,rng):
 mode=rng.choice(['normal','cpu','trafico','perdida','mixto','fallos'])
 for n in rng.sample(list(g.nodes),rng.randint(2,6)):
  d=g.nodes[n]
  if mode in ('cpu','mixto'): d['cpu']=rng.randint(86,100)
  if mode in ('trafico','mixto'): d['trafico']=rng.randint(82,100)
  if mode in ('fallos','mixto'): d['fallos']=rng.randint(2,5)
  if mode=='mixto': d['temperatura']=round(rng.uniform(76,90),1)
 if mode in ('perdida','mixto'):
  from app.graph_service import edge_weight
  for u,v in rng.sample(list(g.edges),min(rng.randint(2,5),g.number_of_edges())):
   e=g.edges[u,v]; e['perdida']=round(rng.uniform(11,20),2); e['latencia']=rng.randint(145,200); e['peso']=edge_weight(e['latencia'],e['trafico'],e['perdida'])

def to_data(g):
 try:
  import torch
  from torch_geometric.data import Data
 except ImportError as exc: raise RuntimeError('Instale torch y torch_geometric para usar la GNN real.') from exc
 nodes=list(g.nodes); idx={n:i for i,n in enumerate(nodes)}
 x=torch.tensor([features(g,n) for n in nodes],dtype=torch.float32); y=torch.tensor([label(g,n) for n in nodes],dtype=torch.long)
 pairs=[]; affinity=[]
 for u,v,d in g.edges(data=True):
  if d.get('bloqueada'): continue
  a=1/(1+float(d['peso'])); pairs += [[idx[u],idx[v]],[idx[v],idx[u]]]; affinity += [a,a]
 data=Data(x=x,edge_index=torch.tensor(pairs,dtype=torch.long).t().contiguous(),edge_weight=torch.tensor(affinity,dtype=torch.float32),y=y)
 data.node_names=nodes; return data

def generate(count=150,seed=42):
 from app.graph_service import build_graph
 result=[]
 for i in range(count):
  g=build_graph(seed+i); inject(g,random.Random((seed+i)*37)); d=to_data(g); d.scenario_seed=seed+i; result.append(d)
 return result
