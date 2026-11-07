import json
from .gcn import create,dependencies

def metrics(y,p):
 from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
 pr,re,f1,_=precision_recall_fscore_support(y,p,average='binary',zero_division=0)
 return {'accuracy':accuracy_score(y,p),'precision':pr,'recall':re,'f1':f1,'confusion_matrix':confusion_matrix(y,p).tolist()}

def evaluate(model,loader,device):
 torch,_,_,_=dependencies(); model.eval(); ys=[]; ps=[]; loss=0
 with torch.no_grad():
  for b in loader:
   b=b.to(device); z=model(b.x,b.edge_index,b.edge_weight); loss+=torch.nn.functional.cross_entropy(z,b.y).item(); ys+=b.y.cpu().tolist(); ps+=z.argmax(1).cpu().tolist()
 out=metrics(ys,ps); out['loss']=loss/max(1,len(loader)); return out

def train(dataset,folder,epochs=150,lr=.01,seed=42,progress=None):
 torch,_,_,_=dependencies(); from torch_geometric.loader import DataLoader
 torch.manual_seed(seed); n=len(dataset); a=int(n*.70); b=int(n*.85)
 tr,va,te=dataset[:a],dataset[a:b],dataset[b:]
 tl=DataLoader(tr,batch_size=16,shuffle=True); vl=DataLoader(va,batch_size=16); el=DataLoader(te,batch_size=16)
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=create(dataset[0].num_node_features).to(device)
 opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=5e-4); best=-1; history=[]; folder.mkdir(parents=True,exist_ok=True); path=folder/'gcn_best.pt'
 for epoch in range(1,epochs+1):
  model.train(); total=0
  for batch in tl:
   batch=batch.to(device); opt.zero_grad(); z=model(batch.x,batch.edge_index,batch.edge_weight); l=torch.nn.functional.cross_entropy(z,batch.y); l.backward(); opt.step(); total+=l.item()
  val=evaluate(model,vl,device); row={'epoch':epoch,'train_loss':total/max(1,len(tl)),'val_loss':val['loss'],'val_f1':val['f1']}; history.append(row)
  if val['f1']>best: best=val['f1']; torch.save({'state_dict':model.state_dict(),'in_channels':dataset[0].num_node_features,'epoch':epoch},path)
  if progress and (epoch==1 or epoch%10==0 or epoch==epochs): progress(row)
 ck=torch.load(path,map_location=device,weights_only=False); model=create(ck['in_channels']).to(device); model.load_state_dict(ck['state_dict'])
 report={'seed':seed,'epochs':epochs,'graphs':n,'train_graphs':len(tr),'val_graphs':len(va),'test_graphs':len(te),'best_epoch':ck['epoch'],'test':evaluate(model,el,device),'history':history}
 (folder/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); return model,report,path

def load(path):
 torch,_,_,_=dependencies(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ck=torch.load(path,map_location=device,weights_only=False); model=create(ck['in_channels']).to(device); model.load_state_dict(ck['state_dict']); model.eval(); return model,device

def predict(model,device,g):
 torch,_,_,_=dependencies(); from .dataset import to_data
 d=to_data(g).to(device)
 with torch.no_grad(): z=model(d.x,d.edge_index,d.edge_weight); prob=torch.softmax(z,1); pred=prob.argmax(1)
 return {n:{'class':int(pred[i]),'probability':float(prob[i,1])} for i,n in enumerate(d.node_names)}
