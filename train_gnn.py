from pathlib import Path
from app.model.dataset import generate
from app.model.trainer import train
if __name__=='__main__':
 data=generate(150,42)
 model,report,path=train(data,Path('models'),150,.01,42,lambda r:print(f"Epoca {r['epoch']:03d} loss={r['train_loss']:.4f} val_f1={r['val_f1']:.4f}"))
 print('Modelo:',path); print('Prueba:',report['test'])
