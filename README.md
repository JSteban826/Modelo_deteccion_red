# Sistema final - Etapa 4, incremento 2

Este incremento integra una **GCN real** con PyTorch Geometric al grafo final, Dijkstra, bloqueos, autómata dinámico y exportación.

## Instalación
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución
```bash
python main.py
```

## Entrenamiento por terminal
```bash
python train_gnn.py
```

El entrenamiento genera 150 grafos de 20 nodos y separa grafos completos en 70 % entrenamiento, 15 % validación y 15 % prueba. Guarda `models/gcn_best.pt` y `models/metrics.json`.

## Arquitectura
- Entrada: 16 características por nodo.
- GCNConv: 16 -> 32.
- ReLU y dropout 0.30.
- GCNConv: 32 -> 16.
- Capa lineal: 16 -> 2 clases.
- Optimizador Adam y CrossEntropyLoss.
- Afinidad de arista para GNN: `1 / (1 + peso_dijkstra)`.
- Clases: normal y anómalo.
- Bloqueado permanece como estado operativo.

La primera instalación de PyTorch puede tardar varios minutos. Para la sustentación conviene entrenar antes y conservar el archivo `gcn_best.pt`.
