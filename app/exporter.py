import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from .config import RESULTS_DIR


def export_scenario(graph, summary, figure, seed):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = RESULTS_DIR / f"escenario_{seed}_{stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    nodes = []
    for node, data in graph.nodes(data=True):
        nodes.append({"nodo": node, **data})
    edges = []
    for u, v, data in graph.edges(data=True):
        edges.append({"origen": u, "destino": v, **data})
    pd.DataFrame(nodes).to_csv(folder / "nodos.csv", index=False)
    pd.DataFrame(edges).to_csv(folder / "aristas.csv", index=False)
    with open(folder / "resumen.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    figure.savefig(folder / "grafo_actual.png", dpi=180, bbox_inches="tight")
    return folder
