import random
import networkx as nx
from .config import DEVICE_COUNTS, PREFIX, WEIGHT_FACTORS, MAX_VALUES


def edge_weight(latency, traffic, loss):
    return round(
        WEIGHT_FACTORS["latencia"] * latency / MAX_VALUES["latencia"]
        + WEIGHT_FACTORS["trafico"] * traffic / MAX_VALUES["trafico"]
        + WEIGHT_FACTORS["perdida"] * loss / MAX_VALUES["perdida"], 4
    )


def _classify_node(data):
    # Etiqueta inicial util para la interfaz. Se sustituira por la GCN en el incremento 3.
    score = sum([
        data["cpu"] >= 85,
        data["memoria"] >= 90,
        data["trafico"] >= 80,
        data["fallos"] >= 2,
        data["temperatura"] >= 75,
    ])
    data["probabilidad"] = min(0.98, round(0.12 + 0.17 * score, 2))
    data["prediccion"] = "Anomalo" if score >= 3 else "Normal"


def build_graph(seed=42):
    rng = random.Random(seed)
    graph = nx.Graph(seed=seed)
    for device_type, count in DEVICE_COUNTS.items():
        for index in range(1, count + 1):
            node = f"{PREFIX[device_type]}{index:02d}"
            attrs = {
                "tipo": device_type,
                "cpu": rng.randint(5, 100), "memoria": rng.randint(10, 100),
                "trafico": rng.randint(0, 100), "fallos": rng.randint(0, 5),
                "temperatura": round(rng.uniform(20, 90), 1),
                "bloqueado": False, "probabilidad": 0.0, "prediccion": "Normal",
            }
            _classify_node(attrs)
            graph.add_node(node, **attrs)

    topology = [
        ("PC01","SW01"),("PC02","SW01"),("PC03","SW01"),("IoT01","SW01"),("IoT02","SW01"),("IMP01","SW01"),("AP01","SW01"),
        ("PC04","SW02"),("PC05","SW02"),("PC06","SW02"),("IoT03","SW02"),("IoT04","SW02"),("IMP02","SW02"),("AP02","SW02"),
        ("SW01","RTR01"),("SW02","RTR02"),("RTR01","RTR02"),
        ("SRV01","SW01"),("SRV01","RTR02"),("SRV02","SW02"),("SRV02","RTR01"),
    ]
    for left, right in topology:
        latency = rng.randint(5, 200)
        traffic = rng.randint(0, 100)
        loss = round(rng.uniform(0, 20), 2)
        bandwidth = rng.choice([100, 250, 500, 1000])
        graph.add_edge(left, right, latencia=latency, trafico=traffic, perdida=loss,
                       ancho_banda=bandwidth, peso=edge_weight(latency, traffic, loss), bloqueada=False)
    return graph


def active_graph(graph):
    nodes = [n for n, d in graph.nodes(data=True) if not d.get("bloqueado", False)]
    active = graph.subgraph(nodes).copy()
    active.remove_edges_from([(u, v) for u, v, d in active.edges(data=True) if d.get("bloqueada", False)])
    return active


def shortest_route(graph, source, target):
    active = active_graph(graph)
    path = nx.dijkstra_path(active, source, target, weight="peso")
    metrics = {
        "peso": nx.dijkstra_path_length(active, source, target, weight="peso"),
        "latencia": sum(active.edges[u, v]["latencia"] for u, v in zip(path, path[1:])),
        "trafico_promedio": sum(active.edges[u, v]["trafico"] for u, v in zip(path, path[1:])) / (len(path) - 1),
        "perdida_promedio": sum(active.edges[u, v]["perdida"] for u, v in zip(path, path[1:])) / (len(path) - 1),
        "saltos": len(path) - 1,
    }
    return path, metrics
