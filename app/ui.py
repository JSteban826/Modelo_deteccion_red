import math
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

import matplotlib.image as mpimg
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from .automata import REGEX, STATES, run, sequence_for_path
from .config import ICON_FILES, STATE_COLORS, WEIGHT_FACTORS
from .exporter import export_scenario
from .graph_service import build_graph, shortest_route


class FinalApp:
    """Interfaz principal del sistema final.

    Conserva la distribución original de tres columnas:
    configuración y ruta a la izquierda, grafo en el centro y detalles/GNN/autómata
    a la derecha. Las mejoras visuales no alteran las funciones del proyecto.
    """

    BG = "#E9EDF3"
    PANEL = "#FFFFFF"
    PRIMARY = "#145DA0"
    DARK = "#0C2D48"
    ACCENT = "#2E8BC0"
    TEXT = "#172536"
    MUTED = "#607080"
    BORDER = "#C7D1DB"
    SUCCESS_BG = "#E6F5EC"
    SUCCESS_FG = "#16763A"
    PENDING_BG = "#FFF3D3"
    PENDING_FG = "#876000"
    ERROR_BG = "#FDE7E7"
    ERROR_FG = "#A52B2B"

    def __init__(self, root):
        self.root = root
        self.root.title("Sistema final - Analisis de infraestructura con grafos, automata y GNN")
        self.root.geometry("1500x900")
        self.root.minsize(1200, 760)
        self.root.configure(bg=self.BG)

        self.seed = tk.IntVar(value=42)
        self.show_metrics = tk.BooleanVar(value=False)
        self.block_mouse = tk.BooleanVar(value=False)

        self.graph = build_graph(self.seed.get())
        self.position = self._fixed_positions()
        self.path = []
        self.metrics = {}
        self.selected_node = None
        self.selected_edge = None

        self.automata_job = None
        self.automata_trace = []
        self.automata_step = 0
        self.automata_accepted = None

        self.gnn_model = None
        self.gnn_device = None
        self.gnn_report = None
        self.scenario_classified = False

        self.icons = {kind: mpimg.imread(path) for kind, path in ICON_FILES.items()}

        self._build_ui()
        self.draw_graph()
        self._set_details_placeholder()
        self._update_status("Programa listo. La latencia aporta el 70 % del peso de Dijkstra.")

    def _fixed_positions(self):
        return {
            "PC01": (-4, 2.7), "PC02": (-4, 1.8), "PC03": (-4, 0.9),
            "IoT01": (-3.1, 3.1), "IoT02": (-3.0, 0.5), "IMP01": (-4, -0.1),
            "AP01": (-3, -0.6), "SW01": (-1.8, 1.4), "RTR01": (-0.6, 1.4),
            "SRV01": (-0.7, 3.0), "RTR02": (0.6, 1.4), "SW02": (1.8, 1.4),
            "SRV02": (0.7, -0.2), "PC04": (4, 2.7), "PC05": (4, 1.8),
            "PC06": (4, 0.9), "IoT03": (3.1, 3.1), "IoT04": (3, 0.5),
            "IMP02": (4, -0.1), "AP02": (3, -0.6),
        }

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=self.BG)
        style.configure("TLabel", background=self.BG, foreground=self.TEXT, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=self.BG, foreground=self.DARK, font=("Segoe UI", 16, "bold"))
        style.configure("Section.TLabel", background=self.BG, foreground=self.PRIMARY, font=("Segoe UI", 11, "bold"))
        style.configure("Card.TFrame", background=self.PANEL, relief="solid", borderwidth=1)
        style.configure("CardTitle.TLabel", background=self.PANEL, foreground=self.PRIMARY, font=("Segoe UI", 11, "bold"))
        style.configure("CardText.TLabel", background=self.PANEL, foreground=self.MUTED, font=("Segoe UI", 9))
        style.configure("Success.TLabel", background=self.SUCCESS_BG, foreground=self.SUCCESS_FG, font=("Segoe UI", 10, "bold"), padding=(10, 6))
        style.configure("Pending.TLabel", background=self.PENDING_BG, foreground=self.PENDING_FG, font=("Segoe UI", 10, "bold"), padding=(10, 6))
        style.configure("Error.TLabel", background=self.ERROR_BG, foreground=self.ERROR_FG, font=("Segoe UI", 10, "bold"), padding=(10, 6))
        style.configure("Primary.TButton", background=self.PRIMARY, foreground="white", font=("Segoe UI", 10, "bold"), padding=(10, 6))
        style.map("Primary.TButton", background=[("active", self.ACCENT), ("pressed", self.DARK)], foreground=[("!disabled", "white")])
        style.configure("Secondary.TButton", background="#DDE8F2", foreground=self.DARK, font=("Segoe UI", 10), padding=(9, 6))
        style.map("Secondary.TButton", background=[("active", "#C9DCEB")])
        style.configure("Train.TButton", background="#6F42C1", foreground="white", font=("Segoe UI", 10, "bold"), padding=(10, 6))
        style.map("Train.TButton", background=[("active", "#8257D5")], foreground=[("!disabled", "white")])
        style.configure("Classify.TButton", background="#278A49", foreground="white", font=("Segoe UI", 10, "bold"), padding=(10, 6))
        style.map("Classify.TButton", background=[("active", "#36A45D")], foreground=[("!disabled", "white")])
        style.configure("Restore.TButton", background="#D88927", foreground="white", font=("Segoe UI", 10), padding=(9, 6))
        style.map("Restore.TButton", background=[("active", "#E49A3D")], foreground=[("!disabled", "white")])
        style.configure("Horizontal.TProgressbar", troughcolor="#E4EAF0", background=self.PRIMARY, bordercolor="#E4EAF0", lightcolor=self.PRIMARY, darkcolor=self.PRIMARY)

    def _build_ui(self):
        self._configure_styles()

        # Barra superior: conserva el orden y la disposición original.
        toolbar = ttk.Frame(self.root, padding=(12, 8))
        toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(toolbar, text="SISTEMA FINAL DE ANALISIS DE RED", style="Title.TLabel").pack(side=tk.LEFT, padx=(0, 22))
        ttk.Button(toolbar, text="Generar escenario", command=self.generate, style="Primary.TButton").pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Calcular ruta", command=self.calculate_route, style="Primary.TButton").pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Ejecutar automata", command=self.start_automata, style="Primary.TButton").pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Exportar resultados", command=self.export, style="Secondary.TButton").pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Restaurar red", command=self.restore, style="Restore.TButton").pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Entrenar GNN", command=self.train_gnn_async, style="Train.TButton").pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Cargar GNN", command=self.load_gnn, style="Secondary.TButton").pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Clasificar GNN", command=self.classify_gnn, style="Classify.TButton").pack(side=tk.LEFT, padx=3)
        ttk.Checkbutton(toolbar, text="Mostrar metricas", variable=self.show_metrics, command=self.draw_graph).pack(side=tk.RIGHT)

        # Tres columnas originales.
        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=(6, 6), pady=(0, 4))
        left = ttk.Frame(body, padding=8, width=260)
        center = ttk.Frame(body, padding=(4, 0))
        right = ttk.Frame(body, padding=8, width=330)
        body.add(left, weight=0)
        body.add(center, weight=1)
        body.add(right, weight=0)

        # Panel izquierdo: configuración, bloqueos y ruta calculada.
        ttk.Label(left, text="CONFIGURACION", style="Section.TLabel").pack(anchor="w", pady=(2, 8))
        ttk.Label(left, text="Semilla").pack(anchor="w")
        ttk.Entry(left, textvariable=self.seed).pack(fill=tk.X, pady=(0, 7))
        nodes = list(self.graph.nodes)
        ttk.Label(left, text="Origen").pack(anchor="w")
        self.source = ttk.Combobox(left, values=nodes, state="readonly")
        self.source.set("PC01")
        self.source.pack(fill=tk.X, pady=(0, 7))
        ttk.Label(left, text="Destino").pack(anchor="w")
        self.target = ttk.Combobox(left, values=nodes, state="readonly")
        self.target.set("SRV02")
        self.target.pack(fill=tk.X, pady=(0, 7))
        ttk.Label(left, text="Parametro principal").pack(anchor="w")
        ttk.Label(left, text="Latencia (70 %)", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 7))
        ttk.Separator(left).pack(fill=tk.X, pady=8)
        ttk.Checkbutton(left, text="Bloquear nodo con mouse", variable=self.block_mouse).pack(anchor="w", pady=(0, 5))
        ttk.Button(left, text="Bloquear nodo seleccionado", command=self.toggle_selected_node, style="Secondary.TButton").pack(fill=tk.X, pady=3)
        ttk.Button(left, text="Bloqueo aleatorio", command=self.random_block, style="Secondary.TButton").pack(fill=tk.X, pady=3)
        ttk.Button(left, text="Bloquear conexion seleccionada", command=self.toggle_selected_edge, style="Secondary.TButton").pack(fill=tk.X, pady=3)
        ttk.Label(left, text="Ruta calculada", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 3))
        self.route_text = tk.Text(left, height=17, width=28, wrap=tk.WORD, bg=self.PANEL, fg=self.TEXT, relief=tk.FLAT, padx=8, pady=8, font=("Consolas", 9))
        self.route_text.pack(fill=tk.BOTH, expand=True)

        # Centro: grafo grande, igual que en la disposición original.
        self.figure = Figure(figsize=(9, 7), dpi=100, facecolor=self.BG)
        self.axis = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=center)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_graph_click)

        # Panel derecho: detalles arriba, estado GNN compacto y autómata abajo.
        ttk.Label(right, text="DETALLES", style="Section.TLabel").pack(anchor="w", pady=(2, 5))
        self.details = tk.Text(right, height=13, width=38, wrap=tk.WORD, bg=self.PANEL, fg=self.TEXT, relief=tk.FLAT, padx=9, pady=9, font=("Segoe UI", 9))
        self.details.pack(fill=tk.BOTH, expand=True)

        self.probability_frame = ttk.Frame(right)
        self.probability_frame.pack(fill=tk.X, pady=(4, 4))
        ttk.Label(self.probability_frame, text="Probabilidad de anomalia").pack(anchor="w")
        self.anomaly_probability = tk.DoubleVar(value=0)
        self.anomaly_bar = ttk.Progressbar(self.probability_frame, variable=self.anomaly_probability, maximum=100, mode="determinate")
        self.anomaly_bar.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=(0, 6))
        self.anomaly_probability_label = ttk.Label(self.probability_frame, text="0.0 %")
        self.anomaly_probability_label.pack(side=tk.RIGHT)

        gnn_card = ttk.Frame(right, style="Card.TFrame", padding=10)
        gnn_card.pack(fill=tk.X, pady=(4, 9))
        ttk.Label(gnn_card, text="ESTADO DE LA GNN", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 6))
        self.gnn_model_status = ttk.Label(gnn_card, text="○ Modelo no cargado", style="Pending.TLabel")
        self.gnn_model_status.pack(fill=tk.X, pady=2)
        self.gnn_classification_status = ttk.Label(gnn_card, text="○ Escenario sin clasificar", style="Pending.TLabel")
        self.gnn_classification_status.pack(fill=tk.X, pady=2)
        self.gnn_summary = ttk.Label(gnn_card, text="Normal: 0 | Anomalo: 0 | Bloqueado: 0", style="CardText.TLabel")
        self.gnn_summary.pack(anchor="w", pady=(7, 1))

        ttk.Label(right, text="AUTOMATA DINAMICO", style="Section.TLabel").pack(anchor="w", pady=(2, 2))
        self.auto_state = ttk.Label(right, text="Estado actual: q0 - En espera", font=("Segoe UI", 10, "bold"))
        self.auto_state.pack(anchor="w", pady=3)
        self.automata_result = ttk.Label(right, text="○ Comunicacion pendiente", style="Pending.TLabel")
        self.automata_result.pack(fill=tk.X, pady=(0, 4))
        self.auto_text = tk.Text(right, height=8, width=38, wrap=tk.WORD, bg=self.PANEL, fg=self.TEXT, relief=tk.FLAT, padx=8, pady=8, font=("Consolas", 9))
        self.auto_text.pack(fill=tk.X)
        buttons = ttk.Frame(right)
        buttons.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(buttons, text="Pausar", command=self.pause_automata, style="Secondary.TButton").pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(buttons, text="Siguiente", command=self.next_automata_step, style="Secondary.TButton").pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(buttons, text="Reiniciar", command=self.reset_automata, style="Secondary.TButton").pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self.status = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status, relief=tk.SUNKEN, anchor="w", padding=5).pack(side=tk.BOTTOM, fill=tk.X)

    def _set_details_placeholder(self):
        self.details.delete("1.0", tk.END)
        self.details.insert(tk.END, "Seleccione un dispositivo o una conexion del grafo para consultar sus atributos.")
        self.anomaly_probability.set(0)
        self.anomaly_probability_label.configure(text="0.0 %")

    def generate(self):
        self.graph = build_graph(self.seed.get())
        self.path = []
        self.metrics = {}
        self.selected_node = None
        self.selected_edge = None
        self.restore_controls()
        self.mark_classification_pending(new_scenario=True)
        self.route_text.delete("1.0", tk.END)
        self._set_details_placeholder()
        self.draw_graph()
        self.reset_automata()
        self._update_status(f"Escenario generado con semilla {self.seed.get()}. Clasifique nuevamente la red.")

    def restore_controls(self):
        nodes = list(self.graph.nodes)
        self.source["values"] = nodes
        self.target["values"] = nodes

    def calculate_route(self):
        try:
            self.path, self.metrics = shortest_route(self.graph, self.source.get(), self.target.get())
            self.route_text.delete("1.0", tk.END)
            self.route_text.insert(tk.END, " -> ".join(self.path) + "\n\n")
            self.route_text.insert(
                tk.END,
                f"Latencia total: {self.metrics['latencia']} ms\n"
                f"Peso: {self.metrics['peso']:.4f}\n"
                f"Saltos: {self.metrics['saltos']}\n"
                f"Trafico promedio: {self.metrics['trafico_promedio']:.1f} %\n"
                f"Perdida promedio: {self.metrics['perdida_promedio']:.2f} %",
            )
            self._update_status("Ruta eficiente calculada con Dijkstra y peso ponderado por latencia.")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self.path = []
            self.metrics = {}
            self.route_text.delete("1.0", tk.END)
            self.route_text.insert(tk.END, "No existe una ruta activa entre los dispositivos seleccionados.")
            self._update_status("No existe ruta disponible.")
        self.draw_graph()

    def draw_graph(self):
        self.axis.clear()
        self.axis.set_facecolor("#F8FAFC")
        self.figure.patch.set_facecolor(self.BG)
        normal_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if not d["bloqueada"]]
        blocked_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d["bloqueada"]]
        nx.draw_networkx_edges(self.graph, self.position, edgelist=normal_edges, ax=self.axis, edge_color="#A4B3C2", width=1.45, alpha=0.85)
        nx.draw_networkx_edges(self.graph, self.position, edgelist=blocked_edges, ax=self.axis, edge_color="#6D747C", style="dashed", width=2.5, alpha=0.95)
        if len(self.path) > 1:
            nx.draw_networkx_edges(self.graph, self.position, edgelist=list(zip(self.path, self.path[1:])), ax=self.axis, edge_color="#0078D4", width=5.5, alpha=1.0)
        labels = {}
        for u, v, data in self.graph.edges(data=True):
            if self.show_metrics.get():
                labels[(u, v)] = f"L:{data['latencia']}ms\nT:{data['trafico']}% P:{data['perdida']:.1f}%\nW:{data['peso']:.3f}"
            else:
                labels[(u, v)] = f"{data['latencia']} ms"
        nx.draw_networkx_edge_labels(
            self.graph,
            self.position,
            edge_labels=labels,
            ax=self.axis,
            font_size=6,
            label_pos=0.5,
            rotate=False,
            bbox={"boxstyle": "round,pad=.15", "fc": "white", "ec": "none", "alpha": 0.80},
        )
        for node, (x, y) in self.position.items():
            data = self.graph.nodes[node]
            state = "Bloqueado" if data["bloqueado"] else data["prediccion"]
            border = "#1677D2" if node == self.selected_node else STATE_COLORS[state]
            self.axis.scatter([x], [y], s=1150, facecolors="white", edgecolors=border, linewidths=4, zorder=4)
            image = OffsetImage(self.icons[data["tipo"]], zoom=0.055)
            self.axis.add_artist(AnnotationBbox(image, (x, y), frameon=False, zorder=5))
            self.axis.text(x, y - 0.46, node, ha="center", va="top", fontsize=8, fontweight="bold", color="#1D2A39", zorder=6)
            if data["bloqueado"]:
                self.axis.text(x + 0.25, y + 0.25, "X", ha="center", va="center", fontsize=13, fontweight="bold", color="#C0392B", zorder=7)
        self.axis.set_xlim(-5, 5)
        self.axis.set_ylim(-1.2, 3.7)
        self.axis.axis("off")
        self.axis.set_title("Latencia visible por arista | Azul: ruta Dijkstra | Verde/Rojo/Gris: estado", fontsize=11, fontweight="bold", color=self.DARK)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def on_graph_click(self, event):
        if event.xdata is None:
            return
        node = min(self.position, key=lambda n: math.hypot(self.position[n][0] - event.xdata, self.position[n][1] - event.ydata))
        distance = math.hypot(self.position[node][0] - event.xdata, self.position[node][1] - event.ydata)
        if distance < 0.38:
            self.selected_node = node
            self.selected_edge = None
            if self.block_mouse.get():
                self.graph.nodes[node]["bloqueado"] = not self.graph.nodes[node]["bloqueado"]
                self.path = []
                self.mark_classification_pending()
            self.show_node_details(node)
            self.draw_graph()
            return
        edge = self.closest_edge(event.xdata, event.ydata)
        if edge:
            self.selected_edge = edge
            self.selected_node = None
            self.show_edge_details(edge)
            self.draw_graph()

    def closest_edge(self, x, y):
        best = None
        best_dist = 0.16
        for u, v in self.graph.edges:
            x1, y1 = self.position[u]
            x2, y2 = self.position[v]
            dx, dy = x2 - x1, y2 - y1
            if dx == dy == 0:
                continue
            t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)))
            px, py = x1 + t * dx, y1 + t * dy
            dist = math.hypot(x - px, y - py)
            if dist < best_dist:
                best = (u, v)
                best_dist = dist
        return best

    def show_node_details(self, node):
        data = self.graph.nodes[node]
        state = "Bloqueado" if data["bloqueado"] else data["prediccion"]
        probability = float(data["probabilidad"]) * 100
        text = (
            f"DISPOSITIVO\n\nID: {node}\nTipo: {data['tipo']}\n"
            f"Estado visual: {state}\nPrediccion: {data['prediccion']}\n"
            f"Probabilidad de anomalia: {probability:.1f} %\n\n"
            f"CPU: {data['cpu']} %\nMemoria: {data['memoria']} %\n"
            f"Trafico: {data['trafico']} %\nFallos: {data['fallos']}\n"
            f"Temperatura: {data['temperatura']} C\nBloqueado: {'Si' if data['bloqueado'] else 'No'}"
        )
        self.details.delete("1.0", tk.END)
        self.details.insert(tk.END, text)
        self.anomaly_probability.set(probability)
        self.anomaly_probability_label.configure(text=f"{probability:.1f} %")

    def show_edge_details(self, edge):
        u, v = edge
        data = self.graph.edges[u, v]
        text = (
            f"CONEXION\n\nOrigen: {u}\nDestino: {v}\n"
            f"Latencia: {data['latencia']} ms\nTrafico: {data['trafico']} %\n"
            f"Perdida: {data['perdida']:.2f} %\nAncho de banda: {data['ancho_banda']} Mbps\n"
            f"Peso Dijkstra: {data['peso']:.4f}\nEstado: {'Bloqueada' if data['bloqueada'] else 'Activa'}\n\n"
            "Formula:\nW = 0.70 Ln + 0.20 Tn + 0.10 Pn"
        )
        self.details.delete("1.0", tk.END)
        self.details.insert(tk.END, text)
        self.anomaly_probability.set(0)
        self.anomaly_probability_label.configure(text="N/A")

    def toggle_selected_node(self):
        if not self.selected_node:
            messagebox.showinfo("Seleccion", "Seleccione un dispositivo sobre el grafo.")
            return
        data = self.graph.nodes[self.selected_node]
        data["bloqueado"] = not data["bloqueado"]
        self.path = []
        self.mark_classification_pending()
        self.show_node_details(self.selected_node)
        self.draw_graph()

    def random_block(self):
        import random
        candidates = [n for n, d in self.graph.nodes(data=True) if not d["bloqueado"]]
        if candidates:
            self.selected_node = random.choice(candidates)
            self.graph.nodes[self.selected_node]["bloqueado"] = True
            self.path = []
            self.mark_classification_pending()
            self.show_node_details(self.selected_node)
            self.draw_graph()

    def toggle_selected_edge(self):
        if not self.selected_edge:
            messagebox.showinfo("Seleccion", "Seleccione una conexion haciendo clic cerca de la linea.")
            return
        data = self.graph.edges[self.selected_edge]
        data["bloqueada"] = not data["bloqueada"]
        self.path = []
        self.mark_classification_pending()
        self.show_edge_details(self.selected_edge)
        self.draw_graph()

    def restore(self):
        for _, data in self.graph.nodes(data=True):
            data["bloqueado"] = False
        for _, _, data in self.graph.edges(data=True):
            data["bloqueada"] = False
        self.path = []
        self.selected_node = None
        self.selected_edge = None
        self.mark_classification_pending()
        self._set_details_placeholder()
        self.draw_graph()
        self._update_status("Red restaurada. Vuelva a calcular y clasificar.")

    def start_automata(self):
        if not self.path:
            self.calculate_route()
        sequence = sequence_for_path(self.path)
        result = run(sequence)
        self.automata_trace = result["trace"]
        self.automata_accepted = result["accepted"]
        self.automata_step = 0
        self.auto_text.delete("1.0", tk.END)
        self.auto_text.insert(tk.END, f"Cadena: {sequence}\nER: {REGEX}\n\n")
        self.automata_result.configure(text="○ Comunicacion en ejecucion", style="Pending.TLabel")
        self.next_automata_step(auto=True)

    def next_automata_step(self, auto=False):
        if self.automata_step >= len(self.automata_trace):
            self._finish_automata_indicator()
            return
        symbol, state = self.automata_trace[self.automata_step]
        self.auto_state.configure(text=f"Estado actual: {state} - {STATES[state]}")
        if symbol is None:
            self.auto_text.insert(tk.END, f"Inicio en {state}\n")
        else:
            self.auto_text.insert(tk.END, f"--{symbol}--> {state} ({STATES[state]})\n")
        self.auto_text.see(tk.END)
        self.automata_step += 1
        if self.automata_step >= len(self.automata_trace):
            self._finish_automata_indicator()
        elif auto:
            self.automata_job = self.root.after(650, lambda: self.next_automata_step(auto=True))

    def _finish_automata_indicator(self):
        if not self.automata_trace:
            return
        final_state = self.automata_trace[-1][1]
        if final_state == "q4":
            self.automata_result.configure(text="● Comunicacion aceptada", style="Success.TLabel")
        elif final_state == "q5":
            self.automata_result.configure(text="● Comunicacion bloqueada", style="Pending.TLabel")
        else:
            self.automata_result.configure(text="● Comunicacion rechazada por error", style="Error.TLabel")

    def pause_automata(self):
        if self.automata_job:
            self.root.after_cancel(self.automata_job)
            self.automata_job = None
            self._update_status("Automata pausado.")

    def reset_automata(self):
        self.pause_automata()
        self.automata_trace = []
        self.automata_step = 0
        self.automata_accepted = None
        self.auto_text.delete("1.0", tk.END)
        self.auto_state.configure(text="Estado actual: q0 - En espera")
        self.automata_result.configure(text="○ Comunicacion pendiente", style="Pending.TLabel")

    def train_gnn_async(self):
        self.gnn_model_status.configure(text="● Entrenando GNN...", style="Pending.TLabel")
        self.gnn_classification_status.configure(text="○ Clasificacion pendiente", style="Pending.TLabel")
        self._update_status("Entrenamiento GNN iniciado en segundo plano.")

        def worker():
            try:
                from .model.dataset import generate
                from .model.trainer import load, train

                dataset = generate(150, self.seed.get())

                def progress(row):
                    self.root.after(
                        0,
                        lambda r=row: self.gnn_model_status.configure(
                            text=f"● Entrenando: epoca {r['epoch']} | F1 {r['val_f1']:.3f}",
                            style="Pending.TLabel",
                        ),
                    )

                _, report, path = train(
                    dataset,
                    Path(__file__).resolve().parents[1] / "models",
                    150,
                    0.01,
                    self.seed.get(),
                    progress,
                )
                self.gnn_model, self.gnn_device = load(path)
                self.gnn_report = report
                test = report["test"]
                self.root.after(
                    0,
                    lambda: self.gnn_model_status.configure(
                        text=f"● Modelo entrenado | Acc {test['accuracy']:.3f} | F1 {test['f1']:.3f}",
                        style="Success.TLabel",
                    ),
                )
                self.root.after(0, self.classify_gnn)
            except Exception as exc:
                self.root.after(0, lambda e=exc: messagebox.showerror("GNN", str(e)))
                self.root.after(0, lambda: self.gnn_model_status.configure(text="● Error de entrenamiento", style="Error.TLabel"))

        threading.Thread(target=worker, daemon=True).start()

    def load_gnn(self):
        try:
            from .model.trainer import load

            path = Path(__file__).resolve().parents[1] / "models" / "gcn_best.pt"
            self.gnn_model, self.gnn_device = load(path)
            self.gnn_model_status.configure(text="● Modelo GNN cargado", style="Success.TLabel")
            self.gnn_classification_status.configure(text="○ Escenario pendiente de clasificacion", style="Pending.TLabel")
            self.scenario_classified = False
            self._update_status("Modelo GNN cargado. Presione Clasificar GNN.")
        except Exception as exc:
            self.gnn_model_status.configure(text="● Error al cargar el modelo", style="Error.TLabel")
            messagebox.showerror("GNN", f"No se pudo cargar el modelo.\n\n{exc}")

    def classify_gnn(self):
        if self.gnn_model is None:
            messagebox.showinfo("GNN", "Entrene o cargue el modelo antes de clasificar.")
            return
        try:
            from .model.trainer import predict

            predictions = predict(self.gnn_model, self.gnn_device, self.graph)
            normales = 0
            anomalos = 0
            bloqueados = 0
            for node, result in predictions.items():
                classification = int(result["class"])
                self.graph.nodes[node]["prediccion"] = "Anomalo" if classification == 1 else "Normal"
                self.graph.nodes[node]["probabilidad"] = float(result["probability"])
                if self.graph.nodes[node]["bloqueado"]:
                    bloqueados += 1
                elif classification == 1:
                    anomalos += 1
                else:
                    normales += 1
            total = normales + anomalos + bloqueados
            normal_pct = normales / total * 100 if total else 0
            anomaly_pct = anomalos / total * 100 if total else 0
            self.gnn_classification_status.configure(text="● Escenario clasificado correctamente", style="Success.TLabel")
            self.gnn_summary.configure(
                text=f"Normal: {normales} ({normal_pct:.0f} %) | Anomalo: {anomalos} ({anomaly_pct:.0f} %) | Bloqueado: {bloqueados}"
            )
            self.scenario_classified = True
            self.draw_graph()
            if self.selected_node:
                self.show_node_details(self.selected_node)
            self._update_status("Clasificacion completada con la GCN real.")
        except Exception as exc:
            self.gnn_classification_status.configure(text="● Error durante la clasificacion", style="Error.TLabel")
            messagebox.showerror("GNN", f"No se pudo clasificar el escenario.\n\n{exc}")

    def mark_classification_pending(self, new_scenario=False):
        self.scenario_classified = False
        if self.gnn_model is not None:
            text = "○ Nuevo escenario pendiente de clasificacion" if new_scenario else "○ Red modificada, vuelva a clasificar"
            self.gnn_classification_status.configure(text=text, style="Pending.TLabel")
        else:
            self.gnn_classification_status.configure(text="○ Escenario sin clasificar", style="Pending.TLabel")
        self.gnn_summary.configure(text="Normal: 0 | Anomalo: 0 | Bloqueado: 0")

    def export(self):
        summary = {
            "semilla": self.seed.get(),
            "origen": self.source.get(),
            "destino": self.target.get(),
            "ruta": self.path,
            "metricas": self.metrics,
            "formula_peso": WEIGHT_FACTORS,
            "escenario_clasificado": self.scenario_classified,
        }
        folder = export_scenario(self.graph, summary, self.figure, self.seed.get())
        messagebox.showinfo("Exportacion", f"Resultados guardados en:\n{folder}")

    def _update_status(self, text):
        self.status.set(text)
