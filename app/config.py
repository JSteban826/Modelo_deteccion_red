from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ICON_DIR = ROOT / "assets" / "icons"
RESULTS_DIR = ROOT / "resultados"

DEVICE_COUNTS = {
    "Computador": 6,
    "Servidor": 2,
    "Router": 2,
    "Switch": 2,
    "IoT": 4,
    "Impresora": 2,
    "PuntoAcceso": 2,
}
PREFIX = {
    "Computador": "PC", "Servidor": "SRV", "Router": "RTR",
    "Switch": "SW", "IoT": "IoT", "Impresora": "IMP", "PuntoAcceso": "AP",
}
ICON_FILES = {
    "Computador": ICON_DIR / "pc.png",
    "Servidor": ICON_DIR / "server.png",
    "Router": ICON_DIR / "router.png",
    "Switch": ICON_DIR / "switch.png",
    "IoT": ICON_DIR / "iot.png",
    "Impresora": ICON_DIR / "printer.png",
    "PuntoAcceso": ICON_DIR / "access_point.png",
}
STATE_COLORS = {"Normal": "#2EAF62", "Anomalo": "#D94C4C", "Bloqueado": "#7C8794"}
WEIGHT_FACTORS = {"latencia": 0.70, "trafico": 0.20, "perdida": 0.10}
MAX_VALUES = {"latencia": 200.0, "trafico": 100.0, "perdida": 20.0}
