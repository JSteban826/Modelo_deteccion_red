STATES = {
    "q0": "En espera", "q1": "Conectando", "q2": "Autorizado",
    "q3": "Transmitiendo", "q4": "Finalizado", "q5": "Bloqueado", "q6": "Error",
}
ALPHABET = {"c", "a", "t", "f", "b", "e", "r"}
FINAL = {"q4"}
TRANSITIONS = {
    ("q0", "c"): "q1", ("q0", "b"): "q5",
    ("q1", "a"): "q2", ("q1", "b"): "q5", ("q1", "e"): "q6",
    ("q2", "t"): "q3", ("q2", "b"): "q5", ("q2", "e"): "q6",
    ("q3", "t"): "q3", ("q3", "f"): "q4", ("q3", "b"): "q5", ("q3", "e"): "q6",
    ("q4", "r"): "q0", ("q5", "r"): "q0", ("q6", "r"): "q0",
}
REGEX = "cat+f"


def sequence_for_path(path):
    if not path or len(path) < 2:
        return "cab"
    return "ca" + "t" * (len(path) - 1) + "f"


def run(sequence):
    state = "q0"
    trace = [(None, state)]
    for symbol in sequence:
        state = TRANSITIONS.get((state, symbol), "q6")
        trace.append((symbol, state))
    return {"sequence": sequence, "trace": trace, "state": state, "accepted": state in FINAL}
