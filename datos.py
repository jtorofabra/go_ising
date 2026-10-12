"""Carga el dataset desde el punto de vista del jugador al que le toca, y el split train/test.

El Go es simétrico al cambiar los colores (salvo el komi), así que en vez de
negras/blancas usamos "el que juega / el rival": basta con multiplicar tablero y
dueño final por el turno t = ±1,

    X' = t * X    (+1 = piedra del que juega, -1 = del rival)
    Y' = t * Y    (+1 = punto del que juega,  -1 = del rival)

go9_dataset.npz se queda en colores reales; a_colores_reales deshace el cambio para dibujar.

Posiciones repetidas: KataGo con 50 visitas es casi determinista, así que muchas
partidas comparten la primera mitad (hay posiciones que salen ~30 veces). Las copias
van juntas a train o a test, y los bootstrap remuestrean grupos de posición idéntica.

Uso: python datos.py   (crea resultados/split.json)
"""

import json
import os
import numpy as np

SPLIT = "resultados/split.json"


def cargar_dataset(path="go9_dataset.npz"):
    """Devuelve (X', Y', t) con tableros y dueños vistos desde el que juega."""
    d = np.load(path)
    t = d["turno"]
    return t[:, None, None] * d["X"], t[:, None, None] * d["Y"], t


def a_colores_reales(m, t):
    """m = t * m'. Vale para un tablero (9, 9) con t escalar o un lote (n, 9, 9) con t de forma (n,)."""
    t = np.asarray(t)
    if t.ndim == 1:
        t = t[:, None, None]
    return t * m


def grupos_por_posicion(X):
    """Número de grupo de cada posición: las posiciones idénticas comparten grupo."""
    _, grupo = np.unique(X.reshape(len(X), -1), axis=0, return_inverse=True)
    return grupo.ravel()


def remuestras_por_grupos(grupos, n_boot, rng):
    """n_boot remuestreos bootstrap de grupos (con reemplazo), como índices de los elementos."""
    unicos, inversa = np.unique(grupos, return_inverse=True)
    miembros = [np.flatnonzero(inversa == g) for g in range(len(unicos))]
    return [np.concatenate([miembros[g] for g in rng.integers(0, len(unicos), len(unicos))])
            for _ in range(n_boot)]


def cargar_split():
    with open(SPLIT) as f:
        s = json.load(f)
    return np.array(s["train"]), np.array(s["test"])


if __name__ == "__main__":
    Xp, Yp, t = cargar_dataset()
    X = np.load("go9_dataset.npz")["X"]

    # 80/20 por grupos de posición idéntica, con semilla fija
    grupos = grupos_por_posicion(X)
    unicos = np.random.default_rng(0).permutation(np.unique(grupos))
    en_test = np.isin(grupos, unicos[:round(0.2 * len(unicos))])
    train, test = np.flatnonzero(~en_test), np.flatnonzero(en_test)
    assert not set(grupos[train]) & set(grupos[test])

    os.makedirs("resultados", exist_ok=True)
    with open(SPLIT, "w") as f:
        json.dump({"train": train.tolist(), "test": test.tolist()}, f)
    print(f"{len(X)} posiciones en {grupos.max() + 1} grupos: {len(train)} train, {len(test)} test")
