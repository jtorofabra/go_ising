"""¿Es un imán el Ising ajustado? Barrido de temperatura e histéresis en el tablero vacío.

Se toman los parámetros del Ising entrenado y se les pone temperatura,

    P_T(s) ∝ exp( (sum_i (h_i + H) s_i + sum_<ij> J_ij s_i s_j) / T ),

con h = 0 (sin piedras), así que solo quedan los acoplamientos J. T = 1 es el
modelo entrenado. Con M = sum_i s_i y N = 81:

  - Barrido de T (H = 0): m = <|M|> / N y chi = (<M^2> - <|M|>^2) / (N T).
    Con h = 0, <M> = 0 por simetría, por eso se usa |M|. El pico de chi marca T_c.
  - Histéresis: H sube de -4 a +4 y vuelve a bajar, sin reiniciar las cadenas.
    La remanencia es <M> / N en H = 0 a la vuelta.

Uso: python iman.py   (~5 min) -> resultados/iman.json
"""

import json
import time
import numpy as np
import torch

from modelo import N, cargar_modelo, gibbs

TEMPERATURAS = np.arange(0.25, 4.01, 0.125)
CADENAS, QUEMADO, BARRIDOS = 256, 300, 1000

T_HISTERESIS = [1.0, 2.5]
CAMPOS_H = np.linspace(-4, 4, 41)  # pasos de 0,2; el índice 20 es H = 0
CADENAS_H, BARRIDOS_POR_PASO = 256, 100  # la mitad de los barridos relaja, la otra mitad mide


def barrido_temperatura(mod, g):
    # todas las temperaturas a la vez: una fila de cadenas por T
    beta = torch.tensor(1 / TEMPERATURAS).repeat_interleave(CADENAS)[:, None, None]
    # arranque frío: la mitad de cadenas de cada T todo +1, la otra mitad todo -1
    signo = torch.tensor([1.0, -1.0]).repeat_interleave(CADENAS // 2).repeat(len(TEMPERATURAS))
    s = signo[:, None, None] * torch.ones(len(signo), N, N)
    h = torch.zeros_like(s)
    s = gibbs(mod, s, h, QUEMADO, g, beta)
    abs_M, M2 = torch.zeros(len(s)), torch.zeros(len(s))
    for _ in range(BARRIDOS):
        s = gibbs(mod, s, h, 1, g, beta)
        M = s.sum(dim=(1, 2))
        abs_M += M.abs() / BARRIDOS
        M2 += M ** 2 / BARRIDOS
    abs_M = abs_M.reshape(len(TEMPERATURAS), -1).mean(dim=1).numpy()
    M2 = M2.reshape(len(TEMPERATURAS), -1).mean(dim=1).numpy()
    return {"m": (abs_M / N ** 2).tolist(), "chi": ((M2 - abs_M ** 2) / (N ** 2 * TEMPERATURAS)).tolist()}


def histeresis(mod, g):
    beta = torch.tensor(1 / np.array(T_HISTERESIS)).repeat_interleave(CADENAS_H)[:, None, None]
    s = -torch.ones(len(beta), N, N)  # con H = -4 el estado natural es todo -1
    m = []
    for H in np.concatenate([CAMPOS_H, CAMPOS_H[::-1]]):
        h = torch.full_like(s, H)
        s = gibbs(mod, s, h, BARRIDOS_POR_PASO // 2, g, beta)
        suma = torch.zeros(len(s))
        for _ in range(BARRIDOS_POR_PASO // 2):
            s = gibbs(mod, s, h, 1, g, beta)
            suma += s.sum(dim=(1, 2))
        m.append((suma / (BARRIDOS_POR_PASO // 2) / N ** 2).reshape(len(T_HISTERESIS), -1).mean(dim=1).numpy())
    m = np.array(m).T
    salida = {}
    for k, T in enumerate(T_HISTERESIS):
        ida, vuelta = m[k, :len(CAMPOS_H)], m[k, len(CAMPOS_H):][::-1]  # la vuelta, ordenada por H creciente
        salida[str(T)] = {"ida": ida.tolist(), "vuelta": vuelta.tolist(),
                          "remanencia": float(vuelta[len(CAMPOS_H) // 2])}
    return salida


if __name__ == "__main__":
    mod = cargar_modelo("ising")
    g = torch.Generator().manual_seed(0)

    t0 = time.time()
    temperatura = barrido_temperatura(mod, g)
    T_c = TEMPERATURAS[np.argmax(temperatura["chi"])]
    print(f"barrido de T: pico de chi en T = {T_c:.3f} ({time.time() - t0:.0f} s)")

    t0 = time.time()
    ciclos = histeresis(mod, g)
    for T, r in ciclos.items():
        print(f"histéresis en T = {T}: remanencia {r['remanencia']:+.3f}")
    print(f"({time.time() - t0:.0f} s)")

    with open("resultados/iman.json", "w") as f:
        json.dump({"temperaturas": TEMPERATURAS.tolist(), **temperatura, "H": CAMPOS_H.tolist(),
                   "histeresis": ciclos}, f, indent=1)
