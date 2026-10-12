"""Métricas en test de los dos modelos (sin contar los puntos neutrales).

  - precisión por punto: sign(m) == dueño real.
  - log-loss por punto, con P(s = +1) = (1 + m) / 2. Solo mira las marginales.
  - log-loss conjunta: -log P(tablero final real | posición), por punto. Mide el
    modelo entero, con sus correlaciones; sin acoplamiento coincide con la anterior.

Errores por bootstrap sobre grupos de posición idéntica de test.

Uso: python evaluar.py   (~1 min) -> resultados/metricas.json
"""

import json
import numpy as np
import torch

from datos import cargar_dataset, cargar_split, grupos_por_posicion, remuestras_por_grupos
from modelo import MODELOS, cargar_modelo, a_tensores

N_BOOT = 1000

if __name__ == "__main__":
    Xp, Yp, t = cargar_dataset()
    _, test = cargar_split()
    X, S, mascara = a_tensores(Xp, Yp, test)
    remuestras = remuestras_por_grupos(grupos_por_posicion(np.load("go9_dataset.npz")["X"])[test],
                                       N_BOOT, np.random.default_rng(0))

    metricas = {}
    for nombre in MODELOS:
        mod = cargar_modelo(nombre)
        m = mod.predecir(X)
        with torch.no_grad():
            h = mod.campo(X)
            conjunta = (mod.log_Z(h) - mod.log_Z(h, fijos=S * mascara)).numpy()
        p = ((1 + m) / 2).clamp(1e-6, 1 - 1e-6)
        log_loss = -torch.where(S > 0, p.log(), (1 - p).log())
        # sumas por posición, para poder remuestrear posiciones
        por_posicion = {"precision": ((m.sign() == S) * mascara).sum(dim=(1, 2)).numpy(),
                        "log_loss": (log_loss * mascara).sum(dim=(1, 2)).numpy(),
                        "log_loss_conjunta": conjunta}
        puntos = mascara.sum(dim=(1, 2)).numpy()

        metricas[nombre] = {}
        for k, v in por_posicion.items():
            boot = [v[r].sum() / puntos[r].sum() for r in remuestras]
            metricas[nombre][k] = {"valor": float(v.sum() / puntos.sum()), "se": float(np.std(boot, ddof=1))}
        print(nombre, {k: f"{v['valor']:.3f} ± {v['se']:.3f}" for k, v in metricas[nombre].items()})

    with open("resultados/metricas.json", "w") as f:
        json.dump(metricas, f, indent=1)
