"""Error de los parámetros por bootstrap sobre los grupos de posición idéntica de train.

Cada remuestreo reajusta el modelo arrancando del óptimo con todos los datos, y el
error es la desviación típica entre remuestreos. El Ising es lento de reajustar
(verosimilitud exacta), así que lleva menos remuestreos.

Uso: python bootstrap.py   (~30 min) -> resultados/bootstrap.json
"""

import json
import time
import numpy as np

from datos import cargar_dataset, cargar_split, grupos_por_posicion, remuestras_por_grupos
from modelo import MODELOS, entrenar, cargar_modelo, a_tensores

N_BOOT = {"sin_acoplamiento": 100, "ising": 10}

if __name__ == "__main__":
    Xp, Yp, t = cargar_dataset()
    train, _ = cargar_split()
    grupos_train = grupos_por_posicion(np.load("go9_dataset.npz")["X"])[train]
    rng = np.random.default_rng(0)

    resultado = {}
    for nombre in MODELOS:
        t0 = time.time()
        optimo = cargar_modelo(nombre)
        muestras = []
        for r in remuestras_por_grupos(grupos_train, N_BOOT[nombre], rng):
            mod = entrenar(nombre, *a_tensores(Xp, Yp, train[r]), inicio=optimo)
            muestras.append(np.concatenate([mod.w.detach(), mod.b.detach(), mod.J.detach()]))
        sd = np.std(muestras, axis=0, ddof=1)
        resultado[nombre] = {"sd_w": sd[:6].tolist(), "sd_b": sd[6:21].tolist(), "sd_J": sd[21:].tolist()}
        print(f"{nombre}: {N_BOOT[nombre]} remuestreos en {time.time() - t0:.0f} s, sd_J = {np.round(sd[21:], 4)}")

    with open("resultados/bootstrap.json", "w") as f:
        json.dump(resultado, f, indent=1)
