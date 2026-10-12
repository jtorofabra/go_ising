"""Funciones para dibujar tableros y mapas de ownership 9×9.

Orientación: fila 0 = fila 1 del tablero (abajo), como en sgfmill.
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

N = 9
LETRAS = "ABCDEFGHJ"  # en Go se salta la I

# divergente: naranja = blancas (-1), gris = neutral (0), azul = negras (+1)
cmap = LinearSegmentedColormap.from_list("ownership", ["#eb6834", "#f0efec", "#2a78d6"])


def preparar_ejes(ax, titulo):
    ax.set_xticks(range(N), LETRAS)
    ax.set_yticks(range(N), range(1, N + 1))
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(-0.5, N - 0.5)  # fila 0 abajo, como en sgfmill
    ax.set_aspect("equal")
    ax.set_title(titulo)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def dibujar_tablero(ax, M, titulo):
    # líneas del tablero y piedras encima
    ax.set_facecolor("#e3c08d")
    for k in range(N):
        ax.plot([0, N - 1], [k, k], color="#52514e", lw=0.8, zorder=1)
        ax.plot([k, k], [0, N - 1], color="#52514e", lw=0.8, zorder=1)
    filas, cols = np.nonzero(M)
    colores = ["#0b0b0b" if M[r, c] == 1 else "#fcfcfb" for r, c in zip(filas, cols)]
    ax.scatter(cols, filas, s=420, c=colores, edgecolors="#0b0b0b", linewidths=0.8, zorder=2)
    preparar_ejes(ax, titulo)


def dibujar_ownership(ax, M, titulo):
    # vale para el dueño real (-1/0/+1) y para magnetizaciones del modelo (entre -1 y 1)
    im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1, origin="lower")
    for k in np.arange(-0.5, N, 1):  # rejilla fina para separar las casillas
        ax.axhline(k, color="white", lw=1)
        ax.axvline(k, color="white", lw=1)
    preparar_ejes(ax, titulo)
    return im
