"""GIF de una partida de test entera, jugada a jugada, con la predicción del Ising al lado.

En cada jugada se calcula la predicción exacta para la posición actual, y entre una
jugada y la siguiente se interpola en varios frames para que el cambio se vea suave.
Al final se enseña el dueño real.

Ojo: el modelo se entrenó con posiciones de mitad de partida, así que al principio y al
final de la partida está fuera de lo que ha visto.

Uso: python gif_partida.py   (~1 min) -> game.gif en la carpeta figures/ del post
"""

import io
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sgfmill import sgf_moves

from datos import cargar_split
from dibujo import dibujar_tablero, dibujar_ownership
from extraer import CARPETA, leer_partidas, tablero_a_matriz
from modelo import cargar_modelo

SALIDA = os.path.expanduser("~/Documents/blog_carpeta/blog/posts/is_go_ferromagnetic/figures/game.gif")
JUGADAS_MIN, JUGADAS_MAX = 45, 70  # una partida de longitud normal
FRAMES_POR_JUGADA = 15
MS_POR_FRAME = 30                   # ~33 fps: 0,45 s por jugada
PAUSA_FINAL_MS = 3000
DPI = 100


def prediccion(mod, M, turno):
    # el modelo trabaja en perspectiva del que juega: X' = t·X, y se deshace con m = t·m'
    X = torch.tensor(turno * M[None], dtype=torch.float64)
    return turno * mod.predecir(X)[0].numpy()


def suave(x):
    # interpolación con arranque y frenada suaves (smoothstep)
    return x * x * (3 - 2 * x)


# --- 1. Elegir una partida de test: su posición de mitad de partida está en el test ---
datos = np.load("go9_dataset.npz")
_, test = cargar_split()
en_test = {datos["X"][i].tobytes(): i for i in test}
for game in leer_partidas(CARPETA):
    board, jugadas = sgf_moves.get_setup_and_moves(game)
    jugadas = [(color, mov) for color, mov in jugadas if mov is not None]
    if not JUGADAS_MIN <= len(jugadas) <= JUGADAS_MAX:
        continue
    # repetir la partida guardando cada posición y quién juega a continuación
    posiciones, turnos, ultimas = [tablero_a_matriz(board)], [1], [None]
    for color, (r, c) in jugadas:
        board.play(r, c, color)
        posiciones.append(tablero_a_matriz(board))
        turnos.append(-1 if color == "b" else 1)
        ultimas.append((r, c))
    mitad = posiciones[len(jugadas) // 2]
    if mitad.tobytes() in en_test:
        idx = en_test[mitad.tobytes()]
        break
resultado = datos["resultado"][idx]
dueno_real = datos["Y"][idx]
print(f"partida de test {idx}: {len(jugadas)} jugadas, {resultado}")

# --- 2. Predicción en cada jugada ---
mod = cargar_modelo("ising")
preds = [prediccion(mod, M, t) for M, t in zip(posiciones, turnos)]

# --- 3. Frames: piedras al instante, predicción interpolada hacia la nueva ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.8))
fig.subplots_adjust(left=0.05, right=0.98, top=0.86, bottom=0.06, wspace=0.15)
titulo = fig.suptitle("", fontsize=13)
frames, duraciones = [], []


def guardar_frame(ms):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    buf.seek(0)
    frames.append(Image.open(buf).convert("RGB"))
    duraciones.append(ms)


for k in range(len(posiciones)):
    ax1.clear()
    dibujar_tablero(ax1, posiciones[k], "stones")
    if ultimas[k] is not None:  # marca la última jugada
        r, c = ultimas[k]
        ax1.scatter([c], [r], s=60, color="#eb6834", zorder=3)
    titulo.set_text("empty board" if k == 0 else f"move {k} of {len(jugadas)}")
    anterior = preds[k - 1] if k > 0 else preds[0]
    for f in range(FRAMES_POR_JUGADA):
        a = suave((f + 1) / FRAMES_POR_JUGADA)
        ax2.clear()
        dibujar_ownership(ax2, (1 - a) * anterior + a * preds[k], "Ising prediction")
        guardar_frame(MS_POR_FRAME)
duraciones[-1] = PAUSA_FINAL_MS // 2

# cierre: el dueño real al lado de la última posición
ax2.clear()
dibujar_ownership(ax2, dueno_real, "actual final territory")
titulo.set_text(f"end of the game ({resultado})")
guardar_frame(PAUSA_FINAL_MS)
plt.close(fig)

# --- 4. GIF (paleta común para que no parpadee y pese menos) ---
muestra = frames[:: max(1, len(frames) // 12)]
mosaico = Image.new("RGB", (frames[0].width, frames[0].height * len(muestra)))
for j, fr in enumerate(muestra):
    mosaico.paste(fr, (0, j * fr.height))
paleta = mosaico.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
gif = [fr.quantize(palette=paleta, dither=Image.Dither.NONE) for fr in frames]
gif[0].save(SALIDA, save_all=True, append_images=gif[1:], duration=duraciones, loop=0, optimize=True)
print(f"{len(frames)} frames, {sum(duraciones) / 1000:.0f} s -> {SALIDA}")
