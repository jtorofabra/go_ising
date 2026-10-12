"""Figuras del post (en inglés), directamente en la carpeta figures/ del blog.

Salen de los resultados ya calculados (modelo.py, bootstrap.py, iman.py), salvo la
del Ising de libro, que es una simulación pequeña aparte.

Uso: python figuras_blog.py
"""

import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datos import cargar_dataset, cargar_split, a_colores_reales
from dibujo import N, LETRAS, cmap, dibujar_tablero, dibujar_ownership
from modelo import cargar_modelo, a_tensores, gibbs, IDX_W, IDX_B

SALIDA = os.path.expanduser("~/Documents/blog_carpeta/blog/posts/is_go_ferromagnetic/figures")
DPI = 150
SEMILLA_PIPELINE = 1  # qué partida sale en el esquema
SEMILLA_EJEMPLOS = 2  # qué partidas salen en las predicciones
TINTA, TINTA_SUAVE, REJILLA = "#262624", "#6b6a64", "#e4e3dc"
AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"axes.edgecolor": TINTA_SUAVE, "axes.labelcolor": TINTA, "xtick.color": TINTA_SUAVE,
                     "ytick.color": TINTA_SUAVE, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10})
os.makedirs(SALIDA, exist_ok=True)


def guardar(fig, nombre):
    fig.savefig(f"{SALIDA}/{nombre}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def anotar(ax, M, fmt, vmax):
    # el valor de cada casilla encima del mapa de color
    for (f, c), v in np.ndenumerate(M):
        ax.text(c, f, fmt(f, c, v), ha="center", va="center", fontsize=10 if M.shape[0] < 9 else 7,
                color="white" if abs(v) > 0.6 * vmax else TINTA)


# --- 1. El Ising de libro: esquema y dos simulaciones, por encima y por debajo de T_c ---
def ising_libro(L, T, barridos, rng):
    # Gibbs en tablero de ajedrez, J = 1, h = 0, bordes periódicos
    s = rng.choice([-1, 1], size=(L, L))
    cuadros = (np.indices((L, L)).sum(axis=0) % 2).astype(bool)
    for _ in range(barridos):
        for sub in (cuadros, ~cuadros):
            vec = np.roll(s, 1, 0) + np.roll(s, -1, 0) + np.roll(s, 1, 1) + np.roll(s, -1, 1)
            p = 1 / (1 + np.exp(-2 * vec / T))
            s = np.where(sub, np.where(rng.random((L, L)) < p, 1, -1), s)
    return s


rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
ax = axes[0]
espines = rng.choice([-1, 1], size=(5, 5))
espines[1:4, 1:4] = 1
for (f, c), v in np.ndenumerate(espines):
    ax.annotate("", xy=(c, f + 0.32 * v), xytext=(c, f - 0.32 * v),
                arrowprops=dict(arrowstyle="-|>", lw=2, color=AZUL if v > 0 else NARANJA, mutation_scale=14))
for df, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # un espín y sus cuatro vecinos, unidos por J
    ax.plot([2, 2 + dc * 0.85], [2, 2 + df * 0.85], color=AQUA, lw=4, alpha=0.5, zorder=0)
ax.text(2.45, 2.55, "J", color=AQUA, fontsize=13, fontweight="bold")
ax.annotate("", xy=(5.3, 3.5), xytext=(5.3, 0.5), arrowprops=dict(arrowstyle="-|>", lw=3, color=TINTA_SUAVE))
ax.text(5.5, 2, "external\nfield h", color=TINTA_SUAVE, va="center")
ax.set_xlim(-0.7, 6.6)
ax.set_ylim(-0.7, 4.7)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("spins on a grid")
T_c = 2 / np.log(1 + np.sqrt(2))
for ax, T, titulo in [(axes[1], 1.5 * T_c, "hot (T > T_c): disordered"),
                      (axes[2], 0.85 * T_c, "cold (T < T_c): spins align on their own")]:
    ax.imshow(ising_libro(64, T, 400, rng), cmap=matplotlib.colors.ListedColormap([NARANJA, AZUL]),
              interpolation="nearest")
    ax.set_title(titulo)
    ax.axis("off")
guardar(fig, "ising_model.png")

# --- datos de test y predicciones ---
Xp, Yp, t = cargar_dataset()
d = np.load("go9_dataset.npz")
_, test = cargar_split()
X, S, mascara = a_tensores(Xp, Yp, test)
ising, sin_J = cargar_modelo("ising"), cargar_modelo("sin_acoplamiento")
m_ising, m_sin_J = ising.predecir(X).numpy(), sin_J.predecir(X).numpy()
h = ising.campo(X).detach().numpy()

# --- 2. Esquema: posición -> campo externo -> espines medios ---
k = int(np.random.default_rng(SEMILLA_PIPELINE).integers(len(test)))
i = test[k]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
dibujar_tablero(axes[0], d["X"][i], "1. starting position")
hr = a_colores_reales(h[k], t[i])
dibujar_ownership(axes[1], hr / np.abs(hr).max(), "2. external magnetic field h")  # escalado a [-1, 1]
for (f, c), v in np.ndenumerate(hr):
    axes[1].text(c, f, f"{v:+.1f}", ha="center", va="center", fontsize=7, color=TINTA)
dibujar_ownership(axes[2], a_colores_reales(m_ising[k], t[i]), "3. expected spins = predicted territory")
for x in (0.355, 0.635):
    fig.text(x, 0.5, "→", fontsize=30, ha="center", va="center", color=TINTA_SUAVE)
fig.text(0.5, 0.02, "blue = black's territory, orange = white's", ha="center", color=TINTA_SUAVE)
guardar(fig, "pipeline.png")

# --- 3. Predicciones frente al final real, en 3 partidas de test al azar ---
rng_ej = np.random.default_rng(SEMILLA_EJEMPLOS)
ejemplos = rng_ej.choice(len(test), size=3, replace=False)
Sn, Mn = S.numpy(), mascara.numpy().astype(bool)


def acierto(m, k):
    return ((np.sign(m[k]) == Sn[k]) & Mn[k]).sum() / Mn[k].sum()


fig, axes = plt.subplots(3, 4, figsize=(16, 12.5))
for fila, k in enumerate(ejemplos):
    i = test[k]
    dibujar_tablero(axes[fila, 0], d["X"][i], f"mid-game position ({'black' if t[i] == 1 else 'white'} to move)")
    dibujar_ownership(axes[fila, 1], d["Y"][i], f"real final territory ({d['resultado'][i]})")
    dibujar_ownership(axes[fila, 2], a_colores_reales(m_sin_J[k], t[i]),
                      f"prediction, no coupling ({acierto(m_sin_J, k):.0%} right)")
    dibujar_ownership(axes[fila, 3], a_colores_reales(m_ising[k], t[i]),
                      f"prediction, Ising ({acierto(m_ising, k):.0%} right)")
guardar(fig, "predictions.png")

# --- 3b. Dos finales sorteados de cada modelo para la primera de esas partidas ---
# las medias se parecen, los sorteos no: sin acoplamiento sale sal y pimienta, con Ising salen bloques
k = ejemplos[0]
g = torch.Generator().manual_seed(SEMILLA_EJEMPLOS)
tiradas_sin_J = np.where(rng_ej.random((2, N, N)) < (1 + m_sin_J[k]) / 2, 1.0, -1.0)
s0 = torch.randint(0, 2, (2, N, N), generator=g).double() * 2 - 1
tiradas_ising = gibbs(ising, s0, ising.campo(X[[k, k]]).detach(), 1000, g).numpy()
fig, axes = plt.subplots(2, 2, figsize=(9, 9.5))
for col in range(2):
    dibujar_ownership(axes[0, col], a_colores_reales(tiradas_sin_J[col], t[test[k]]), f"no coupling, sample {col + 1}")
    dibujar_ownership(axes[1, col], a_colores_reales(tiradas_ising[col], t[test[k]]), f"Ising, sample {col + 1}")
guardar(fig, "samples.png")

# --- 4. Kernel w con y sin acoplamiento, y kernel J ---
P = json.load(open("resultados/parametros.json"))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
vmax = max(np.abs(P[m]["w"]).max() for m in P)
for ax, m, titulo in [(axes[0], "sin_acoplamiento", "stone kernel w, no coupling"),
                      (axes[1], "ising", "stone kernel w, Ising")]:
    W = np.array(P[m]["w"])[IDX_W.numpy()]
    ax.imshow(W, cmap=cmap, vmin=-vmax, vmax=vmax)
    anotar(ax, W, lambda f, c, v: f"{v:+.2f}", vmax)
    ax.set_xticks(range(5), range(-2, 3))
    ax.set_yticks(range(5), range(-2, 3))
    ax.set_title(titulo)
ax = axes[2]
jo, jd = P["ising"]["J"]
K = np.array([[jd, jo, jd], [jo, 0, jo], [jd, jo, jd]])
ax.imshow(K, cmap=cmap, vmin=-vmax, vmax=vmax)
anotar(ax, K, lambda f, c, v: "·" if (f, c) == (1, 1) else f"{v:+.2f}", vmax)
ax.set_xticks(range(3), range(-1, 2))
ax.set_yticks(range(3), range(-1, 2))
ax.set_title("coupling kernel J, Ising")
guardar(fig, "parameters.png")

# --- 4b. Sesgo b con su error bootstrap ---
SD = json.load(open("resultados/bootstrap.json"))
fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
vmax = max(np.abs(P[m]["b"]).max() for m in P)
for ax, m, titulo in [(axes[0], "sin_acoplamiento", "bias b, no coupling"), (axes[1], "ising", "bias b, Ising")]:
    B = np.array(P[m]["b"])[IDX_B.numpy()]
    E = np.array(SD[m]["sd_b"])[IDX_B.numpy()]
    im = ax.imshow(B, cmap=cmap, vmin=-vmax, vmax=vmax, origin="lower")
    anotar(ax, B, lambda f, c, v: f"{v:+.2f}\n±{E[f, c]:.2f}", vmax)
    ax.set_xticks(range(N), LETRAS)
    ax.set_yticks(range(N), range(1, N + 1))
    ax.set_title(titulo)
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="bias towards the player to move")
guardar(fig, "bias.png")

# --- 5. Barrido de temperatura en el tablero vacío (letra más grande desde aquí) ---
plt.rcParams.update({"font.size": 13, "lines.linewidth": 2.5})
IM = json.load(open("resultados/iman.json"))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
a1.plot(IM["temperaturas"], IM["m"], color=TINTA, lw=2, marker="o", ms=3)
a2.plot(IM["temperaturas"], IM["chi"], color=TINTA, lw=2, marker="o", ms=3)
for a in (a1, a2):
    a.axvline(1, color=TINTA_SUAVE, ls=":", lw=1)
    a.text(0.97, 0.5, "our model\n(T = 1)", transform=a.get_xaxis_transform(), ha="right", fontsize=11,
           color=TINTA_SUAVE)
    a.set_xlabel("temperature T")
    a.grid(color=REJILLA, lw=0.8)
a1.set_ylabel("magnetization ⟨|M|⟩ / N")
a2.set_ylabel("susceptibility χ per point")
guardar(fig, "temperature_sweep.png")

# --- 6. Histéresis en el tablero vacío ---
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.axhline(0, color=TINTA_SUAVE, lw=0.8)
ax.axvline(0, color=TINTA_SUAVE, lw=0.8)
for T, color in [("1.0", TINTA), ("2.5", NARANJA)]:
    r = IM["histeresis"][T]
    etiqueta = f"T = {float(T):g}, remanence {abs(r['remanencia']):.2f}"
    ax.plot(IM["H"], r["ida"], color=color, lw=2, label=etiqueta)
    ax.plot(IM["H"], r["vuelta"], color=color, lw=2, ls="--")
ax.set_xlabel("uniform external field H")
ax.set_ylabel("magnetization ⟨M⟩ / N")
ax.set_title("solid: field going up, dashed: going down", fontsize=11, color=TINTA_SUAVE)
ax.grid(color=REJILLA, lw=0.8)
ax.set_xlim(-2, 2)
ax.legend(frameon=False, fontsize=11, loc="upper left", bbox_to_anchor=(0, 0.88))
guardar(fig, "hysteresis.png")

print("figuras del blog en", SALIDA)
