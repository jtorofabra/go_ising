"""Modelo de Ising para el dueño final de cada punto, dado el tablero a mitad de partida.

    P(s | X) ∝ exp( sum_i h_i s_i + sum_<ij> J_ij s_i s_j ),    h = conv2d(X', w) + b

Los parámetros respetan las 8 simetrías del tablero:
  - w: kernel 5×5 con 6 valores, uno por clase de desplazamiento
       (0,0), (0,1), (1,1), (0,2), (1,2), (2,2).
  - b: sesgo 9×9 con 15 valores, uno por órbita de casillas.
  - J: J_orto para los 4 vecinos ortogonales y J_diag para los 4 diagonales.

Dos modelos:
  - sin_acoplamiento: J = 0, cada punto va por libre y m = tanh h (una logística por punto).
  - ising: con J, entrenado por máxima verosimilitud exacta.

Z se calcula exacta por matriz de transferencia: una columna de 9 espines tiene
2^9 = 512 estados y J solo une columnas contiguas. Las predicciones son las
magnetizaciones exactas m_i = d log Z / d h_i.

Uso: python modelo.py   (~5 min) -> resultados/parametros.json
"""

import json
import time
import numpy as np
import torch
import torch.nn.functional as F

from datos import cargar_dataset, cargar_split

torch.set_default_dtype(torch.float64)
N = 9
MODELOS = {"sin_acoplamiento": False, "ising": True}  # nombre -> ¿tiene J?
PARAMETROS = "resultados/parametros.json"


# --- Parámetros simétricos -----------------------------------------------------

def _clase(a, b):
    # desplazamiento (a, b) -> clase bajo rotaciones y reflexiones
    a, b = abs(a), abs(b)
    return min(a, b), max(a, b)


CLASES_W = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2)]
IDX_W = torch.tensor([[CLASES_W.index(_clase(dy, dx)) for dx in range(-2, 3)] for dy in range(-2, 3)])
ORBITAS_B = sorted({_clase(r - 4, c - 4) for r in range(N) for c in range(N)})
IDX_B = torch.tensor([[ORBITAS_B.index(_clase(r - 4, c - 4)) for c in range(N)] for r in range(N)])


def _estados(n):
    # (2^n, n): todos los estados ±1 de una columna de n espines
    k = torch.arange(2 ** n)[:, None]
    return ((k >> torch.arange(n)) & 1).double() * 2 - 1


class ModeloIsing(torch.nn.Module):
    def __init__(self, acoplado):
        super().__init__()
        self.acoplado = acoplado
        self.w = torch.nn.Parameter(torch.zeros(6))
        self.b = torch.nn.Parameter(torch.zeros(15))
        self.J = torch.nn.Parameter(torch.zeros(2), requires_grad=acoplado)  # (J_orto, J_diag)

    def kernel_J(self):
        jo, jd = self.J
        cero = torch.zeros(())
        return torch.stack([torch.stack([jd, jo, jd]),
                            torch.stack([jo, cero, jo]),
                            torch.stack([jd, jo, jd])])

    def campo(self, X):
        # X: (n, 9, 9) -> h: (n, 9, 9)
        h = F.conv2d(X[:, None], self.w[IDX_W][None, None], padding=2)[:, 0]
        return h + self.b[IDX_B]

    def vecinos(self, s):
        # sum_j J_ij s_j, con 0 fuera del tablero
        return F.conv2d(s[:, None], self.kernel_J()[None, None], padding=1)[:, 0]

    def log_Z(self, h, fijos=None):
        """log Z por posición, columna a columna (matriz de transferencia).

        fijos: ±1 en los espines fijados y 0 donde se suma. Da el log de la suma
        restringida a los finales compatibles.
        """
        E = _estados(N)                                      # (512, 9)
        jo, jd = self.J
        intra = jo * (E[:, :-1] * E[:, 1:]).sum(dim=1)       # enlaces dentro de la columna
        entre = jo * E @ E.T + jd * (E[:, :-1] @ E[:, 1:].T + E[:, 1:] @ E[:, :-1].T)  # (512, 512)
        campo = torch.einsum("nfc,kf->nck", h, E) + intra    # (n, columnas, 512)
        if fijos is not None:
            # estados de una columna que contradicen algún espín fijado: peso 0
            choques = (fijos.abs().sum(dim=1)[:, :, None] - torch.einsum("nfc,kf->nck", fijos, E)) / 2
            campo = campo.masked_fill(choques > 0.5, -torch.inf)
        # recursión hacia delante en escala logarítmica, restando máximos para no desbordar
        desp = entre.max()
        T = torch.exp(entre - desp)
        a = campo[:, 0]
        for c in range(1, N):
            amax = a.max(dim=1, keepdim=True).values
            a = torch.log(torch.exp(a - amax) @ T) + amax + desp + campo[:, c]
        return torch.logsumexp(a, dim=1)

    def perdida(self, X, S, mascara):
        """-log P(final real | posición), por punto (los neutrales no cuentan)."""
        h = self.campo(X)
        if not self.acoplado:
            nll = F.binary_cross_entropy_with_logits(2 * h, (S + 1) / 2, reduction="none")
            return (nll * mascara).sum() / mascara.sum()
        return (self.log_Z(h) - self.log_Z(h, fijos=S * mascara)).sum() / mascara.sum()

    def predecir(self, X):
        """Magnetizaciones exactas m_i = <s_i> = d log Z / d h_i."""
        h = self.campo(X).detach()
        if not self.acoplado:
            return torch.tanh(h)
        with torch.enable_grad():
            h.requires_grad_(True)
            m, = torch.autograd.grad(self.log_Z(h).sum(), h)
        return m


@torch.no_grad()
def gibbs(mod, s, h, barridos, g, beta=1.0):
    """barridos de Gibbs sobre los espines s (n, 9, 9) con campo h, a temperatura 1/beta.

    Las 4 subredes (fila par/impar × columna par/impar) no tienen vecinos dentro,
    ni ortogonales ni diagonales, así que cada una se actualiza de golpe.
    """
    filas, cols = torch.arange(N)[:, None] % 2, torch.arange(N)[None, :] % 2
    subredes = [(filas == a) & (cols == b) for a in (0, 1) for b in (0, 1)]
    for _ in range(barridos):
        for sub in subredes:
            p = torch.sigmoid(2 * beta * (h + mod.vecinos(s)))  # P(s_i = +1 | resto)
            s = torch.where(sub, (torch.rand(s.shape, generator=g) < p).double() * 2 - 1, s)
    return s


def a_tensores(Xp, Yp, idx):
    X = torch.tensor(Xp[idx], dtype=torch.float64)
    S = torch.tensor(Yp[idx], dtype=torch.float64)
    return X, S, (S != 0).double()


def cargar_modelo(nombre):
    with open(PARAMETROS) as f:
        p = json.load(f)[nombre]
    mod = ModeloIsing(MODELOS[nombre])
    with torch.no_grad():
        for k in ("w", "b", "J"):
            getattr(mod, k).copy_(torch.tensor(p[k]))
    return mod


def entrenar(nombre, X, S, mascara, inicio=None, max_pasos=500, tol=1e-12):
    """LBFGS con todos los datos a la vez hasta que la pérdida deja de bajar."""
    mod = ModeloIsing(MODELOS[nombre])
    if inicio is not None:
        mod.load_state_dict(inicio.state_dict())
    # ojo: con max_iter=1 LBFGS se atasca lejos del óptimo
    opt = torch.optim.LBFGS([p for p in mod.parameters() if p.requires_grad], lr=1, max_iter=5,
                            history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-10, tolerance_change=1e-14)

    def cierre():
        opt.zero_grad()
        l = mod.perdida(X, S, mascara)
        l.backward()
        return l

    previa = np.inf
    for _ in range(max_pasos):
        actual = opt.step(cierre).item()
        if abs(previa - actual) < tol:
            break
        previa = actual
    return mod


if __name__ == "__main__":
    Xp, Yp, t = cargar_dataset()
    train, test = cargar_split()
    datos_train = a_tensores(Xp, Yp, train)
    datos_test = a_tensores(Xp, Yp, test)

    parametros = {}
    for nombre in MODELOS:
        t0 = time.time()
        mod = entrenar(nombre, *datos_train)
        parametros[nombre] = {k: getattr(mod, k).tolist() for k in ("w", "b", "J")}
        with torch.no_grad():
            p_train, p_test = mod.perdida(*datos_train).item(), mod.perdida(*datos_test).item()
        print(f"{nombre}: {time.time() - t0:.0f} s, pérdida train {p_train:.4f}, test {p_test:.4f}, "
              f"J = {np.round(parametros[nombre]['J'], 3)}")

    with open(PARAMETROS, "w") as f:
        json.dump(parametros, f, indent=1)
