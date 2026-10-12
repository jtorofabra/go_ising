"""Convierte las partidas SGF de KataGo en matrices numpy.

De cada partida guarda:
  X         tablero a mitad de partida (+1 negra, -1 blanca, 0 vacío)
  Y         dueño final de cada punto (+1 negras, -1 blancas, 0 neutral)
  turno     quién va a jugar en la posición X (+1 negras, -1 blancas)
  resultado resultado del SGF, p.ej. "W+6"
  komi      komi de esa partida

Necesita analysis.cfg (con reportAnalysisWinratesAs = BLACK) y red.bin.gz en esta carpeta.

Uso: python extraer.py
"""

import glob
import json
import re
import subprocess
import numpy as np
from sgfmill import sgf, sgf_moves

N = 9
CARPETA = "partidas_9x9"
SALIDA = "go9_dataset.npz"
LETRAS = "ABCDEFGHJ"  # coordenadas GTP: en Go se salta la I


def tablero_a_matriz(board):
    # +1 negra, -1 blanca, 0 vacío
    M = np.zeros((N, N), dtype=np.int8)
    for color, (r, c) in board.list_occupied_points():
        M[r, c] = 1 if color == "b" else -1
    return M


def dueno_directo(M):
    # conteo por área del tablero TAL CUAL: piedras de su color, huecos del color
    # que los rodea (0 si tocan los dos)
    Y = M.copy()
    visto = np.zeros_like(M, dtype=bool)
    for r in range(N):
        for c in range(N):
            if M[r, c] != 0 or visto[r, c]:
                continue
            region, bordes, pila = [], set(), [(r, c)]
            visto[r, c] = True
            while pila:
                i, j = pila.pop()
                region.append((i, j))
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = i + di, j + dj
                    if 0 <= a < N and 0 <= b < N:
                        if M[a, b] == 0 and not visto[a, b]:
                            visto[a, b] = True
                            pila.append((a, b))
                        elif M[a, b] != 0:
                            bordes.add(M[a, b])
            dueno = bordes.pop() if len(bordes) == 1 else 0
            for (i, j) in region:
                Y[i, j] = dueno
    return Y


def leer_partidas(carpeta):
    # KataGo escribe un archivo por hilo con varias partidas, una por línea
    for path in sorted(glob.glob(f"{carpeta}/*.sgf*")):
        with open(path, "rb") as f:
            for linea in f.read().splitlines():
                if linea.strip().startswith(b"("):
                    yield sgf.Sgf_game.from_bytes(linea)


def cuadra(Yf, res, komi):
    # True si la cuenta por área da el mismo margen que el resultado del SGF
    m = re.match(r"([BW])\+([\d.]+)", str(res))
    if m is None:
        return False  # empates o resultados raros: no se pueden verificar
    margen = float(m.group(2)) * (1 if m.group(1) == "B" else -1)
    return abs((Yf.sum() - komi) - margen) < 1e-6


# --- ETIQUETAS: CÓMO SE DECIDE EL DUEÑO FINAL -------------------------------------
#
# 1. Primero contamos el tablero final tal cual (dueno_directo). Si la cuenta
#    da exactamente el resultado del SGF, no quedaron piedras muertas y esa
#    etiqueta es EXACTA. Es lo que pasa en la mayoría de partidas.
#
# 2. Si no cuadra, es porque quedaron piedras muertas: el motor pasó sin
#    capturarlas porque sabe que están muertas, y al puntuar KataGo retira las
#    que están dentro del territorio "pass-alive" del rival. Contándolas como
#    vivas le daríamos esos puntos al perdedor.
#
#    En esas partidas, en vez de descartarlas (lo que sesgaría el dataset hacia
#    partidas "limpias", que suelen ser las más igualadas), le preguntamos a
#    KataGo el dueño de cada punto en la posición final. Para una partida
#    terminada su estimación es prácticamente ±1 y ya trata las piedras muertas
#    como del rival.
#
# Así solo molestamos a KataGo en las partidas que lo necesitan (mucho más rápido).
# ----------------------------------------------------------------------------------

katago = None  # solo se arranca si alguna partida lo necesita


def arrancar_katago():
    # lanzo KataGo en modo análisis una sola vez (tarda unos segundos en cargar la red)
    print("arrancando KataGo para las partidas con piedras muertas...", flush=True)
    return subprocess.Popen(
        ["katago", "analysis", "-config", "analysis.cfg", "-model", "red.bin.gz"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1,
    )


def ownership_katago(jugadas, komi, ident):
    # le paso la partida entera y le pido de quién es cada punto al final
    global katago
    if katago is None:
        katago = arrancar_katago()

    moves = [["B" if col == "b" else "W", f"{LETRAS[c]}{r + 1}"] for col, (r, c) in jugadas]
    query = {
        "id": str(ident),
        "moves": moves,
        "rules": {"ko": "POSITIONAL", "scoring": "AREA", "tax": "NONE",
                  "suicide": False, "hasButton": False},  # las mismas reglas que en match.cfg
        "komi": komi,
        "boardXSize": N, "boardYSize": N,
        "includeOwnership": True,
        "maxVisits": 10,  # partida terminada: la red ve directamente qué está vivo y qué muerto
    }
    katago.stdin.write(json.dumps(query) + "\n")
    katago.stdin.flush()  # me aseguro de que la consulta sale ya

    while True:  # leo hasta encontrar la respuesta de esta posición
        linea = katago.stdout.readline()
        if not linea:
            raise RuntimeError("KataGo se ha cerrado; mira la carpeta analysis_logs/")
        resp = json.loads(linea)
        if resp.get("id") != str(ident):
            continue
        if "error" in resp:
            raise RuntimeError(f"KataGo devolvió un error: {resp}")
        if "ownership" in resp:
            break

    own = np.array(resp["ownership"]).reshape(N, N)
    own = own[::-1]  # KataGo empieza por la fila de arriba; la giro para que la fila 0 sea la de abajo
    # redondeo: > 0.5 negras, < -0.5 blancas, en medio neutral
    return np.where(own > 0.5, 1, np.where(own < -0.5, -1, 0)).astype(np.int8)


if __name__ == "__main__":
    X, Y, turnos, resultados, komis = [], [], [], [], []
    descartadas, directo_fallaba = 0, 0
    for n, game in enumerate(leer_partidas(CARPETA)):
        board, jugadas = sgf_moves.get_setup_and_moves(game)
        jugadas = [(color, mov) for color, mov in jugadas if mov is not None]  # fuera los pases
        if len(jugadas) < 10:
            continue  # partida rara o vacía

        mitad = len(jugadas) // 2
        for k, (color, (r, c)) in enumerate(jugadas):
            if k == mitad:
                x_mitad = tablero_a_matriz(board)
                turno = 1 if color == "b" else -1  # quién va a jugar ahora
            board.play(r, c, color)  # aplica la jugada y las capturas

        res, komi = game.get_root().get("RE"), game.get_komi()
        y_directo = dueno_directo(tablero_a_matriz(board))
        if cuadra(y_directo, res, komi):
            Yf = y_directo  # tablero final limpio: la cuenta directa ya es exacta
        else:
            directo_fallaba += 1  # quedaron piedras muertas: le pregunto a KataGo
            Yf = ownership_katago(jugadas, komi, n)

        # control de calidad: si ni con KataGo cuadra (seki, empate...), fuera
        if not cuadra(Yf, res, komi):
            descartadas += 1
            continue

        X.append(x_mitad)
        turnos.append(turno)
        Y.append(Yf)
        resultados.append(res)
        komis.append(komi)

        if len(X) % 100 == 0:
            print(f"{len(X)} partidas procesadas...", flush=True)

    if katago is not None:
        katago.stdin.close()
        katago.wait()

    np.savez(SALIDA, X=np.array(X), Y=np.array(Y), turno=np.array(turnos),
             resultado=np.array(resultados), komi=np.array(komis))
    total = len(X) + descartadas
    print(f"{len(X)} partidas guardadas en {SALIDA}")
    print(f"{directo_fallaba} de {total} necesitaron a KataGo (piedras muertas en el tablero final)")
    print(f"{descartadas} de {total} descartadas por el control de calidad")
