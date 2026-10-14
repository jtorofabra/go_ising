# go_ising

Code for the blog post *On the ferromagnetic properties of the game of Go* @ juantorofabra.com .

Read the post **[HERE](https://juantorofabra.com/posts/is_go_ferromagnetic/is_go_ferromagnetic.html)**.

The idea: predict the final owner of every point of a 9×9 Go board from the mid-game position, using an Ising model. The stones act as an external field h and the owners of neighbouring points are coupled through J. Then we check whether the Ising model learned from Go games behaves like a magnet.

The code and its comments are in Spanish.

## Model

Each point is a spin s = ±1 (its final owner, seen from the player to move):

```
P(s | X) ∝ exp( Σ_i h_i s_i + Σ_<ij> J_ij s_i s_j ),     h = w * X + b
```

- `w`: 5×5 kernel over the stones (6 values after symmetry).
- `b`: bias per square (15 values).
- `J`: one value for orthogonal neighbours and one for diagonal ones.

Two models are compared: no coupling (J = 0, every point on its own) and the Ising model. Z is computed exactly with a transfer matrix (512 states per column), so the Ising model is trained by exact maximum likelihood and the predictions are the exact magnetizations. The details are in the annexes of the post.

## How to reproduce

```
python -m venv .venv
source .venv/bin/activate
pip install sgfmill numpy matplotlib torch pillow
```

The games were generated with KataGo v1.18.2 (Homebrew, Metal backend) and the network `kata1-tf2-b10c384-s2941M-d5872M`. The network isn't in the repo: download it from [katagotraining.org](https://katagotraining.org/networks/) and save it as `red.bin.gz` in this folder. Then:

```
katago match -config match.cfg -log-file match.log -sgf-output-dir partidas_9x9/
```

After that, in order:

| Script | What it does | Output | Time |
|---|---|---|---|
| `extraer.py` | Replays every game and saves the mid-game position and the final owner | `go9_dataset.npz` | a few minutes |
| `datos.py` | 80/20 train/test split by groups of identical positions | `resultados/split.json` | seconds |
| `modelo.py` | Trains both models | `resultados/parametros.json` | ~5 min |
| `bootstrap.py` | Parameter errors | `resultados/bootstrap.json` | ~30 min |
| `evaluar.py` | Test accuracy and log-loss | `resultados/metricas.json` | ~1 min |
| `iman.py` | Temperature sweep and hysteresis on the empty board | `resultados/iman.json` | ~5 min |
| `figuras_blog.py` | The figures of the post | the post's `figures/` | ~1 min |
| `gif_partida.py` | GIF of a game, move by move | the post's `figures/game.gif` | ~1 min |

`dibujo.py` has the functions that draw boards and ownership maps. The two figure scripts write straight into the blog folder (`SALIDA` at the top of each one).

## Data

- 6972 self-play games with komi 7.5, area scoring and no resignation, in `partidas_9x9/` (KataGo writes one file per thread, one game per line).
- One position is kept per game, the one at mid-game. In `go9_dataset.npz`: `X` (board: +1 black, −1 white, 0 empty), `Y` (final owner: +1 black, −1 white, 0 neutral), `turno` (player to move), `resultado` (result) and `komi`. Row 0 is board row 1 (bottom).
- Labels: if counting the final board as it is gives the result in the SGF, that count is used. If not, dead stones were left on the board (1047 games, 15 %) and the final owner comes from KataGo in analysis mode. One game didn't match even then and was dropped, which leaves 6971.
- With 50 visits KataGo is almost deterministic and many games share their first half: 6971 positions in only 4686 distinct groups. That's why the split (5535 train, 1436 test) and the bootstraps work with groups of identical positions.
- Everything is seen from the player to move (`X' = t·X`, `Y' = t·Y`). This is done when loading, so the file stays in real colours.

## Results

On the test set, bootstrap errors in brackets:

| | Accuracy per point | Log-loss per point | Log-loss of the whole board |
|---|---|---|---|
| No coupling | 0.808 (5) | 0.452 (7) | 0.452 (7) |
| Ising | 0.798 (5) | 0.453 (9) | 0.224 (2) |

- Point by point the coupling doesn't help, but the log-loss of the whole board is cut in half: the model knows that territory comes in blocks.
- J_orto = 0.77 and J_diag = −0.04: only orthogonal neighbours couple. For reference, the infinite 2D Ising model without field orders above J ≈ 0.44.
- With coupling, `w` becomes very local (orthogonal neighbours get a negative weight) and `b` is small and mostly compatible with zero. Without coupling, `b` is positive over the whole board: that's the advantage of having the move.
- On the empty board the fitted Ising model is a ferromagnet: the susceptibility peaks at T_c ≈ 1.5, and at T = 1 (the trained model) the remanence is 0.95. At T = 2.5 there is no loop.

## Lessons

- The KataGo option is `-config`, not `-config-file`, and `numBots = 1` fails ("no matchups specified").
- PyTorch's LBFGS with `max_iter=1` per step gets stuck far from the optimum; with `max_iter=5` and tight tolerances it converges.
- Training by pseudo-likelihood (each spin given its real neighbours) is much cheaper, but it gives a J that is too large for the joint model, and it predicts worse than no coupling at all. Exact likelihood is needed, and it's only feasible because the board is narrow: on 13×13 there would be 8192 states per column.
