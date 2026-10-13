# snake-ai

Neuroevolution of Snake-playing agents, with a live dashboard for watching and
browsing the population.

A population of small feed-forward networks plays Snake. Each generation is
scored, the strongest genomes reproduce with mutation, and the process repeats.
There is no gradient descent anywhere: weights are searched, not trained.

```
gen    17  best  22  mean  7.400  eat%  85.0  fit  16494.7  steps  942.0  div 0.219
gen    18  best  31  mean  5.400  eat%  70.0  fit  26049.4  steps  691.2  div 0.227
gen    65  best  49  mean 16.000  eat%  75.0  fit  48100.5  steps 1341.9  div 0.233
```

## Running

```bash
pip install -e .
python -m snake_ai train
```

The dashboard is served at <http://127.0.0.1:8765>. Training runs until
interrupted with Ctrl+C, which finishes the current generation and writes a
checkpoint.

```bash
python -m snake_ai train --generations 200 --open   # bounded run, open a browser
python -m snake_ai train --resume                   # continue the newest checkpoint
python -m snake_ai train --headless --no-hold       # no dashboard, never pause
python -m snake_ai train --workers 1                # serial, for a determinism check
python -m snake_ai inspect                          # describe a checkpoint
python -m snake_ai evaluate                         # score a champion on held-out seeds
python -m snake_ai export                           # write exports/snake_ai.pt
python -m snake_ai export --pick best-average       # export the run's steadiest genome instead
```

Every flag, dashboard shortcut and config knob is in [COMMANDS.md](COMMANDS.md).
How the project got here, and what each change measured, is in
[HISTORY.md](HISTORY.md).

## Layout

```
COMMANDS.md          command reference
HISTORY.md           what changed, and what it measured
TODO.md              open work
snake_ai/
  config.py          every tunable, as frozen dataclasses
  features.py        state -> network input
  space.py           reachable-space measurement
  trainer.py         generation loop, history and checkpoint scheduling
  held_out.py        scoring and comparing champions on held-out seeds
  cli.py             command line entry points
  game/              board geometry, snake body, food, episode runner
  evolution/         individuals, operators, selection, fitness, population
  nn/                activations and the policy network
  persistence/       .npz checkpoints and standalone .pt export
  telemetry/         generation history and console reporting
  web/               dashboard server, hold gate, static frontend
tests/               game rules, features, fitness, evolution, persistence, held-out scoring
models/              the best model so far, as .pt with its .json sidecar
runs/                per-run output (gitignored)
exports/             exported models (gitignored)
```

## How it works

**Actions are relative.** The network chooses turn left, go straight, or turn
right. A reversal is unreachable by construction, so no output is spent on a move
the rules would reject.

**Inputs are egocentric.** All eleven inputs are expressed relative to the
snake's heading, so a rule such as "turn away from a blocked side" is learned
once rather than four times in four world orientations.

| Input | Meaning |
| --- | --- |
| `danger_left/straight/right` | whether that move is immediately fatal |
| `space_left/straight/right` | share of the free board reachable behind that move |
| `food_forward`, `food_lateral` | food offset in body coordinates |
| `food_distance` | normalised Manhattan distance |
| `length_ratio` | how much of the board the snake occupies |
| `starvation_progress` | how close the episode is to timing out |

The space inputs are what let an agent avoid sealing itself into a pocket.
Immediate danger cannot express that: entering a dead end is safe for exactly one
move and fatal several moves later.

They are measured over time rather than against a snapshot. The search runs
outward in move order and lets each body cell clear on the move its segment
leaves, so a snake following its own tail sees open board where a frozen scan
sees a dead end. Two earlier encodings are kept for comparison; on 60 held-out
seeds after 60 generations their champions score 46.2 (`egocentric_v3`), 36.4
(`egocentric_v2`) and 19.6 (`egocentric_v1`), each faster and worse than the one
above it. The best model so far comes from a longer `egocentric_v3` run and
averages 81.5 on the same seeds. It is in `models/`, and HISTORY.md has the details.

**Two rules keep selection honest.**

Every individual is scored only on episodes played in the current generation.
Scores are cleared across the whole population before evaluation, so an elite
that has survived twenty generations enters selection with the same clean slate
as a newborn. Without that, a long-lived elite accumulates far more evaluation
attempts than a newcomer and wins on sampling luck rather than on policy.

A genome's score is the mean of its evaluation episodes, never the best of them.
Taking the best of several trials rewards a favourable food sequence instead of a
better policy.

Both rules have regression tests in `tests/test_evolution.py`.

## Dashboard

- **Progress** — best, mean and feeding rate across every generation of the run.
- **Population** — every agent's episode replayed on its own board, sortable by
  fitness, score or survival. Click an agent for its fitness breakdown.
- **Generation browser** — scrub, step or jump to any retained generation.
  Selecting one detaches the view from the live edge; *Follow latest* reattaches.
- **Training control** — see below.
- **Export model** — see below.

Keyboard: `←`/`→` step generations, `space` play/pause, `c` continue, `l` follow
latest, `esc` close the agent panel.

### Holding at generation boundaries

Evolution runs much faster than anyone can read. While the dashboard is open,
training pauses at each generation boundary so a generation cannot be replaced
before it has been looked at. Pausing happens only between generations, never
mid-episode, so no partial evaluation is ever recorded.

Presence is a lease, refreshed by a heartbeat from the page. If the tab is closed
or the connection drops, the lease expires and training resumes on its own rather
than waiting for a browser that is never coming back. *Run free* disables holding
for the session; `--no-hold` disables it for the whole run.

### Checkpoints and export

Training writes compressed `.npz` checkpoints and never imports torch. A
checkpoint records the feature spec and architecture it was trained under, and
loading refuses a mismatch rather than reshaping the genome to fit.

Exporting a standalone `.pt` model is a separate action from the dashboard's
**Export model** panel. Pick a generation, press *Calculate* to see the
statistics, achievements, parameter count and estimated size — this writes
nothing — then press *Export .pt* to produce `exports/snake_ai.pt`, a timestamped
copy under `exports/archive/`, and a readable `exports/snake_ai.json` sidecar.

Because each generation archives its champion genome, any retained generation can
be exported, not only the ones a checkpoint happened to land on.

The exported archive is self-contained — weights as named tensors, architecture,
input names, action space and training statistics — so running it later needs
numpy or torch, not this project:

```python
import torch, numpy as np

model = torch.load("exports/snake_ai.pt", weights_only=False)
x = np.zeros(model["feature_spec"]["size"], dtype=np.float32)
for i in range(len(model["architecture"]["hidden_layers"]) + 1):
    x = x @ model["parameters"][f"weight_{i}"].numpy() + model["parameters"][f"bias_{i}"].numpy()
    if i < len(model["architecture"]["hidden_layers"]):
        x = np.tanh(x)
turn = model["action_space"][int(np.argmax(x))]
```

`.pt` export requires torch: `pip install -e ".[export]"`. Without it the export
panel says so and the rest of the dashboard is unaffected.

## History retention

Storage is tiered so browsing any generation stays possible without memory
growing with the length of the run:

| Tier | Contents | Retention |
| --- | --- | --- |
| `history.jsonl` | one summary line per generation | every generation, forever |
| `replays/gen_*.json.gz` | replay frames and champion genome | `replay_retention`, records pinned |
| memory ring | most recent generations | `memory_window` |

Record-setting generations are never evicted, so milestones stay reviewable
however long the run goes on.

## Configuration

Everything tunable is in `snake_ai/config.py` as frozen dataclasses. The
values that most affect results:

| Setting | Default | Effect |
| --- | --- | --- |
| `evolution.population_size` | 150 | genomes per generation |
| `evolution.evaluation_episodes` | 4 | episodes averaged per genome |
| `evolution.crossover_rate` | 0.25 | share of offspring from two parents |
| `evolution.mutation_rate` | 0.12 | share of genes perturbed |
| `network.hidden_layers` | (16, 12) | hidden layer widths |
| `features.spec` | `egocentric_v3` | state encoding |
| `fitness.food_reward` | 250 | weight on eating |

Crossover is a minority operator on purpose. Blending two networks whose hidden
units are not aligned produces a child resembling neither parent, and preserves
measurably less parent behaviour than mutation does.

## Tests

```bash
pip install -e ".[dev]"
pytest
```
