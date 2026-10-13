# Command reference

Quick reference for running, inspecting and exporting. See `README.md` for how
the system works.

All commands assume the repo root as the working directory. `python -m snake_ai`
works everywhere; the `snake-ai` console script does the same thing but needs its
install directory on PATH.

## Everyday

| Goal | Command |
| --- | --- |
| Train and watch | `python -m snake_ai train --open` |
| Train without pausing for the dashboard | `python -m snake_ai train --open --no-hold` |
| Train with no dashboard at all | `python -m snake_ai train --headless` |
| Bounded run | `python -m snake_ai train --generations 120` |
| Continue the newest checkpoint | `python -m snake_ai train --resume` |
| Continue a specific checkpoint | `python -m snake_ai train --resume runs/<run>/checkpoints/latest.npz` |
| Describe the newest checkpoint | `python -m snake_ai inspect` |
| Describe a specific checkpoint | `python -m snake_ai inspect runs/<run>/checkpoints/gen_000050.npz` |
| Score the newest champion on held-out seeds | `python -m snake_ai evaluate` |
| Compare runs or configurations | `python -m snake_ai evaluate runs/<run-a> runs/<run-b> ...` |
| Export the run's steadiest snake | `python -m snake_ai export --pick best-average --name v3-seed99-gen155-81` |
| Export the stored champion, fast | `python -m snake_ai export --name <name>` |
| List feature encodings | `python -m snake_ai specs` |

Stop a run with **Ctrl+C**. It finishes the generation in progress and writes
`latest.npz` before exiting, so nothing is lost mid-generation.

## train flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--generations N` | unlimited | Stop after N generations |
| `--population N` | 150 | Genomes per generation |
| `--episodes N` | 4 | Evaluation episodes per genome, averaged |
| `--board N` | 14 | Square board, N cells a side |
| `--spec NAME` | `egocentric_v3` | State encoding; see the table under Recipes |
| `--avoidable-penalty F` | 0 | Cost of dying when a safe move existed, as a fraction of food earned |
| `--death-penalty F` | 0 | Cost of dying at all, as a fraction of food earned; running out the clock is exempt |
| `--workers N` | 0 | Evaluation processes; 0 uses every core but one, 1 is serial |
| `--tournament N` | 4 | Tournament size; higher is stronger selection pressure |
| `--elites N` | 6 | Genomes carried forward unchanged each generation |
| `--immigrants N` | 4 | Fresh random genomes injected each generation |
| `--mutation-rate F` | 0.12 | Share of genes perturbed; read HISTORY.md before lowering it |
| `--mutation-strength F` | 0.08 | Size of each perturbation |
| `--min-mutation-rate F` | 0.06 | Floor of the adaptive schedule; binds only if set above the base rate |
| `--max-mutation-rate F` | 0.35 | Ceiling the schedule escalates to while stagnating |
| `--crossover-rate F` | 0.25 | Share of offspring built from two parents |
| `--seed N` | random | Makes a run reproducible |
| `--port N` | 8765 | Dashboard port |
| `--resume [PATH]` | off | Bare `--resume` picks the newest checkpoint |
| `--report-every N` | 1 | Print every Nth generation; records always print |
| `--headless` | off | No dashboard |
| `--no-hold` | off | Never pause at generation boundaries |
| `--no-speciation` | already off | Kept for symmetry with the config |
| `--open` | off | Open the dashboard in a browser |

Flags override `snake_ai/config.py` for that run only; nothing is written back.

## evaluate

Scores each run's champion on the fixed held-out block: 60 episodes on seeds
900000–900059, one episode per seed, exploration off. Every held-out figure in
HISTORY.md comes from this block, so any of them can be re-checked here.

```bash
python -m snake_ai evaluate                                    # newest run
python -m snake_ai evaluate run-20260911-133159                # by run id
python -m snake_ai evaluate runs/run-20260911-1517*            # a pattern
python -m snake_ai evaluate runs/<a> runs/<b> --json out.json  # keep every episode
```

Targets can be run directories, run ids, checkpoint files or glob patterns.
Patterns work in PowerShell too; they are expanded by the command itself.

Champions are grouped by the configuration in each run's `run.json`, ignoring
the seed and the worker count, and labelled by how they differ from the current
defaults. When there is more than one group, the first one named is the
reference and every other group is compared against it.

| Output | Meaning |
| --- | --- |
| `train best` | Best single episode during training, on the seeds it was selected on |
| `mean` `median` `p25` `worst` `best` | Across the champion's held-out episodes |
| `50+` | Held-out episodes scoring 50 or more |
| `deaths` | The two most common ways its episodes ended |
| `mean of champions` | Average over the group's runs, with a 95% interval |
| `spread between runs` | Standard deviation of the runs' means — the training-seed noise |

Intervals come from the spread between runs, never between episodes of one
champion; the second is several times smaller and is the mistake HISTORY.md
records. A configuration run at a single seed gets no interval at all, because
nothing has measured its seed noise.

| Flag | Default | Notes |
| --- | --- | --- |
| `--episodes N` | 60 | Held-out episodes per champion; other values are not comparable with HISTORY.md |
| `--first-seed N` | 900000 | Start of the held-out block |
| `--workers N` | 0 | 0 uses every core but one, 1 is serial; results are identical |
| `--same-config` | off | Treat every champion as one configuration |
| `--json PATH` | off | Also write every episode's score and death, per champion |

Runs older than `run.json` have no recorded configuration and are kept in a
group of their own. `--same-config` pools them with the rest when you know they
match.

## Recipes

Fastest path to a visible learning curve:

```bash
python -m snake_ai train --open --no-hold
```

A run worth keeping. Champions improve until roughly generation 110 and gain
decimals after, while stopping at 60 leaves about twenty held-out points behind:

```bash
python -m snake_ai train --generations 120 --headless
```

Two runs in five never take off, and by generation 40 they are already the
lowest-scoring of a batch. So start several short, then continue only the better
half:

```bash
python -m snake_ai train --seed 201 --generations 40 --headless
```

Repeat that for four seeds, then keep the ones still improving -- recent records,
low `stagnation`, mutation not pinned at its ceiling in `history.jsonl`. Do not
cull on peak score: a run with the lowest peak at generation 40 has gone on to
take off. Take the survivors the rest of the way:

```bash
python -m snake_ai train --resume runs/<survivor>/checkpoints/latest.npz --generations 120 --headless
```

Reproduce an exact run:

```bash
python -m snake_ai train --seed 42 --generations 50 --headless
```

Quick experiment, small and fast:

```bash
python -m snake_ai train --population 40 --episodes 2 --generations 30 --headless
```

Fewer episodes per genome means faster generations but noisier scores, so a
lucky genome is more likely to win selection. Good for a smoke test, not for
comparing configurations.

Harder problem, bigger board:

```bash
python -m snake_ai train --board 20 --open
```

Compare the two state encodings:

```bash
python -m snake_ai train --spec egocentric_v1 --seed 7 --generations 60 --headless
python -m snake_ai train --spec egocentric_v2 --seed 7 --generations 60 --headless
```

| Spec | Inputs | Space measured as | Held-out mean, one seed |
| --- | --- | --- | --- |
| `egocentric_v4` | 14 | v3, plus space relative to body length | 40.0 |
| `egocentric_v3` | 11 | reachable within `space_horizon` moves, body clearing as it goes | 46.2 |
| `egocentric_v2` | 11 | reachable now, against a frozen body | 36.4 |
| `egocentric_v1` | 11 | reachable now, capped scan | 19.6 |

v3 is the default. Each figure is one training run's champion at 60 generations
over the 60 held-out seeds, and the ordering between v2, v3 and v4 is unproven:
the same configuration at six seeds spans 30.4 to 68.4, wider than any gap
between them.

The v1 gap is established, though. Given 200 generations -- 3.3 times the length
and more wall clock than the default needed -- v1 reached 26.8 against the
default's 73.1. Its cheaper generations come from snakes that die sooner, so the
speed buys nothing. See [HISTORY.md](HISTORY.md).

Comparing two configurations. One seed is nowhere near enough: the identical
default configuration, run at six seeds, produced champions from 30.4 to 68.4 on
the same held-out seeds. A single-seed comparison cannot resolve anything under
about 40 points. Run each configuration at several seeds and compare the
averages of their champions:

```bash
python -m snake_ai train --seed 7  --generations 120 --headless --population 300
python -m snake_ai train --seed 11 --generations 120 --headless --population 300
python -m snake_ai train --seed 23 --generations 120 --headless --population 300
```

A shared seed makes generation 0 identical across configurations, but runs
diverge from the first breeding step, so it does not remove seed luck.

Then name both sets of runs in one call. They are grouped by configuration and
the second is compared against the first:

```bash
python -m snake_ai evaluate runs/<control-7> runs/<control-11> runs/<control-23> runs/<pop300-7> runs/<pop300-11> runs/<pop300-23>
```

Death penalties. Both flags exist and both are off by default. Each was run at a
single seed and scored 18.9 and 37.5 held-out against a control of 46.2 — large
gaps, but single-seed gaps, so they settle nothing on their own. What the charges
did to behaviour is the informative part: see [HISTORY.md](HISTORY.md).

```bash
python -m snake_ai train --seed 7 --generations 60 --headless --avoidable-penalty 0.3
python -m snake_ai train --seed 7 --generations 60 --headless --death-penalty 0.25
```

Prove parallelism changed nothing:

```bash
python -m snake_ai train --seed 7 --generations 6 --headless --workers 1
python -m snake_ai train --seed 7 --generations 6 --headless --workers 7
```

Then diff the two `history.jsonl` files, ignoring `elapsed`. Every other field
must match exactly. If they do not, something is drawing randomness inside a
worker, and selection is no longer comparing genomes on equal terms.

Run on a spare port alongside another run:

```bash
python -m snake_ai train --port 8766 --open
```

## export

```bash
python -m snake_ai export --checkpoint runs/<run>/checkpoints/latest.npz --pick best-average --name <name>
```

| Flag | Default | Notes |
| --- | --- | --- |
| `--checkpoint PATH` | newest run | Which run to export from |
| `--name NAME` | `snake_ai` | File stem, so several models stay apart; e.g. `v3-seed99-gen155-81` |
| `--pick best-average` | `champion` | Plays every genome the run holds and exports the steadiest. Takes a few minutes |
| `--workers N` | 0 | For `--pick best-average`; 0 uses every core but one |

The stored champion is whichever genome played the run's single best game. Under
the 10,000-step cap that picked a 44.1 snake while the same population held an
81.5 one, so for a model you intend to use, pick by average. The choice is
recorded in the export's `.json` sidecar under `selection`, and its figures are
optimistic -- score the result with `evaluate` for an honest number.

The sidecar's training statistics and achievements cover the whole lineage. A
resumed run starts a new directory, so the export follows `resumed_from` back
through each `run.json` and counts every generation behind the model, not only
the ones the last run played.

## Dashboard

<http://127.0.0.1:8765>

| Key | Action |
| --- | --- |
| `←` / `→` | Previous / next generation |
| `space` | Play or pause the replay |
| `c` | Continue one generation while held |
| `l` | Follow latest |
| `esc` | Close the agent panel |

**Training control** — while the page is open, training pauses at each
generation boundary so a generation cannot be replaced before you have seen it.
*Continue* releases one generation, *Continue ×5* releases five, *Run free*
disables pausing, *Hold each generation* turns it back on. Closing the tab
releases the hold on its own after about twelve seconds.

**Generation browser** — the scrubber, arrows and jump box move through any
retained generation. Selecting one detaches the view from the live edge;
*Follow latest* reattaches it.

**Export model** — enter a generation, press *Calculate* to see what would be
written (this touches no files), then *Export .pt*.

## Files a run produces

```
runs/run-YYYYMMDD-HHMMSS/
  run.json                       seed and full configuration, written at start
  history.jsonl                  one summary line per generation, kept forever
  replays/gen_NNNNNN.json.gz     replay frames + champion genome
  checkpoints/latest.npz         newest, rewritten on every record
  checkpoints/gen_NNNNNN.npz     periodic, every 25 generations
exports/
  <name>.pt                      exported model; --name keeps several apart
  <name>.json                    readable sidecar for that export
  archive/<name>_genNNNNNN_*.pt  timestamped copies
```

Both directories are gitignored. Deleting `runs/` resets everything.

Read a run's history without the dashboard:

```bash
python -c "import json;[print(json.loads(l)['generation'], json.loads(l)['max_episode_score']) for l in open('runs/<run>/history.jsonl')]"
```

## Console columns

```
gen    18  best  31  mean  5.105  eat%  78.7  fit  22267.6  steps  520.8  div 0.129  mut 0.120  stag   0  starvation  record
```

| Column | Meaning |
| --- | --- |
| `gen` | Generation number |
| `best` | Highest score any single episode reached |
| `mean` | Mean score across the whole population |
| `eat%` | Share of the population that ate at least once |
| `fit` | Best fitness this generation |
| `steps` | Mean episode length |
| `div` | Genome diversity; falling toward 0 means the population is converging |
| `mut` | Current mutation rate, widened automatically while stagnant |
| `stag` | Generations since the best score last improved |
| last word | Most common cause of death |
| `record` | This generation beat the all-time best |

## Config

`snake_ai/config.py`, frozen dataclasses. Highest-impact values:

| Setting | Default | Effect |
| --- | --- | --- |
| `evolution.workers` | 0 | Evaluation processes; 0 uses every core but one |
| `evolution.population_size` | 150 | Genomes per generation |
| `evolution.evaluation_episodes` | 4 | Episodes averaged per genome |
| `evolution.crossover_rate` | 0.25 | Share of offspring from two parents |
| `evolution.mutation_rate` | 0.12 | Share of genes perturbed |
| `network.hidden_layers` | (16, 12) | Hidden layer widths |
| `features.spec` | `egocentric_v3` | State encoding |
| `features.flood_fill_limit` | 96 | Reachable-space cap (`egocentric_v1` only) |
| `features.space_horizon` | 32 | Moves searched ahead (`egocentric_v3` only) |
| `fitness.food_reward` | 250 | Weight on eating |
| `fitness.avoidable_death_fraction` | 0 | Charge for dying when an escape existed |
| `fitness.death_fraction` | 0 | Charge for dying at all |
| `history.replay_retention` | 400 | Replays kept on disk |
| `history.replay_steps` | 0 | Steps kept per replay; 0 keeps the whole game |
| `episode.max_steps` | 10,000 | Cap on a game's length; starvation ends loops long before it |
| `checkpoint.save_every` | 25 | Periodic checkpoint interval |
| `server.hold_when_viewing` | true | Pause while the dashboard is open |

Changing `features.spec` or `network.hidden_layers` changes the genome shape and
invalidates existing checkpoints. Loading refuses the mismatch rather than
reshaping a genome to fit.

## Development

```bash
pip install -e ".[dev]"      # editable install with pytest
pip install -e ".[export]"   # add torch for .pt export
pytest                       # 89 tests
pytest -q tests/test_evolution.py
ruff check snake_ai tests
ruff check snake_ai tests --fix
```

## Using an exported model

```python
import torch, numpy as np

model = torch.load("exports/snake_ai.pt", weights_only=False)
print(model["feature_spec"]["inputs"])   # the 11 inputs, in order
print(model["action_space"])             # ['left', 'straight', 'right']

x = np.zeros(model["feature_spec"]["size"], dtype=np.float32)
hidden = model["architecture"]["hidden_layers"]
for i in range(len(hidden) + 1):
    x = x @ model["parameters"][f"weight_{i}"].numpy() + model["parameters"][f"bias_{i}"].numpy()
    if i < len(hidden):
        x = np.tanh(x)
turn = model["action_space"][int(np.argmax(x))]
```

The archive carries its own weights, architecture, input names and action space,
so it does not need this project to run.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `snake-ai: command not found` | Script directory not on PATH. Use `python -m snake_ai` |
| `dashboard unavailable on 127.0.0.1:8765` | Port in use. Pass `--port 8766` |
| Training seems frozen | The dashboard is holding it. Press *Continue* or *Run free* |
| `cannot resume: checkpoint was trained with feature spec ...` | Encoding changed since that checkpoint. Start a new run |
| Export panel says `torch not installed` | `pip install torch` |
| Generation not available in the browser | Its replay was evicted. Records are never evicted |
