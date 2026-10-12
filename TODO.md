# Backlog

Open work only. Settled changes and their measurements live in
[HISTORY.md](HISTORY.md).

## Measurement — do this before any other experiment

- **Single-seed comparisons cannot resolve anything under ~40 points.** The
  default configuration at six training seeds produced champions from 30.4 to
  68.4 on the same held-out seeds, a standard deviation of 14.6 between runs.
  Every comparison in HISTORY.md was run at one seed, so every one of them is
  now "not established" — including the encoding ladder.

  Score and compare with the project's own command, which groups runs by
  configuration and reports the spread between them:

  ```bash
  python -m snake_ai evaluate runs/<a> runs/<b> runs/<c>
  ```

- **Cut the variance before buying more runs.** At sd 14.6, resolving a 10-point
  difference needs 34 seeds per configuration — 68 runs, roughly 50 hours.

  **Champion selection: dropped under the 3,000-step cap, essential under
  10,000.** Under the old cap, ranking each population by average instead of by
  best single game gained +0.4, and a perfect selector was worth 1.9. With the
  cap at 10,000 a single game can run much longer, and the peak rule crowned a
  44.1 snake over an 81.5 one in the same population. Export with
  `--pick best-average`; see HISTORY.md.

  **Runs stall or get truncated — this is where the evidence points.** Seeds 99
  and 123 were still setting records at generations 59 and 58 when the limit
  stopped them, and they produced two of the three best champions. Seeds 23 and
  42 stopped improving at generation 41 and spent 18 generations pinned at the
  0.35 mutation ceiling without recovering. The six held-out means may be two
  clusters (30-31 and 46-68) rather than one spread.

  Measured, by scoring each run's archived champion every ten generations (see
  HISTORY.md): the spread between runs *grows* with time, 7.3 at generation 30
  against 14.6 at generation 60. Longer runs will raise the ceiling and widen
  the spread, not narrow it, so there is no run length that makes cheap
  comparisons possible.

  Runs come in three kinds and the kind appears to be luck: climbing at the
  limit (seeds 99 and 123, +10.8 and +19.9 over their last 20 generations),
  crawling (23 and 42, about +5), and frozen for 20 generations (7 and 11).

  Done, and it worked: extending three runs to generation 100 gained +19.1 and
  +24.0 held-out on two of them, each measured against itself. The third lost
  15.6 because its champion regressed, which is a separate matter and is closed
  (see below). Running longer is the only change this project has verified as a
  double-digit gain, and no run has been shown to reach its ceiling.

  Champion selection was closed here at 1.9 points, measured under the 3,000-step
  cap, and has since been reopened by the 10,000-step cap. See "The end game" in
  HISTORY.md.

  Settled: the climb ends near generation 110. Both runs taken on to 140 gained
  1.9 and 0.8, against 19.1 and 24.0 for the previous forty generations.

  **The recipe: several seeds, 100 to 120 generations, keep the best champion.**
  Sixty generations leaves about twenty points unclaimed; past 120 is decimals.

  Answered, and the answer is worse than the question assumed. Five runs at
  generation 120 spread 21.2, wider than at 60: 28.4, 34.0, 67.2, 68.2, 73.1 --
  two clusters rather than a spread. Two runs in five never take off, and no
  comparison by average is affordable (10 points needs 71 runs a side).

  What replaces it: judge a configuration by how often it takes off and by how
  good its successful runs get, and stop paying for runs that have not. Runs
  that take off agree within about three points, so three successful runs a side
  resolve roughly eight points.

  Cull on whether a run is still improving, not on its peak. The peak rule was
  tested once and failed: v4 seed 302 held the lowest peak at generation 40 and
  took off to 65.5, while the run with a higher peak had gone 27 generations
  without a record. Look at stagnation, recent records and whether mutation has
  escalated to its ceiling.

- **Then compare configurations, at five to eight seeds each**, expecting to
  resolve 15-20 points and no less. v2 against v3 is the one worth settling,
  since the default rests on it.

  One comparison has now cleared the bar: `egocentric_v1` at 200 generations
  reached 26.8 against the default's 73.1, a 46.3-point gap on single seeds, so
  the first rung of the encoding ladder is real even if the later ones are
  unproven. See HISTORY.md.

  A second is settled the other way: `egocentric_v4` against v3, three successful
  runs each at generation 120, came out 68.2 against 69.5 with an interval of
  -8.9 to +6.3. No difference, and no difference in how the champions die.
  Encodings are finished as a lever on the ceiling.

  Costs: a default-config run takes ~45 minutes with seven workers; four run side
  by side at two workers each took about 2h40m.

- **The held-out set is not the bottleneck.** 60 seeds pins one champion to
  within 1.5 points; the spread between training runs is ten times that. Spend
  compute on more runs, not longer evaluation.

## Behaviour

- **Self-collision is the ceiling.** 85% of the `egocentric_v3` champion's
  deaths, at a held-out mean of 46.2. Every encoding change so far has removed a
  different cause of death and left this one standing.

  Diagnosed over 60 held-out episodes:

  | | |
  | --- | --- |
  | died with no legal move left | 35% |
  | died with a safe move available | 65% |

  The second figure looks damning but is not: across ~108,000 steps that is one
  avoidable fatal turn per ~3,300 steps, so immediate safety is already 99.97%
  correct. What precedes it matters more. Reachable space as a multiple of body
  length, in the run-up to death:

  | moves before death | space / length |
  | --- | --- |
  | 40 | 3.25 |
  | 20 | 3.07 |
  | 10 | 1.98 |
  | 5 | 1.67 |
  | 1 | 1.25 |

  The snake steadily squeezes itself into a region barely larger than its own
  body, and **it cannot read that**: the space inputs are normalised by free
  board cells, not by body length. At length 49 with 61 cells reachable the input
  says "0.41 of the board" when the survival-critical fact is "1.25 body
  lengths". Below 1.0 the snake no longer fits in the space it has left, and that
  threshold is invisible to it.

  `egocentric_v4` gave the network exactly that quantity and, tested properly at
  generation 120, changed nothing: 68.2 against v3's 69.5, and its champions die
  of self-collision 89% of the time against v3's 81% once the step cap is out of
  the way. Information alone does not change behaviour that selection does not
  reward. See [HISTORY.md](HISTORY.md).

  What is left: capacity and reward. The network is 16 and 12 hidden units, 435
  parameters, which may be too small to plan around its own body. And the food
  term is worth ~44,500 at score 46 against a survival cap of 120, so nothing in
  the fitness function pays for the caution that would avoid a trap.

- **Penalising death does not work, and the reason narrows the search.** Two
  charges were tried and both lost badly on held-out seeds: 18.9 and 37.5 against
  the control's 46.2. See [HISTORY.md](HISTORY.md).

  The avoidable-death charge cut self-collision from 85% of deaths to 3%, the
  largest behavioural change any single edit has produced, and the agents were
  still worse. Reaching a high score requires risking a long body; there is no
  safe route to 60, so penalising death penalises the behaviour that produces
  high scores. Both charges also flattened the population — the ratio of
  generation-best to population-mean fell from 6.1x to 2.7x — removing the spread
  selection needs.

  Do not retry this by tuning the fractions. The direction is wrong, not the
  magnitude. Anything that makes dying costly will trade the ceiling for the
  floor in the same way.

- **Movement is direct rather than space-filling.** Agents head for food in
  fairly straight lines. A snake that used the board better would survive longer
  once long, but this has to emerge from selection rather than be hard-coded.
  Worth trying: a coverage term that scales with snake length, so spreading out
  only starts paying once the snake is long enough for it to matter.

- **The population converges early, but that is not what separates runs.**
  Diversity falls to ~0.1 within a dozen generations in every run, and the
  control set its last record at generation 34 of 60 — 26 flat generations.
  Across five runs at generation 120, though, diversity points the wrong way: the
  best run had the lowest (0.121) and the worst the highest (0.183). Whatever
  decides a run's fate, it is not how far diversity falls.

  The stalled runs do share one signature: long episodes with little food. Seed
  23 averaged 1,692 steps per episode at generation 40 with a best episode of 32,
  so it had learned to survive rather than to eat. Worth understanding before
  anything else in this section.

  The death-penalty runs sharpened this. Both kept setting records much later
  (generations 54 and 51) while scoring worse, which suggests the usual plateau
  is the search converging rather than the representation running out. The
  control's population carries individuals 6.1x better than its own average, so
  the spread exists; selection just stops exploiting it.

  Halving mutation made it far worse (15.9 held-out): diversity collapsed by
  generation 10. The inverse, 1.5x, went +19.3 at one seed and -5.8 at another —
  not established. See HISTORY.md.

  Every search knob is now a CLI flag: `--population`, `--episodes`,
  `--tournament`, `--elites`, `--immigrants`, `--mutation-rate`,
  `--mutation-strength`, `--min-mutation-rate`, `--max-mutation-rate`,
  `--crossover-rate`.

## Dashboard

- **Its export still uses the stored champion.** The dashboard's *Export .pt*
  takes the genome with the best single game, the rule that crowned a 44.1 snake
  over an 81.5 one under the 10,000-step cap. It should offer the pick by average
  that `export --pick best-average` uses, or at least warn that the stored
  champion is unreliable.

- **"Best ever" is the peak, and it drives mutation.** The record the dashboard
  shows, and the stagnation counter that widens mutation, both follow the best
  single game. Under the longer cap that number jumps on flukes; whether it
  should follow a steadier measure is untested.

- **All-time best replay.** The grid shows the selected generation's agents;
  there is no way to replay the all-time champion on demand. The genome is
  already archived per generation, so this needs an endpoint and a control.

- **Compare two generations.** Putting generation N and M side by side would
  make it much easier to see whether a change helped.

- **Fitness term breakdown over time.** Individual agents show their fitness
  terms, but there is no view of how the balance shifts across a run, which is
  what would show a shaping term being farmed.

## Engineering

- **Speciation is off by default.** Implemented, including the fitness sharing
  that applies the clustering, but Euclidean distance between flat weight vectors
  is a weak similarity measure for neural genomes. Either find an encoding where
  the clusters mean something, or remove it.

- **No resume across feature-spec changes.** Changing the encoding invalidates
  every checkpoint, by design. A conversion path would need a principled way to
  map an old genome onto a new input layer, which may not exist. Note that
  `egocentric_v4` appends inputs rather than altering the existing eleven, so a
  v3 genome could in principle be widened rather than discarded.

- **Space and danger disagree about the tail in v1 and v2.** `would_collide`
  treats the tail as safe because it vacates on the same tick, but their space
  scanner marks every body cell blocked, so a legal move can report zero space.
  Affects 0.9% of steps. `egocentric_v3` does not have this problem, since its
  scan already clears the tail on move one, so this only matters if the older
  encodings are revived.
