# Project history

What changed, why, and what it measured. Newest first.

Every performance figure here is a run champion scored on **60 held-out seeds**,
900000–900059, one episode each. Training never picks them deliberately; an
accidental overlap has odds of about one in 300,000 per run.
`python -m snake_ai evaluate <run>` reproduces any of these figures. In-run peak
score is recorded too, because it twice pointed the opposite way to the truth
and that is worth remembering.

## Summary

| | best score | held-out mean | notes |
| --- | --- | --- | --- |
| original code | 2 | — | flat across 30 generations |
| rebuild, `egocentric_v1` | 41 | 19.6 | 11 egocentric inputs |
| `egocentric_v2` | 69 | 36.4 | uncapped space scan |
| `egocentric_v3` | 62 | 46.2 | space measured over time — current default; margin over v2 within seed noise |
| `egocentric_v4` | 61 | 40.0 | + space relative to body length; loss within seed noise, not established |
| death penalties | 46 | 18.9 / 37.5 | two fitness charges, one seed each; off by default |
| gentler mutation | 30 | 15.9 | rate 0.06, capped 0.10; diversity collapsed by generation 10 |
| 1.5x mutation | 84 / 66 | 65.5 / 50.2 | two seeds, opposite directions; not established |
| default config, 6 seeds | 35 to 97 | 30.4 to 68.4 | one configuration, six training seeds at 60 generations; spread 14.6 |
| length to generation 100 | 92 / 104 | 65.3 / 72.4 | two runs extended from 60; +19.1 and +24.0 |
| length to generation 140 | 92 / 105 | 67.2 / **73.1** | +1.9 and +0.8 more; the climb ends near 110 |
| `egocentric_v1`, 200 generations | 47 | 26.8 | 3.3x the generations, still 46 points behind |
| `egocentric_v4`, 3 runs at 120 | 98 | 68.2 | against v3's 69.5; no difference, same way of dying |
| end-game fixes, 15 generations at 10,000 steps | 153 | **81.5** | best genome by average; the stored champion fell to 44.1 |
| parallel evaluation | — | — | 3.67x faster, results bit-identical |

The held-out figures above are each a single training run. The unchanged default
configuration, run at six seeds, produced champions from 30.4 to 68.4 — a spread
wider than every difference this file attributes to a change. Nothing in the
table is established as an improvement; see "Training-seed variance" below for
what that does and does not invalidate. The best snake the project has produced,
81.5 held-out, came from letting end games play out; see "The end game" below.

---

## The end game - a replay that stopped early, a cap on the best games, and a champion rule that broke

Watching the best snake in the dashboard, it played well until it was long, then
while it chased its tail waiting for a way to the food, the game appeared to end.
Three explanations were tested at once with the 73.1 champion on the 60 held-out
games, each played under four rule sets. Only the *endings* were switched off:
one of the snake's inputs is hunger as a share of the starvation limit, so
changing the limit itself would have changed what it saw from the first move.
The 54 games no rule cut short came out move-for-move identical under all four.

**The replay.** The dashboard recorded only the first 1,200 steps of each game,
and 58 of the 60 games ran longer. When a replay ran out, its tile greyed out as
if the snake had died. The median score on screen at that point was 53; the
games went on to a median of 76, so about 40% of each game -- 21 points -- was
never shown. That is what was being watched.

**Starvation was not the problem.** It ended 3 of 60 games, at 25, 36 and 61 --
mid-game, not end game -- and given 47,000 further moves none of the three ever
ate again. The limit ends snakes stuck in a loop, not snakes waiting for an
opening. It is also nowhere near tight for a long snake: between lengths 70 and
89 the longest wait for a meal was 424 moves against a limit of 1,112.

**The 3,000-step cap was, on the best games.** It ended exactly three games, the
three best (94, 85 and 91), which went on to 101, 89 and 92 without it before
colliding with themselves a few hundred moves later.

Fixes: replays now record the whole game (`history.replay_steps = 0`), a replay
that is cut short says so on its tile instead of reading as a death, and the cap
is 10,000 (`episode.max_steps`). Starvation already ends a snake circling without
food, so the cap only ever stops snakes that are still eating.

### Fifteen generations under the new rules

The 73.1 snake's run was resumed from generation 140 to 155. The replay fix held:
of 2,250 replays none was truncated, 662 ran past 3,000 steps, the longest
reached 10,000, and a generation's replays grew from about 47 KB to 126 KB.

The stored champion came out at **44.1 held-out**, with its best training game up
from 105 to 153. Every champion the run held, in order:

| held from | how it took the title | held-out | 25th percentile | starved, of 60 |
| --- | --- | --- | --- | --- |
| generation 140 | inherited, the 73.1 snake | 73.3 | 65.8 | 3 |
| generation 142 | one game of 114 | 51.3 | **1.0** | 19 |
| generation 144 | one game of 126 | 71.3 | 63.8 | 0 |
| generation 154 | one game of 153 | **44.1** | **1.0** | 23 |

Cliffs at records rather than a slide across generations. Twice the rule crowned
a genome that loops and starves within moments in a quarter of its games and now
and then plays one enormous game. The population itself had improved: its best
genome by average over eight probe games scored **81.5 held-out**.

This reverses "Champion selection: measured twice, dropped" below. That result --
a perfect selector worth 1.9 points -- was measured under the 3,000-step cap and
holds there. A longer cap lets one game run far longer and score far higher, so a
single-game peak becomes a much noisier measure; here the peak rule cost 37
points. The conclusion depended on the cap, and was not re-checked when the cap
changed.

### Choosing by average, at export

`export --pick best-average` plays every genome a run holds -- its population, its
stored champion and every earlier champion -- eight games each, re-plays the five
leaders and the stored champion on forty more, and exports the best average over
all 48. The selection games start at seed 800,000, clear of the held-out block.
It runs only when a model is wanted, so it costs training nothing.

### Did the new cap make the better snakes?

The same pick under the same rules, on the same lineage before and after:

| population | best genome, held-out | its 25th percentile | stored champion |
| --- | --- | --- | --- |
| generation 140, trained under the 3,000-step cap | 72.8 | 66.8 | 73.3 |
| generation 155, after 15 generations at 10,000 | **81.5** | 73.0 | 44.1 |

The better genomes did not predate the change: the old population's best was the
same as its champion. They appeared in the fifteen generations under the new cap,
**+8.7** -- where the same lineage had gained 0.8 in the forty generations from 100
to 140 under the old one. The likely mechanism is that games stopped at 3,000
steps hid which snakes play the end game well, so selection could not reward it.
One lineage, so this is strong evidence rather than a measured effect size.

**The best snake the project has produced is now `v3-seed99-gen155-81.pt`: 81.5
held-out, 25th percentile 73.0, and 50 or more in 57 of 60 games.**

## egocentric_v4 at convergence - the same ceiling as v3

The first v4 attempt was judged on single 60-generation runs, which the sections
below show settle nothing, so it was re-run properly. The design and the
prediction were written down before any run started.

**Design.** Four v4 seeds to generation 40; the two with the highest peaks taken
on to 120; the result compared against the three v3 runs that took off, all at
generation 120 and on the same held-out seeds. Measured: the mean of runs that
took off (a held-out mean of 50 or more), how those champions die, and how many
runs took off.

**Prediction**, from the diagnosis of the first attempt: lower self-collision,
higher starvation, and a mean at or below v3's 69.5.

| champion, generation 120 | held-out | 50+ games | self | starvation | wall | step limit |
| --- | --- | --- | --- | --- | --- | --- |
| v4 seed 301 | 70.8 | 55 | 97% | 0% | 0% | 3% |
| v4 seed 304 | 68.4 | 53 | 73% | 7% | 2% | 18% |
| v4 seed 302 | 65.5 | 53 | 63% | 10% | 10% | 17% |
| v3 seed 11 | 68.2 | 49 | 73% | 7% | 12% | 8% |
| v3 seed 7 | 67.2 | 55 | 75% | 0% | 20% | 5% |
| v3 seed 99 | **73.1** | 57 | 78% | 5% | 12% | 5% |

**v4 68.2 against v3 69.5: a difference of -1.3, 95% interval -8.9 to +6.3.** No
difference at the roughly eight-point resolution three runs a side allows. v4
does not raise the ceiling, and it does not lower it either, so the original
rejection at 60 generations was an artefact of stopping early.

The prediction failed where it mattered. The mean did come in below 69.5, inside
the noise. Self-collision was **not** lower -- 78% against 76% -- and starvation
rose from 4% to 6%, nothing like the 43% of the first attempt. The story recorded
further down, of cautious snakes that stop colliding and starve instead, does not
survive to convergence. By generation 120 selection drives both encodings to the
same failure.

**The step cap was checked and is not hiding anything.** v4's champions hit the
3,000-step limit in 13% of games against v3's 6%, and those games end with the
snake alive, so all six champions were re-scored with the cap at 12,000. Every
one moved by less than a point, no game ran past 3,509 steps, and the comparison
became -0.9 with an interval of -8.1 to +6.2. With the cap out of the way the
capped games resolved into self-collisions: **89% of v4's deaths against 81% of
v3's.** The encoding built to prevent self-collision ends with more of it.

**Take-off:** all three v4 runs taken to 120 took off, and a fourth was culled at
40; v3 took off in three of five. Suggestive, not distinguishable at this size.

**The peak-based cull failed its first real test.** Seed 302 held the lowest peak
at generation 40 -- 49, against 59, 69 and 78 -- and would have been discarded
under the rule as designed. It was resumed on the strength of its full history
(still setting records, stagnation of 2, mutation not escalated) and took off to
65.5. The run culled with the *higher* peak, seed 303 at 59, had gone 27
generations without a record with mutation pinned at its ceiling. One case, but
it points the same way as everything else here: whether a run is still improving
says more than how high its peak is.

### What it closes

Encodings are finished as a lever on the ceiling. v1 is clearly worse; v3 and v4
converge to the same held-out score and die the same way, a long snake trapping
itself four times in five. Giving the network its reachable space as a multiple
of body length -- precisely the quantity that predicts those deaths -- changed
neither the score nor the way it dies.

So the ceiling near 70 is not about what the snake can see. The candidates left
are what it can compute with what it sees, and what selection rewards: a network
of 16 and 12 hidden units holding 435 parameters, and a fitness function whose
food term dwarfs every other consideration.

## Convergence: two modes, not a spread

Five runs of the identical default configuration, all measured at generation 120.
Three were extended from 60 for this; seeds 7 and 99 had already gone further, so
their generation-120 champions came from the replay archives rather than new runs.

| seed | 23 | 42 | 7 | 11 | 99 |
| --- | --- | --- | --- | --- | --- |
| held-out mean | 28.4 | 34.0 | 67.2 | 68.2 | **73.1** |
| episodes scoring 50+ | 0 | 1 | 55 | 49 | 57 |

**The spread between runs is 21.2, against 14.6 at generation 60.** Running to
convergence made it wider, not narrower, which corrects the guess recorded below
that runs tighten as they converge -- that came from two runs which both happened
to take off.

The shape is the real finding. The five results are 28, 34, 67, 68, 73: two
clusters with nothing in between. A run either takes off or it does not, and two
of five did not. Extending those two from generation 60 to 120 moved them by -2.0
and +2.6, so they are not slow, they are finished.

That closes comparison by average for good:

| difference to detect | seeds per configuration |
| --- | --- |
| 10 points | 71 |
| 15 points | 32 |
| 20 points | 18 |
| 25 points | 12 |

### What does not explain the split

**Diversity does not.** Of the five runs, the best had the *lowest* genome
diversity at generation 60 (0.121) and the worst had the highest (0.183). TODO.md
carried early convergence as its strongest lead; this is evidence against it.

**Eating does not.** All five kept 80% to 96% of the population eating at least
once, throughout.

What does show up is stalling. Seed 23 averaged 1,692 steps per episode at
generation 40 while its best episode that generation scored 32 -- it survives
without eating. Seed 99 ran similar episode lengths while scoring 50.

### Telling them apart early

By generation 30 the two runs that would stall already held the two lowest peak
scores, 31 and 32 against 45, 50 and 66, and the gap still holds at generation
40. On this sample, stopping the bottom half of a batch at generation 40 keeps
exactly the three runs that took off and saves about a third of the compute.

Five runs is a thin basis for a rule. It is a working heuristic, and the test of
it is whether it keeps choosing correctly on the next batch.

## Running longer - the only verified double-digit gain

Three finished runs were resumed from their 60-generation checkpoints and taken
to generation 100. Each is compared against itself, so training-seed luck plays
no part in the difference.

| run | generation 60 | generation 100 | change |
| --- | --- | --- | --- |
| seed 7 | 46.2 | **65.3** | +19.1 |
| seed 99 | 48.4 | **72.4** | +24.0 |
| seed 123 | 68.4 | 52.8 | -15.6 |

Seed 7 had gone 25 generations without a record and its held-out quality had sat
flat for 20, which this file called frozen. It was not. Eleven generations after
resuming it set a record, then five more, and finished 19 points better. A run
that looks finished at generation 60 is not finished, and no run here has been
shown to reach its ceiling.

The best champion the project has produced is seed 99's, later extended to
generation 140: **73.1 held-out**, best episode 94, and 50 or more in 57 of its
60 games.

Seed 123 lost ground for an unrelated reason, and it is the one case worth
reading in full.

### Where the gain stops

Both surviving runs were then taken from generation 100 to 140.

| run | generation 60 | 100 | 140 |
| --- | --- | --- | --- |
| seed 7 | 46.2 | 65.3 | 67.2 |
| seed 99 | 48.4 | 72.4 | **73.1** |

Nineteen and twenty-four points for the first forty extra generations; 1.9 and
0.8 for the next forty. Seed 99 set one record at generation 100 and then nothing
for 39 generations. **The climb is over by about generation 110**, so stopping at
60 left roughly twenty points unclaimed and running to 200 would have bought
decimals.

Both of these runs took off, and comparing only them suggested that spread
tightens as runs converge. Five runs measured at generation 120 say the opposite:
21.2, wider than at generation 60. See "Convergence: two modes, not a spread"
above.

### egocentric_v1 with three times the generations

The encoding ladder is re-graded as unproven below, so v1 was given the length
treatment: same seed 7, 200 generations, 4h23m, 120,000 episodes.

| | generations | best episode | held-out |
| --- | --- | --- | --- |
| `egocentric_v1` | 60 | 41 | 19.6 |
| `egocentric_v1` | 200 | 47 | 26.8 |
| `egocentric_v3`, the default | 140 | 105 | **73.1** |

v1 improves with length like everything else, by 7.2 points, and stays **46.3
points behind**. That clears the 41-point bar a single-seed comparison has to
beat, which makes it the first cross-configuration difference in this file that
exceeds the noise rather than hiding inside it.

It also settles the cheaper-encoding argument. v1 costs about 1.3 minutes per
generation against the default's 4.5, but that saving comes from its snakes
dying sooner, and spending it on three times the generations still finishes 46
points short. Speed that comes from weakness does not convert into score.

None of this rescues the rest of the ladder: v2 -> v3 was 9.8 points and stays
unproven.

### Champion selection: measured twice, dropped

Seed 123's peak score rose from 96 to 105 while its held-out mean fell from 68.4
to 52.8. Scoring its archived champion generation by generation put the whole
loss on two exact moments -- the records at generations 60 and 66 -- with 33 flat
generations after. Each record replaced the genome the run kept with one that
had a better single game and played worse everywhere else.

That is the winner's curse: the maximum of tens of thousands of noisy draws is
biased upward. The published deep-neuroevolution GA avoids it by re-scoring its
top ten candidates on 30 extra episodes and keeping the best mean.

So the rule was built, and then verified against the nine runs on disk by
replaying every selection rule over every champion those runs ever held:

| rule for choosing what to keep | mean held-out, 9 runs |
| --- | --- |
| best single episode (what the runs did) | 52.4 |
| best average over 20 episodes | 52.3 |
| best average over 80 episodes | 54.0 |
| perfect hindsight, an oracle no rule can beat | **54.3** |

**A perfect selector is worth 1.9 points** against a spread between runs of 14.6,
and five of the nine runs have a single candidate that every rule picks. The
implementation was reverted.

The seed-123 failure was real -- 12.8 points lost on that run -- but it is the
only case of its kind in nine runs, and a rule that catches it gives back as much
elsewhere. An earlier and weaker version of this test, re-ranking each final
population on eight probe episodes, had already found +0.4 points; the dramatic
single case was what made it look worth building anyway.

*Superseded under the 10,000-step cap:* there the peak rule cost 37 points, and a
pick by average became the way to export a model. See "The end game" above.

## Training-seed variance — the result that re-grades this file

The default configuration was run at six training seeds with nothing else
changed, and every champion was scored on the same 60 held-out seeds.

| training seed | 23 | 42 | 7 | 99 | 11 | 123 |
| --- | --- | --- | --- | --- | --- | --- |
| in-run best | 35 | 41 | 62 | 79 | 97 | 96 |
| held-out mean | 30.4 | 31.4 | 46.2 | 48.4 | 56.0 | **68.4** |
| held-out median | 30.0 | 31.0 | 46.5 | 46.5 | 61.0 | **72.0** |
| worst of 60 | 21 | 26 | 26 | 14 | 4 | 10 |
| episodes scoring 50+ | 0 | 0 | 24 | 27 | 42 | **53** |

**One configuration, six seeds, held-out means from 30.4 to 68.4.** The standard
deviation between runs is 14.6. Reproduce it with:

```bash
python -m snake_ai evaluate <the six runs> --same-config
```

Every "standard error" elsewhere in this file was computed across held-out
seeds for one fixed champion. That quantity averages **1.5** here: it measures
how consistently one genome plays, which is not in question. The spread between
training runs is **14.6**, about ten times larger, and it is the one that decides
whether a change did anything. A difference has to clear it to mean anything.

| seeds per configuration | smallest difference it can resolve |
| --- | --- |
| 1 | 41 points |
| 3 | 24 points |
| 5 | 18 points |
| 8 | 15 points |

Every comparison in this file was run at one seed, so every one of them sits
inside the first row:

| comparison, single seed | effect | status |
| --- | --- | --- |
| v1 -> v2 | +16.8 | not established |
| v2 -> v3 | +9.8 | not established |
| v4 vs v3 | -6.2 | not established |
| avoidable-death charge | -27.3 | not established |
| death charge | -8.7 | not established |
| gentler mutation | -30.3 | not established |
| 1.5x mutation | +19.3 / -5.8 | not established |

Not established means unmeasured, not disproved. What survives is everything
observed *inside* a run rather than inferred from a difference of champion
means: diversity collapsing to ~0.1 by generation 10 under gentle mutation,
self-collision falling from 85% of deaths to 3% under the avoidable-death
charge, the generation-best to population-mean ratio falling from 6.1x to 2.7x
under both charges. Those are direct observations and they stand. What does not
stand is any claim that one configuration produces better champions than
another, including the encoding ladder this project was built on.

`egocentric_v3` remains the default because it is the principled encoding, not
because it has been shown better than v2.

Two cautions about the 14.6 itself. It comes from six runs, so its own 95%
interval runs from 9.1 to 35.7 — the true spread could be much worse. And it was
measured for this configuration; one that collapses early, as gentler mutation
did, may vary far less.

What a real comparison would cost, at 95% confidence and 80% power:

| difference to detect | seeds per configuration | runs | time at ~45 min each |
| --- | --- | --- | --- |
| 10 points | 34 | 68 | ~50 hours |
| 15 points | 15 | 30 | ~22 hours |
| 25 points | 6 | 12 | ~9 hours |

That is the real price of the method used so far, and it is why the next step is
to cut the variance rather than to buy more runs at this variance.

One encouraging detail: within this configuration, in-run best tracked held-out
mean closely (r = 0.95). The seed does not fool the measurement — it changes how
good the run actually is.

### Champion selection: tested, not the cause

The champion a run stores is the genome with the highest single episode out of
~36,000. Taking the maximum of many noisy measurements is biased upward — the
winner's curse — and the published deep-neuroevolution GA avoids exactly this by
re-evaluating its top ten individuals on 30 extra episodes and keeping the best
mean instead.

Measured here on the six finished runs, without retraining: each final
population was ranked two ways on the same eight probe episodes, by best single
episode and by best average, and both picks were scored on the 60 held-out
seeds.

| rule | 7 | 11 | 23 | 42 | 99 | 123 | mean | spread |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| by peak | 46.7 | 57.4 | 24.3 | 30.6 | 50.6 | 66.2 | 46.0 | 15.9 |
| by mean | 46.4 | 60.1 | 26.7 | 35.3 | 53.9 | 55.8 | 46.4 | 13.0 |
| stored champion | 46.2 | 56.0 | 30.4 | 31.4 | 48.4 | 68.4 | 46.8 | 14.6 |

**+0.4 points for the mean rule, standard error 2.3** — nothing against a spread
between runs of 13 to 16. Five seeds gained up to 4.7 and seed 123 lost 10.4.

The test understates the effect: "by peak" here is a maximum over a few hundred
episodes while the stored champion is a maximum over ~36,000, and the pool is
the final population rather than every genome the run ever produced. Even so,
there is no gain here worth chasing. A plausible mechanism, named in the
literature and fixed in published systems, turned out not to be this project's
problem — which is the same lesson as the rest of this section.

### Where the variance actually comes from

| training seed | 23 | 42 | 7 | 11 | 99 | 123 |
| --- | --- | --- | --- | --- | --- | --- |
| last record at generation | 41 | 41 | 34 | 36 | **59** | **58** |
| generations stuck at the end | 18 | 18 | 25 | 23 | 0 | 1 |
| mutation rate at the end | 0.35 | 0.35 | 0.35 | 0.35 | 0.12 | 0.14 |
| held-out mean | 30.4 | 31.4 | 46.2 | 56.0 | 48.4 | **68.4** |

The two best runs were still setting records when the 60-generation limit
stopped them, with mutation still at its base rate. The two worst stopped
improving at generation 41 and spent 18 generations pinned at the 0.35 ceiling
without recovering. Much of the spread is therefore not measurement noise at
all: runs either escape or stall, and the ones that escape were cut off
mid-climb. The six means may be two clusters, 30-31 and 46-68, rather than one
distribution — with six runs there is no way to tell.

### Champion quality over time, measured rather than inferred

Every generation archives the best genome the run has produced so far, so
scoring those on the held-out seeds gives the curve that in-run peak score only
hinted at: what the champion would have been worth had the run stopped there.

| stopped after generation | 20 | 30 | 40 | 50 | 60 | last 20 |
| --- | --- | --- | --- | --- | --- | --- |
| seed 7 | 24.6 | 30.3 | 46.2 | 46.2 | 46.2 | +0.0 |
| seed 11 | 37.5 | 42.9 | 56.0 | 56.0 | 56.0 | +0.0 |
| seed 23 | 25.4 | 25.4 | 24.9 | 30.4 | 30.4 | +5.5 |
| seed 42 | 17.2 | 27.1 | 26.1 | 31.4 | 31.4 | +5.2 |
| seed 99 | 28.3 | 28.3 | 37.5 | 47.4 | 48.4 | +10.8 |
| seed 123 | 30.3 | 40.0 | 48.6 | 60.1 | **68.4** | +19.9 |

**The spread grows with time.** Between runs it is 7.3 at generation 30 and 14.6
at generation 60. This is not a fixed noise floor that more generations would
average away — runs fan out as they go. Running longer will raise the best
result and widen the spread at the same time.

**Two runs were still climbing steeply when the limit stopped them.** Seed 123
gained 19.9 points over its last 20 generations and seed 99 gained 10.8. Seeds 7
and 11 were frozen for those 20 generations: not slowed, stopped. Seeds 23 and
42, the two that looked stalled from their record history, did gain about 5
points late, so "stalled" is really three behaviours — frozen, crawling, and
climbing — and which one a run gets appears to be luck.

**Generation 30 is a coarse screen at best.** Quality at 30 correlates 0.85 with
quality at 60 across these six runs, but the leader changed (seed 11 led at 30,
seed 123 at 60) and the differences at 30 are half the size. Good enough to
discard an obviously bad idea, not to choose between two reasonable ones.

Also visible: a new in-run record sometimes made the champion *worse* on held-out
seeds. Seed 23 fell from 25.4 to 24.9 and seed 42 from 27.1 to 26.1 when a new
peak-scoring genome replaced the previous champion. That is the winner's curse
in miniature, and it costs about a point.

## Gentler mutation — rejected, and a measurement error

Mutating the control's champion 160 times per setting, each mutant scored on the
champion's own seeds, gave a clean-looking curve:

| rate / strength | mutants beating the champion | mean mutant score |
| --- | --- | --- |
| 0.02 / 0.015 | 80.0% | 46.53 |
| 0.06 / 0.040 | 27.5% | 34.86 |
| 0.12 / 0.080 (default) | 1.9% | 18.95 |
| 0.35 / 0.200 (stagnation ceiling) | 0.6% | 2.60 |

It read as "the default destroys 98% of offspring; halve it." A 60-generation
run at seed 7 with rate 0.06, strength 0.04 and the ceiling clamped to 0.10:

| held-out, 60 seeds | control | gentler |
| --- | --- | --- |
| mean score | **46.2** | 15.9 |
| best of 60 | **69** | 26 |
| episodes scoring 40+ | **39** | 0 |

30.3 behind at 17.7 standard errors — the worst result of anything tried.

**The probe measured the wrong thing.** Mutating one evolved champion answers an
exploitation question: do small steps improve *this* genome? Evolution depends
just as much on exploration, and the probe could not see it. Diversity shows what
it missed:

| generations | control | gentler |
| --- | --- | --- |
| 0-9 | 0.367 | 0.318 |
| 10-19 | **0.231** | 0.061 |
| 50-59 | **0.159** | 0.085 |

Gentle mutation made the population clonal by generation 10 and it converged,
quickly, on something mediocre. The offspring the probe counted as destroyed are
the search sampling widely; elitism keeps the best genome safe from the damage.
A local measurement around one point says little about a population-level search.

## Death penalties — both rejected

`egocentric_v4` showed that caution was bred out because nothing paid for it, so
two charges were added, each a fraction of the food already earned so the cost of
dying scales with what the episode had at stake. Both default to 0, making the
control identical to the previous default configuration.

- `--avoidable-penalty 0.3` charges only when the fatal move had a legal
  alternative. The engine flags these, verified against the v3 champion at 63%,
  matching an independent manual measurement exactly.
- `--death-penalty 0.25` charges on any death by wall, self or starvation,
  exempting episodes that run out the clock.

Three runs at seed 7, 60 generations each.

**In-run, both looked like clear wins.** A raised the population mean by 69% and
kept setting records to generation 54 where the control went flat at 34:

| in-run, last 5 generations | control | A | B |
| --- | --- | --- | --- |
| mean score | 9.12 | **15.39** | 11.96 |
| agents that ate | 88.7% | **95.7%** | 86.9% |
| mean steps survived | 890 | **1,788** | 1,340 |
| last record at generation | 34 | **54** | 51 |

**Held out on 60 unseen seeds, both collapsed:**

| held-out | control | A | B |
| --- | --- | --- | --- |
| mean score | **46.2** | 18.9 | 37.5 |
| median | **46.5** | 18.0 | 38.0 |
| worst of 60 | 26 | **0** | 28 |
| 25th percentile | 35.8 | **1.0** | 36.0 |
| standard deviation | 11.4 | 16.3 | **2.6** |
| episodes scoring 50+ | **24** | 1 | 0 |
| self-collision deaths | 85% | **3%** | 67% |

A is 27.3 behind at 10.5 standard errors, B 8.7 behind at 5.7.

**The charges did what they were designed to do.** A almost eliminated
self-collision, from 85% of deaths to 3%. That is the largest behavioural change
any single edit has produced. It simply did not help: those agents starved and
hit walls instead, and a quarter of held-out seeds scored 1 or less.

B is the more interesting failure. Its held-out standard deviation is **2.6**
against the control's 11.4 — it reliably scores 36 to 42 and never more. Risk
aversion bought consistency and capped achievement.

**Why both failed.** Reaching a high score in Snake requires taking risks with a
long body; there is no safe route to 60. Penalising death penalises the behaviour
that produces high scores. The population statistics show the mechanism:

| last 10 generations | population mean | generation best | ratio |
| --- | --- | --- | --- |
| control | 8.90 | 54.2 | **6.1x** |
| A | 16.08 | 42.7 | 2.7x |
| B | 10.91 | 41.9 | 3.8x |

Both charges raised the floor and flattened the distribution. Selection needs
individuals far better than average to make progress, and that is exactly the
spread they removed.

**Methodological note.** In-run *population mean* pointed the wrong way here, as
in-run *peak* had twice before. A had the best in-run mean of the three and the
worst held-out score by a wide margin. Nothing measured during a run has yet been
a reliable guide; only held-out champion evaluation has.

The control also reproduced the earlier serial `egocentric_v3` run exactly —
identical in every field except a new fitness-term key holding 0.0 — which is a
third independent confirmation that parallel evaluation changes nothing.

## Parallel evaluation

Genome evaluation is a pure function of (genome, seed) and genomes never
interact, so a generation parallelises across processes. Scoring moved into
`evolution/evaluation.py` and both paths call it, so serial and parallel cannot
drift apart through code changes.

Three rules make the results identical rather than merely similar: seeds are
fixed by the parent before any task is dispatched, no worker touches the
population RNG, and results are applied in member order rather than completion
order.

Verified at the CLI, six generations at seed 7, sixteen fields per generation
compared: **zero differences**. Confirmed again by accident when the 60-
generation v4 run reproduced the earlier 48-generation serial run exactly for its
first 48 generations.

Throughput on eight cores with seven workers: 3,254 -> 11,958 steps/s, and a
60-generation v4 run fell from 118 to 43 minutes.

Dispatch is one genome at a time rather than fixed chunks. Episode length spans
7 to 1,276 steps within a single generation, so chunks strand workers. Measured
interleaved over five rounds it won four of them by a mean of 1.5s per
generation, but the machine drifted 4.7x slower across that benchmark on
identical work, so the size of the win is unresolved. Correctness does not depend
on it either way.

## egocentric_v4 — space relative to body length (rejected at 60 generations; equal to v3 at 120)

v3's space inputs divide reachable cells by the free board, which is the wrong
denominator for survival: 61 reachable cells is roomy at length 20 and fatal at
length 55. v4 appends three inputs dividing by body length instead, scaled so
0.5 is exactly break-even. 14 inputs, 483 parameters, no throughput cost.

The mechanism worked. **Self-collision fell from 85% of deaths to 35%** — the
failure the inputs were built to address, more than halved. But the agent
over-corrected into caution and starvation came back from 3% to 43%, at a mean
score of 39 with the snake still alive and long.

| held-out, 60 seeds | v3 | v4 |
| --- | --- | --- |
| mean score | **46.2** | 41.2 |
| median | **46.5** | 42.5 |
| 25th percentile | 35.8 | **38.0** |
| worst of 60 | **26** | 1 |
| best of 60 | **69** | 54 |
| episodes scoring 40+ | 39 | **41** |
| episodes scoring 50+ | **24** | 10 |
| self-collision deaths | 85% | **35%** |
| starvation deaths | **3%** | 43% |

v4 is behind on the mean by 5.0 at 2.5 standard errors — separated, but far more
weakly than v1→v2 (7.3) or v2→v3 (4.9).

The first v4 run was cut off at generation 48, so it was re-run to 60. It still
loses: held-out mean 40.0 against v3's 46.2, a gap of 6.18 at 2.7 standard
errors. Truncation was not the explanation.

| held-out, 60 seeds | v3 | v4 (60 generations) |
| --- | --- | --- |
| mean score | **46.2** | 40.0 |
| median | **46.5** | 44.5 |
| 25th percentile | 35.8 | 35.5 |
| worst of 60 | **26** | 12 |
| episodes scoring 40+ | 39 | 39 |
| episodes scoring 50+ | **24** | 14 |

v4 matches v3 through the middle of the distribution and loses at both ends: a
worse floor and far fewer high-scoring episodes.

**Why it failed is the useful part.** The inputs worked — the champion at
generation 44 died of self-collision 35% of the time against v3's 85%. But the
champion at generation 50 was back to 73%. The caution the inputs made possible
was strongest mid-run and eroded as selection continued, because nothing in the
fitness function pays for it: `survival_cap` is 120 against a food term worth
~44,500 at score 46, so being careful is nearly free and being aggressive scores
better on the episodes selection actually sees.

Giving the network the information was not enough. The next lever is the fitness
function, not another encoding.

## egocentric_v3 — space measured over time

Space was measured against a frozen body, which asks the wrong question at
length: the cells directly ahead of a snake following its own tail are vacated
exactly as it arrives. v3 searches outward in move order and lets each body cell
clear on the move its segment leaves, bounded by `features.space_horizon`.

On a coiled snake the frozen scan reports 2 reachable cells; the timed scan
reports 81.

| | v2 | v3 |
| --- | --- | --- |
| held-out mean | 36.4 | **46.2** |
| held-out median | 36.0 | **46.5** |
| worst of 60 | 15 | **26** |
| 25th percentile | 29.0 | **35.8** |
| episodes scoring 50+ | 6 | **24** |
| starvation deaths | 17% | **3%** |
| steps per second | 5,958 | 3,715 |

+9.8 mean, reported at the time as 4.9 standard errors — of the wrong quantity,
and inside seed noise; see "Training-seed variance". It did what it was built for —
starvation nearly vanished, because a snake that can tell a tail chase from a
trap stops stalling, and its episodes ran nearly twice as long. It did not touch
self-collision, which rose to 85% of deaths purely by being what was left.

In-run peak said v3 *lost*: 62 against v2's 69.

Also checked: `step_limit` appeared at 10% of v3's deaths, and the truncated
episodes were the best ones (mean score 59). Raising `max_steps` from 3,000 to
12,000 moved the held-out mean from 46.2 to 46.4 — they die of self-collision
later. No change made.

## egocentric_v2 — uncapped space scan

v1 capped the flood fill at `flood_fill_limit`. When two candidate moves opened
onto the same free region, the first consumed the budget and the second started a
fresh truncated scan, reporting a different number for identical space. Three of
eleven inputs carried mostly scan-order noise.

Measured against a strong policy: the board is genuinely divided into
different-sized regions on **3.1%** of steps, while v1 reported a difference on
**53.7%**.

| | v1 | v2 |
| --- | --- | --- |
| held-out mean | 19.6 | **36.4** |
| held-out median | 24.5 | **36.0** |
| 25th percentile | 0.0 | **29.0** |
| worst of 60 | 0 | **15** |
| episodes scoring 40+ | 2 | **24** |
| steps per food | 58.7 | **32.2** |
| steps per second | **9,382** | 5,958 |

7.3 standard errors apart. v1's champion scores **zero on a quarter of unseen
seeds**; v2's worst episode still scores 15. v2 is 1.6x slower per step and wins
anyway.

The diagnosis that produced v2 was wrong. It was built expecting space inputs to
saturate at long lengths; measurement showed saturation was constant across
lengths, and that v2 reported *less* apparent signal than v1 — because most of
v1's signal was fabricated. The change was right for a reason that only became
visible after building it.

## The rebuild

The original code plateaued at score 2 and stayed there. Three defects each
independently prevented learning.

**Fitness was a lifetime ratchet.** `individual.fitness = max(fitness, new)`,
`clone()` copied the figure forward, and nothing reset it. After six generations
the top five by fitness had never been evaluated under their current identity —
a champion lineage had accumulated 15 evaluation episodes against a newcomer's 3,
so selection compared best-of-15 with best-of-3 and the incumbent won on sampling
luck. `load_checkpoint` restored those figures, so a restart resurrected the
frozen leaderboard.

**Scores took the best of several trials, not the mean**, which rewards a
favourable food sequence rather than a better policy, and compounds the ratchet.

**The network z-scored its own input vector.** Holding the snake still and moving
only the food changed the 3x3 danger readings by 0.41, against a whole-vector
standard deviation of 0.365 — the corruption was larger than the signal.

Fixing only those three, on an otherwise unchanged codebase: best score 3 → 7
over 30 generations, and mean score climbed instead of staying flat.

| | original | rebuilt |
| --- | --- | --- |
| best score, ~20 generations | 2 | 31 |
| mean score | 0.19 | 6.44 |
| episodes that ate | 17% | 91% |
| mean survival | ~20 steps | ~520 steps |
| throughput | 2,955 steps/s | 11,345 steps/s |
| memory | 5.08 MB per generation, unbounded | flat |
| genome | 1,795 params | 435 params |
| inputs | 37 (13 effective) | 11 |

### Other measurements from the rebuild

- **Debug strings cost 56% of runtime.** 84 `debug_ga(f"...")` call sites built
  f-strings that were discarded, because the argument is evaluated before the
  function checks its flag. Several interpolated numpy arrays, invoking numpy's
  array-printing machinery. Removing them: 27 → 61 episodes/s.
- **The feature vector was 37 inputs carrying 13.** Thirteen pairs at |r| =
  1.0000; `wall_up` and `cwall_up` were the same computation; the centre cell of
  the 3x3 body map was the head and therefore always zero.
- **Crossover preserved less parent behaviour than mutation** — 76% child/parent
  decision agreement at best against 86–97% for mutation — so crossover dropped
  from 0.7 to a minority operator.
- **torch was a 0.48 GB dependency storing zero tensors**, used only to pickle a
  dict of numpy arrays. Checkpoints moved to `.npz`; `.pt` became an explicit
  export.
- **The flood fill was 90% of runtime** on first implementation. Rewriting it as
  one component-labelling pass over a padded integer grid took the whole system
  from 17 to 82 episodes/s.
- **`torch_available()` took 129 seconds** under GIL contention with the trainer,
  because it imported torch to answer. It now uses `importlib.util.find_spec`.
- **Replay downsampling made bodies unreconstructable** — sampled head positions
  are not adjacent, so snakes rendered as scattered squares. Replays now store
  one direction byte per step; 258 steps fit in 344 base64 characters.

---

## Method notes

**Score held-out seeds, not the run.** In-run peak has pointed the wrong way
twice. It is one lucky episode out of 36,000. Median and worst-case separate
policies; peak does not.

**One variable at a time, and more than one seed.** Encoding comparisons ran at
the same seed for the same generation count. A shared seed makes generation 0
identical, but runs diverge from the first breeding step, so it does not remove
seed luck — one configuration at six seeds spanned 30.4 to 68.4.

**Report the right standard error.** The one used for most of this file measured
noise across held-out seeds for a single champion: 1.5 points. The noise that
decides a comparison is the spread between training runs: 14.6 points, and it
shows up only by running each configuration at several seeds. v2 -> v3 was
reported at 4.9 standard errors of the wrong quantity; measured against the
right one it is a quarter of what a single-seed comparison can resolve.

**Score with the project's own command.** Held-out figures up to the
training-seed experiment came from one-off scripts. `python -m snake_ai evaluate`
reproduces all four champions of that experiment exactly — mean, median, worst,
best and causes of death — so the earlier figures stand, and new ones should
come from the command rather than another script.

**Check the ceiling before building the fix.** Champion selection had a clear
mechanism, a published precedent and a 15.6-point failure to point at. Measuring
what a perfect selector would have been worth -- 1.9 points -- took one scoring
pass over runs already on disk and would have closed the question before any code
was written.

**Check which artefact is being measured.** Two conclusions in this project were
drawn from a two-generation-old checkpoint picked up by a glob that sorted to the
wrong run. Both dissolved when re-run against the intended one.
