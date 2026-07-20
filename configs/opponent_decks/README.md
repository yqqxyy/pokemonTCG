# Opponent deck hypotheses

These 60-card ID lists are small public archetype snapshots used to exercise the
MCTS opponent-belief interface. They are not assumed to be a complete or current
ladder metagame:

- `meta_a.csv`: public Meta A / stable-submission snapshot;
- `fishcat_v8.csv`: public Fishcat V8 submission snapshot;
- `mcts_sample.csv`: deck from the public RL/MCTS sample notebook.

The official sample deck remains under `data/official` and is intentionally not
copied into Git. Supply any newer local deck list with another repeated
`--opponent-deck NAME=PATH` argument.
