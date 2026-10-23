# Path-ordered geometric attention

Attention whose aggregation is an ordered composition of learned rotations along the path between
two positions, rather than a weighted average over them.

The practical consequence: the mechanism can carry **the transformation between two moments**, not
only the state at each. On a task that requires applying the changes made between two points in a
sequence to a different starting state, every standard memory fails and this one does not — including
baselines that track the sequence's state almost perfectly.

| Cherry-pick: apply the changes between markers A and B to a new starting state | exact match | state tracking |
|---|---|---|
| **Path-ordered attention (on host activations)** | **0.996** | 1.000 |
| **Path-ordered attention (learned embeddings)** | **0.930** | 1.000 |
| rule: correct edits, order ignored | 0.580 | — |
| transformer, pre-LN + warmup | 0.056 | 0.887 |
| gated recurrent network | 0.030 | 0.997 |
| transformer with rotary positions | 0.025 | 0.945 |

The recurrent baseline tracks the state correctly at 99.7% of turns and still answers 3% of replay
questions. Two states do not determine the transformation between them.

---

## The idea

Standard attention combines what it finds as a weighted average, which is order-blind: the same set
of earlier positions produces the same result whatever order they occurred in.

Here, each link between adjacent positions carries a learned rotation, generated from the content of
the two positions it joins. The relation between positions *i* and *j* is the ordered product of the
rotations along the path between them, and the key and value of the earlier position are transported
by that product before being attended to.

Three properties follow. Rotations preserve length, so a distant position arrives undiminished.
Rotations in two or more dimensions do not commute, so the result depends on what happened in
between and in what order. And every position can still attend directly to any earlier one.

Full description: [Zenodo link].

---

## Reproducing

```bash
pip install -r requirements.txt

python session_cherry.py --selftest                  # verify the task and its labels
python session_cherry.py --arms holo3:0.05 gru standard rope:16 --n-sess 20000 --epochs 30
python checkpoint_baselines.py                       # error against span length
python variable_tracking.py                          # length extrapolation
```

Each script's `--selftest` checks its own labels and prints worked examples before any training, so
a failure in the task construction shows up before GPU time is spent.

Single T4, a few hours per run.

---

## What else was measured

**Span length.** Mean absolute error as the span of composed changes grows, training on spans 1–12:

| span | 4 | 8 | 12 | 16 | 24 | 32 | 48 |
|---|---|---|---|---|---|---|---|
| constant guess | .204 | .212 | .225 | .232 | .225 | .227 | .221 |
| **path-ordered** | .046 | .046 | .051 | .061 | .075 | .092 | **.127** |
| transformer | .047 | .141 | .224 | .236 | .228 | .229 | .222 |
| rotary positions | .043 | .134 | .239 | .249 | .261 | .299 | .291 |
| recurrent | .062 | .110 | .180 | .221 | .242 | .260 | .256 |

Error grows roughly linearly, consistent with information being kept and accumulating a small error
per step. The baselines reach the constant guess by span 16, which is what losing the information
looks like.

**Length extrapolation.** Variable tracking with distractors, trained at length 8:

| | 8 (trained) | 16 | 32 |
|---|---|---|---|
| transformer | 0.990 | 0.284 | 0.137 |
| **path-ordered** | 0.762 | 0.588 | **0.523** |

---

## The baseline audit

The transformer baseline initially scored 0.030 on the cherry-pick, and tracked the sequence state
at only 0.634 — a model that had learned nothing. Post-LN layers with no warmup do not train
reliably at that depth.

Rebuilt with pre-normalization, a final norm and a 500-step warmup applied to every arm, it reached
0.887 on state tracking. Its cherry-pick score went from 0.030 to 0.056, still below the
order-ignoring rule at 0.580. Rotary positions, which train more easily, reached 0.945 on state
tracking and 0.025 on the cherry-pick.

So the gap is not an artifact of a badly trained baseline. Three properly trained memories learn the
state and none of them carries the transformation.

---

## Limits

**It is not better at everything.** On tasks that only require holding the current value and
overwriting it — no reaching back, no composition — a gated recurrent network wins. Measured at
0.934 against 0.854 over 48 overwrite events, and the failure mode is drift: this mechanism rotates
slowly through long stretches where nothing relevant happens, where a gate can hold a value exactly.

**Exact identifiers belong elsewhere.** On real git histories, "did this commit change dependency
*K*" is answered at 0.962 by string matching and at best 0.24 by any memory tested, including this
one. Names and identifiers are the symbolic layer's job; this is for order, reach and change.

**One seed per configuration.** The numbers above are single runs. Treat differences of a few points
as noise; the large gaps are not.

**Comparison against PaTH and DeltaProduct is not included.** Those methods compose transformations
along a path too, with reflections rather than rotations, and are the closest related work. A direct
comparison on the same tasks has not been run.

---

## Related work

Yang et al., *PaTH Attention*, NeurIPS 2025, arXiv:2505.16381 — accumulated Householder
transformations as position encoding; names rotations as future work.
DeltaProduct; GRAPE, arXiv:2512.07805; *Why Do Accumulated Transformations Extrapolate?*,
arXiv:2606.24975.

---

## License and status

Code: [choose one — Apache 2.0 and MIT are the usual choices for research code].
The mechanism is the subject of a pending patent application; the license covers the code in this
repository.

Questions and corrections welcome — open an issue.
