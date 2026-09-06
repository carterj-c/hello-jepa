# hello-jepa

Infrastructure smoke test: does a JEPA trained with VICReg learn anything from a
synthetic financial time series whose ground truth we control?

This tests the *pipeline*, not a modelling result. The synthetic data exists so
that "did it learn anything" has a checkable answer.

## Hypothesis

> **H0** — A JEPA model using VICReg does not meaningfully explain more than a
> simple logistic regression on a raw flattened window.

**Result: H0 is not rejected, in every configuration tried.** No trained run beat
the raw-window baseline's 0.817, and on the temporal metric every trained run
scored *below* its own untrained control.

## Data (`src/HMM_data.py`)

A 3-state hidden Markov chain (bull / bear / kangaroo) driving four channels: log
returns, realized volatility, log volume, and a slow regime-independent drift.
Each channel is its own AR(1) with an **independent** shock — a shared shock would
let any model solve the task by isolating one common factor, so a module-level
assert keeps cross-channel correlation below 0.05.

| regime | share | dwell | σ (returns) | φ (lag-1) | log-vol mean |
|---|---|---|---|---|---|
| bull | 0.58 | 100 | 0.008 | +0.05 | 0.00 |
| bear | 0.11 | 29 | 0.022 | +0.35 | 0.55 |
| kangaroo | 0.31 | 36 | 0.022 | −0.30 | 0.35 |

T = 20,000 steps, 500 burn-in, cut into 19,437 windows of (64, 4). Split 70/30 by
time with 127 windows dropped between train and test; standardization fit on
training rows only. No window shares a timestep across the split.

**Why bear-vs-kangaroo is reported separately.** Bull is 58% of the data and has
2.75× lower volatility, so overall accuracy is dominated by an easy class. Bear and
kangaroo have *identical* return volatility and differ mainly in the **sign of their
lag-1 autocorrelation**, so separating them requires modelling how consecutive
timesteps relate, not just how large they are. It is the column that says whether
temporal structure was learned.

That claim is tested, not assumed: the `window-mean` baseline uses per-channel means
only and contains no temporal information by construction. It scores 0.695 against a
chance level of 0.692 — the log-volume mean offset (0.55 vs 0.35) leaks no usable
amplitude shortcut.

## Results

Every row is a multinomial logistic probe on a frozen representation, fit on train,
evaluated on the time-held-out test split.

| representation | test acc | bear vs kang |
|---|---|---|
| majority class (chance) | 0.606 | 0.692 |
| window-mean (amplitude only) | 0.644 | 0.695 |
| vol only | 0.670 | 0.692 |
| vol + lag-1 autocorr | 0.691 | **0.797** |
| mean + vol + autocorr | 0.691 | 0.790 |
| **raw flattened window (the H0 bar)** | **0.817** | 0.725 |
| HMM (windowed, causal) | 0.712 | 0.501 |
| HMM (full Viterbi, non-causal) | 0.720 | 0.496 |
| random-init encoder (null control) | 0.735 | **0.713** |
| JEPA + VICReg, cov=1 (defaults) | 0.765 | 0.708 |
| JEPA + VICReg, cov=0.1 | 0.754 | 0.702 |
| JEPA + VICReg, cov=0.01 | 0.752 | 0.690 |
| JEPA + VICReg, cov ablated | 0.747 | 0.687 |

![results](figures/sweep/results.png)

The random-init control is what makes this readable. A frozen, *untrained*
transformer already scores 0.735, so comparing against chance (0.606) would have
shown a spurious +0.16 and the run would have been called a success.

## The coefficient sweep, and why it failed

The first run's diagnosis was that the covariance regularizer supplied 90.6% of the
total loss reduction while the prediction term supplied 3.1% — so the obvious fix
was to shrink `cov_coeff` and let prediction drive optimization. **That fix was
tried and it made things worse, monotonically.**

| run | cov_coeff | final pred MSE | test acc | bear vs kang |
|---|---|---|---|---|
| baseline | 1.0 | 0.909 | **0.765** | **0.708** |
| cov ×0.1 | 0.1 | 0.679 | 0.754 | 0.702 |
| cov ×0.01 | 0.01 | 0.628 | 0.752 | 0.690 |
| cov ablated | 0.0 | 0.630 | 0.747 | 0.687 |
| *random-init null* | — | — | *0.735* | *0.713* |

Prediction MSE improved by 31%. Probe accuracy fell. Bear-vs-kangaroo fell. **Better
prediction bought a worse representation**, and the ordering is perfectly monotone in
`cov_coeff`.

The obvious explanation — that decorrelated features merely condition a linear probe
better — is **wrong**. PCA-whitening the embeddings before probing changes nothing
(0.765 → 0.764 for cov=1; 0.747 → 0.745 for cov=0). The covariance term is
preserving real information, not just improving conditioning.

## Prediction MSE is a confounded metric here

![prediction MSE](figures/sweep/sweep_prediction_mse.png)

Every run's MSE **bottoms out at epoch 1–3 and then gets worse**, while effective
rank climbs over the same span:

| run | best MSE | @ epoch | final MSE | eff_rank (best → final) |
|---|---|---|---|---|
| baseline | 0.773 | 1 | 0.909 | 5.6 → 13.6 |
| cov ×0.1 | 0.401 | 2 | 0.679 | 6.6 → 12.0 |
| cov ×0.01 | 0.323 | 3 | 0.628 | 7.7 → 12.4 |
| cov ablated | 0.320 | 3 | 0.630 | 7.6 → 12.0 |

The target is not fixed — it is the EMA encoder's own output, and its richness grows
during training. A low MSE against a rank-6 target is not better than a higher MSE
against a rank-13 target; it may simply mean there is less to predict. This is the
same trade visible across the sweep: shrinking `cov_coeff` lowers MSE *and* lowers
effective rank *and* lowers probe accuracy, together.

**So the original "the predictor explains only 9.1% of target variance" framing was
itself misleading**, and this sweep is what exposed it. Raw MSE cannot be compared
across runs or across epochs while target rank is moving. Any future progress metric
has to be rank-normalized, or measured against a frozen target.

## Not collapse

`emb_std` stayed near 1.0 in all runs and never approached 0, and effective rank
*rose* over training (max possible 128). The representations are spread out; they
are simply not predictive of regime beyond what an untrained network already gives.

![collapse diagnostics](figures/inv25_var15_cov1_e30_s0/collapse.png)

## Two things the baselines show

- **The HMM is the strongest principled baseline and is structurally blind.** 0.712
  overall — it has the correct 3-state structure, since the data really was generated
  by a Markov chain — but **0.501 on bear-vs-kangaroo, a coin flip**. A Gaussian HMM
  treats observations as conditionally independent given the state, so it cannot
  represent within-regime autocorrelation. Unlimited non-causal context does not help
  (full Viterbi: 0.496). A model-class limit, not a context-length one.
- **The raw-window baseline is strong overall but weak temporally** (0.817 / 0.725)
  because lag-1 autocorrelation is a *second-order* statistic — a product of two
  timesteps — which a linear model on raw inputs cannot compute. Hand-feeding it gets
  0.797. That gap is the room a working sequence model should be filling.

## Verdict on the infrastructure

The pipeline works: generator validated against its own parameters, no leakage, EMA
target encoder and stop-gradient training stably, collapse instrumentation live,
probes and controls in place, runs versioned and reproducible. It produced an honest
negative result and then falsified its own first explanation for that result, which
is what a smoke test is for.

## Next steps

Ranked by what would actually change the answer:

1. **Stop training at the MSE minimum (epoch 2–3) and probe there.** The cheapest
   test of whether late training is destroying the representation. Currently every
   probe is taken at epoch 29, well past the point where prediction peaked.
2. **Fix the progress metric before tuning anything else.** Rank-normalize the
   prediction loss, or evaluate against a frozen target encoder, so MSE across runs
   becomes comparable. Right now there is no trustworthy scalar to tune against.
3. **Revisit the masking scheme.** 8 patches with a 2–4 patch target block may leave
   too little context; the same block is used for the whole batch.
4. Only then compare against 0.817 again. The number to care about is
   bear-vs-kangaroo above 0.725.

## Layout

```
src/HMM_data.py     synthetic generator, validation, windowing, time-split
src/jepa.py         patch encoder, EMA target encoder, predictor, training loop
src/probes.py       baselines, HMM, null control, linear probes
src/experiment.py   versioned sweep runner
src/figures.py      figure generation
src/vicreg_loss/    vendored from github.com/jolibrain/vicreg-loss (MIT, JoliBrain)

runs/baselines.json          cached non-JEPA baselines
runs/<run_id>/metrics.json   history + probe results for one run
runs/manifest.json           every run_id
figures/<run_id>/*.png       per-run figures
figures/sweep/*.png          cross-run comparisons
```

Run ids encode their own configuration — `inv25_var15_cov0.1_e30_s0` is
`inv_coeff=25, var_coeff=15, cov_coeff=0.1, epochs=30, seed=0` — and every figure
carries its run id and timestamp in the footer, so any chart traces back to the run
that produced it.

```bash
uv run python src/HMM_data.py            # regenerate + print generator validation
uv run python -m src.experiment          # run the full sweep (cached; --refresh to redo)
uv run python -m src.figures             # regenerate all figures
```

Seeded throughout (seed 0); the numbers above reproduce.
