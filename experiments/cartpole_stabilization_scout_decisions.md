# Cartpole Stabilization Scout Decision Log

This append-only log records the single 50k stabilization run used to choose delay-intervention boundaries.

## 2026-08-24 - Frozen scout design

- Use a fresh fixed-$h=3$ Cartpole agent with constant delay zero. Adaptive horizon selection and the delay schedule are disabled so neither can be mistaken for ordinary learning saturation.
- Evaluate ten deterministic episodes every 2,500 collected transitions, beginning before seed-data pretraining has finished. With 500 transitions per Cartpole episode, each interval is approximately five training episodes.
- Define the final reference level as the mean of the last five evaluation means. Report two plateau estimates: the first four-point sequence at or above 95% of that reference, and the first five-point window with coefficient of variation at most 2%, relative slope at most 0.5% per 1,000 transitions, and mean at or above 95% of the final reference.
- Use the resulting transition and episode estimates to place perturbation boundaries. Do not tune a boundary from a single transient evaluation point.
- The scientific training, environment, evaluation, checkpoint, and renderer code is unchanged from the validated pilot revision `e4274dc`. This scout adds only a named 50k launcher mode, analysis code, tests, and tracking files; the prior one-GPU and EGL-rendering evidence therefore remains applicable, while the new revision still requires the full NCC CPU suite before submission.

## 2026-08-24 - Live cadence observation

- Job 4437 is healthy and produced finite evaluation means of 169.53 at 5k transitions and 232.96 at 10k transitions.
- Although the launcher requests a 2.5k interval, collection advances the global transition counter in increments of eight. Since 2,500 is not divisible by eight, the exact modulo trigger is first reachable at 5,000 and then every 5,000 transitions. Keep this run as a valid coarse scout, but interpret its plateau resolution as 5k transitions, or approximately ten aggregate 500-step episodes.
- If the coarse result cannot place a perturbation boundary unambiguously, use an interval divisible by eight (for example 2,000 or 2,400 transitions) in a follow-up measurement-only scout; do not silently reinterpret the missing 2.5k points.

## 2026-08-24 - Postprocessing validator diagnosis

- Training, both requested anchor checkpoints, both rendered anchor rollouts, the ten finite evaluations, and the run manifest completed at 50k. Slurm nevertheless recorded exit code 2 because the validator inferred the old 24k smoke contract solely from `min_steps < 500000`.
- Repair validation by naming the contract explicitly as `full`, `smoke`, or `stabilization`. This is an audit-only change: preserve job 4437 and its scientific commit and revalidate the existing artifacts rather than rerunning training.
