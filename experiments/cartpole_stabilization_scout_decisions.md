# Cartpole Stabilization Scout Decision Log

This append-only log records the single 50k stabilization run used to choose delay-intervention boundaries.

## 2026-08-24 - Frozen scout design

- Use a fresh fixed-$h=3$ Cartpole agent with constant delay zero. Adaptive horizon selection and the delay schedule are disabled so neither can be mistaken for ordinary learning saturation.
- Evaluate ten deterministic episodes every 2,500 collected transitions, beginning before seed-data pretraining has finished. With 500 transitions per Cartpole episode, each interval is approximately five training episodes.
- Define the final reference level as the mean of the last five evaluation means. Report two plateau estimates: the first four-point sequence at or above 95% of that reference, and the first five-point window with coefficient of variation at most 2%, relative slope at most 0.5% per 1,000 transitions, and mean at or above 95% of the final reference.
- Use the resulting transition and episode estimates to place perturbation boundaries. Do not tune a boundary from a single transient evaluation point.
- The scientific training, environment, evaluation, checkpoint, and renderer code is unchanged from the validated pilot revision `e4274dc`. This scout adds only a named 50k launcher mode, analysis code, tests, and tracking files; the prior one-GPU and EGL-rendering evidence therefore remains applicable, while the new revision still requires the full NCC CPU suite before submission.
