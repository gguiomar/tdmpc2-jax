# Dense-RHS Experiment Decisions

This file is append-only campaign memory for `goal-experiment-steward`.

## Bootstrap State

- `1179` is the current clean no-RHS quadruped baseline: `924.65 ± 16.91`, runtime `01:15:58`.
- `1387` is the current clean Dense-RHS winner: `915.34 ± 24.33`, runtime about `01:29:59`, horizon path `2 -> 2 -> 3 -> 2 -> 3 -> 4`.
- `1600` credible transition with `p=.65,c=.06` stayed at `h=3` and underperformed; it was too conservative.
- `1618` credible transition with `p=.56,c=.03` moved to `h=4` but later switched to `h=5`, bucketed to `8`, and collapsed final eval despite good late training episodes.

## Open RFCs

- `expected_improvement_transition`: replace hard switch probability threshold with expected net benefit so the model can accept moderate but valuable transitions and reject high-risk late switches.
- `cap_or_uncertain_late_large_horizon`: avoid late large-horizon moves unless uncertainty-adjusted evidence is strong; this must be written as a general uncertainty rule, not a task-specific schedule.

## 2026-05-01 Expected-Improvement Transition RFC

Implemented `expected_improvement_transition` behind config flags:

- `dense_rhs.credible_transition_rule=expected_improvement`
- `dense_rhs.transition_risk_weight`
- `dense_rhs.transition_min_expected_net`

Decision equation:

`X_h = U(h) - U(h_current) - C(h, h_current)`

`EI_h = E[max(X_h, 0)]`

`EL_h = E[max(-X_h, 0)]`

`B_h = EI_h - transition_risk_weight * EL_h`

The proposed horizon is selected by `B_h` over the candidate set, and the switch is accepted when `B_h > transition_min_expected_net`. This keeps the transition rule Bayesian but avoids the brittle hard switch-probability threshold that made `1600` too conservative and `1618` too permissive late in training.

Added one smoke profile: `dense_rhs_ei_smoke_80k_vec8_s15_risk15_c003_start20k`. It must not launch until the code is committed/synced because this is not marked as a dirty smoke.

## 2026-05-01T10:15:53+00:00

Launched `dense_rhs_ei_smoke_80k_vec8_s15_risk15_c003_start20k` as job `1644`.

## 2026-05-01T14:29:12+00:00

Processed `dense_rhs_ei_smoke_80k_vec8_s15_risk15_c003_start20k` job `1644`: `ei_too_conservative`.

- SLURM state: `COMPLETED`
- Runtime: `00:39:23`
- Final eval: `241.28457641601562` ± `123.68336486816406`
- Best eval: `241.28457641601562` at step `80000`
- Horizon path: `3->3->3`
- Reason: EI smoke stayed at h=3 with non-positive expected net benefit at every query; lower risk/cost should be tested before a full run.
- Follow-up: Queued follow-up `dense_rhs_ei_smoke_80k_vec8_s15_risk05_c002_start20k`.
