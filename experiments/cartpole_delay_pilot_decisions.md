# Cartpole Delay Pilot Decision Log

This append-only log records launch gates, job failures, controlled fixes, retries, and scientific decisions for the seven-profile Cartpole delay pilot. Existing CoRL campaign ledgers and decisions remain immutable.

## 2026-08-23 — Campaign isolation

- Created a dedicated pilot goal and ledger rather than extending the completed CoRL v3 campaign.
- Frozen the intended matrix at six delay-intervention runs (additive, multiplicative, scripted; seeds 1 and 23) plus one canonical fixed-horizon, no-delay control.
- Full 500k launches are blocked until CPU tests, one GPU smoke, and one EGL GIF-rendering smoke pass from the exact committed revision.
- The canonical parity reference is the retained six-seed JAX fixed-$h=3$ Cartpole result (mean 873.402, sample SD 8.648); no externally sourced official Cartpole target is present in the repository.
- Any deterministic code, non-finite, or OOM failure requires a recorded diagnosis and single-GPU reproducer before retry. Scientific configuration changes require a named revision and invalidate unmatched partial runs.

## 2026-08-23 — Pre-launch scientific correction

- Replaced the proposed clean single-seed parity rerun with a fixed-$h=3$ controller under the same observable $0\rightarrow4\rightarrow0$ delay intervention. This is the control needed to isolate adaptive horizon selection; the retained six-seed clean campaign remains the stronger parity reference.
- Disabled phase pruning for this nonstationary pilot. Candidate evaluation remains local (incumbent and immediate neighbors), but every horizon in $2{:}8$ stays accessible after both intervention boundaries.
- Designated additive seed 1 as the calibration arm. It saves the exact replay batches, all 64 roughness directions for every horizon, model-stage timing for $M\in\{0,2,4,8,16,32,64\}$, all-horizon paired $K=128$ query-planner evaluations over 256 steps, and separate all-horizon $K=32$ deployed-planner evaluations over the full 500-step episode at 100k, 250k, and 450k. These shadow results are an additive-agent mechanism check, not an oracle for the other learned agents.
- Frozen artifact challenges at constant delay 0 and constant delay 4, with the same reset pool and planner seed across runs, seeds, and checkpoints. The paired GIF is qualitative; numeric trajectory summaries are the quantitative observable.
- The arm called `multiplicative` is the cleaned incumbent-relative log utility, not the historical per-query min--max product. The pilot compares two coherent fixed-reference utilities; it does not directly rerun the legacy score.

## 2026-08-23 — Independent pre-launch audit

- Corrected the all-horizon shadow evaluation so horizons above the deployed horizon receive a full $h_{\max}=8$ plan buffer while retaining the deployed planner's 512/24/64/6 settings.
- Renamed that $K=32$ shadow result a conditional learned-model/deployed-planner reference. It now reports paired gap uncertainty and replica-wise best frequencies; it is not described as a population oracle.
- Preserved deployed, $K=128$ query-planner reference, and $K=32$ conditional-reference returns as three explicit raw sources. Terminal validation requires every planned anchor, horizon, probe direction, and reference replica.
- Added directional bootstrap intervals for every nested probe count. $M=64$ remains a finite-stack reference rather than ground truth.
- Bound acceptance to the resolved Hydra configuration as well as the commit/profile identity. Each Slurm job uses a detached exact-commit source snapshot so a later diagnosed fix cannot mutate a running job's scripts.
- Hardened the steward with attempt-specific Slurm identities, single-attempt non-idempotent submission, durable lost-response reconciliation, normalized terminal states, and launch-revision validation. The intended four-then-three launch transition passed a synthetic controller test.
- Raised the wall-time request to 24 hours for the calibration arm; this does not reserve additional GPUs and avoids losing a nearly complete run to the heavier reference and rendering stages.

## 2026-08-23 — Exact smoke and transport-only correction

- Exact-revision smoke job 4375 at commit `989b63b5382fd9da9d0d90e7b90872faea0987d3` completed 0:0 in 16:59. It passed all 59 NCC CPU tests, the online horizon query, every frozen timing/reference grid, two EGL GIF anchors, and terminal artifact validation.
- The first controller tick submitted no jobs because the local password-helper transcript appeared on stdout and was conservatively mistaken for a pre-existing Slurm record. The ledger remained empty.
- The follow-up revision filters only the local Expect `spawn` and password-prompt lines. No file under `tdmpc2_jax/`, no Slurm experiment script, configuration, finalizer, renderer, or data-producing path changed.
- The 4375 GPU/EGL evidence is therefore carried forward across this orchestration-only revision by an explicit exception. CPU/controller tests and a live read-only NCC transport check must pass at the new revision before launch; the gate records both the launch revision and the evidence revision.
