# Pendulum score-formulation screen decisions

## Frozen design

All active arms restore the same full 34k Pendulum checkpoint and traverse hidden action delays `0 -> 2 -> 6 -> 4 -> 0` in five 4k-transition blocks. The adaptive candidate set is `h=2..8`; deployed queries occur every 2k transitions; evaluation return is measured every 400 transitions. Six 128-replica shadow frontiers and S0--S3 shadow decisions are observational and cannot change deployment.

The screen compares: S0, the current additive score; S1, a paired-return LCB minus discounted per-step excess roughness; S2, a return-first credible set with roughness and switching cost used only for tie-breaking; and S3, paired-return LCB minus curvature-conditioned local model risk and Bellman inconsistency. Historical anchors give identical oracle agreement for return weights 10, 20, and 40, so the predeclared lowest tied value, 10, is used.

Promotion uses external evidence only: normalized evaluation-return AUC, final clean return, mean 128-replica oracle regret, phase-wise adaptation, switching lag, chatter, recovery, and compute. An arm must lie within 5% of the best AUC and clean return and pass the stability rules. Among eligible arms, lowest oracle regret wins; compute breaks a remaining tie.

## Checkpoint reconstruction gate

The archived 34k model anchor contains the agent and horizon state but no replay buffer. It is therefore not a valid training continuation source. Before the screen, one deterministic 30k-to-34k source job reproduces the original clean adaptive continuation and writes a full composite checkpoint containing the agent, replay buffer, horizon state, and global step. Every S0--S3 arm must fork that single validated source directory. No arm may mix the 34k model with the later 46k replay buffer.

## Source reconstruction result

Slurm job 4632 completed at scientific revision `856f878` and produced a finite, validator-approved 34k composite checkpoint with the agent, replay buffer, global step, and adaptive horizon state. The reconstructed and archived 32k controller decisions agree exactly on the scientific action: both propose and select `h=2`. Across the nine common clean evaluation points before 34k, the return-mean MAE is 5.32 points (maximum 16.5), and the training traces remain closely aligned; this is consistent with ordinary GPU numerical variation rather than a different continuation.

The archived run's evaluation return drops at exactly 34k because its hidden delay-four phase begins at that boundary, while the source reconstruction intentionally remains at delay zero. In the MJX loop, the composite checkpoint is saved before the boundary evaluation, so this expected evaluation difference does not contaminate the 34k training state. The source provenance gate therefore passes. The raw 32k score values are not required to match bit-for-bit because the stochastic deployment probe is sensitive to small numerical trajectory differences; equality of the chosen/proposed horizon plus the aligned pre-boundary learning trace is the relevant gate.

## Smoke attempt 1 diagnosis

Slurm job 4633 completed training, the terminal composite checkpoint, all six d=0/4/6 trajectory trios, and all EGL GIF/PNG renders, but strict validation correctly rejected its query cadence: it emitted only the first query at 34.4k. The fork logic reset `next_query_step` to the new protocol but left `query_interval_steps=4000` inside the restored horizon state, so the next query was scheduled beyond the 36k smoke endpoint. The smallest repair makes a requested schedule reset adopt both the new first query and the new interval. This is a protocol-enforcement fix; it does not change any frozen score, delay, model, or controller setting. Attempt 1 is retained as failed evidence and will not be used scientifically.
