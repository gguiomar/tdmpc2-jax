# Pendulum score-formulation screen decisions

## Frozen design

All active arms restore the same full 34k Pendulum checkpoint and traverse hidden action delays `0 -> 2 -> 6 -> 4 -> 0` in five 4k-transition blocks. The adaptive candidate set is `h=2..8`; deployed queries occur every 2k transitions; evaluation return is measured every 400 transitions. Six 128-replica shadow frontiers and S0--S3 shadow decisions are observational and cannot change deployment.

The screen compares: S0, the current additive score; S1, a paired-return LCB minus discounted per-step excess roughness; S2, a return-first credible set with roughness and switching cost used only for tie-breaking; and S3, paired-return LCB minus curvature-conditioned local model risk and Bellman inconsistency. Historical anchors give identical oracle agreement for return weights 10, 20, and 40, so the predeclared lowest tied value, 10, is used.

Promotion uses external evidence only: normalized evaluation-return AUC, final clean return, mean 128-replica oracle regret, phase-wise adaptation, switching lag, chatter, recovery, and compute. An arm must lie within 5% of the best AUC and clean return and pass the stability rules. Among eligible arms, lowest oracle regret wins; compute breaks a remaining tie.

## Checkpoint reconstruction gate

The archived 34k model anchor contains the agent and horizon state but no replay buffer. It is therefore not a valid training continuation source. Before the screen, one deterministic 30k-to-34k source job reproduces the original clean adaptive continuation and writes a full composite checkpoint containing the agent, replay buffer, horizon state, and global step. Every S0--S3 arm must fork that single validated source directory. No arm may mix the 34k model with the later 46k replay buffer.
