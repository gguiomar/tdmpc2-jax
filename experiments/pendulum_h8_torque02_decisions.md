# Pendulum horizon-8 torque-0.2 decisions

## Frozen design

The experiment reuses one stable nominal Pendulum checkpoint at 30k transitions. The branch restores the agent, optimizer, replay buffer, global step, and horizon-controller state, then changes only the fixed actuator-strength scale from 1.0 to 0.2. Horizon evidence is sampled from the newest 4k transitions, while learner updates retain the full replay history. The unrestricted proposal and deployed decision both consider horizons 2 through 8. The controller starts at horizon 3 and may jump directly to horizon 8; this endpoint test isolates score pressure from the one-step transition restriction.

Seed 7 is frozen because prior nominal Pendulum runs learned reliably by approximately 30k transitions. The branch runs only if the last three base evaluations have mean at least 700 and coefficient of variation at most 0.15.

The online queries occur at 34k, 38k, and 42k. A higher-replication full-horizon reference query is recorded at 42k. A result is scientifically valid even if horizon 8 is not selected; selection is an outcome rather than an artifact-validity condition.
