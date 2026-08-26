# Pendulum horizon-8 torque-0.2 decisions

## Frozen design

The experiment reuses one stable nominal Pendulum checkpoint at 30k transitions. The branch restores the agent, optimizer, replay buffer, global step, and horizon-controller state, then changes only the fixed actuator-strength scale from 1.0 to 0.2. Horizon evidence is sampled from the newest 4k transitions, while learner updates retain the full replay history. The unrestricted proposal and deployed decision both consider horizons 2 through 8. The controller starts at horizon 3 and may jump directly to horizon 8; this endpoint test isolates score pressure from the one-step transition restriction.

Seed 7 is frozen because prior nominal Pendulum runs learned reliably by approximately 30k transitions. The branch runs only if the last three base evaluations have mean at least 700 and coefficient of variation at most 0.15.

The online queries occur at 34k, 38k, and 42k. A higher-replication full-horizon reference query is recorded at 42k. A result is scientifically valid even if horizon 8 is not selected; selection is an outcome rather than an artifact-validity condition.

## Launch

The clean scientific revision is `b74fad7fe81158c1d076f9584712322c5499c3df`. The remote CPU gate passed 28 tests. Slurm job 4513 is the nominal 30k base. Slurm job 4514 is the torque-0.2 continuation with dependency `afterok:4513`. The frozen base and branch configuration hashes are `88426a0298c5b8e1` and `39d50886194d893b`, respectively.

## Attempt-1 artifact repair

Job 4513 completed in 21 minutes and passed the frozen base stability gate. Job 4514 completed all branch training, three online queries, the 42k reference query, and the final 46k evaluation. It then failed artifact validation because its 16k checkpoint interval was interpreted against the absolute global step and therefore did not accept a save at 46k. This is an artifact-only failure after scientific execution. The smallest repair changes the branch save interval to 23k, which divides the absolute terminal step 46k, and reruns only the affected continuation from the preserved valid 30k parent.

The attempt-1 diagnostic outcome was negative for the horizon-8 hypothesis. Online proposed and selected horizons were 2 at 34k, 38k, and 42k; the 42k high-replication reference also proposed 2. Evaluation return was 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.2, and 100.0 at 32k through 46k in 2k increments. The torque-0.2 endpoint appears close to infeasible for the inherited policy and model, rather than a regime that reveals a useful long horizon.
