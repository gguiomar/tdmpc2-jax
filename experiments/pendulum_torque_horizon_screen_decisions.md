# Pendulum torque-by-horizon screen decisions

## Frozen design

The screen forks the validated seed-7 Pendulum checkpoint at 30k transitions. Every child restores the same agent, optimizer, replay buffer, and global step. The only scientific interventions are the fixed actuator-strength scale in `{0.4, 0.6, 0.8}` and the fixed TD-MPC2 training/planning horizon in `{3, 8}`. Dense-RHS and all horizon queries are disabled.

Each child continues from 30k to 46k transitions and is evaluated every 2k transitions with ten deterministic 500-step episodes. The comparison of interest is paired within torque: `h=8 - h=3`. This is a regime-location screen, not a final statistical comparison; seed confirmation follows only after a torque level shows both retained controllability and a material long-horizon advantage.

The parent is accepted only if its existing validator confirms the 30k checkpoint and the frozen stability rule. Before the six children can start, a separate torque-0.6, h=8 continuation to 30.8k must restore the composite checkpoint, train, evaluate, save, and pass artifact validation on one GPU.

## Interpretation rule

A torque endpoint is useful for the next adaptive experiment only if control remains non-degenerate and h=8 improves over h=3. A joint collapse of both horizons is an infeasible endpoint, not evidence for either horizon. Similar strong returns at both horizons indicate that torque level does not identify horizon necessity. The best next adaptive condition is the weakest controllable torque with the clearest positive `h=8 - h=3` gap.
