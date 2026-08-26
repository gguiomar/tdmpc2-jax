# Pendulum frontier-atlas decisions

## Why this batch comes before another adaptive run

The completed torque screen did not identify a robust long-horizon rescue band. At torque 0.4 and 0.6, both controllers were weak and h=8 was worse; at torque 0.8, control remained possible but h=8 did not dominate the evaluation curve. Launching another adaptive run at an arbitrary torque would therefore confound a bad decision rule with an environment in which a long horizon is not useful.

The next scientific run is instead one frozen-checkpoint frontier atlas. It restores the same validated seed-7 checkpoint at 30k transitions and never updates the network. For every environment condition, horizons 2 through 8 are evaluated on the same 32 reset/noise replicas for 500 steps. This isolates planning-horizon value and zero-shot generalization from learning drift.

## Frozen atlas

The atlas changes one variable at a time around the nominal Pendulum: actuator-strength scale, joint-damping scale, gravity scale, hidden action delay, and observation-noise scale. Shared nominal values are deduplicated, giving 24 conditions. The physical damping and gravity scales are applied to the MuJoCo model before it is lowered to MJX. The fixed observation-noise hook changes only observation noise, without enabling actuator, wind, push, slip, or jitter randomization. Hidden delay deliberately leaves the observation dimension unchanged so the 30k checkpoint remains compatible.

All horizon candidates use the same controller-query planner budget: population 256, 12 policy-prior samples, 32 elites, four MPPI iterations, temperature 0.5, and planning capacity eight. This matches the budget that an online adaptive query would use. The four Slurm shards are computational partitions of one scientific atlas, not independent training runs.

## Stage gate

A condition is called horizon-rescuable when the best horizon in `{6,7,8}` has a positive paired 90% lower confidence bound relative to h=3. The full curve is retained; this binary rule is only the gate for spending the next three online runs. If an ordered axis also shows a progressive best-horizon change, its two stable endpoints become the online schedule.

Only after that gate passes do we launch the matched online trio from the same 30k checkpoint: fixed h=3, fixed h=8, and the full adaptive controller. If no condition is rescuable, those jobs remain unsubmitted and we revise the intervention or source checkpoint rather than interpreting another failed switch.

