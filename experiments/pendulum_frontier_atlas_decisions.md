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

## First smoke failure and minimal repair

Revision `c02bffadfbac6c789e0b112cd0a37043d56f6970` passed 48 authoritative CPU tests on NCC and revalidated the stable 30k source. Smoke job 4552 then failed before checkpoint restore or GPU evaluation because the new analyzer was invoked as a file under `scripts/`; that makes `scripts/`, rather than the repository root, Python's import root and prevented importing `tdmpc2_jax.frontier_atlas`. The `afterok` gate correctly kept array 4553 and reducer 4554 from running, so no scientific result or GPU atlas artifact was produced.

The smallest repair changes only the four analyzer invocations in the GPU and reducer batch scripts to module form: `python -m scripts.analyze_pendulum_frontier_atlas`. The frozen checkpoint, conditions, paired resets, horizons, planner budget, and statistical rule are unchanged. The failed attempt remains preserved as attempt 1; the repaired chain will use attempt 2.

## Repaired launch

Revision `4d1f3ace085a294026fef2a744c86be936f80bad` passed the module-import check, the same 48 CPU tests, and source validation. Smoke job 4555 completed in 46 seconds, restored the 30k checkpoint, evaluated all seven horizons for four paired replicas over 32 steps, wrote seven summary rows and 28 paired-return rows, and passed both artifact validators. Its nominal evaluation kernel took 20.818 seconds including first compilation.

The successful `afterok` gate released array tasks 4556_0 through 4556_3 at 15:31 UTC. Each task owns six of the 24 canonical conditions and one H200. Reducer job 4557 remains dependent on all four valid shards. The online fixed-h3, fixed-h8, and adaptive traversal runs remain unsubmitted until the aggregate applies the frozen horizon-rescue rule.

## Terminal atlas result

All four shards completed and passed their finite-value, identity, cell-coverage, paired-replica, checkpoint, and marker checks. Each shard produced 42 summary rows and 1,344 paired-return rows. The four-GPU makespan was 5m58s. Reducer 4557 then validated all 24 canonical conditions, 168 summary cells, and 5,376 paired returns and wrote `AGGREGATE_VALID`.

Only hidden action delay 3 and 4 passed the predeclared long-horizon rescue rule. At delay 3, h=6 exceeded h=3 by 19.69 return units with paired 90% LCB 10.41, but h=2 remained the overall best horizon (106.81). This is a rescue of h=3, not evidence that the adaptive optimum should move long. At delay 4, h=7 was the overall best horizon: mean 71.47 versus 54.91 for h=3 and 57.38 for h=2. The best-long minus h=3 paired difference was 16.56 with 90% LCB 7.78. Thus delay 4 is the only atlas cell that directly supports a short-to-long selected-horizon transition.

The other axes did not pass the rule. Nominal, torque, damping, and gravity favored h=2 or h=3 whenever the checkpoint remained functional; severe torque, damping, and gravity changes collapsed all horizons together. Observation noise 0.03 and 0.1 had positive point estimates for a long horizon versus h=3, but their paired 90% lower bounds remained negative. The result therefore isolates delay 4 as the clean next endpoint rather than broadly claiming that harder dynamics require longer planning.

The online traversal now needs one final design freeze. The empirically matched long oracle is h=7, not h=8. The minimal scientifically aligned trio is therefore fixed h=3, fixed h=7, and adaptive over h=2..8 under a nominal to delay-4 to nominal schedule from the 30k checkpoint. Keeping the earlier fixed-h8 label would test a legacy endpoint, but would not use the atlas-optimal long horizon.
