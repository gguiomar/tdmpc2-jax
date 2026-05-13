# CoRL Compact Dense-RHS V3 Decisions

- 2026-05-13: Created the expanded v3 steward plan. The target matrix is 10 environments x 2 regimes x 2 methods x 6 seeds = 240 valid cells. The v3 ledger reuses the 66 finite v2 main rows and excludes the 6 non-finite fish-swim chaos rows.
- 2026-05-13: V3 uses phased autonomous launch gates: first repair fish-swim chaos, then add seeds 23/31/42 for the original six environments, then run acrobot-swingup and walker-run, then run pendulum-swingup and reacher-hard only after their MJX ports and gates pass.
- 2026-05-13: Fish-swim chaos requires an explicit finite-rollout gate at `runs/results/mjx_gates_chaos/fish-swim/mjx_gate.json` before any fish chaos rerun or downstream full-matrix launch. This prevents silently rerunning the known NaN failure mode.
