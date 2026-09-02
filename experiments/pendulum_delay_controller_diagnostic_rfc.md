# Pendulum delay/controller diagnostic RFC

## Frozen arms

| Arm | Actual deployed horizon | Dense-RHS role |
|---|---|---|
| B1 | fixed h=2 | observational shadow only |
| B2 | fixed h=3 | observational shadow only |
| B3 | 2→2→6→4→2 | observational shadow only |
| B4 | 2→3→7→5→2 | observational shadow only |
| B5 | exact argmax of paired mean return over h=2,...,8 | active deployment |

All arms fork the exact same 34k agent, replay buffer, and training step; they reset controller evidence and run to 54k under delay 0→2→6→4→0. Evaluation is every 400 transitions. Queries are every 2k transitions, with all seven candidates retained. Independent references are collected at phase midpoints and terminal recovery with 128 paired replicas per horizon, the deployed planner budget, and 500-step episodes.

## Invariants that separate bugs from scientific behavior

At every query, the validator must recover the configured delay, all seven ordered candidates, finite paired returns, and the decision from raw evidence. For B1–B4, the counterfactual scorer may move but the deployed, planner, learner, evaluation, checkpoint, and artifact horizons must follow the script. For B5, the deployed horizon must equal the lowest-horizon exact argmax of the 128 paired means; every primary roughness/spread/Bellman/curvature/switch term is exactly zero or excluded.

The diagnostic also compares the online query ranking with an independent full-planner, 500-step reference. A disagreement is classified as a proxy-objective mismatch, not a scoring-rule success or failure. A positive immediate probe followed by worse learned return is classified as probe-to-control transfer failure.

## Interpretation limits

The delay queue remains hidden to preserve the experiment already studied. Consequently, the learned one-step dynamics receive the issued command even though the next state may depend on an unobserved historical command. Longer planning cannot by itself repair that partial observability. The forced B3/B4 arms are therefore causal interventions, not assumed-positive controllers, and one seed supports only checkpoint-conditional conclusions.
