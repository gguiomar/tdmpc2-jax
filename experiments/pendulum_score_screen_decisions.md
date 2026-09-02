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

## Repair gate and smoke attempt 2

The isolated NCC checkout was safely fast-forwarded to repair revision `76392f1`. The full frozen CPU gate passed all 51 tests on NCC. With all four H200 GPUs physically idle at submission time, corrected S3 smoke attempt 2 was submitted as Slurm job 4822. It retains the original smoke config hash and common validated 34k source; only the restored query-interval contract differs from failed attempt 1.

## Smoke attempt 2 diagnosis

Slurm job 4822 verified the interval repair by emitting queries at 34.4k, 34.8k, 35.2k, and 35.6k. It then completed training, the 36k composite checkpoint, and all six EGL GIF/PNG renders, but strict validation rejected the missing 36k terminal query. The MJX loop explicitly required `global_step < max_steps` when applying a query after a collection boundary. Consequently, a query due exactly at the frozen endpoint was skipped even though checkpoint, evaluation, reference-probe, and artifact logic all include that endpoint. This would also omit the scientifically required 54k deployment query in every full arm. The next repair will allow a due query at `global_step == max_steps`, test that boundary contract directly, and preserve the frozen schedule and score definitions.

Repair revision `7c1ad18` implements that terminal-boundary contract and adds a focused regression test. The complete NCC CPU gate passed 52 tests. With all four H200 GPUs physically idle, fresh smoke attempt 3 was submitted as Slurm job 4823 using the unchanged scientific config hash and validated 34k source.

Smoke attempt 3 passed the complete gate. Slurm job 4823 emitted the exact five-query cadence through 36k, produced finite metrics and the full terminal composite checkpoint, rendered all six d=0/4/6 anchor trios as GIF and PNG, and wrote `RUN_VALID`. The four frozen scientific arms are therefore eligible to launch at revision `7c1ad18` from the common validated 34k source. Full-profile config hashes are the SHA-256 of the canonical sorted goal projection `{campaign, mode=full, shared, profile}`.

At launch time all four H200 GPUs were physically idle and no unrelated job was running. The campaign limit remains three concurrent GPUs. S0, S1, and S2 were submitted as Slurm jobs 4824, 4825, and 4826. S3 remains the sole pending frozen arm and will be submitted when one campaign slot is released.

## S0 full-arm result and S3 slot release

Slurm job 4824 completed successfully in 01:05:35. Independent validation confirmed the expected revision and config hash, 50 finite evaluation points, all ten deployment queries through the exact 54k terminal boundary, all six 128-replica reference anchors through 54k, a complete 54k composite checkpoint, and six d=0/4/6 trajectory anchors with GIF/PNG pairs. The current additive controller selected `h=2` at every query. This terminal result releases one of the three frozen campaign slots, so S3 is now eligible for immediate submission from the same validated 34k source.

The node-wide GPU snapshot showed an idle physical H200 after S0 completed. The isolated checkout remained clean at `7c1ad18`, the S3 attempt-1 output path was absent, and the validated 34k parent checkpoint was present. The frozen S3 curvature--Bellman arm was therefore submitted through the sole Slurm launcher as job 4827, preserving the predeclared config hash and three-job campaign limit.

S1 job 4825 then completed successfully in 01:07:21. Validation confirmed the exact revision and config identity, 50 finite evaluations, the complete deployment and reference cadence through 54k, the full composite checkpoint, all six artifact anchors, trajectory trios, and GIF/PNG media. Its selected-horizon sequence was `2,2,3,2,2,2,2,2,2,2`: one temporary switch at the 40k query followed by immediate recovery to `h=2`.

S2 job 4826 completed successfully in 01:13:04 and passed the same full validation contract. Its selected-horizon sequence was `4,3,3,3,3,3,3,3,5,2`: the return-first rule spent most of the schedule at `h=3`, briefly moved to `h=5` at 52k, and recovered to `h=2` at the terminal clean query. S3 job 4827 remains the sole active full arm.

## S3 full-arm result and screen completion

Slurm job 4827 completed successfully in 01:07:39. Independent validation confirmed revision `7c1ad18`, the frozen config hash, 50 finite evaluations, exact deployment queries at 36k:2k:54k, reference probes at 36k, 40k, 44k, 48k, 52k, and 54k, the complete 54k agent/buffer/global-step/horizon-state checkpoint, all six anchor media sets, and `RUN_VALID`. The curvature--Bellman controller selected `h=4` at all ten deployment queries. All four frozen screen arms are now complete and valid; reduction and the external promotion rule are the next actions.

## Reduction repairs and screen decision

The first reduction attempt exposed a run-discovery typo: the glob expected two underscores after the profile token although the frozen run IDs have one. Repair `f5cfcf2` added a focused discovery test and passed the expanded 53-test CPU gate. The next attempt exposed removal of `numpy.trapz` in the installed NumPy version; repair `2cfba14` moved the time-weighted AUC calculation to `numpy.trapezoid` and passed 54 tests. The first successful summary then revealed that the confirmation list could still include an ineligible arm. Repair `867fdbe` restricts confirmation ranking to promotion-eligible profiles and passed the final 55-test CPU gate. These are analysis-only repairs; the four frozen scientific runs remain identified by revision `7c1ad18` and unchanged config hashes.

The corrected reducer selects S0 (current additive) as the only promotion-eligible profile. S1 has the highest normalized return AUC (512.06 versus 510.10 for S0) and comparable final clean return (831.89 versus 836.44), but fails the predeclared no-A-B-A chatter rule. S2 has the lowest mean shadow-oracle regret (1.32 return units) but misses the 95% AUC and clean-return gates. S3 locks at `h=4`, has the lowest AUC (443.73), and the highest oracle regret (4.89). S0 therefore wins by satisfying all performance, clean-recovery, and stability gates, despite selecting `h=2` throughout.

No delay-six oracle selected `h=8`, so the frozen-checkpoint `h=9..20` frontier is not triggered. The two-by-two confirmation continuation is also not scientifically valid yet: only S0 is promotion-eligible, and the isolated campaign contains only the validated seed-7 34k source checkpoint rather than independent sources for two confirmation seeds. No confirmation jobs are launched. Compact metrics, manifests, validation summaries, GIF/PNG media, and aggregate plots were synced to `runs/results/pendulum_score_screen`; all four plots were visually inspected successfully.

## Report completion

Version 1.5 of the Cartpole Delay Pilot report now includes the complete Pendulum score screen: frozen methods and provenance, scientific and analysis revisions, Slurm jobs 4824--4827, return and regret statistics, controller trajectories, shadow-oracle boundary evidence, compute cost, environment images, promotion outcome, and limitations. The 29-page PDF compiled successfully. Pages containing every newly added table and figure were rendered and visually inspected; the result table was tightened and rechecked with no overfull box. The final artifact is `/Users/ggmar/Documents/git/robusthorizonsearch/reports/experiment_plans/cartpole_delay_six_run_pilot_plan.pdf`.
