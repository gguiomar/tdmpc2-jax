# Pendulum delay/controller diagnostic decisions

This log is append-only.

## 2026-09-02 — frozen scientific question

We will distinguish four explanations for the absent horizon response: no useful static long-horizon advantage, an incorrect delay-to-horizon mapping, safeguards suppressing a real immediate-return signal, and an implementation/proxy-objective mismatch.

The common parent is the validated seed-7 34k composite checkpoint from Slurm job 4632. Every arm resets accumulated Dense-RHS controller evidence on restore and explicitly sets its deployed horizon before the first anchor, rollout, evaluation, or update.

There is no established static optimum under this schedule. B1 and B2 therefore test fixed h=2 and the canonical TD-MPC2 h=3. B3 forces h=max(2,d); B4 forces h=max(2,d+1), because an action issued now first affects a transition d steps later. B5 is an unguarded paired mean-return argmax over h=2,...,8. In B5, uncertainty, roughness, curvature, Bellman risk, switching cost, tolerance, hysteresis, and confidence gates are readouts only and cannot affect deployment.

The score query is an episodic fresh-reset objective. Boundary queries are analyzed separately from the +2k midpoint because the incoming environment delay changes immediately while the recent replay/model evidence still comes from the preceding phase.

## 2026-09-02 — prelaunch audit repairs

The prelaunch audit found two provenance leaks that would have made the diagnostic ambiguous. First, checkpoint forks inherited the parent controller's accumulated score evidence and original query clock; the diagnostic now resets both while retaining the same agent, replay buffer, and global step. Second, the conditional 500-step oracle initially reused the deployment environment, which would have reduced B1--B4 from 128 to 32 replicas. It now uses the dedicated 128-replica reference environment for every arm.

The only active adaptive arm, B5, implements the exact lowest-index `float32` argmax of paired mean return. Its decision bypasses uncertainty, roughness, Bellman/curvature terms, switching costs, confidence gates, hysteresis, and transition-size limits. Those quantities remain logged as diagnostics. The single Slurm launcher hard-codes the scientific arguments, enforces the frozen run ID for each profile, and computes its config identity from the exact committed launcher revision rather than accepting a caller-supplied hash.

## 2026-09-02 — source and CPU gate

The clean isolated NCC checkout was created at frozen scientific revision `e25f20437fd79f773de3c73d27a8f3d591a523f8`. Source job 4632 matches run `pendscore__source34k__s7`, config hash `0bdd02caf67c1fc95ee0efac5b07ee92f315ee81612f8965eba8abdf1cfe5bb8`, and a valid 34k checkpoint containing agent, buffer state, horizon state, and global step. The full CPU gate passed 106 tests with one non-fatal headless GLFW warning.

With three H200 GPUs physically idle and no active campaign jobs, the single required B4 compressed GPU/EGL smoke was submitted as Slurm job 4834. Its frozen config hash is `94a71c723d2cc524a989864a75b65e82cfade0757fa062c714a3f9c2a7eda590` and its fresh output is `penddiag__smoke_b4__s7/attempt_1`.

## 2026-09-02 — GPU/EGL smoke gate passed

Slurm job 4834 completed in 22m46s with exit code 0. The validator accepted all five exact queries at 34400, 34800, 35200, 35600, and 36000; scripted deployed horizons were 3, 7, 5, 2, and 2. The run produced its terminal composite checkpoint, all six anchor trajectory sets, six GIFs, six PNGs, `TRAINING_COMPLETE`, `MEDIA_COMPLETE`, `validation_summary.json`, and `RUN_VALID`. CPU, GPU, and EGL gates are therefore closed and the frozen full B1--B5 launch stage is allowed.

Three H200 GPUs were physically idle after the smoke completed. The two static baselines and active return-only controller were therefore launched first: B1 as job 4835, B2 as job 4836, and B5 as job 4837. B3 and B4 remain pending and will be submitted into fresh attempt paths as physical slots release.

## 2026-09-02 — static baselines valid; forced-control arms launched

Jobs 4835 (B1 fixed h=2) and 4836 (B2 fixed h=3) completed with exit code 0. Both have `TRAINING_COMPLETE`, `MEDIA_COMPLETE`, `validation_summary.json`, and `RUN_VALID`; the validator accepted the exact ten-query cadence through 54k, frozen run/config identities, reference evidence, terminal checkpoint, anchors, trajectories, GIFs, and PNGs.

Their released GPUs were physically idle, the remote checkout remained clean at `e25f20437fd79f773de3c73d27a8f3d591a523f8`, and both target attempt paths were absent. The remaining forced-control arms were therefore submitted from the same 34k source: B3 delay-match as job 4838 and B4 causal-coverage as job 4839. B5 job 4837 remains active. This leaves three campaign GPUs active and preserves one GPU for the other user's allocation.

## 2026-09-02 — return-only arm valid

Job 4837 (B5 active 128-replica return argmax) completed with exit code 0 and `RUN_VALID`. The validator accepted all ten queries through 54k, the exact return-only selection invariant, both 128-replica reference namespaces, terminal checkpoint, anchors, trajectories, and six GIF/PNG pairs. Its deployed horizon sequence was 2, 2, 3, 3, 3, 2, 2, 4, 4, 5 at queries 36k through 54k. Interpretation is deferred until B3/B4 complete and the predeclared aggregate reducer can compare all five arms.

## 2026-09-02 — forced-control arms valid

Jobs 4838 (B3 delay-match) and 4839 (B4 causal-coverage) completed with exit code 0 and `RUN_VALID`. For each arm, the validator accepted the exact ten-query cadence from 36k through 54k, finite paired evidence for every candidate horizon, both 128-replica reference namespaces including the conditional 500-step reference, exact deployment/training/evaluation/episode/anchor horizon identity, the terminal composite checkpoint, every anchor and delay-0/delay-4/delay-6 trajectory, and nonempty GIF/PNG media.

All five frozen full profiles B1--B5 are now valid. Scientific interpretation remains deferred to the predeclared aggregate reducer and its static, forced-control, and return-argmax gates.

## 2026-09-03 — aggregate diagnostic decision

The compact B1--B5 metrics, manifests, validation summaries, trajectories, and media were synchronized locally. The predeclared reducer completed with `AGGREGATE_VALID`. B1 fixed h=2 is the checkpoint-conditional static winner: its full-curve AUC exceeds B2 fixed h=3 by 5.497%, while its final clean-phase mean is 99.21% of B2 and therefore clears the 95% safeguard.

Neither forced controller is promoted. Relative to B1, B3 delay-match reaches 82.85% of delayed-phase AUC and 92.89% of clean return, failing both gates. B4 causal-coverage reaches 73.72% and 99.41%, failing the delayed-phase gate. B5 exact return argmax reaches 118.49% of B1 delayed-phase AUC and 101.52% of its clean return, but its independent-reference regret fraction is 2.393%, above the predeclared 2% ceiling. B5 is retained only as a promising mechanism diagnostic, not promoted as a controller.

The five full runs consumed 11:50:22 GPU time; including the validated smoke gives 12:13:08. The version 1.6 report records the exact methods, revisions, job IDs, results, trajectories, media, gates, compute cost, failure interpretation, and hidden-delay limitations and was compiled and visually checked on 2026-09-03.
