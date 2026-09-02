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
