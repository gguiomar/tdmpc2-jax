# Pendulum torque-by-horizon screen decisions

## Frozen design

The screen forks the validated seed-7 Pendulum checkpoint at 30k transitions. Every child restores the same agent, optimizer, replay buffer, and global step. The only scientific interventions are the fixed actuator-strength scale in `{0.4, 0.6, 0.8}` and the fixed TD-MPC2 training/planning horizon in `{3, 8}`. Dense-RHS and all horizon queries are disabled.

Each child continues from 30k to 46k transitions and is evaluated every 2k transitions with ten deterministic 500-step episodes. The comparison of interest is paired within torque: `h=8 - h=3`. This is a regime-location screen, not a final statistical comparison; seed confirmation follows only after a torque level shows both retained controllability and a material long-horizon advantage.

The parent is accepted only if its existing validator confirms the 30k checkpoint and the frozen stability rule. Before the six children can start, a separate torque-0.6, h=8 continuation to 30.8k must restore the composite checkpoint, train, evaluate, save, and pass artifact validation on one GPU.

## Pre-submission launcher repair

The first launcher invocation at revision `5c2da9116e0c5ba3ba6c5bad987ce3e0804899bf` stopped before `sbatch` because the lightweight NCC control shell has no bare `python` command. No job or output directory was created. The launcher now invokes the explicit project-venv interpreter for its read-only parent validation. The Slurm job script already activated this same environment, so the scientific configuration is unchanged.

## Interpretation rule

A torque endpoint is useful for the next adaptive experiment only if control remains non-degenerate and h=8 improves over h=3. A joint collapse of both horizons is an infeasible endpoint, not evidence for either horizon. Similar strong returns at both horizons indicate that torque level does not identify horizon necessity. The best next adaptive condition is the weakest controllable torque with the clearest positive `h=8 - h=3` gap.

## Slurm launch

Revision `515f35c18eded4e6c0e7c8175e5913a781263730` passed the parent and CPU gates. Restore-smoke job 4540 is running at torque 0.6 and fixed h=8. Full jobs 4541--4546 encode the frozen torque-by-horizon matrix and were submitted with dependency `afterok:4540`; none can start if the smoke fails.

Job 4540 then completed in 2m41s. It restored the composite 30k parent, trained and evaluated through 30.8k, wrote the terminal checkpoint, preserved fixed h=8 in both horizon audit scalars, and passed manifest/artifact validation. Its single evaluation return was 90.9. Slurm released jobs 4541--4546; jobs 4541--4544 started immediately on all four GPUs and jobs 4545--4546 remained queued under the four-GPU per-user limit.

## Terminal screen result

All six full jobs completed with exit code 0 and passed checkpoint, manifest, finite-metric, terminal-evaluation, and fixed-horizon validation. The h=3 jobs took 10m19--10m23 and the h=8 jobs took 11m36--11m40; four-GPU scheduling completed the full matrix in 22m03.

Torque 0.4 was a failure regime: h=3 had mean/best/final evaluation returns 12.55/100.0/100.0, while h=8 had 0.45/3.2/3.2. Torque 0.6 remained weak and intermittent: h=3 had 80.41/196.1/100.0 and h=8 had 18.55/100.0/100.0. Torque 0.8 was controllable, but did not show a robust long-horizon advantage: h=3 had 651.19/836.7/662.9 and h=8 had 559.60/766.8/704.8. Thus torque alone has not exposed a horizon-rescuable band in this coarse screen; h=8 has a higher final point at 0.8 but a lower curve mean and best point.
