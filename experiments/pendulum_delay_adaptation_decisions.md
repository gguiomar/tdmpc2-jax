# Pendulum delay-adaptation decisions

## Frozen scientific question

The frontier atlas found one clean horizon-rescuable endpoint: hidden action delay four, where fixed h=7 was the overall best tested horizon and beat h=3 with a positive paired 90% lower confidence bound. The online experiment therefore traverses nominal delay zero, delay four, and delay zero again while continuing to learn from the same validated 30k Pendulum checkpoint.

The matched scientific trio is fixed h=3, fixed h=7, and the full adaptive controller over h=2..8. All three use the same checkpoint, seed, optimizer and replay state, 30k--46k transition budget, delay boundaries, deployment-planner budget, evaluation resets, and media anchors. The delay remains hidden so the checkpoint observation shape is unchanged; this is an adaptation-under-latent-regime-change test, not a fully observed Markov augmentation.

## Readouts and interpretation

Training loss and mean online reward plus ten-episode deterministic evaluation summaries are written every 400 collected transitions. This is the densest cadence that is exactly compatible with eight vector environments and the collection loop, and yields forty genuine post-checkpoint evaluation points per run. It replaces sparse 2k readouts without interpolating measurements.

The adaptive controller queries at 32k, 36k, 40k, and 44k. The first query is a nominal pre-delay baseline; the middle two occur during delay four; the final query tests recovery after delay returns to zero. At every query a separate 128-replica shadow probe is retained. The shadow probe is observational and cannot change the selected horizon.

The primary success pattern is short before the intervention, h>=6 during delay four, then h<=3 after recovery. If that switch does not occur, the run remains diagnostic: proposed versus selected horizons, confidence, return, roughness, return-spread terms, and the high-replica shadow proposal distinguish a weak signal from a blocking decision rule.

## Visual evidence

Every run saves immutable model anchors and deterministic paired delay-0/delay-4 trajectories at 30k, 34k, 36k, 40k, 42k, 44k, and 46k. EGL rendering produces a GIF and first-frame PNG per anchor. These trajectories do not mutate training state and use fixed reset states and planner randomness across controllers and anchors.

## First smoke failure and minimal repair

Smoke job 4561 completed the 30k restore, the 30k--32k hidden-delay traversal, one adaptive query, five dense evaluation points, the 128-replica observational reference, five anchor checkpoints, ten raw trajectories, and all five EGL GIF/PNG pairs. It then failed strict validation because the smoke reused the full profile's 23k Orbax save cadence. Step 32k is not a 23k cadence boundary, so the terminal composite checkpoint manager declined that save even though the dedicated 32k model anchor was valid.

Attempt 2 changes only gate mechanics. The smoke uses a 32k composite-save cadence so its terminal step is accepted. It also omits the high-replica shadow reference, which is not needed to gate restore, online adaptation, metric, artifact, or EGL correctness and had dominated the gate runtime. The full fixed-h3, fixed-h7, and adaptive scientific configurations are unchanged; the adaptive full run still retains all four 128-replica reference probes.

## Completed outcome

Attempt 2 passed at commit `9e5f0187578aaf1f2638f376b81fe5d13a8eaefc`. Smoke job 4573 passed the restore, adaptive-query, terminal-checkpoint, finite-metric, anchor, paired-trajectory, and EGL-media gates. The matched full jobs then completed and validated: fixed h=3 job 4574 in 37:43, fixed h=7 job 4575 in 40:20, and adaptive job 4576 in 51:11. Reducer job 4577 produced a valid aggregate. Every full run has forty measured train/evaluation points, seven model/media anchors, seven GIFs, and seven first-frame PNGs.

The primary h>=6 success criterion failed. The adaptive controller moved from h=3 to h=2 at 32k and then selected and proposed h=2 at 36k, 40k, and 44k. All four independent 128-replica score-based shadow proposals were also h=2, ruling out the 32-replica deployment budget as the cause of the missing switch.

The return-only reference tells a more useful story. Its high-precision argmax sequence was h=3 at the 32k clean anchor, h=4 at both delayed anchors (36k and 40k), and h=2 at the 44k clean-recovery anchor. At 36k, the corresponding mean returns for h=2..8 were 38.48, 46.74, 58.02, 39.16, 36.27, 32.23, and 23.19. At 40k they were 40.00, 47.41, 62.05, 60.77, 49.88, 46.59, and 45.07. Thus hidden delay produces a reproducible but moderate h=3/2 to h=4 shift, not support for h=7 or h=8.

The score decomposition explains why the controller misses even that moderate shift. At 36k, candidates h=3..8 receive small positive normalized return terms (approximately 0.02--0.11), but roughness terms fall from -0.92 at h=3 to -3.83 at h=8; return-spread terms are comparatively small. The total score therefore declines monotonically away from h=2. The same qualitative roughness dominance appears at 40k. This is a structural score-scaling result, not a confidence-gate or replica-precision failure.

Mean evaluation returns by phase were:

- fixed h=3: 744.92 pre-delay, 162.91 at delay four, 830.23 in recovery;
- fixed h=7: 762.70 pre-delay, 112.33 at delay four, 843.04 in recovery;
- adaptive: 794.09 pre-delay, 134.88 at delay four, 782.48 in recovery.

The fixed h=3 controller is best during the delayed online phase; h=7 is worst. The final clean returns are descriptively tied at 820.5 (fixed h=3), 821.2 (fixed h=7), and 815.7 (adaptive). Because the fixed runs continue learning at different horizons, their phase comparison mixes planning and learner divergence; the within-checkpoint 128-replica frontier is the cleaner planning-horizon comparison.

## Next decision

Do not push this intervention to h=8 and do not diagnose the failure as insufficient return replicas. The next controller ablation should preserve the same delayed anchors and compare the deployed additive score with (i) return-only selection, (ii) a down-weighted or normalized roughness term, and (iii) a local one-step Bellman check. The current run supplies the paired candidate returns needed to choose those weights offline before launching another training batch.
