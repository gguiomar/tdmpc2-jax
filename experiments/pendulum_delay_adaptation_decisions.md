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
