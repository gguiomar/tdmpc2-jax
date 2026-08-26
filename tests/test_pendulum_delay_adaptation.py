import csv
import json
from pathlib import Path

from scripts.analyze_pendulum_delay_adaptation import (
    FULL_ANCHORS,
    READOUT_INTERVAL,
    SOURCE_STEP,
    validate,
)


def _write_fixed_run(root: Path) -> Path:
  run_dir = root / 'fixed_h3'
  run_id = 'penddelay__fixed_h3__s7'
  run_dir.mkdir()
  manifest = {
      'schema_version': 1,
      'experiment': 'pendulum-delay-adaptation-v1',
      'run_id': run_id,
      'mode': 'full',
      'profile': 'fixed_h3',
      'controller': 'fixed',
      'fixed_horizon': 3,
      'git_commit': 'abc',
      'config_hash': 'hash',
      'seed': 7,
      'environment': 'pendulum-swingup',
      'parent': {
          'run_dir': '/source',
          'run_id': 'parent',
          'checkpoint_step': SOURCE_STEP,
          'config_hash': 'parent-hash',
      },
      'final_step': 46_000,
      'delay_schedule': {
          'base': 0,
          'active': 4,
          'start': 34_000,
          'end': 42_000,
          'observed': False,
      },
      'readout_interval_steps': READOUT_INTERVAL,
      'evaluation_episodes': 10,
      'query_steps': [],
      'artifact_anchors': list(FULL_ANCHORS),
  }
  (run_dir / 'run_manifest.json').write_text(json.dumps(manifest))
  (run_dir / 'TRAINING_COMPLETE').touch()
  (run_dir / 'MEDIA_COMPLETE').touch()
  (run_dir / 'checkpoint' / '46000').mkdir(parents=True)
  metrics = run_dir / 'metrics'
  metrics.mkdir()
  with (metrics / 'scalars.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=('step', 'tag', 'value'))
    writer.writeheader()
    for step in range(30_400, 46_001, READOUT_INTERVAL):
      writer.writerow({'step': step, 'tag': 'eval/return_mean', 'value': 10.0})
      writer.writerow({'step': step, 'tag': 'train/online_reward_mean', 'value': 0.5})
      writer.writerow({'step': step, 'tag': 'eval/selected_horizon', 'value': 3})
  with (metrics / 'horizon_queries.csv').open('w', newline='') as handle:
    csv.writer(handle).writerow(('step', 'selected_horizon'))
  rollout_root = run_dir / 'artifacts' / 'rollouts' / run_id
  for step in FULL_ANCHORS:
    (run_dir / 'artifacts' / 'anchor_checkpoints' / str(step)).mkdir(parents=True)
    rollout_dir = rollout_root / f'step_{step:06d}'
    rollout_dir.mkdir(parents=True)
    (rollout_dir / 'metadata.json').write_text(json.dumps({
        'global_step': step,
        'selected_horizon': 3,
        'environment': {'task': 'pendulum-swingup'},
    }))
    for name in (
        'trajectory_delay0.npz',
        'trajectory_delay4.npz',
        'pendulum_delay0_vs_delay4.gif',
        'pendulum_delay0_vs_delay4_frame.png',
    ):
      (rollout_dir / name).write_bytes(b'fixture')
  return run_dir


def test_fixed_profile_validator_requires_dense_measured_cadence_and_media(tmp_path):
  summary = validate(_write_fixed_run(tmp_path))
  assert summary['valid']
  assert summary['evaluation_points'] == 40
  assert summary['train_reward_points'] == 40
  assert summary['final_eval_return'] == 10.0
  assert len(summary['media']) == len(FULL_ANCHORS)
