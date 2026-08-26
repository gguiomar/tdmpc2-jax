import csv
import json
from pathlib import Path

import pytest

from scripts.analyze_pendulum_torque_horizon_screen import validate


def _write_screen_run(tmp_path: Path, *, mode: str = 'full', horizon: int = 8):
  final_step = 46_000 if mode == 'full' else 30_800
  run_dir = tmp_path / 'run'
  (run_dir / 'metrics').mkdir(parents=True)
  (run_dir / 'checkpoint' / str(final_step)).mkdir(parents=True)
  (run_dir / 'TRAINING_COMPLETE').touch()
  manifest = {
      'run_id': 'test',
      'phase': 'fixed_horizon_screen',
      'mode': mode,
      'config_hash': 'abc',
      'actuator_strength_scale': 0.6,
      'fixed_horizon': horizon,
      'dense_rhs_enabled': False,
      'parent': {'checkpoint_step': 30_000},
  }
  (run_dir / 'run_manifest.json').write_text(json.dumps(manifest))
  with (run_dir / 'metrics' / 'scalars.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=('step', 'tag', 'value'))
    writer.writeheader()
    for tag, value in (
        ('eval/return_mean', 750.0),
        ('eval/selected_horizon', float(horizon)),
        ('eval/training_bucket_horizon', float(horizon)),
    ):
      writer.writerow({'step': final_step, 'tag': tag, 'value': value})
  return run_dir


def test_screen_validator_accepts_fixed_horizon_run(tmp_path):
  summary = validate(_write_screen_run(tmp_path))
  assert summary['valid']
  assert summary['fixed_horizon'] == 8
  assert summary['final_upright_fraction'] == pytest.approx(0.75)


def test_screen_validator_rejects_horizon_drift(tmp_path):
  run_dir = _write_screen_run(tmp_path, horizon=8)
  rows = list(csv.DictReader((run_dir / 'metrics' / 'scalars.csv').open()))
  rows[-1]['value'] = '3'
  with (run_dir / 'metrics' / 'scalars.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=('step', 'tag', 'value'))
    writer.writeheader()
    writer.writerows(rows)
  with pytest.raises(ValueError, match='was not preserved'):
    validate(run_dir)
