import csv
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from scripts.analyze_pendulum_frontier_atlas import validate
from tdmpc2_jax.envs.mjx_dmc import MJXDMCBatchEnv
from tdmpc2_jax.frontier_atlas import (
    ATLAS_HORIZONS,
    ATLAS_VERSION,
    condition_overrides,
    frontier_conditions,
    shard_conditions,
)


def test_canonical_atlas_has_24_unique_one_factor_conditions():
  conditions = frontier_conditions()
  assert len(conditions) == 24
  assert conditions[0].condition_id == 'nominal'
  assert len({condition.condition_id for condition in conditions}) == 24
  assert [condition.index for condition in conditions] == list(range(24))
  for condition in conditions:
    overrides = condition_overrides(condition)
    changed = {
        key for key, value in overrides.items()
        if (
            (key == 'base_action_delay' and value != 0) or
            (key in ('actuator_strength_scale', 'joint_damping_scale', 'gravity_scale') and value != 1.0) or
            (key == 'fixed_observation_noise_scale' and value is not None)
        )
    }
    if condition.axis == 'nominal':
      assert not changed
    elif condition.axis == 'observation_noise_scale':
      assert changed == {'fixed_observation_noise_scale'}
      assert overrides['enable_observation_noise'] is True
    else:
      assert changed == {condition.axis}


def test_four_shards_are_disjoint_and_cover_canonical_atlas():
  shards = [shard_conditions(index, 4) for index in range(4)]
  assert all(len(shard) == 6 for shard in shards)
  flattened = [condition for shard in shards for condition in shard]
  assert {condition.condition_id for condition in flattened} == {
      condition.condition_id for condition in frontier_conditions()
  }


def test_fixed_observation_noise_is_available_without_domain_randomization():
  env = object.__new__(MJXDMCBatchEnv)
  env.enable_domain_randomization = False
  env.enable_observation_noise = True
  env.fixed_observation_noise_scale = 0.06
  params = env._sample_reset_params_jax(jax.random.PRNGKey(0), (3,))
  np.testing.assert_allclose(np.asarray(params['obs_noise_scale']), 0.06)
  np.testing.assert_allclose(np.asarray(params['actuator_strength']), 1.0)
  np.testing.assert_allclose(np.asarray(params['wind_force']), 0.0)


def _write_valid_shard(tmp_path: Path):
  run_dir = tmp_path / 'run'
  artifacts = run_dir / 'artifacts' / 'frontier_atlas'
  artifacts.mkdir(parents=True)
  (run_dir / 'checkpoint' / '30000').mkdir(parents=True)
  conditions = shard_conditions(0, 1, limit=1)
  planner = {
      'population_size': 256,
      'policy_prior_samples': 12,
      'num_elites': 32,
      'mppi_iterations': 4,
      'temperature': 0.5,
  }
  manifest = {
      'atlas_version': ATLAS_VERSION,
      'run_id': 'test',
      'mode': 'smoke',
      'commit': 'abc',
      'config_hash': 'hash',
      'seed': 7,
      'source': {'run_dir': '/source', 'checkpoint_step': 30000},
      'shard_index': 0,
      'num_shards': 1,
      'condition_limit': 1,
      'condition_ids': [condition.condition_id for condition in conditions],
      'horizons': list(ATLAS_HORIZONS),
      'replicas': 4,
      'eval_steps': 32,
      'planner': planner,
  }
  (run_dir / 'run_manifest.json').write_text(json.dumps(manifest))
  runtime = {
      'atlas_version': ATLAS_VERSION,
      'source_step': 30000,
      'shard_index': 0,
      'num_shards': 1,
      'condition_ids': ['nominal'],
      'horizons': list(ATLAS_HORIZONS),
      'replicas': 4,
      'eval_steps': 32,
      'planner': planner | {'planning_hmax': 8},
      'condition_elapsed_s': {'nominal': 1.0},
      'total_elapsed_s': 1.0,
  }
  (artifacts / 'runtime.json').write_text(json.dumps(runtime))
  with (artifacts / 'summary.csv').open('w', newline='') as handle:
    fields = (
        'atlas_version', 'source_step', 'shard_index', 'num_shards',
        'condition_index', 'condition_id', 'axis', 'value', 'horizon',
        'return_mean', 'return_std', 'return_se', 'replicas', 'eval_steps',
        'condition_elapsed_s',
    )
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for horizon in ATLAS_HORIZONS:
      writer.writerow({
          'atlas_version': ATLAS_VERSION,
          'source_step': 30000,
          'shard_index': 0,
          'num_shards': 1,
          'condition_index': 0,
          'condition_id': 'nominal',
          'axis': 'nominal',
          'value': 1.0,
          'horizon': horizon,
          'return_mean': 10.0,
          'return_std': 1.0,
          'return_se': 0.5,
          'replicas': 4,
          'eval_steps': 32,
          'condition_elapsed_s': 1.0,
      })
  with (artifacts / 'paired_returns.csv').open('w', newline='') as handle:
    fields = (
        'condition_index', 'condition_id', 'axis', 'value', 'horizon',
        'replica', 'return',
    )
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for horizon in ATLAS_HORIZONS:
      for replica in range(4):
        writer.writerow({
            'condition_index': 0,
            'condition_id': 'nominal',
            'axis': 'nominal',
            'value': 1.0,
            'horizon': horizon,
            'replica': replica,
            'return': 10.0 + replica,
        })
  (run_dir / 'ATLAS_COMPLETE').touch()
  (run_dir / 'RUN_VALID').touch()
  return run_dir


def test_atlas_validator_accepts_complete_paired_shard(tmp_path):
  summary = validate(_write_valid_shard(tmp_path))
  assert summary['valid']
  assert summary['summary_rows'] == 7
  assert summary['paired_rows'] == 28


def test_atlas_validator_rejects_nonfinite_return(tmp_path):
  run_dir = _write_valid_shard(tmp_path)
  path = run_dir / 'artifacts' / 'frontier_atlas' / 'paired_returns.csv'
  rows = list(csv.DictReader(path.open()))
  rows[0]['return'] = 'nan'
  with path.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
  with pytest.raises(ValueError, match='Non-finite paired return'):
    validate(run_dir)

