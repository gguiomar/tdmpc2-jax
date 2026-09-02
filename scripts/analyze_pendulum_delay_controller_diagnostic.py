#!/usr/bin/env python3
"""Manifest, validation, and reduction for the Pendulum controller diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SOURCE_STEP = 34_000
SOURCE_RUN_ID = 'pendscore__source34k__s7'
SOURCE_CONFIG_HASH = (
    '0bdd02caf67c1fc95ee0efac5b07ee92f315ee81612f8965eba8abdf1cfe5bb8'
)
HORIZONS = tuple(range(2, 9))
READOUT_INTERVAL = 400
ZERO_ATOL = 1e-6
FULL_QUERIES = tuple(range(36_000, 54_001, 2_000))
FULL_ANCHORS = (34_000, 38_000, 42_000, 46_000, 50_000, 54_000)
REFERENCE_STEPS = (36_000, 40_000, 44_000, 48_000, 52_000, 54_000)
PHASES = (
    ('clean_i', 34_000, 38_000, 0),
    ('delay_2', 38_000, 42_000, 2),
    ('delay_6', 42_000, 46_000, 6),
    ('delay_4', 46_000, 50_000, 4),
    ('clean_ii', 50_000, 54_001, 0),
)
PROFILES = {
    'b1': {
        'run_id': 'penddiag__b1_fixed_h2__s7',
        'controller': 'fixed_h2',
        'variant': 'current_additive',
    },
    'b2': {
        'run_id': 'penddiag__b2_fixed_h3__s7',
        'controller': 'fixed_h3',
        'variant': 'current_additive',
    },
    'b3': {
        'run_id': 'penddiag__b3_delay_match__s7',
        'controller': 'delay_match',
        'variant': 'current_additive',
    },
    'b4': {
        'run_id': 'penddiag__b4_causal_coverage__s7',
        'controller': 'causal_coverage',
        'variant': 'current_additive',
    },
    'b5': {
        'run_id': 'penddiag__b5_return_argmax__s7',
        'controller': 'return_argmax',
        'variant': 'return_argmax',
    },
}


def _read_csv(path: Path) -> list[dict[str, str]]:
  with path.open(newline='') as handle:
    return list(csv.DictReader(handle))


def _write_json(path: Path, value) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
  temporary.replace(path)


def _finite(value) -> bool:
  try:
    return math.isfinite(float(value))
  except (TypeError, ValueError):
    return False


def _contract(mode: str, profile: str) -> dict:
  if profile not in PROFILES:
    raise ValueError(f'Unknown profile {profile!r}')
  if mode == 'smoke':
    if profile != 'b4':
      raise ValueError('Only B4 may be used for the frozen smoke gate')
    return {
        'final_step': 36_000,
        'queries': (34_400, 34_800, 35_200, 35_600, 36_000),
        'anchors': (34_000, 34_400, 34_800, 35_200, 35_600, 36_000),
        'delay_boundaries': (34_400, 34_800, 35_200, 35_600),
        'delay_values': (0, 2, 6, 4, 0),
        'script_steps': (0, 34_400, 34_800, 35_200, 35_600),
        'script_values': (2, 3, 7, 5, 2),
        'reference_steps': (),
    }
  if mode != 'full':
    raise ValueError(f'Unknown mode {mode!r}')
  script = {
      'b1': ((0,), (2,)),
      'b2': ((0,), (3,)),
      'b3': ((0, 38_000, 42_000, 46_000, 50_000), (2, 2, 6, 4, 2)),
      'b4': ((0, 38_000, 42_000, 46_000, 50_000), (2, 3, 7, 5, 2)),
      'b5': ((), ()),
  }[profile]
  return {
      'final_step': 54_000,
      'queries': FULL_QUERIES,
      'anchors': FULL_ANCHORS,
      'delay_boundaries': (38_000, 42_000, 46_000, 50_000),
      'delay_values': (0, 2, 6, 4, 0),
      'script_steps': script[0],
      'script_values': script[1],
      'reference_steps': REFERENCE_STEPS,
  }


def _piecewise(step: int, boundaries, values) -> int:
  selected = int(values[0])
  for boundary, value in zip(boundaries, values[1:]):
    if int(step) < int(boundary):
      break
    selected = int(value)
  return selected


def _expected_horizon(profile: str, mode: str, step: int) -> int | None:
  contract = _contract(mode, profile)
  if not contract['script_values']:
    return None
  return _piecewise(step, contract['script_steps'][1:], contract['script_values'])


def _expected_run_id(mode: str, profile: str) -> str:
  if mode == 'smoke':
    return 'penddiag__smoke_b4__s7'
  return PROFILES[profile]['run_id']


def _expected_config_hash(commit: str,
                          mode: str,
                          profile: str,
                          run_id: str,
                          final_step: int) -> str:
  payload = (
      'pendulum-delay-controller-diagnostic-v1'
      f'|commit={commit}|mode={mode}|profile={profile}|run={run_id}'
      f'|source={SOURCE_STEP}|final={final_step}|seed=7'
  )
  return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _validate_source_parent(parent_run_dir: Path) -> dict:
  parent_run_dir = parent_run_dir.resolve()
  manifest_path = parent_run_dir / 'run_manifest.json'
  if not manifest_path.is_file():
    raise ValueError(f'Frozen source manifest is missing: {manifest_path}')
  parent = json.loads(manifest_path.read_text())
  if parent.get('run_id') != SOURCE_RUN_ID:
    raise ValueError(
        f'Frozen source run mismatch: {parent.get("run_id")!r} != '
        f'{SOURCE_RUN_ID!r}'
    )
  if parent.get('config_hash') != SOURCE_CONFIG_HASH:
    raise ValueError(
        'Parent config hash mismatch: '
        f"{parent.get('config_hash')!r} != {SOURCE_CONFIG_HASH!r}"
    )
  if int(parent.get('final_step', -1)) != SOURCE_STEP:
    raise ValueError(
        f'Frozen source final step is {parent.get("final_step")!r}, '
        f'expected {SOURCE_STEP}'
    )
  checkpoint = parent_run_dir / 'checkpoint' / str(SOURCE_STEP)
  required = ('agent', 'buffer_state', 'global_step', 'horizon_state')
  missing = [name for name in required if not (checkpoint / name).exists()]
  if missing:
    raise ValueError(f'Frozen source checkpoint is missing items: {missing}')
  return parent


def write_manifest(args) -> int:
  contract = _contract(args.mode, args.profile)
  if int(args.parent_checkpoint_step) != SOURCE_STEP:
    raise ValueError('Frozen parent checkpoint must be step 34000')
  expected_run_id = _expected_run_id(args.mode, args.profile)
  if args.run_id != expected_run_id:
    raise ValueError(
        f'Frozen run ID for {args.mode}/{args.profile} is '
        f'{expected_run_id!r}, got {args.run_id!r}'
    )
  parent = _validate_source_parent(args.parent_run_dir)
  expected_config_hash = _expected_config_hash(
      args.commit,
      args.mode,
      args.profile,
      args.run_id,
      contract['final_step'],
  )
  if args.config_hash != expected_config_hash:
    raise ValueError(
        f'Config hash mismatch: {args.config_hash!r} != '
        f'{expected_config_hash!r}'
    )
  manifest = {
      'schema_version': 1,
      'campaign': 'pendulum_delay_controller_diagnostic',
      'run_id': args.run_id,
      'mode': args.mode,
      'profile': args.profile,
      'controller': PROFILES[args.profile]['controller'],
      'score_variant': PROFILES[args.profile]['variant'],
      'shadow_only': args.profile != 'b5',
      'git_commit': args.commit,
      'config_hash': args.config_hash,
      'seed': 7,
      'parent': {
          'run_dir': str(args.parent_run_dir.resolve()),
          'run_id': parent.get('run_id'),
          'config_hash': parent.get('config_hash'),
          'checkpoint_step': SOURCE_STEP,
      },
      'source_step': SOURCE_STEP,
      'final_step': contract['final_step'],
      'query_steps': list(contract['queries']),
      'artifact_anchors': list(contract['anchors']),
      'reference_steps': list(contract['reference_steps']),
      'delay_schedule': {
          'boundaries': list(contract['delay_boundaries']),
          'values': list(contract['delay_values']),
          'observed': False,
      },
      'scripted_horizon': {
          'steps': list(contract['script_steps']),
          'values': list(contract['script_values']),
      },
      'candidate_horizons': list(HORIZONS),
      'controller_evidence_reset_on_restore': True,
      'deployment_replicas': 128 if args.profile == 'b5' else 32,
      'query_env_steps': 256,
      'query_planner': {
          'population_size': 512,
          'policy_prior_samples': 24,
          'num_elites': 64,
          'mppi_iterations': 6,
      },
      'reference_replicas': 128,
      'conditional_reference_env_steps': 500,
  }
  _write_json(args.run_dir.resolve() / 'run_manifest.json', manifest)
  return 0


def _scalars(run_dir: Path) -> tuple[list[dict[str, str]], dict[str, dict[int, float]]]:
  rows = _read_csv(run_dir / 'metrics' / 'scalars.csv')
  by_tag: dict[str, dict[int, float]] = {}
  for row in rows:
    if not _finite(row.get('value')):
      raise ValueError(f'Non-finite scalar: {row}')
    by_tag.setdefault(row['tag'], {})[int(row['step'])] = float(row['value'])
  return rows, by_tag


def _validate_checkpoint(run_dir: Path, step: int) -> None:
  checkpoint = run_dir / 'checkpoint' / str(step)
  required = ('agent', 'buffer_state', 'global_step', 'horizon_state')
  missing = [name for name in required if not (checkpoint / name).exists()]
  if missing:
    raise ValueError(f'Missing terminal checkpoint items at {step}: {missing}')


def _expected_eval_steps(final_step: int) -> tuple[int, ...]:
  return tuple(range(SOURCE_STEP + READOUT_INTERVAL, final_step + 1, READOUT_INTERVAL))


def _require_exact_steps(tags: dict[str, dict[int, float]],
                         tag: str,
                         expected: tuple[int, ...]) -> None:
  actual = tuple(sorted(tags.get(tag, {})))
  if actual != expected:
    raise ValueError(f'{tag} cadence mismatch: expected {expected}, got {actual}')


def _replica_values(tags: dict[str, dict[int, float]],
                    namespace: str,
                    horizon: int,
                    step: int) -> dict[int, float]:
  prefix = f'{namespace}dense_rhs/candidate_{horizon}_return_replica_'
  replicas = {}
  for tag, values in tags.items():
    if not tag.startswith(prefix):
      continue
    suffix = tag[len(prefix):]
    if suffix.isdigit() and step in values:
      replicas[int(suffix)] = values[step]
  return replicas


def _require_replicas(tags: dict[str, dict[int, float]],
                      namespace: str,
                      horizon: int,
                      step: int,
                      expected_count: int) -> None:
  replicas = _replica_values(tags, namespace, horizon, step)
  expected_indices = set(range(expected_count))
  if set(replicas) != expected_indices:
    raise ValueError(
        f'Expected replica indices 0..{expected_count - 1} for '
        f'{namespace}h={horizon} at {step}; got {sorted(replicas)}'
    )
  if not all(math.isfinite(value) for value in replicas.values()):
    raise ValueError(f'Non-finite replicas for {namespace}h={horizon} at {step}')


def _deployed_horizon_at(profile: str,
                         mode: str,
                         step: int,
                         query_rows: list[dict[str, str]]) -> int:
  scripted = _expected_horizon(profile, mode, step)
  if scripted is not None:
    return scripted
  deployed = 2
  for row in query_rows:
    if int(row['step']) > int(step):
      break
    deployed = int(float(row['deployed_horizon']))
  return deployed


def _validate_media(run_dir: Path,
                    manifest: dict,
                    contract: dict,
                    deployed_by_step: dict[int, int]) -> list[dict]:
  media = []
  rollout_root = run_dir / 'artifacts' / 'rollouts' / manifest['run_id']
  for step in contract['anchors']:
    anchor_checkpoint = run_dir / 'artifacts' / 'anchor_checkpoints' / str(step)
    required_anchor_items = ('agent', 'metadata', 'horizon_state')
    missing_anchor_items = [
        name for name in required_anchor_items
        if not (anchor_checkpoint / name).exists()
    ]
    if missing_anchor_items:
      raise ValueError(
          f'Anchor checkpoint {step} is missing items: {missing_anchor_items}'
      )
    anchor = rollout_root / f'step_{step:06d}'
    metadata_path = anchor / 'metadata.json'
    if not metadata_path.is_file():
      raise ValueError(f'Missing rollout metadata at {step}')
    metadata = json.loads(metadata_path.read_text())
    if int(metadata.get('global_step', -1)) != int(step):
      raise ValueError(f'Rollout metadata step mismatch at {step}')
    environment = metadata.get('environment', {})
    if environment.get('task') != 'pendulum-swingup':
      raise ValueError(f'Rollout task mismatch at {step}')
    if tuple(environment.get('action_delay_schedule_boundaries', ())) != tuple(
        contract['delay_boundaries']
    ):
      raise ValueError(f'Rollout delay boundaries mismatch at {step}')
    if tuple(environment.get('action_delay_schedule_values', ())) != tuple(
        contract['delay_values']
    ):
      raise ValueError(f'Rollout delay values mismatch at {step}')
    if set(metadata.get('trajectories', {})) != {'delay0', 'delay4', 'delay6'}:
      raise ValueError(f'Rollout trajectory manifest mismatch at {step}')
    expected_h = _expected_horizon(manifest['profile'], manifest['mode'], step)
    if expected_h is None:
      expected_h = deployed_by_step.get(step, 2 if step == SOURCE_STEP else None)
    if expected_h is not None and int(metadata['selected_horizon']) != expected_h:
      raise ValueError(f'Anchor {step} deployed h={metadata["selected_horizon"]}, expected {expected_h}')
    for condition in ('delay0', 'delay4', 'delay6'):
      trajectory = anchor / f'trajectory_{condition}.npz'
      if not trajectory.is_file() or trajectory.stat().st_size <= 0:
        raise ValueError(f'Missing {condition} trajectory at {step}')
    gif = anchor / 'pendulum_delay0_vs_delay4_vs_delay6.gif'
    png = anchor / 'pendulum_delay0_vs_delay4_vs_delay6_frame.png'
    if (
        not gif.is_file() or gif.stat().st_size <= 0 or
        not png.is_file() or png.stat().st_size <= 0
    ):
      raise ValueError(f'Missing GIF/PNG at {step}')
    media.append({'step': step, 'gif': str(gif), 'png': str(png)})
  return media


def validate_run(run_dir: Path) -> dict:
  run_dir = run_dir.resolve()
  manifest = json.loads((run_dir / 'run_manifest.json').read_text())
  profile, mode = manifest['profile'], manifest['mode']
  contract = _contract(mode, profile)
  if manifest.get('campaign') != 'pendulum_delay_controller_diagnostic':
    raise ValueError('Wrong campaign manifest')
  if manifest.get('run_id') != _expected_run_id(mode, profile):
    raise ValueError('Run ID does not match the frozen mode/profile')
  if manifest.get('controller') != PROFILES[profile]['controller']:
    raise ValueError('Controller does not match the frozen profile')
  if manifest.get('score_variant') != PROFILES[profile]['variant']:
    raise ValueError('Score variant does not match frozen profile')
  expected_config_hash = _expected_config_hash(
      manifest.get('git_commit', ''),
      mode,
      profile,
      manifest.get('run_id', ''),
      contract['final_step'],
  )
  if manifest.get('config_hash') != expected_config_hash:
    raise ValueError('Config hash does not match the frozen identity projection')
  expected_shadow_only = profile != 'b5'
  if manifest.get('shadow_only') is not expected_shadow_only:
    raise ValueError('Shadow-only mode does not match the frozen profile')
  parent = manifest.get('parent', {})
  if (
      parent.get('run_id') != SOURCE_RUN_ID or
      parent.get('config_hash') != SOURCE_CONFIG_HASH or
      int(parent.get('checkpoint_step', -1)) != SOURCE_STEP
  ):
    raise ValueError('Parent provenance does not match validated source job 4632')
  _validate_source_parent(Path(parent.get('run_dir', '')))
  if int(manifest.get('source_step', -1)) != SOURCE_STEP:
    raise ValueError('Source step mismatch')
  if int(manifest.get('final_step', -1)) != contract['final_step']:
    raise ValueError('Final step mismatch')
  if tuple(manifest.get('query_steps', ())) != contract['queries']:
    raise ValueError('Manifest query cadence mismatch')
  if tuple(manifest.get('artifact_anchors', ())) != contract['anchors']:
    raise ValueError('Manifest artifact anchors mismatch')
  if tuple(manifest.get('reference_steps', ())) != contract['reference_steps']:
    raise ValueError('Manifest reference cadence mismatch')
  delay_schedule = manifest.get('delay_schedule', {})
  if (
      tuple(delay_schedule.get('boundaries', ())) != contract['delay_boundaries'] or
      tuple(delay_schedule.get('values', ())) != contract['delay_values'] or
      delay_schedule.get('observed') is not False
  ):
    raise ValueError('Manifest delay schedule mismatch')
  scripted = manifest.get('scripted_horizon', {})
  if (
      tuple(scripted.get('steps', ())) != contract['script_steps'] or
      tuple(scripted.get('values', ())) != contract['script_values']
  ):
    raise ValueError('Manifest scripted-horizon schedule mismatch')
  expected_replicas = 128 if profile == 'b5' else 32
  expected_manifest = {
      'candidate_horizons': list(HORIZONS),
      'controller_evidence_reset_on_restore': True,
      'deployment_replicas': expected_replicas,
      'query_env_steps': 256,
      'reference_replicas': 128,
      'conditional_reference_env_steps': 500,
  }
  for field, expected in expected_manifest.items():
    if manifest.get(field) != expected:
      raise ValueError(
          f'Manifest {field} drifted: {manifest.get(field)!r} != {expected!r}'
      )
  if not (run_dir / 'TRAINING_COMPLETE').is_file():
    raise ValueError('TRAINING_COMPLETE is missing')
  if not (run_dir / 'MEDIA_COMPLETE').is_file():
    raise ValueError('MEDIA_COMPLETE is missing')
  _validate_checkpoint(run_dir, contract['final_step'])

  scalar_rows, tags = _scalars(run_dir)
  query_rows = _read_csv(run_dir / 'metrics' / 'horizon_queries.csv')
  query_steps = tuple(int(row['step']) for row in query_rows)
  if query_steps != contract['queries']:
    raise ValueError(f'Query cadence mismatch: {query_steps}')
  for row in query_rows:
    for field, value in row.items():
      if field != 'phase_name' and not _finite(value):
        raise ValueError(f'Non-finite query field {field}: {row}')

  selected = []
  shadow = []
  for row in query_rows:
    step = int(row['step'])
    deployed = int(float(row['deployed_horizon']))
    selected.append(deployed)
    shadow.append(int(float(row['shadow_selected_horizon'])))
    if deployed != int(float(row['selected_horizon'])):
      raise ValueError(f'Deployed/selected mismatch at {step}')
    expected_h = _expected_horizon(profile, mode, step)
    if expected_h is not None and deployed != expected_h:
      raise ValueError(f'{profile} deployed h={deployed} at {step}; expected {expected_h}')
    expected_delay = _piecewise(
        step, contract['delay_boundaries'], contract['delay_values']
    )
    actual_delay = tags.get('environment/effective_action_delay', {}).get(step)
    if actual_delay is None or int(actual_delay) != expected_delay:
      raise ValueError(f'Delay metric mismatch at {step}: {actual_delay} != {expected_delay}')
    if int(float(row['num_candidate_horizons'])) != len(HORIZONS):
      raise ValueError(f'Candidate count mismatch at {step}')
    for horizon in HORIZONS:
      for suffix in (
          'env_mean', 'env_std', 'decision_score', 'deployment_score',
          'return_term', 'roughness_term', 'return_std_term',
          'learner_proxy_term',
      ):
        tag = f'dense_rhs/candidate_{horizon}_{suffix}'
        if step not in tags.get(tag, {}):
          raise ValueError(f'Missing {tag} at {step}')
      _require_replicas(
          tags,
          '',
          horizon,
          step,
          expected_replicas,
      )
    bucket = tags.get('dense_rhs/training_bucket_horizon', {}).get(step)
    if bucket is None or int(bucket) != deployed:
      raise ValueError(f'Training/planning horizon mismatch at {step}: {bucket} != {deployed}')
    shadow_flag = tags.get('dense_rhs/shadow_only', {}).get(step)
    if shadow_flag is None or bool(round(shadow_flag)) != (profile != 'b5'):
      raise ValueError(f'Shadow-only flag mismatch at {step}')
    deployed_scalar = tags.get('dense_rhs/deployed_horizon', {}).get(step)
    shadow_scalar = tags.get(
        'dense_rhs/shadow_controller_selected_horizon', {}
    ).get(step)
    if deployed_scalar is None or int(deployed_scalar) != deployed:
      raise ValueError(f'Deployed scalar/row mismatch at {step}')
    if shadow_scalar is None or int(shadow_scalar) != shadow[-1]:
      raise ValueError(f'Counterfactual scalar/row mismatch at {step}')

    if profile == 'b5':
      logged_means = np.asarray([
          tags[f'dense_rhs/candidate_{h}_env_mean'][step] for h in HORIZONS
      ], dtype=np.float32)
      replica_means = []
      for horizon in HORIZONS:
        replicas = _replica_values(tags, '', horizon, step)
        replica_means.append(np.mean(
            np.asarray(
                [replicas[index] for index in range(expected_replicas)],
                dtype=np.float32,
            ),
            dtype=np.float32,
        ))
      replica_means = np.asarray(replica_means, dtype=np.float32)
      if not np.allclose(
          logged_means,
          replica_means,
          rtol=1e-5,
          atol=1e-4,
      ):
        raise ValueError(
            f'B5 logged means do not match the 128 retained replicas at '
            f'{step}: logged={logged_means.tolist()}, '
            f'recomputed={replica_means.tolist()}'
        )
      expected_argmax = HORIZONS[int(np.argmax(replica_means))]
      decision_horizons = {
          'deployed': deployed,
          'selected': int(float(row['selected_horizon'])),
          'shadow': int(float(row['shadow_selected_horizon'])),
          'proposed': int(float(row['proposed_horizon'])),
          'best': int(float(row['best_h'])),
      }
      if any(value != expected_argmax for value in decision_horizons.values()):
        raise ValueError(
            f'B5 decision {decision_horizons} does not equal the exact '
            f'float32 raw-mean argmax h={expected_argmax} at '
            f'{step}: {replica_means.tolist()}'
        )
      for horizon in HORIZONS:
        for suffix in ('roughness_term', 'return_std_term', 'learner_proxy_term'):
          value = tags[f'dense_rhs/candidate_{horizon}_{suffix}'][step]
          if abs(value) > ZERO_ATOL:
            raise ValueError(f'B5 non-return term {suffix}={value} at h={horizon}, step={step}')
      for field in ('roughness_term_best', 'return_std_term_best', 'learner_proxy_term_best'):
        value = float(row[field])
        if abs(value) > ZERO_ATOL:
          raise ValueError(f'B5 query alias {field}={value} at step={step}')

  expected_eval_steps = _expected_eval_steps(contract['final_step'])
  for tag in (
      'eval/return_mean',
      'eval/selected_horizon',
      'eval/training_bucket_horizon',
  ):
    _require_exact_steps(tags, tag, expected_eval_steps)
  eval_curve = sorted(tags['eval/return_mean'].items())
  for step in expected_eval_steps:
    expected_h = _deployed_horizon_at(profile, mode, step, query_rows)
    selected_h = int(tags['eval/selected_horizon'][step])
    bucket_h = int(tags['eval/training_bucket_horizon'][step])
    if selected_h != expected_h or bucket_h != expected_h:
      raise ValueError(
          f'Evaluation/deployed horizon mismatch at {step}: '
          f'selected={selected_h}, bucket={bucket_h}, expected={expected_h}'
      )

  episode_rows = _read_csv(run_dir / 'metrics' / 'episodes.csv')
  for row in episode_rows:
    for field in (
        'step', 'env_index', 'episode_index', 'episode_return',
        'episode_length', 'selected_horizon',
    ):
      if not _finite(row.get(field)):
        raise ValueError(f'Non-finite episode field {field}: {row}')
    episode_horizon = int(float(row['selected_horizon']))
    if episode_horizon not in HORIZONS:
      raise ValueError(f'Episode horizon outside frozen candidates: {row}')
    expected_h = _deployed_horizon_at(
        profile,
        mode,
        int(float(row['step'])),
        query_rows,
    )
    if episode_horizon != expected_h:
      raise ValueError(
          f'Episode horizon drifted from deployment: {row}; '
          f'expected h={expected_h}'
      )

  if mode == 'full':
    completed = tags.get('reference_probe/completed', {})
    if tuple(sorted(completed)) != contract['reference_steps']:
      raise ValueError(f'Reference cadence mismatch: {tuple(sorted(completed))}')
    for step in contract['reference_steps']:
      for horizon in HORIZONS:
        for namespace in ('reference_probe/', 'conditional_reference_probe/'):
          mean_tag = f'{namespace}dense_rhs/candidate_{horizon}_env_mean'
          if step not in tags.get(mean_tag, {}):
            raise ValueError(f'Missing {mean_tag} at {step}')
          _require_replicas(tags, namespace, horizon, step, 128)

  media = _validate_media(
      run_dir,
      manifest,
      contract,
      {
          int(row['step']): int(float(row['deployed_horizon']))
          for row in query_rows
      },
  )
  summary = {
      'valid': True,
      'run_id': manifest['run_id'],
      'profile': profile,
      'mode': mode,
      'git_commit': manifest['git_commit'],
      'config_hash': manifest['config_hash'],
      'query_steps': list(query_steps),
      'selected_horizons': selected,
      'shadow_selected_horizons': shadow,
      'evaluation_points': len(eval_curve),
      'scalar_rows': len(scalar_rows),
      'media': media,
  }
  _write_json(run_dir / 'validation_summary.json', summary)
  return summary


def _mean_over_window(curve, start: int, end: int) -> float:
  values = [value for step, value in curve if start <= step < end]
  return float(np.mean(values)) if values else math.nan


def _auc_over_window(curve,
                     start: int | None = None,
                     end: int | None = None) -> float:
  selected = [
      (step, value) for step, value in curve
      if (start is None or step >= start) and (end is None or step < end)
  ]
  if not selected:
    return math.nan
  if len(selected) == 1:
    return float(selected[0][1])
  steps = np.asarray([step for step, _ in selected], dtype=np.float64)
  values = np.asarray([value for _, value in selected], dtype=np.float64)
  return float(np.trapezoid(values, steps) / (steps[-1] - steps[0]))


def _rank_correlation(left, right) -> float:
  left = np.asarray(left, dtype=np.float64)
  right = np.asarray(right, dtype=np.float64)
  if left.size < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
    return 0.0
  left_rank = np.argsort(np.argsort(left)).astype(np.float64)
  right_rank = np.argsort(np.argsort(right)).astype(np.float64)
  return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _delay_at(step: int) -> int:
  return _piecewise(
      step,
      (38_000, 42_000, 46_000, 50_000),
      (0, 2, 6, 4, 0),
  )


def _aba_chatter(sequence: list[int]) -> bool:
  return any(
      sequence[index] == sequence[index + 2] and
      sequence[index] != sequence[index + 1]
      for index in range(len(sequence) - 2)
  )


def _run_summary(run_dir: Path) -> dict:
  manifest = json.loads((run_dir / 'run_manifest.json').read_text())
  _, tags = _scalars(run_dir)
  queries = _read_csv(run_dir / 'metrics' / 'horizon_queries.csv')
  curve = sorted(tags['eval/return_mean'].items())
  auc = _auc_over_window(curve)
  delayed_auc = _auc_over_window(curve, 38_000, 50_000)
  phase_returns = {
      name: _mean_over_window(curve, start, end)
      for name, start, end, _ in PHASES
  }
  phase_aucs = {
      name: _auc_over_window(curve, start, end)
      for name, start, end, _ in PHASES
  }
  reference_regrets = []
  reference_regret_fractions = []
  reference_oracles = {}
  for step in REFERENCE_STEPS:
    means = {
        h: tags.get(
            f'conditional_reference_probe/dense_rhs/candidate_{h}_env_mean', {}
        ).get(step)
        for h in HORIZONS
    }
    if any(value is None for value in means.values()):
      continue
    oracle = max(means, key=means.get)
    reference_oracles[str(step)] = oracle
    deployed = int(next(float(row['deployed_horizon']) for row in queries if int(row['step']) == step))
    regret = float(means[oracle] - means[deployed])
    reference_regrets.append(regret)
    reference_regret_fractions.append(
        regret / max(abs(float(means[oracle])), 1e-12)
    )
  selected_horizons = [
      int(float(row['deployed_horizon'])) for row in queries
  ]
  shadow_horizons = [
      int(float(row['shadow_selected_horizon'])) for row in queries
  ]
  rank_correlations = []
  for row in queries:
    step = int(row['step'])
    rank_correlations.append(_rank_correlation(
        [tags[f'dense_rhs/candidate_{h}_decision_score'][step] for h in HORIZONS],
        [tags[f'dense_rhs/candidate_{h}_env_mean'][step] for h in HORIZONS],
    ))
  phase_median_horizons = {}
  switches_per_phase = {}
  for name, start, end, _ in PHASES:
    phase_sequence = [
        int(float(row['deployed_horizon'])) for row in queries
        if start <= int(row['step']) < end
    ]
    phase_median_horizons[name] = (
        float(np.median(phase_sequence)) if phase_sequence else math.nan
    )
    switches_per_phase[name] = sum(
        left != right for left, right in zip(phase_sequence, phase_sequence[1:])
    )
  return {
      'run_dir': str(run_dir),
      'run_id': manifest['run_id'],
      'profile': manifest['profile'],
      'controller': manifest['controller'],
      'normalized_return_auc': auc,
      'delayed_phase_auc': delayed_auc,
      'phase_mean_returns': phase_returns,
      'phase_normalized_aucs': phase_aucs,
      'phase_median_deployed_horizons': phase_median_horizons,
      'final_clean_return': phase_returns['clean_ii'],
      'mean_reference_oracle_regret': (
          float(np.mean(reference_regrets)) if reference_regrets else math.nan
      ),
      'mean_reference_oracle_regret_fraction': (
          float(np.mean(reference_regret_fractions))
          if reference_regret_fractions else math.nan
      ),
      'reference_oracles': reference_oracles,
      'selected_horizons': selected_horizons,
      'shadow_horizons': shadow_horizons,
      'switches_per_phase': switches_per_phase,
      'deployed_aba_chatter': _aba_chatter(selected_horizons),
      'counterfactual_aba_chatter': _aba_chatter(shadow_horizons),
      'mean_score_return_rank_correlation': float(np.mean(rank_correlations)),
      'mean_query_compute_s': float(np.mean([
          float(row['query_total_s']) for row in queries
      ])),
      'eval_curve': curve,
  }


def _validated_full_run_dirs(root: Path) -> dict[str, Path]:
  runs = {}
  for path in sorted(root.glob('penddiag__*/attempt_*')):
    if not (path / 'RUN_VALID').is_file():
      continue
    manifest_path = path / 'run_manifest.json'
    validation_path = path / 'validation_summary.json'
    if not manifest_path.is_file() or not validation_path.is_file():
      raise ValueError(f'Validated run is missing manifest/summary: {path}')
    manifest = json.loads(manifest_path.read_text())
    validation = json.loads(validation_path.read_text())
    if manifest.get('mode') != 'full':
      continue
    profile = manifest.get('profile')
    if profile not in PROFILES:
      raise ValueError(f'Unknown validated profile {profile!r}: {path}')
    if (
        not validation.get('valid') or
        validation.get('mode') != 'full' or
        validation.get('profile') != profile or
        validation.get('run_id') != manifest.get('run_id') or
        validation.get('git_commit') != manifest.get('git_commit') or
        validation.get('config_hash') != manifest.get('config_hash')
    ):
      raise ValueError(f'Cached validation identity mismatch: {path}')
    if profile in runs:
      raise ValueError(
          f'Multiple RUN_VALID attempts for {profile}: {runs[profile]} and {path}'
      )
    # Re-run the full validator when the non-compact source and artifacts are
    # present. A synced compact cache is instead authenticated by the marker
    # and identity-bound validation summary written before RUN_VALID.
    parent_dir = Path(manifest.get('parent', {}).get('run_dir', ''))
    terminal = path / 'checkpoint' / str(manifest.get('final_step'))
    if parent_dir.is_dir() and terminal.is_dir():
      validate_run(path)
    runs[profile] = path
  missing = sorted(set(PROFILES) - set(runs))
  if missing:
    raise ValueError(f'Missing valid full profiles: {missing}')
  commits = {
      json.loads((path / 'run_manifest.json').read_text())['git_commit']
      for path in runs.values()
  }
  if len(commits) != 1:
    raise ValueError(f'Full profiles do not share one scientific revision: {commits}')
  return runs


def reduce_runs(root: Path, output_dir: Path) -> dict:
  import matplotlib.pyplot as plt

  run_dirs = _validated_full_run_dirs(root)
  runs = {profile: _run_summary(path) for profile, path in run_dirs.items()}
  output_dir.mkdir(parents=True, exist_ok=True)

  fig, axis = plt.subplots(figsize=(10, 5))
  for profile in sorted(runs):
    curve = runs[profile]['eval_curve']
    axis.plot([x for x, _ in curve], [y for _, y in curve], label=profile.upper())
  for boundary in (38_000, 42_000, 46_000, 50_000):
    axis.axvline(boundary, color='0.75', linewidth=1)
  axis.set(xlabel='Training transitions', ylabel='Evaluation return', title='Pendulum delay/controller diagnostic')
  axis.legend(ncol=5)
  fig.tight_layout()
  fig.savefig(output_dir / 'evaluation_returns.png', dpi=180)
  plt.close(fig)

  fig, axes = plt.subplots(len(runs), 1, figsize=(10, 9), sharex=True, sharey=True)
  for axis, profile in zip(axes, sorted(runs)):
    axis.step(FULL_QUERIES, runs[profile]['selected_horizons'], where='post', label='deployed')
    axis.step(FULL_QUERIES, runs[profile]['shadow_horizons'], where='post', linestyle='--', label='counterfactual')
    axis.set_ylabel(profile.upper())
    axis.set_yticks(HORIZONS)
  axes[0].legend(ncol=2)
  axes[-1].set_xlabel('Training transitions')
  fig.tight_layout()
  fig.savefig(output_dir / 'horizon_trajectories.png', dpi=180)
  plt.close(fig)

  fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=True)
  axes = axes.flat
  for axis, profile in zip(axes, sorted(runs)):
    _, tags = _scalars(run_dirs[profile])
    matrix = np.asarray([
        [tags[f'dense_rhs/candidate_{h}_env_mean'][step] for h in HORIZONS]
        for step in FULL_QUERIES
    ]).T
    image = axis.imshow(matrix, aspect='auto', origin='lower', cmap='viridis')
    axis.set_title(f'{profile.upper()} raw candidate means')
    axis.set_yticks(range(len(HORIZONS)), HORIZONS)
    axis.set_xticks(range(len(FULL_QUERIES)), [str(s // 1000) for s in FULL_QUERIES])
    axis.set_ylabel('Candidate h')
    fig.colorbar(image, ax=axis, fraction=0.045)
  axes[-1].axis('off')
  for axis in axes[:5]:
    axis.set_xlabel('Query (k transitions)')
  fig.tight_layout()
  fig.savefig(output_dir / 'candidate_return_heatmaps.png', dpi=180)
  plt.close(fig)

  fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True)
  axes = axes.flat
  for axis, profile in zip(axes, sorted(runs)):
    _, tags = _scalars(run_dirs[profile])
    queries = _read_csv(run_dirs[profile] / 'metrics' / 'horizon_queries.csv')
    chosen = [
        int(float(row[
            'deployed_horizon' if profile == 'b5' else 'shadow_selected_horizon'
        ]))
        for row in queries
    ]
    for suffix, label in (
        ('return_term', 'return'),
        ('roughness_term', 'roughness'),
        ('return_std_term', 'return spread'),
    ):
      axis.plot(
          FULL_QUERIES,
          [
              tags[f'dense_rhs/candidate_{horizon}_{suffix}'][step]
              for horizon, step in zip(chosen, FULL_QUERIES)
          ],
          marker='o',
          label=label,
      )
    axis.axhline(0, color='0.4', linewidth=0.8)
    axis.set_title(
        f'{profile.upper()} '
        f'{"deployed" if profile == "b5" else "counterfactual"} decision terms'
    )
    axis.grid(alpha=0.2)
  axes[0].legend(ncol=3, fontsize=8)
  axes[-1].axis('off')
  for axis in axes[:5]:
    axis.set_xlabel('Training transitions')
    axis.set_ylabel('Score contribution')
  fig.tight_layout()
  fig.savefig(output_dir / 'score_components.png', dpi=180)
  plt.close(fig)

  static_auc_gap = abs(
      runs['b1']['normalized_return_auc'] - runs['b2']['normalized_return_auc']
  ) / max(
      abs(runs['b1']['normalized_return_auc']),
      abs(runs['b2']['normalized_return_auc']),
      1e-12,
  )
  if static_auc_gap < 0.02:
    static_winner = 'b2'
    static_reason = 'AUC difference below 2%; retain canonical h=3'
  else:
    candidate = max(
        ('b1', 'b2'), key=lambda profile: runs[profile]['normalized_return_auc']
    )
    other = 'b2' if candidate == 'b1' else 'b1'
    if runs[candidate]['final_clean_return'] >= 0.95 * runs[other]['final_clean_return']:
      static_winner = candidate
      static_reason = 'AUC winner passes the 95% final-clean safeguard'
    else:
      static_winner = None
      static_reason = 'AUC winner fails the 95% final-clean safeguard'
  comparison_static = static_winner or max(
      ('b1', 'b2'), key=lambda profile: runs[profile]['normalized_return_auc']
  )
  forced_gates = {
      profile: {
          'passes': bool(
              runs[profile]['delayed_phase_auc'] >=
              1.05 * runs[comparison_static]['delayed_phase_auc'] and
              runs[profile]['final_clean_return'] >=
              0.95 * runs[comparison_static]['final_clean_return']
          ),
          'delayed_auc_ratio_to_static': (
              runs[profile]['delayed_phase_auc'] /
              max(abs(runs[comparison_static]['delayed_phase_auc']), 1e-12)
          ),
          'final_clean_ratio_to_static': (
              runs[profile]['final_clean_return'] /
              max(abs(runs[comparison_static]['final_clean_return']), 1e-12)
          ),
      }
      for profile in ('b3', 'b4')
  }
  b5_gate = {
      'passes': bool(
          runs['b5']['delayed_phase_auc'] >=
          1.05 * runs[comparison_static]['delayed_phase_auc'] and
          runs['b5']['final_clean_return'] >=
          0.95 * runs[comparison_static]['final_clean_return'] and
          runs['b5']['mean_reference_oracle_regret_fraction'] <= 0.02
      ),
      'delayed_auc_ratio_to_static': (
          runs['b5']['delayed_phase_auc'] /
          max(abs(runs[comparison_static]['delayed_phase_auc']), 1e-12)
      ),
      'final_clean_ratio_to_static': (
          runs['b5']['final_clean_return'] /
          max(abs(runs[comparison_static]['final_clean_return']), 1e-12)
      ),
      'independent_oracle_regret_fraction': (
          runs['b5']['mean_reference_oracle_regret_fraction']
      ),
      'chatter_diagnostic_only': runs['b5']['deployed_aba_chatter'],
  }
  summary = {
      'schema_version': 1,
      'campaign': 'pendulum_delay_controller_diagnostic',
      'static_decision': {
          'winner': static_winner,
          'comparison_static': comparison_static,
          'relative_auc_gap': static_auc_gap,
          'reason': static_reason,
      },
      'forced_control_gates': forced_gates,
      'return_argmax_gate': b5_gate,
      'runs': runs,
  }
  _write_json(output_dir / 'summary.json', summary)
  with (output_dir / 'summary.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=(
        'profile', 'controller', 'normalized_return_auc',
        'delayed_phase_auc', 'final_clean_return',
        'mean_reference_oracle_regret',
        'mean_reference_oracle_regret_fraction',
        'mean_score_return_rank_correlation', 'mean_query_compute_s',
    ))
    writer.writeheader()
    for profile in sorted(runs):
      writer.writerow({key: runs[profile][key] for key in writer.fieldnames})
  with (output_dir / 'phase_metrics.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=(
        'profile', 'phase', 'delay', 'mean_eval_return',
        'normalized_eval_auc', 'median_deployed_horizon', 'switches',
    ))
    writer.writeheader()
    for profile in sorted(runs):
      for phase, _, _, delay in PHASES:
        writer.writerow({
            'profile': profile,
            'phase': phase,
            'delay': delay,
            'mean_eval_return': runs[profile]['phase_mean_returns'][phase],
            'normalized_eval_auc': runs[profile]['phase_normalized_aucs'][phase],
            'median_deployed_horizon': (
                runs[profile]['phase_median_deployed_horizons'][phase]
            ),
            'switches': runs[profile]['switches_per_phase'][phase],
        })
  candidate_fields = (
      'profile', 'step', 'delay', 'horizon', 'deployed_horizon',
      'counterfactual_horizon', 'env_mean', 'env_std', 'decision_score',
      'deployment_score', 'return_term', 'roughness_term',
      'return_std_term', 'conditional_reference_mean',
  )
  with (output_dir / 'candidate_evidence.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=candidate_fields)
    writer.writeheader()
    for profile in sorted(runs):
      _, tags = _scalars(run_dirs[profile])
      queries = _read_csv(run_dirs[profile] / 'metrics' / 'horizon_queries.csv')
      for row in queries:
        step = int(row['step'])
        for horizon in HORIZONS:
          writer.writerow({
              'profile': profile,
              'step': step,
              'delay': _delay_at(step),
              'horizon': horizon,
              'deployed_horizon': int(float(row['deployed_horizon'])),
              'counterfactual_horizon': int(float(row['shadow_selected_horizon'])),
              'env_mean': tags[f'dense_rhs/candidate_{horizon}_env_mean'][step],
              'env_std': tags[f'dense_rhs/candidate_{horizon}_env_std'][step],
              'decision_score': tags[
                  f'dense_rhs/candidate_{horizon}_decision_score'
              ][step],
              'deployment_score': tags[
                  f'dense_rhs/candidate_{horizon}_deployment_score'
              ][step],
              'return_term': tags[
                  f'dense_rhs/candidate_{horizon}_return_term'
              ][step],
              'roughness_term': tags[
                  f'dense_rhs/candidate_{horizon}_roughness_term'
              ][step],
              'return_std_term': tags[
                  f'dense_rhs/candidate_{horizon}_return_std_term'
              ][step],
              'conditional_reference_mean': tags.get(
                  f'conditional_reference_probe/dense_rhs/'
                  f'candidate_{horizon}_env_mean', {}
              ).get(step, ''),
          })
  (output_dir / 'AGGREGATE_VALID').touch()
  return summary


def main() -> int:
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest='command', required=True)
  manifest = subparsers.add_parser('write-manifest')
  manifest.add_argument('--run-dir', type=Path, required=True)
  manifest.add_argument('--run-id', required=True)
  manifest.add_argument('--mode', choices=('smoke', 'full'), required=True)
  manifest.add_argument('--profile', choices=tuple(PROFILES), required=True)
  manifest.add_argument('--commit', required=True)
  manifest.add_argument('--config-hash', required=True)
  manifest.add_argument('--parent-run-dir', type=Path, required=True)
  manifest.add_argument('--parent-checkpoint-step', type=int, required=True)
  validate = subparsers.add_parser('validate')
  validate.add_argument('--run-dir', type=Path, required=True)
  reduce_parser = subparsers.add_parser('reduce')
  reduce_parser.add_argument('--root', type=Path, required=True)
  reduce_parser.add_argument('--output-dir', type=Path, required=True)
  args = parser.parse_args()
  if args.command == 'write-manifest':
    return write_manifest(args)
  if args.command == 'validate':
    print(json.dumps(validate_run(args.run_dir), indent=2, sort_keys=True))
    return 0
  print(json.dumps(reduce_runs(args.root, args.output_dir), indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
