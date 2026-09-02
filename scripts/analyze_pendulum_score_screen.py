#!/usr/bin/env python3
"""Validate and reduce the sequential-delay Pendulum score screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


SOURCE_STEP = 34_000
FULL_FINAL_STEP = 54_000
FULL_QUERY_STEPS = tuple(range(36_000, 54_001, 2_000))
FULL_ANCHORS = (34_000, 38_000, 42_000, 46_000, 50_000, 54_000)
REFERENCE_STEPS = (36_000, 40_000, 44_000, 48_000, 52_000, 54_000)
PHASES = (
    ('clean_i', 34_000, 38_000, 0),
    ('delay_2', 38_000, 42_000, 2),
    ('delay_6', 42_000, 46_000, 6),
    ('delay_4', 46_000, 50_000, 4),
    ('clean_ii', 50_000, 54_001, 0),
)
PROFILE_VARIANTS = {
    's0': 'current_additive',
    's1': 'calibrated_local_roughness',
    's2': 'return_first',
    's3': 'curvature_bellman',
}


def _full_run_candidates(root: Path, profile: str) -> list[Path]:
  """Return full-run attempts for one profile's frozen run-id prefix."""
  return sorted(root.glob(f'pendscore__{profile}_*/attempt_*'))


def _read_csv(path: Path) -> list[dict[str, str]]:
  with path.open(newline='') as handle:
    return list(csv.DictReader(handle))


def _write_json(path: Path, value) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
  temporary.replace(path)


def _finite(value: str) -> bool:
  try:
    return math.isfinite(float(value))
  except (TypeError, ValueError):
    return False


def _contract(mode: str, profile: str) -> dict:
  if mode == 'source':
    return {
        'final_step': SOURCE_STEP,
        'query_steps': (32_000,),
        'anchors': (),
        'schedule_boundaries': (),
        'schedule_values': (),
    }
  if mode == 'smoke':
    return {
        'final_step': 36_000,
        'query_steps': (34_400, 34_800, 35_200, 35_600, 36_000),
        'anchors': (34_000, 34_400, 34_800, 35_200, 35_600, 36_000),
        'schedule_boundaries': (34_400, 34_800, 35_200, 35_600),
        'schedule_values': (0, 2, 6, 4, 0),
    }
  if mode == 'full':
    return {
        'final_step': FULL_FINAL_STEP,
        'query_steps': FULL_QUERY_STEPS,
        'anchors': FULL_ANCHORS,
        'schedule_boundaries': (38_000, 42_000, 46_000, 50_000),
        'schedule_values': (0, 2, 6, 4, 0),
    }
  raise ValueError(f'Unknown mode {mode!r} for profile {profile!r}')


def write_manifest(args) -> int:
  run_dir = args.run_dir.resolve()
  contract = _contract(args.mode, args.profile)
  manifest = {
      'schema_version': 1,
      'campaign': 'pendulum_score_formulation_screen',
      'run_id': args.run_id,
      'mode': args.mode,
      'profile': args.profile,
      'score_variant': PROFILE_VARIANTS[args.profile],
      'git_commit': args.commit,
      'config_hash': args.config_hash,
      'parent': {
          'run_dir': str(args.parent_run_dir.resolve()),
          'checkpoint_step': args.parent_checkpoint_step,
      },
      'source_step': args.parent_checkpoint_step,
      'final_step': contract['final_step'],
      'query_steps': list(contract['query_steps']),
      'artifact_anchors': list(contract['anchors']),
      'delay_schedule': {
          'boundaries': list(contract['schedule_boundaries']),
          'values': list(contract['schedule_values']),
          'observed': False,
      },
      'horizons': list(range(2, 9)),
      'seed': 7,
      'calibration': {
          'return_weight_grid': [10, 20, 40],
          'selected_return_weight': 10,
          'tie_rule': 'lowest weight after identical 4/4 historical oracle agreement',
          'return_scale': 50.0,
          'roughness_discount': 0.99,
          'return_first_tolerance': 5.0,
          'horizon_switch_cost': 0.05,
          'bellman_risk_weight': 0.1,
          'curvature_risk_scale': 0.01,
      },
  }
  _write_json(run_dir / 'run_manifest.json', manifest)
  return 0


def _validate_checkpoint(run_dir: Path, final_step: int) -> None:
  checkpoint = run_dir / 'checkpoint' / str(final_step)
  if not checkpoint.is_dir():
    raise ValueError(f'Missing terminal checkpoint {checkpoint}')
  required = ('agent', 'global_step', 'buffer_state', 'horizon_state')
  missing = [name for name in required if not (checkpoint / name).exists()]
  if missing:
    raise ValueError(f'Terminal checkpoint is missing items: {missing}')


def validate_run(run_dir: Path) -> dict:
  run_dir = run_dir.resolve()
  manifest = json.loads((run_dir / 'run_manifest.json').read_text())
  profile = manifest['profile']
  mode = manifest['mode']
  if profile not in PROFILE_VARIANTS:
    raise ValueError(f'Unknown profile {profile!r}')
  if manifest['score_variant'] != PROFILE_VARIANTS[profile]:
    raise ValueError('Manifest score variant does not match profile')
  contract = _contract(mode, profile)
  if int(manifest['final_step']) != int(contract['final_step']):
    raise ValueError('Manifest final step does not match frozen contract')
  if tuple(manifest['query_steps']) != tuple(contract['query_steps']):
    raise ValueError('Manifest query cadence does not match frozen contract')
  if tuple(manifest['delay_schedule']['values']) != tuple(contract['schedule_values']):
    raise ValueError('Manifest delay values do not match frozen contract')
  _validate_checkpoint(run_dir, contract['final_step'])

  scalar_rows = _read_csv(run_dir / 'metrics' / 'scalars.csv')
  for row in scalar_rows:
    if not _finite(row['value']):
      raise ValueError(f'Non-finite scalar metric: {row}')
  query_rows = _read_csv(run_dir / 'metrics' / 'horizon_queries.csv')
  query_steps = tuple(int(row['step']) for row in query_rows)
  if query_steps != tuple(contract['query_steps']):
    raise ValueError(f'Query cadence mismatch: {query_steps}')
  for row in query_rows:
    for key in (
        'selected_horizon', 'proposed_horizon', 'best_fitness',
        'deployment_score_best', 'return_term_best', 'roughness_term_best',
    ):
      if not _finite(row[key]):
        raise ValueError(f'Non-finite query field {key}: {row}')
  selected = [int(float(row['selected_horizon'])) for row in query_rows]
  if any(value < 2 or value > 8 for value in selected):
    raise ValueError(f'Selected horizon outside [2,8]: {selected}')

  evaluation_rows = [
      row for row in scalar_rows if row['tag'] == 'eval/return_mean'
  ]
  expected_evaluations = (contract['final_step'] - manifest['source_step']) // 400
  if len(evaluation_rows) != expected_evaluations:
    raise ValueError(
        f'Expected {expected_evaluations} evaluation rows, got '
        f'{len(evaluation_rows)}'
    )

  media = []
  if contract['anchors']:
    rollout_root = run_dir / 'artifacts' / 'rollouts' / manifest['run_id']
    for step in contract['anchors']:
      anchor = rollout_root / f'step_{step:06d}'
      metadata_path = anchor / 'metadata.json'
      if not metadata_path.is_file():
        raise ValueError(f'Missing anchor metadata {metadata_path}')
      metadata = json.loads(metadata_path.read_text())
      if set(metadata.get('trajectories', {})) != {'delay0', 'delay4', 'delay6'}:
        raise ValueError(f'Anchor {step} lacks the d=0/4/6 trajectory trio')
      for condition in ('delay0', 'delay4', 'delay6'):
        if not (anchor / f'trajectory_{condition}.npz').is_file():
          raise ValueError(f'Missing {condition} trajectory at {step}')
      gif = anchor / 'pendulum_delay0_vs_delay4_vs_delay6.gif'
      png = anchor / 'pendulum_delay0_vs_delay4_vs_delay6_frame.png'
      if not gif.is_file() or not png.is_file():
        raise ValueError(f'Missing rendered media at {step}')
      media.append({'step': step, 'gif': str(gif), 'png': str(png)})

  summary = {
      'valid': True,
      'run_id': manifest['run_id'],
      'mode': mode,
      'profile': profile,
      'score_variant': manifest['score_variant'],
      'git_commit': manifest['git_commit'],
      'config_hash': manifest['config_hash'],
      'final_step': contract['final_step'],
      'evaluation_points': len(evaluation_rows),
      'query_steps': list(query_steps),
      'selected_horizons': selected,
      'media': media,
  }
  _write_json(run_dir / 'validation_summary.json', summary)
  return summary


def _scalars_by_tag(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
  result: dict[str, list[tuple[int, float]]] = {}
  for row in _read_csv(run_dir / 'metrics' / 'scalars.csv'):
    result.setdefault(row['tag'], []).append((int(row['step']), float(row['value'])))
  for values in result.values():
    values.sort()
  return result


def _delay_at(step: int) -> int:
  delay = 0
  for boundary, value in zip((38_000, 42_000, 46_000, 50_000), (2, 6, 4, 0)):
    if step >= boundary:
      delay = value
  return delay


def _rank_correlation(x: Iterable[float], y: Iterable[float]) -> float:
  x = np.asarray(list(x), dtype=np.float64)
  y = np.asarray(list(y), dtype=np.float64)
  if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
    return 0.0
  x_rank = np.argsort(np.argsort(x)).astype(np.float64)
  y_rank = np.argsort(np.argsort(y)).astype(np.float64)
  return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _trapezoidal_mean(values: np.ndarray, steps: np.ndarray) -> float:
  """Return the step-weighted mean using NumPy's current trapezoid API."""
  return float(np.trapezoid(values, steps) / (steps[-1] - steps[0]))


def _confirmation_profiles(runs: dict, eligible: list[str], limit: int = 2) -> list[str]:
  """Rank only promotion-eligible profiles for confirmation."""
  return sorted(
      eligible,
      key=lambda profile: runs[profile]['metrics']['mean_oracle_regret'],
  )[:limit]


def _run_metrics(run_dir: Path) -> dict:
  manifest = json.loads((run_dir / 'run_manifest.json').read_text())
  scalars = _scalars_by_tag(run_dir)
  queries = _read_csv(run_dir / 'metrics' / 'horizon_queries.csv')
  eval_curve = scalars['eval/return_mean']
  eval_steps = np.asarray([step for step, _ in eval_curve], dtype=np.float64)
  eval_values = np.asarray([value for _, value in eval_curve], dtype=np.float64)
  auc = _trapezoidal_mean(eval_values, eval_steps)
  phase_returns = {}
  phase_horizons = {}
  switches_per_phase = {}
  for name, start, end, delay in PHASES:
    phase_returns[name] = float(np.mean([
        value for step, value in eval_curve if start <= step < end
    ]))
    horizon_values = [
        int(float(row['selected_horizon'])) for row in queries
        if start <= int(row['step']) < end
    ]
    phase_horizons[name] = float(np.median(horizon_values)) if horizon_values else math.nan
    switches_per_phase[name] = sum(
        int(float(row['selected_horizon'])) != int(float(row['previous_horizon']))
        for row in queries if start <= int(row['step']) < end
    )

  selected_sequence = [int(float(row['selected_horizon'])) for row in queries]
  chatter = any(
      selected_sequence[index] == selected_sequence[index + 2] and
      selected_sequence[index] != selected_sequence[index + 1]
      for index in range(len(selected_sequence) - 2)
  )
  stable = (
      all(count <= 2 for count in switches_per_phase.values()) and
      not chatter and
      abs(phase_horizons['clean_ii'] - phase_horizons['clean_i']) <= 1
  )

  regrets = []
  oracle_horizons = {}
  for step in REFERENCE_STEPS:
    means = {
        horizon: dict(scalars.get(
            f'reference_probe/dense_rhs/candidate_{horizon}_env_mean', []
        )).get(step)
        for horizon in range(2, 9)
    }
    if any(value is None for value in means.values()):
      continue
    oracle_horizon = max(means, key=means.get)
    oracle_horizons[str(step)] = oracle_horizon
    selected_horizon = int(next(
        float(row['selected_horizon']) for row in queries
        if int(row['step']) == step
    ))
    regrets.append(float(means[oracle_horizon] - means[selected_horizon]))

  shadow = {}
  for arm in ('s0', 's1', 's2', 's3'):
    tag = f'dense_rhs/shadow_{arm}_selected_horizon'
    shadow[arm] = {
        str(step): int(value) for step, value in scalars.get(tag, [])
        if step in FULL_QUERY_STEPS
    }
  phase_delays = [phase[3] for phase in PHASES]
  median_horizons = [phase_horizons[phase[0]] for phase in PHASES]
  return {
      'run_id': manifest['run_id'],
      'profile': manifest['profile'],
      'score_variant': manifest['score_variant'],
      'normalized_return_auc': auc,
      'final_clean_return': phase_returns['clean_ii'],
      'phase_mean_returns': phase_returns,
      'phase_median_horizons': phase_horizons,
      'switches_per_phase': switches_per_phase,
      'chatter': chatter,
      'stable': stable,
      'delay_horizon_rank_association': _rank_correlation(
          phase_delays, median_horizons
      ),
      'mean_oracle_regret': float(np.mean(regrets)) if regrets else math.nan,
      'oracle_regrets': regrets,
      'oracle_horizons': oracle_horizons,
      'selected_horizons': {
          row['step']: int(float(row['selected_horizon'])) for row in queries
      },
      'shadow_selected_horizons': shadow,
      'eval_curve': eval_curve,
  }


def reduce_runs(root: Path, output_dir: Path) -> dict:
  import matplotlib.pyplot as plt
  from PIL import Image, ImageDraw

  runs = {}
  for profile in PROFILE_VARIANTS:
    candidates = _full_run_candidates(root, profile)
    valid = [path for path in candidates if (path / 'RUN_VALID').is_file()]
    if not valid:
      raise ValueError(f'No valid full run found for {profile}')
    run_dir = valid[-1]
    validation = validate_run(run_dir)
    if validation['mode'] != 'full':
      raise ValueError(f'{run_dir} is not a full run')
    runs[profile] = {'run_dir': run_dir, 'metrics': _run_metrics(run_dir)}

  output_dir.mkdir(parents=True, exist_ok=True)
  colors = {'s0': '#4c78a8', 's1': '#f58518', 's2': '#54a24b', 's3': '#b279a2'}

  fig, ax = plt.subplots(figsize=(10, 5.5))
  for profile, record in runs.items():
    curve = record['metrics']['eval_curve']
    ax.plot(
        [step for step, _ in curve],
        [value for _, value in curve],
        label=profile.upper(),
        color=colors[profile],
        linewidth=2,
    )
  for name, start, end, delay in PHASES:
    ax.axvspan(start, end, alpha=0.07 if delay else 0.02, color='black')
    ax.text((start + end) / 2, 0.98, f'd={delay}', transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=9)
  ax.set(xlabel='Training transitions', ylabel='Evaluation return',
         title='Sequential-delay score screen: evaluation performance')
  ax.legend(ncol=4)
  ax.grid(alpha=0.2)
  fig.tight_layout()
  fig.savefig(output_dir / 'score_screen_learning_curves.png', dpi=200)
  plt.close(fig)

  fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                           gridspec_kw={'height_ratios': [2, 1]})
  for profile, record in runs.items():
    sequence = record['metrics']['selected_horizons']
    steps = [int(step) for step in sequence]
    axes[0].step(steps, [sequence[str(step)] for step in steps], where='post',
                 marker='o', label=profile.upper(), color=colors[profile])
  schedule_steps = [34_000, 38_000, 42_000, 46_000, 50_000, 54_000]
  schedule_values = [0, 2, 6, 4, 0, 0]
  axes[1].step(schedule_steps, schedule_values, where='post', color='black', linewidth=2)
  axes[0].set(ylabel='Selected horizon', ylim=(1.5, 8.5),
              title='Controller response to the hidden delay sequence')
  axes[1].set(xlabel='Training transitions', ylabel='Delay', ylim=(-0.5, 6.5))
  axes[0].legend(ncol=4)
  for axis in axes:
    axis.grid(alpha=0.2)
  fig.tight_layout()
  fig.savefig(output_dir / 'score_screen_horizon_adaptation.png', dpi=200)
  plt.close(fig)

  profiles = list(PROFILE_VARIANTS)
  aucs = [runs[p]['metrics']['normalized_return_auc'] for p in profiles]
  regrets = [runs[p]['metrics']['mean_oracle_regret'] for p in profiles]
  fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
  axes[0].bar(profiles, aucs, color=[colors[p] for p in profiles])
  axes[0].set(title='Mean evaluation-return AUC', ylabel='Return')
  axes[1].bar(profiles, regrets, color=[colors[p] for p in profiles])
  axes[1].set(title='Mean shadow-oracle regret', ylabel='Return units')
  for axis in axes:
    axis.grid(axis='y', alpha=0.2)
  fig.tight_layout()
  fig.savefig(output_dir / 'score_screen_performance_stability.png', dpi=200)
  plt.close(fig)

  best_auc = max(aucs)
  best_clean = max(runs[p]['metrics']['final_clean_return'] for p in profiles)
  eligible = []
  for profile in profiles:
    metrics = runs[profile]['metrics']
    metrics['promotion_eligible'] = (
        metrics['normalized_return_auc'] >= 0.95 * best_auc and
        metrics['final_clean_return'] >= 0.95 * best_clean and
        metrics['stable']
    )
    if metrics['promotion_eligible']:
      eligible.append(profile)
  winner = min(
      eligible,
      key=lambda profile: runs[profile]['metrics']['mean_oracle_regret'],
  ) if eligible else max(profiles, key=lambda profile: runs[profile]['metrics']['normalized_return_auc'])

  montage_images = []
  for step in FULL_ANCHORS:
    path = (
        runs[winner]['run_dir'] / 'artifacts' / 'rollouts' /
        runs[winner]['metrics']['run_id'] / f'step_{step:06d}' /
        'pendulum_delay0_vs_delay4_vs_delay6_frame.png'
    )
    image = Image.open(path).convert('RGB')
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 115, 22), fill='white')
    draw.text((5, 5), f'{step // 1000}k', fill='black')
    montage_images.append(image)
  width = max(image.width for image in montage_images)
  height = max(image.height for image in montage_images)
  montage = Image.new('RGB', (width * 3, height * 2), 'white')
  for index, image in enumerate(montage_images):
    montage.paste(image, ((index % 3) * width, (index // 3) * height))
  montage.save(output_dir / 'score_screen_environment_montage.png')

  summary = {
      'schema_version': 1,
      'winner': winner,
      'promotion_eligible': eligible,
      'runs': {profile: runs[profile]['metrics'] for profile in profiles},
      'confirmation_profiles': _confirmation_profiles(runs, eligible),
      'h8_boundary_hit': any(
          8 in record['metrics']['oracle_horizons'].values()
          for record in runs.values()
      ),
  }
  for record in summary['runs'].values():
    record.pop('eval_curve', None)
  _write_json(output_dir / 'summary.json', summary)
  (output_dir / 'AGGREGATE_VALID').touch()
  return summary


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  subparsers = parser.add_subparsers(dest='command', required=True)

  write_parser = subparsers.add_parser('write-manifest')
  write_parser.add_argument('--run-dir', type=Path, required=True)
  write_parser.add_argument('--run-id', required=True)
  write_parser.add_argument('--mode', choices=('source', 'smoke', 'full'), required=True)
  write_parser.add_argument('--profile', choices=tuple(PROFILE_VARIANTS), required=True)
  write_parser.add_argument('--commit', required=True)
  write_parser.add_argument('--config-hash', required=True)
  write_parser.add_argument('--parent-run-dir', type=Path, required=True)
  write_parser.add_argument('--parent-checkpoint-step', type=int, required=True)

  validate_parser = subparsers.add_parser('validate')
  validate_parser.add_argument('--run-dir', type=Path, required=True)

  reduce_parser = subparsers.add_parser('reduce')
  reduce_parser.add_argument('--root', type=Path, required=True)
  reduce_parser.add_argument('--output-dir', type=Path, required=True)

  args = parser.parse_args(argv)
  if args.command == 'write-manifest':
    return write_manifest(args)
  if args.command == 'validate':
    summary = validate_run(args.run_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
  summary = reduce_runs(args.root, args.output_dir)
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
