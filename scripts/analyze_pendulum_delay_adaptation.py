#!/usr/bin/env python3
"""Manifest, validation, and plots for the Pendulum delay-adaptation trio."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path


SOURCE_STEP = 30_000
FULL_FINAL_STEP = 46_000
FULL_DELAY_START = 34_000
FULL_DELAY_END = 42_000
FULL_QUERY_STEPS = (32_000, 36_000, 40_000, 44_000)
FULL_ANCHORS = (30_000, 34_000, 36_000, 40_000, 42_000, 44_000, 46_000)
SMOKE_FINAL_STEP = 32_000
SMOKE_DELAY_START = 30_800
SMOKE_DELAY_END = 31_600
SMOKE_QUERY_STEPS = (31_200,)
SMOKE_ANCHORS = (30_000, 30_800, 31_200, 31_600, 32_000)
READOUT_INTERVAL = 400
PROFILES = {
    'fixed_h3': {'controller': 'fixed', 'horizon': 3},
    'fixed_h7': {'controller': 'fixed', 'horizon': 7},
    'adaptive': {'controller': 'adaptive', 'horizon': None},
}


def _read_csv(path: Path) -> list[dict[str, str]]:
  if not path.is_file():
    raise FileNotFoundError(path)
  with path.open(newline='') as handle:
    return list(csv.DictReader(handle))


def _write_json(path: Path, payload) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def _manifest(run_dir: Path) -> dict:
  return json.loads((run_dir / 'run_manifest.json').read_text())


def _latest_scalars(run_dir: Path) -> dict[str, dict[int, float]]:
  values: dict[str, dict[int, float]] = {}
  for row in _read_csv(run_dir / 'metrics' / 'scalars.csv'):
    step = int(float(row['step']))
    value = float(row['value'])
    if not math.isfinite(value):
      raise ValueError(f'Non-finite scalar {row["tag"]} at step {step}')
    values.setdefault(row['tag'], {})[step] = value
  return values


def _curve(scalars: dict[str, dict[int, float]], tag: str) -> list[tuple[int, float]]:
  return sorted(scalars.get(tag, {}).items())


def _expected_steps(final_step: int) -> tuple[int, ...]:
  return tuple(range(SOURCE_STEP + READOUT_INTERVAL, final_step + 1, READOUT_INTERVAL))


def _profile_contract(mode: str, profile: str) -> dict:
  if mode == 'smoke':
    if profile != 'adaptive':
      raise ValueError('The gate smoke must exercise the adaptive controller.')
    return {
        'final_step': SMOKE_FINAL_STEP,
        'delay_start': SMOKE_DELAY_START,
        'delay_end': SMOKE_DELAY_END,
        'query_steps': SMOKE_QUERY_STEPS,
        'anchors': SMOKE_ANCHORS,
    }
  return {
      'final_step': FULL_FINAL_STEP,
      'delay_start': FULL_DELAY_START,
      'delay_end': FULL_DELAY_END,
      'query_steps': FULL_QUERY_STEPS if profile == 'adaptive' else (),
      'anchors': FULL_ANCHORS,
  }


def write_manifest(args: argparse.Namespace) -> None:
  contract = _profile_contract(args.mode, args.profile)
  profile = PROFILES[args.profile]
  parent_dir = Path(args.parent_run_dir).resolve()
  parent_manifest = json.loads((parent_dir / 'run_manifest.json').read_text())
  payload = {
      'schema_version': 1,
      'experiment': 'pendulum-delay-adaptation-v1',
      'run_id': args.run_id,
      'mode': args.mode,
      'profile': args.profile,
      'controller': profile['controller'],
      'fixed_horizon': profile['horizon'],
      'git_commit': args.commit,
      'config_hash': args.config_hash,
      'seed': 7,
      'environment': 'pendulum-swingup',
      'parent': {
          'run_dir': str(parent_dir),
          'run_id': parent_manifest['run_id'],
          'checkpoint_step': SOURCE_STEP,
          'config_hash': parent_manifest['config_hash'],
      },
      'final_step': contract['final_step'],
      'delay_schedule': {
          'base': 0,
          'active': 4,
          'start': contract['delay_start'],
          'end': contract['delay_end'],
          'observed': False,
      },
      'readout_interval_steps': READOUT_INTERVAL,
      'evaluation_episodes': 10,
      'query_steps': list(contract['query_steps']),
      'artifact_anchors': list(contract['anchors']),
  }
  _write_json(Path(args.run_dir) / 'run_manifest.json', payload)


def _validate_media(run_dir: Path, manifest: dict) -> list[dict]:
  task_prefix = manifest['environment'].split('-', maxsplit=1)[0]
  rollout_root = run_dir / 'artifacts' / 'rollouts' / manifest['run_id']
  records = []
  for step in manifest['artifact_anchors']:
    anchor_checkpoint = run_dir / 'artifacts' / 'anchor_checkpoints' / str(step)
    rollout_dir = rollout_root / f'step_{int(step):06d}'
    if not anchor_checkpoint.is_dir():
      raise ValueError(f'Missing anchor checkpoint {anchor_checkpoint}')
    metadata_path = rollout_dir / 'metadata.json'
    if not metadata_path.is_file():
      raise ValueError(f'Missing rollout metadata {metadata_path}')
    metadata = json.loads(metadata_path.read_text())
    if int(metadata['global_step']) != int(step):
      raise ValueError(f'Rollout step mismatch at {rollout_dir}')
    if metadata['environment']['task'] != manifest['environment']:
      raise ValueError(f'Rollout task mismatch at {rollout_dir}')
    for condition in ('delay0', 'delay4'):
      trajectory = rollout_dir / f'trajectory_{condition}.npz'
      if not trajectory.is_file() or trajectory.stat().st_size <= 0:
        raise ValueError(f'Missing trajectory {trajectory}')
    gif = rollout_dir / f'{task_prefix}_delay0_vs_delay4.gif'
    png = rollout_dir / f'{task_prefix}_delay0_vs_delay4_frame.png'
    for media_path in (gif, png):
      if not media_path.is_file() or media_path.stat().st_size <= 0:
        raise ValueError(f'Missing rendered media {media_path}')
    records.append({
        'step': int(step),
        'selected_horizon': int(metadata['selected_horizon']),
        'gif': str(gif),
        'png': str(png),
    })
  return records


def validate(run_dir: Path, *, require_valid_marker: bool = False) -> dict:
  run_dir = Path(run_dir)
  manifest = _manifest(run_dir)
  if manifest.get('experiment') != 'pendulum-delay-adaptation-v1':
    raise ValueError('Experiment identity mismatch.')
  profile_name = manifest['profile']
  if profile_name not in PROFILES:
    raise ValueError(f'Unknown profile {profile_name!r}.')
  profile = PROFILES[profile_name]
  contract = _profile_contract(manifest['mode'], profile_name)
  if int(manifest['parent']['checkpoint_step']) != SOURCE_STEP:
    raise ValueError('Parent checkpoint provenance mismatch.')
  if int(manifest['final_step']) != contract['final_step']:
    raise ValueError('Final-step contract mismatch.')
  schedule = manifest['delay_schedule']
  if schedule != {
      'base': 0,
      'active': 4,
      'start': contract['delay_start'],
      'end': contract['delay_end'],
      'observed': False,
  }:
    raise ValueError('Delay schedule drifted from the frozen contract.')
  if manifest['artifact_anchors'] != list(contract['anchors']):
    raise ValueError('Artifact-anchor contract mismatch.')
  if not (run_dir / 'TRAINING_COMPLETE').is_file():
    raise ValueError('TRAINING_COMPLETE marker missing.')
  if not (run_dir / 'MEDIA_COMPLETE').is_file():
    raise ValueError('MEDIA_COMPLETE marker missing.')
  if not (run_dir / 'checkpoint' / str(contract['final_step'])).is_dir():
    raise ValueError('Terminal composite checkpoint missing.')
  scalars = _latest_scalars(run_dir)
  expected_steps = _expected_steps(contract['final_step'])
  eval_curve = _curve(scalars, 'eval/return_mean')
  train_reward_curve = _curve(scalars, 'train/online_reward_mean')
  if tuple(step for step, _ in eval_curve) != expected_steps:
    raise ValueError(
        f'Evaluation cadence mismatch: expected {expected_steps}, got '
        f'{tuple(step for step, _ in eval_curve)}'
    )
  if tuple(step for step, _ in train_reward_curve) != expected_steps:
    raise ValueError('Online training-reward cadence mismatch.')
  eval_horizons = _curve(scalars, 'eval/selected_horizon')
  if tuple(step for step, _ in eval_horizons) != expected_steps:
    raise ValueError('Evaluation horizon audit cadence mismatch.')
  query_rows = _read_csv(run_dir / 'metrics' / 'horizon_queries.csv')
  queries = []
  if profile['controller'] == 'fixed':
    if query_rows:
      raise ValueError('Fixed profile unexpectedly wrote adaptive queries.')
    expected_horizon = int(profile['horizon'])
    if any(int(round(value)) != expected_horizon for _, value in eval_horizons):
      raise ValueError('Fixed-horizon audit drifted during the run.')
  else:
    for row in query_rows:
      step = int(float(row['step']))
      if step not in contract['query_steps']:
        continue
      query = {
          'step': step,
          'previous_horizon': int(float(row['previous_horizon'])),
          'proposed_horizon': int(float(row['proposed_horizon'])),
          'selected_horizon': int(float(row['selected_horizon'])),
          'prob_best_h': float(row['prob_best_h']),
          'return_term_best': float(row['return_term_best']),
          'roughness_term_best': float(row['roughness_term_best']),
          'return_std_term_best': float(row['return_std_term_best']),
          'robust_return_best': float(row['robust_return_best']),
      }
      if not all(math.isfinite(float(value)) for key, value in query.items() if key != 'step'):
        raise ValueError(f'Non-finite adaptive query at {step}.')
      queries.append(query)
    if tuple(query['step'] for query in queries) != tuple(contract['query_steps']):
      raise ValueError('Adaptive query cadence mismatch.')
    for step in contract['query_steps']:
      completed = scalars.get('reference_probe/completed', {}).get(int(step))
      if completed != 1.0:
        raise ValueError(f'Reference probe incomplete at step {step}.')
  media = _validate_media(run_dir, manifest)
  if require_valid_marker and not (run_dir / 'RUN_VALID').is_file():
    raise ValueError('RUN_VALID marker missing.')
  summary = {
      'valid': True,
      'run_id': manifest['run_id'],
      'mode': manifest['mode'],
      'profile': profile_name,
      'config_hash': manifest['config_hash'],
      'git_commit': manifest['git_commit'],
      'evaluation_points': len(eval_curve),
      'train_reward_points': len(train_reward_curve),
      'final_eval_return': eval_curve[-1][1],
      'queries': queries,
      'media': media,
  }
  _write_json(run_dir / 'validation_summary.json', summary)
  return summary


def _phase_stats(curve: list[tuple[int, float]]) -> dict[str, float]:
  phases = {
      'nominal_pre': [value for step, value in curve if step < FULL_DELAY_START],
      'delay4': [value for step, value in curve if FULL_DELAY_START <= step < FULL_DELAY_END],
      'nominal_recovery': [value for step, value in curve if step >= FULL_DELAY_END],
  }
  return {
      name: float(sum(values) / len(values))
      for name, values in phases.items()
  }


def aggregate(run_dirs: list[Path], output_dir: Path) -> dict:
  import matplotlib.pyplot as plt
  import numpy as np
  from PIL import Image, ImageDraw

  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=False)
  runs = {}
  for run_dir in run_dirs:
    summary = validate(run_dir, require_valid_marker=True)
    if summary['mode'] != 'full':
      raise ValueError('Aggregate accepts only full scientific profiles.')
    profile = summary['profile']
    if profile in runs:
      raise ValueError(f'Duplicate profile {profile}.')
    runs[profile] = {'dir': Path(run_dir), 'summary': summary}
  if set(runs) != set(PROFILES):
    raise ValueError(f'Expected profiles {sorted(PROFILES)}, got {sorted(runs)}.')
  commits = {record['summary']['git_commit'] for record in runs.values()}
  if len(commits) != 1:
    raise ValueError('Profiles did not run at one exact commit.')

  labels = {
      'fixed_h3': 'fixed h=3',
      'fixed_h7': 'fixed h=7',
      'adaptive': 'adaptive h=2..8',
  }
  colors = {'fixed_h3': '#4c78a8', 'fixed_h7': '#f58518', 'adaptive': '#54a24b'}
  curves = {}
  for profile, record in runs.items():
    scalars = _latest_scalars(record['dir'])
    curves[profile] = {
        'eval': _curve(scalars, 'eval/return_mean'),
        'train_reward': _curve(scalars, 'train/online_reward_mean'),
        'scalars': scalars,
    }

  def shade_schedule(ax):
    ax.axvspan(FULL_DELAY_START, FULL_DELAY_END, color='#f4a261', alpha=0.17)
    ax.axvline(FULL_DELAY_START, color='0.35', linestyle='--', linewidth=1)
    ax.axvline(FULL_DELAY_END, color='0.35', linestyle='--', linewidth=1)

  fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.6), sharex=True)
  for profile in ('fixed_h3', 'fixed_h7', 'adaptive'):
    eval_curve = curves[profile]['eval']
    reward_curve = curves[profile]['train_reward']
    axes[0].plot(*zip(*eval_curve), label=labels[profile], color=colors[profile], linewidth=1.8)
    axes[1].plot(*zip(*reward_curve), label=labels[profile], color=colors[profile], linewidth=1.5)
  for ax in axes:
    shade_schedule(ax)
    ax.grid(alpha=0.25)
  axes[0].set_ylabel('Evaluation return\n(10 episodes)')
  axes[0].legend(frameon=False, ncol=3)
  axes[1].set_ylabel('Online reward / step')
  axes[1].set_xlabel('Training transitions')
  fig.suptitle('Pendulum hidden-delay traversal: dense measured readouts')
  fig.tight_layout()
  fig.savefig(output_dir / 'learning_curves.png', dpi=200, bbox_inches='tight')
  plt.close(fig)

  adaptive_queries = runs['adaptive']['summary']['queries']
  adaptive_scalars = curves['adaptive']['scalars']
  query_steps = [row['step'] for row in adaptive_queries]
  selected = [row['selected_horizon'] for row in adaptive_queries]
  proposed = [row['proposed_horizon'] for row in adaptive_queries]
  reference = [
      adaptive_scalars.get('reference_probe/dense_rhs/proposed_horizon', {}).get(step, np.nan)
      for step in query_steps
  ]
  fig, ax_h = plt.subplots(figsize=(9.0, 4.6))
  schedule_steps = [SOURCE_STEP, FULL_DELAY_START, FULL_DELAY_END, FULL_FINAL_STEP]
  schedule_values = [0, 4, 0, 0]
  ax_d = ax_h.twinx()
  ax_d.step(schedule_steps, schedule_values, where='post', color='#e45756', linewidth=2.4, alpha=0.75, label='hidden action delay')
  ax_h.step([SOURCE_STEP] + query_steps, [3] + selected, where='post', marker='o', color=colors['adaptive'], linewidth=2.2, label='selected horizon')
  ax_h.scatter(query_steps, proposed, marker='x', s=70, color='#7a5195', label='controller proposal')
  ax_h.scatter(query_steps, reference, marker='D', facecolors='none', edgecolors='black', s=65, label='128-replica shadow proposal')
  ax_h.set_ylim(1.5, 8.5)
  ax_h.set_yticks(range(2, 9))
  ax_d.set_ylim(-0.25, 4.75)
  ax_d.set_yticks(range(0, 5))
  ax_h.set_xlabel('Training transitions')
  ax_h.set_ylabel('Planning horizon')
  ax_d.set_ylabel('Action delay')
  ax_h.grid(alpha=0.25)
  handles, labels_scatter = ax_h.get_legend_handles_labels()
  ax_h.legend(handles + ax_d.get_lines(), labels_scatter + [ax_d.get_lines()[0].get_label()], frameon=False, loc='upper left')
  fig.suptitle('Adaptive control state versus the latent delay intervention')
  fig.tight_layout()
  fig.savefig(output_dir / 'control_delay_trace.png', dpi=200, bbox_inches='tight')
  plt.close(fig)

  fig, ax = plt.subplots(figsize=(8.4, 4.5))
  x = np.arange(len(query_steps))
  width = 0.24
  ax.bar(x - width, [row['return_term_best'] for row in adaptive_queries], width, label='return term')
  ax.bar(x, [row['roughness_term_best'] for row in adaptive_queries], width, label='roughness term')
  ax.bar(x + width, [row['return_std_term_best'] for row in adaptive_queries], width, label='return-spread term')
  ax.axhline(0, color='0.25', linewidth=0.8)
  ax.set_xticks(x, [f'{step // 1000}k' for step in query_steps])
  ax.set_xlabel('Adaptive query')
  ax.set_ylabel('Recorded score component')
  ax.grid(axis='y', alpha=0.25)
  ax.legend(frameon=False, ncol=3)
  fig.suptitle('Controller evidence at each delay phase')
  fig.tight_layout()
  fig.savefig(output_dir / 'query_component_terms.png', dpi=200, bbox_inches='tight')
  plt.close(fig)

  phase_stats = {profile: _phase_stats(curves[profile]['eval']) for profile in runs}
  final_scores = {profile: curves[profile]['eval'][-1][1] for profile in runs}
  fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
  phase_names = ('nominal_pre', 'delay4', 'nominal_recovery')
  x = np.arange(len(phase_names))
  width = 0.24
  for offset, profile in enumerate(('fixed_h3', 'fixed_h7', 'adaptive')):
    axes[0].bar(x + (offset - 1) * width, [phase_stats[profile][phase] for phase in phase_names], width, label=labels[profile], color=colors[profile])
  axes[0].set_xticks(x, ['pre-delay', 'delay 4', 'recovery'])
  axes[0].set_ylabel('Mean evaluation return')
  axes[0].grid(axis='y', alpha=0.25)
  axes[0].legend(frameon=False, fontsize=8)
  axes[1].bar([labels[p] for p in ('fixed_h3', 'fixed_h7', 'adaptive')], [final_scores[p] for p in ('fixed_h3', 'fixed_h7', 'adaptive')], color=[colors[p] for p in ('fixed_h3', 'fixed_h7', 'adaptive')])
  axes[1].set_ylabel('Final evaluation return at 46k')
  axes[1].tick_params(axis='x', rotation=18)
  axes[1].grid(axis='y', alpha=0.25)
  fig.suptitle('Phase performance and terminal scores')
  fig.tight_layout()
  fig.savefig(output_dir / 'phase_and_final_scores.png', dpi=200, bbox_inches='tight')
  plt.close(fig)

  media_dir = output_dir / 'media'
  media_dir.mkdir()
  for profile, record in runs.items():
    for media in record['summary']['media']:
      for kind in ('gif', 'png'):
        source = Path(media[kind])
        destination = media_dir / f'{profile}_step{media["step"]}_{source.name}'
        shutil.copy2(source, destination)

  montage_steps = (30_000, 36_000, 40_000, 44_000, 46_000)
  montage_sources = {
      item['step']: Path(item['png'])
      for item in runs['adaptive']['summary']['media']
      if item['step'] in montage_steps
  }
  images = [Image.open(montage_sources[step]).convert('RGB') for step in montage_steps]
  thumb_width = 360
  thumbs = []
  for step, source in zip(montage_steps, images):
    ratio = thumb_width / source.width
    thumb = source.resize((thumb_width, int(source.height * ratio)))
    canvas = Image.new('RGB', (thumb.width, thumb.height + 30), 'white')
    canvas.paste(thumb, (0, 30))
    ImageDraw.Draw(canvas).text((8, 8), f'adaptive checkpoint {step // 1000}k', fill='black')
    thumbs.append(canvas)
  montage = Image.new('RGB', (thumb_width * len(thumbs), max(image.height for image in thumbs)), 'white')
  for index, image in enumerate(thumbs):
    montage.paste(image, (index * thumb_width, 0))
  montage.save(output_dir / 'environment_montage.png')

  delayed_selected = [row['selected_horizon'] for row in adaptive_queries if FULL_DELAY_START <= row['step'] < FULL_DELAY_END]
  recovered_selected = [row['selected_horizon'] for row in adaptive_queries if row['step'] >= FULL_DELAY_END]
  payload = {
      'valid': True,
      'git_commit': next(iter(commits)),
      'profiles': {profile: record['summary'] for profile, record in runs.items()},
      'phase_mean_eval_returns': phase_stats,
      'final_eval_returns': final_scores,
      'adaptive_selected_horizons': {str(row['step']): row['selected_horizon'] for row in adaptive_queries},
      'adaptive_proposed_horizons': {str(row['step']): row['proposed_horizon'] for row in adaptive_queries},
      'adaptive_moved_long_during_delay': any(value >= 6 for value in delayed_selected),
      'adaptive_recovered_short': bool(recovered_selected and recovered_selected[-1] <= 3),
  }
  _write_json(output_dir / 'summary.json', payload)
  (output_dir / 'AGGREGATE_VALID').touch()
  return payload


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser()
  sub = parser.add_subparsers(dest='command', required=True)
  manifest = sub.add_parser('write-manifest')
  manifest.add_argument('--run-dir', required=True)
  manifest.add_argument('--run-id', required=True)
  manifest.add_argument('--mode', choices=('smoke', 'full'), required=True)
  manifest.add_argument('--profile', choices=tuple(PROFILES), required=True)
  manifest.add_argument('--commit', required=True)
  manifest.add_argument('--config-hash', required=True)
  manifest.add_argument('--parent-run-dir', required=True)
  validation = sub.add_parser('validate')
  validation.add_argument('--run-dir', required=True)
  validation.add_argument('--require-valid-marker', action='store_true')
  aggregation = sub.add_parser('aggregate')
  aggregation.add_argument('--output-dir', required=True)
  aggregation.add_argument('run_dirs', nargs='+')
  return parser


def main() -> None:
  args = _parser().parse_args()
  if args.command == 'write-manifest':
    write_manifest(args)
  elif args.command == 'validate':
    print(json.dumps(validate(
        Path(args.run_dir),
        require_valid_marker=args.require_valid_marker,
    ), indent=2, sort_keys=True))
  else:
    print(json.dumps(aggregate(
        [Path(path) for path in args.run_dirs],
        Path(args.output_dir),
    ), indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
