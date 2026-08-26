#!/usr/bin/env python3
"""Manifest, validation, and summary tools for the fixed-horizon torque screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


PARENT_STEP = 30_000
SMOKE_FINAL_STEP = 30_800
FULL_FINAL_STEP = 46_000
VALID_TORQUE_SCALES = (0.4, 0.6, 0.8)
VALID_HORIZONS = (3, 8)


def _read_csv(path: Path) -> list[dict[str, str]]:
  if not path.exists():
    raise FileNotFoundError(path)
  with path.open(newline='') as handle:
    return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def _manifest(run_dir: Path) -> dict[str, Any]:
  return json.loads((run_dir / 'run_manifest.json').read_text())


def _scalars(run_dir: Path) -> list[dict[str, str]]:
  return _read_csv(run_dir / 'metrics' / 'scalars.csv')


def _finite_series(run_dir: Path, tag: str) -> list[tuple[int, float]]:
  latest: dict[int, float] = {}
  for row in _scalars(run_dir):
    if row.get('tag') != tag:
      continue
    step = int(float(row['step']))
    value = float(row['value'])
    if not math.isfinite(value):
      raise ValueError(f'Non-finite {tag} at step {step}')
    latest[step] = value
  return sorted(latest.items())


def write_manifest(args: argparse.Namespace) -> None:
  parent_dir = Path(args.parent_run_dir).resolve()
  parent_manifest = _manifest(parent_dir)
  payload = {
      'schema_version': 1,
      'run_id': args.run_id,
      'phase': 'fixed_horizon_screen',
      'mode': args.mode,
      'git_commit': args.commit,
      'config_hash': args.config_hash,
      'seed': int(args.seed),
      'environment': 'pendulum-swingup',
      'actuator_strength_scale': float(args.torque_scale),
      'fixed_horizon': int(args.horizon),
      'dense_rhs_enabled': False,
      'parent': {
          'run_dir': str(parent_dir),
          'run_id': parent_manifest['run_id'],
          'config_hash': parent_manifest['config_hash'],
          'checkpoint_step': PARENT_STEP,
      },
  }
  _write_json(Path(args.run_dir) / 'run_manifest.json', payload)


def validate(run_dir: Path) -> dict[str, Any]:
  manifest = _manifest(run_dir)
  if manifest.get('phase') != 'fixed_horizon_screen':
    raise ValueError('Screen manifest phase mismatch')
  mode = str(manifest.get('mode'))
  if mode not in ('smoke', 'full'):
    raise ValueError(f'Unsupported screen mode {mode!r}')
  torque = float(manifest['actuator_strength_scale'])
  horizon = int(manifest['fixed_horizon'])
  if mode == 'full' and torque not in VALID_TORQUE_SCALES:
    raise ValueError(f'Unexpected full-screen torque scale {torque}')
  if horizon not in VALID_HORIZONS:
    raise ValueError(f'Unexpected fixed horizon {horizon}')
  if bool(manifest.get('dense_rhs_enabled')):
    raise ValueError('Dense-RHS must be disabled in a fixed-horizon screen')
  parent = manifest.get('parent') or {}
  if int(parent.get('checkpoint_step', -1)) != PARENT_STEP:
    raise ValueError('30k parent checkpoint provenance missing')
  if not (run_dir / 'TRAINING_COMPLETE').exists():
    raise ValueError('TRAINING_COMPLETE marker missing')
  expected_step = SMOKE_FINAL_STEP if mode == 'smoke' else FULL_FINAL_STEP
  if not (run_dir / 'checkpoint' / str(expected_step)).is_dir():
    raise ValueError(f'Expected terminal checkpoint {expected_step} missing')

  curve = _finite_series(run_dir, 'eval/return_mean')
  if not curve or curve[-1][0] != expected_step:
    raise ValueError(f'Expected terminal evaluation at {expected_step}, got {curve[-3:]}')
  selected = _finite_series(run_dir, 'eval/selected_horizon')
  training = _finite_series(run_dir, 'eval/training_bucket_horizon')
  if not selected or not training:
    raise ValueError('Fixed-horizon audit scalars missing')
  if any(abs(value - horizon) > 1e-6 for _, value in selected + training):
    raise ValueError(
        f'Fixed horizon {horizon} was not preserved: selected={selected}, training={training}'
    )

  summary = {
      'valid': True,
      'mode': mode,
      'run_id': manifest['run_id'],
      'torque_scale': torque,
      'fixed_horizon': horizon,
      'parent_checkpoint_step': PARENT_STEP,
      'terminal_step': expected_step,
      'evaluation_curve': curve,
      'best_eval_return': max(value for _, value in curve),
      'final_eval_return': curve[-1][1],
      'final_upright_fraction': curve[-1][1] / 1000.0,
      'config_hash': manifest['config_hash'],
  }
  _write_json(run_dir / 'screen_validation.json', summary)
  return summary


def combine(run_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
  summaries = [validate(run_dir) for run_dir in run_dirs]
  full = [summary for summary in summaries if summary['mode'] == 'full']
  observed = {(row['torque_scale'], row['fixed_horizon']) for row in full}
  expected = {(torque, horizon) for torque in VALID_TORQUE_SCALES for horizon in VALID_HORIZONS}
  if observed != expected:
    raise ValueError(f'Expected full matrix {sorted(expected)}, got {sorted(observed)}')
  comparisons = []
  for torque in VALID_TORQUE_SCALES:
    by_horizon = {row['fixed_horizon']: row for row in full if row['torque_scale'] == torque}
    comparisons.append({
        'torque_scale': torque,
        'h3_final_return': by_horizon[3]['final_eval_return'],
        'h8_final_return': by_horizon[8]['final_eval_return'],
        'h8_minus_h3_final': (
            by_horizon[8]['final_eval_return'] - by_horizon[3]['final_eval_return']
        ),
        'h3_best_return': by_horizon[3]['best_eval_return'],
        'h8_best_return': by_horizon[8]['best_eval_return'],
    })
  payload = {'runs': full, 'comparisons': comparisons}
  _write_json(output_dir / 'pendulum_torque_horizon_screen_summary.json', payload)

  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), sharey=True)
  for axis, torque in zip(axes, VALID_TORQUE_SCALES):
    for horizon in VALID_HORIZONS:
      row = next(
          item for item in full
          if item['torque_scale'] == torque and item['fixed_horizon'] == horizon
      )
      axis.plot(
          *zip(*row['evaluation_curve']), marker='o', label=f'h={horizon}'
      )
    axis.set_title(f'torque scale {torque:.1f}')
    axis.set_xlabel('Transitions')
    axis.grid(alpha=0.25)
  axes[0].set_ylabel('Evaluation return')
  axes[-1].legend(frameon=False)
  fig.suptitle('Pendulum checkpoint fork: fixed-horizon torque screen')
  fig.tight_layout()
  output_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(
      output_dir / 'pendulum_torque_horizon_screen.png', dpi=180,
      bbox_inches='tight',
  )
  plt.close(fig)
  return payload


def parser() -> argparse.ArgumentParser:
  result = argparse.ArgumentParser()
  sub = result.add_subparsers(dest='command', required=True)
  manifest = sub.add_parser('write-manifest')
  manifest.add_argument('--run-dir', required=True)
  manifest.add_argument('--run-id', required=True)
  manifest.add_argument('--mode', choices=('smoke', 'full'), required=True)
  manifest.add_argument('--commit', required=True)
  manifest.add_argument('--config-hash', required=True)
  manifest.add_argument('--seed', type=int, required=True)
  manifest.add_argument('--torque-scale', type=float, required=True)
  manifest.add_argument('--horizon', type=int, choices=VALID_HORIZONS, required=True)
  manifest.add_argument('--parent-run-dir', required=True)
  validate_parser = sub.add_parser('validate')
  validate_parser.add_argument('--run-dir', required=True)
  combine_parser = sub.add_parser('combine')
  combine_parser.add_argument('--run-dir', action='append', required=True)
  combine_parser.add_argument('--output-dir', required=True)
  return result


def main() -> None:
  args = parser().parse_args()
  if args.command == 'write-manifest':
    write_manifest(args)
  elif args.command == 'validate':
    print(json.dumps(validate(Path(args.run_dir)), indent=2, sort_keys=True))
  else:
    payload = combine(
        [Path(path) for path in args.run_dir], Path(args.output_dir)
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
