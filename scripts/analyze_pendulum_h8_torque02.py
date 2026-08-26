#!/usr/bin/env python3
"""Validate and summarize the checkpoint-forked Pendulum torque-0.2 test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


BASE_FINAL_STEP = 30_000
BRANCH_FINAL_STEP = 46_000
QUERY_STEPS = (34_000, 38_000, 42_000)


def _read_csv(path: Path) -> list[dict[str, str]]:
  if not path.exists():
    raise FileNotFoundError(path)
  with path.open(newline='') as handle:
    return list(csv.DictReader(handle))


def _scalars(run_dir: Path) -> list[dict[str, str]]:
  return _read_csv(run_dir / 'metrics' / 'scalars.csv')


def _eval_curve(run_dir: Path) -> list[tuple[int, float]]:
  latest: dict[int, float] = {}
  for row in _scalars(run_dir):
    if row.get('tag') != 'eval/return_mean':
      continue
    step = int(float(row['step']))
    value = float(row['value'])
    if not math.isfinite(value):
      raise ValueError(f'Non-finite evaluation return at step {step}')
    latest[step] = value
  return sorted(latest.items())


def _scalar_at(run_dir: Path, tag: str, step: int) -> float | None:
  value = None
  for row in _scalars(run_dir):
    if row.get('tag') == tag and int(float(row['step'])) == int(step):
      candidate = float(row['value'])
      if not math.isfinite(candidate):
        raise ValueError(f'Non-finite scalar {tag} at step {step}')
      value = candidate
  return value


def _checkpoint_exists(run_dir: Path, step: int) -> bool:
  return (run_dir / 'checkpoint' / str(step)).is_dir()


def _manifest(run_dir: Path) -> dict[str, Any]:
  return json.loads((run_dir / 'run_manifest.json').read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def write_manifest(args: argparse.Namespace) -> None:
  parent = None
  if args.parent_run_dir:
    parent_dir = Path(args.parent_run_dir).resolve()
    parent_manifest = _manifest(parent_dir)
    parent = {
        'run_dir': str(parent_dir),
        'run_id': parent_manifest['run_id'],
        'config_hash': parent_manifest['config_hash'],
        'checkpoint_step': BASE_FINAL_STEP,
    }
  payload = {
      'schema_version': 1,
      'run_id': args.run_id,
      'phase': args.phase,
      'git_commit': args.commit,
      'config_hash': args.config_hash,
      'seed': int(args.seed),
      'environment': 'pendulum-swingup',
      'actuator_strength_scale': float(args.torque_scale),
      'parent': parent,
  }
  _write_json(Path(args.run_dir) / 'run_manifest.json', payload)


def validate_base(run_dir: Path) -> dict[str, Any]:
  manifest = _manifest(run_dir)
  if manifest['phase'] != 'base' or manifest['actuator_strength_scale'] != 1.0:
    raise ValueError('Base manifest identity mismatch')
  if not (run_dir / 'TRAINING_COMPLETE').exists():
    raise ValueError('Base TRAINING_COMPLETE marker missing')
  if not _checkpoint_exists(run_dir, BASE_FINAL_STEP):
    raise ValueError('Base 30k checkpoint missing')
  curve = _eval_curve(run_dir)
  final = [(step, value) for step, value in curve if step <= BASE_FINAL_STEP][-3:]
  if len(final) != 3 or final[-1][0] != BASE_FINAL_STEP:
    raise ValueError(f'Expected three terminal base evaluations ending at 30k, got {final}')
  values = [value for _, value in final]
  mean = sum(values) / len(values)
  variance = sum((value - mean) ** 2 for value in values) / len(values)
  cv = math.sqrt(variance) / max(abs(mean), 1e-12)
  stable = mean >= 700.0 and cv <= 0.15
  summary = {
      'valid': True,
      'stable': stable,
      'stability_rule': 'last-three eval mean >= 700 and CV <= 0.15',
      'terminal_eval_points': final,
      'terminal_eval_mean': mean,
      'terminal_eval_cv': cv,
      'checkpoint_step': BASE_FINAL_STEP,
      'config_hash': manifest['config_hash'],
  }
  _write_json(run_dir / 'base_validation.json', summary)
  if not stable:
    raise ValueError(f'Base did not satisfy the frozen stability rule: {summary}')
  return summary


def validate_branch(run_dir: Path) -> dict[str, Any]:
  manifest = _manifest(run_dir)
  if manifest['phase'] != 'branch' or manifest['actuator_strength_scale'] != 0.2:
    raise ValueError('Branch manifest identity mismatch')
  if not manifest.get('parent') or manifest['parent']['checkpoint_step'] != BASE_FINAL_STEP:
    raise ValueError('Branch parent checkpoint provenance missing')
  if not (run_dir / 'TRAINING_COMPLETE').exists():
    raise ValueError('Branch TRAINING_COMPLETE marker missing')
  if not _checkpoint_exists(run_dir, BRANCH_FINAL_STEP):
    raise ValueError('Branch 46k checkpoint missing')
  query_rows = _read_csv(run_dir / 'metrics' / 'horizon_queries.csv')
  queries = []
  for row in query_rows:
    step = int(float(row['step']))
    if step not in QUERY_STEPS:
      continue
    queries.append({
        'step': step,
        'previous_horizon': int(float(row['previous_horizon'])),
        'proposed_horizon': int(float(row['proposed_horizon'])),
        'selected_horizon': int(float(row['selected_horizon'])),
        'return_term_best': float(row['return_term_best']),
        'roughness_term_best': float(row['roughness_term_best']),
        'return_std_term_best': float(row['return_std_term_best']),
        'prob_best_h': float(row['prob_best_h']),
    })
  if tuple(row['step'] for row in queries) != QUERY_STEPS:
    raise ValueError(f'Expected query steps {QUERY_STEPS}, got {queries}')
  for query in queries:
    if not all(
        math.isfinite(float(query[key]))
        for key in ('return_term_best', 'roughness_term_best', 'return_std_term_best', 'prob_best_h')
    ):
      raise ValueError(f'Non-finite query metric: {query}')
  curve = _eval_curve(run_dir)
  if not curve or curve[-1][0] != BRANCH_FINAL_STEP:
    raise ValueError('Final 46k evaluation missing')
  reference_proposed = _scalar_at(
      run_dir,
      'reference_probe/dense_rhs/proposed_horizon',
      42_000,
  )
  reference_selected = _scalar_at(
      run_dir,
      'reference_probe/dense_rhs/selected_horizon',
      42_000,
  )
  if _scalar_at(run_dir, 'reference_probe/completed', 42_000) != 1.0:
    raise ValueError('High-precision reference probe at 42k is incomplete')
  selected = [row['selected_horizon'] for row in queries]
  proposed = [row['proposed_horizon'] for row in queries]
  reached_h8 = 8 in selected
  proposed_h8 = 8 in proposed
  retained_h8 = len(selected) >= 2 and selected[-2:] == [8, 8]
  if reference_proposed == 8.0 and reached_h8:
    classification = 'oracle_and_controller_h8'
  elif reference_proposed == 8.0:
    classification = 'oracle_h8_controller_blocked'
  elif reached_h8:
    classification = 'controller_h8_reference_disagrees'
  else:
    classification = 'torque02_did_not_prefer_h8'
  summary = {
      'valid': True,
      'queries': queries,
      'evaluation_curve': curve,
      'reference_proposed_horizon_at_42k': reference_proposed,
      'reference_selected_horizon_at_42k': reference_selected,
      'proposed_h8': proposed_h8,
      'reached_h8': reached_h8,
      'retained_h8_for_last_two_queries': retained_h8,
      'classification': classification,
      'final_eval_return': curve[-1][1],
      'config_hash': manifest['config_hash'],
  }
  _write_json(run_dir / 'branch_validation.json', summary)
  return summary


def combine(base_dir: Path, branch_dir: Path, output_dir: Path) -> dict[str, Any]:
  base = validate_base(base_dir)
  branch = validate_branch(branch_dir)
  payload = {'base': base, 'branch': branch}
  _write_json(output_dir / 'pendulum_h8_torque02_summary.json', payload)

  import matplotlib.pyplot as plt

  base_curve = _eval_curve(base_dir)
  branch_curve = branch['evaluation_curve']
  query_steps = [row['step'] for row in branch['queries']]
  selected = [row['selected_horizon'] for row in branch['queries']]
  proposed = [row['proposed_horizon'] for row in branch['queries']]
  fig, (ax_return, ax_horizon) = plt.subplots(
      2, 1, figsize=(8.2, 5.6), sharex=True,
      gridspec_kw={'height_ratios': [1.6, 1.0]},
  )
  ax_return.plot(*zip(*base_curve), marker='o', label='nominal torque (base)')
  ax_return.plot(*zip(*branch_curve), marker='o', label='torque scale 0.2')
  ax_return.axvline(BASE_FINAL_STEP, color='0.35', linestyle='--', linewidth=1)
  ax_return.set_ylabel('Evaluation return')
  ax_return.legend(frameon=False, loc='best')
  ax_return.grid(alpha=0.25)
  ax_horizon.step(query_steps, selected, where='post', marker='o', label='selected')
  ax_horizon.plot(query_steps, proposed, linestyle='none', marker='x', markersize=8, label='proposed')
  if branch['reference_proposed_horizon_at_42k'] is not None:
    ax_horizon.scatter(
        [42_000], [branch['reference_proposed_horizon_at_42k']],
        facecolors='none', edgecolors='black', s=70, label='reference proposal',
    )
  ax_horizon.set_ylim(1.5, 8.5)
  ax_horizon.set_yticks(range(2, 9))
  ax_horizon.set_xlabel('Training transitions')
  ax_horizon.set_ylabel('Horizon')
  ax_horizon.grid(alpha=0.25)
  ax_horizon.legend(frameon=False, ncol=3, loc='best')
  fig.suptitle('Pendulum checkpoint fork: actuator-strength scale 1.0 to 0.2')
  fig.tight_layout()
  output_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_dir / 'pendulum_h8_torque02.png', dpi=180, bbox_inches='tight')
  plt.close(fig)
  return payload


def parser() -> argparse.ArgumentParser:
  result = argparse.ArgumentParser()
  sub = result.add_subparsers(dest='command', required=True)
  manifest = sub.add_parser('write-manifest')
  manifest.add_argument('--run-dir', required=True)
  manifest.add_argument('--run-id', required=True)
  manifest.add_argument('--phase', choices=('base', 'branch'), required=True)
  manifest.add_argument('--commit', required=True)
  manifest.add_argument('--config-hash', required=True)
  manifest.add_argument('--seed', type=int, required=True)
  manifest.add_argument('--torque-scale', type=float, required=True)
  manifest.add_argument('--parent-run-dir')
  validate = sub.add_parser('validate')
  validate.add_argument('--run-dir', required=True)
  validate.add_argument('--phase', choices=('base', 'branch'), required=True)
  combined = sub.add_parser('combine')
  combined.add_argument('--base-dir', required=True)
  combined.add_argument('--branch-dir', required=True)
  combined.add_argument('--output-dir', required=True)
  return result


def main() -> None:
  args = parser().parse_args()
  if args.command == 'write-manifest':
    write_manifest(args)
  elif args.command == 'validate':
    run_dir = Path(args.run_dir)
    summary = validate_base(run_dir) if args.phase == 'base' else validate_branch(run_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
  else:
    payload = combine(Path(args.base_dir), Path(args.branch_dir), Path(args.output_dir))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
