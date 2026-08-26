#!/usr/bin/env python3
"""Manifest, validation, and aggregation for the Pendulum frontier atlas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from tdmpc2_jax.frontier_atlas import (
    ATLAS_HORIZONS,
    ATLAS_VERSION,
    frontier_conditions,
    shard_conditions,
)


SOURCE_STEP = 30_000


def _read_csv(path: Path):
  with path.open(newline='') as handle:
    return list(csv.DictReader(handle))


def _write_csv(path: Path, rows):
  rows = list(rows)
  if not rows:
    raise ValueError(f'Refusing to write empty CSV {path}.')
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


def write_manifest(args):
  run_dir = Path(args.run_dir)
  run_dir.mkdir(parents=True, exist_ok=False)
  conditions = shard_conditions(
      args.shard_index,
      args.num_shards,
      limit=args.condition_limit,
  )
  manifest = {
      'atlas_version': ATLAS_VERSION,
      'run_id': args.run_id,
      'mode': args.mode,
      'commit': args.commit,
      'config_hash': args.config_hash,
      'seed': int(args.seed),
      'source': {
          'run_dir': str(Path(args.source_run_dir).resolve()),
          'checkpoint_step': SOURCE_STEP,
      },
      'shard_index': int(args.shard_index),
      'num_shards': int(args.num_shards),
      'condition_limit': args.condition_limit,
      'condition_ids': [condition.condition_id for condition in conditions],
      'horizons': list(ATLAS_HORIZONS),
      'replicas': int(args.replicas),
      'eval_steps': int(args.eval_steps),
      'planner': {
          'population_size': 256,
          'policy_prior_samples': 12,
          'num_elites': 32,
          'mppi_iterations': 4,
          'temperature': 0.5,
      },
  }
  (run_dir / 'run_manifest.json').write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + '\n'
  )


def validate(run_dir: Path, *, require_marker: bool = True):
  run_dir = Path(run_dir)
  manifest = json.loads((run_dir / 'run_manifest.json').read_text())
  if manifest.get('atlas_version') != ATLAS_VERSION:
    raise ValueError('Atlas version mismatch.')
  if int(manifest['source']['checkpoint_step']) != SOURCE_STEP:
    raise ValueError('Atlas does not identify the frozen 30k source checkpoint.')
  if manifest.get('horizons') != list(ATLAS_HORIZONS):
    raise ValueError('Atlas horizon grid drifted from 2..8.')
  expected_conditions = shard_conditions(
      int(manifest['shard_index']),
      int(manifest['num_shards']),
      limit=manifest.get('condition_limit'),
  )
  expected_ids = [condition.condition_id for condition in expected_conditions]
  if manifest.get('condition_ids') != expected_ids:
    raise ValueError('Manifest condition assignment does not match canonical shard.')
  artifacts = run_dir / 'artifacts' / 'frontier_atlas'
  runtime = json.loads((artifacts / 'runtime.json').read_text())
  summary = _read_csv(artifacts / 'summary.csv')
  paired = _read_csv(artifacts / 'paired_returns.csv')
  replicas = int(manifest['replicas'])
  expected_summary = len(expected_ids) * len(ATLAS_HORIZONS)
  expected_paired = expected_summary * replicas
  if len(summary) != expected_summary:
    raise ValueError(
        f'Expected {expected_summary} summary rows, found {len(summary)}.'
    )
  if len(paired) != expected_paired:
    raise ValueError(
        f'Expected {expected_paired} paired rows, found {len(paired)}.'
    )
  summary_cells = set()
  paired_cells = defaultdict(set)
  for row in summary:
    cell = (row['condition_id'], int(row['horizon']))
    summary_cells.add(cell)
    for field in ('return_mean', 'return_std', 'return_se', 'condition_elapsed_s'):
      if not math.isfinite(float(row[field])):
        raise ValueError(f'Non-finite {field} in {cell}.')
    if int(row['source_step']) != SOURCE_STEP:
      raise ValueError(f'Source step mismatch in {cell}.')
    if int(row['replicas']) != replicas:
      raise ValueError(f'Replica count mismatch in {cell}.')
    if int(row['eval_steps']) != int(manifest['eval_steps']):
      raise ValueError(f'Evaluation length mismatch in {cell}.')
  for row in paired:
    cell = (row['condition_id'], int(row['horizon']))
    replica = int(row['replica'])
    paired_cells[cell].add(replica)
    if not math.isfinite(float(row['return'])):
      raise ValueError(f'Non-finite paired return in {cell}, replica {replica}.')
  expected_cells = {
      (condition_id, horizon)
      for condition_id in expected_ids
      for horizon in ATLAS_HORIZONS
  }
  if summary_cells != expected_cells or set(paired_cells) != expected_cells:
    raise ValueError('Atlas cells are missing or duplicated.')
  expected_replicas = set(range(replicas))
  if any(values != expected_replicas for values in paired_cells.values()):
    raise ValueError('Paired replica indices are incomplete.')
  if runtime.get('condition_ids') != expected_ids:
    raise ValueError('Runtime condition list drifted from the manifest.')
  if runtime.get('planner') != manifest.get('planner') | {'planning_hmax': 8}:
    raise ValueError('Runtime planner settings drifted from the manifest.')
  if not (run_dir / 'checkpoint' / str(SOURCE_STEP)).is_dir():
    raise ValueError('Copied source checkpoint 30k is missing.')
  if not (run_dir / 'ATLAS_COMPLETE').is_file():
    raise ValueError('ATLAS_COMPLETE marker is missing.')
  if require_marker and not (run_dir / 'RUN_VALID').is_file():
    raise ValueError('RUN_VALID marker is missing.')
  return {
      'valid': True,
      'mode': manifest['mode'],
      'shard_index': int(manifest['shard_index']),
      'condition_count': len(expected_ids),
      'summary_rows': len(summary),
      'paired_rows': len(paired),
      'elapsed_s': float(runtime['total_elapsed_s']),
  }


def aggregate(args):
  run_dirs = [Path(path) for path in args.run_dirs]
  manifests = []
  summary_rows = []
  paired_rows = []
  for run_dir in run_dirs:
    validate(run_dir)
    manifest = json.loads((run_dir / 'run_manifest.json').read_text())
    if manifest['mode'] != 'full':
      raise ValueError(f'Aggregate accepts only full shards, got {run_dir}.')
    manifests.append(manifest)
    artifacts = run_dir / 'artifacts' / 'frontier_atlas'
    summary_rows.extend(_read_csv(artifacts / 'summary.csv'))
    paired_rows.extend(_read_csv(artifacts / 'paired_returns.csv'))
  condition_ids = [condition.condition_id for condition in frontier_conditions()]
  observed_ids = sorted({row['condition_id'] for row in summary_rows})
  if observed_ids != sorted(condition_ids):
    raise ValueError('Full atlas shards do not cover the canonical 24 conditions.')
  if len({manifest['config_hash'] for manifest in manifests}) != len(manifests):
    raise ValueError('Each shard must carry its own immutable configuration hash.')
  if len({manifest['commit'] for manifest in manifests}) != 1:
    raise ValueError('Atlas shards were not executed at one commit.')

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=False)
  summary_rows.sort(key=lambda row: (int(row['condition_index']), int(row['horizon'])))
  paired_rows.sort(
      key=lambda row: (
          int(row['condition_index']), int(row['horizon']), int(row['replica'])
      )
  )
  _write_csv(output_dir / 'summary.csv', summary_rows)
  _write_csv(output_dir / 'paired_returns.csv', paired_rows)

  returns = defaultdict(dict)
  condition_meta = {}
  for row in paired_rows:
    condition_id = row['condition_id']
    horizon = int(row['horizon'])
    returns[condition_id].setdefault(horizon, []).append(float(row['return']))
    condition_meta[condition_id] = {
        'index': int(row['condition_index']),
        'axis': row['axis'],
        'value': float(row['value']),
    }
  findings = []
  for condition_id in condition_ids:
    horizon_returns = {
        horizon: np.asarray(values, dtype=np.float64)
        for horizon, values in returns[condition_id].items()
    }
    means = {horizon: float(np.mean(values)) for horizon, values in horizon_returns.items()}
    best_horizon = max(ATLAS_HORIZONS, key=lambda horizon: (means[horizon], -horizon))
    long_horizon = max((6, 7, 8), key=lambda horizon: (means[horizon], -horizon))
    delta = horizon_returns[long_horizon] - horizon_returns[3]
    delta_mean = float(np.mean(delta))
    delta_se = float(np.std(delta, ddof=1) / np.sqrt(delta.size))
    delta_lcb90 = delta_mean - 1.6448536269514722 * delta_se
    findings.append({
        'condition_id': condition_id,
        **condition_meta[condition_id],
        'best_horizon': int(best_horizon),
        'best_return_mean': means[best_horizon],
        'h3_return_mean': means[3],
        'best_long_horizon': int(long_horizon),
        'long_minus_h3_mean': delta_mean,
        'long_minus_h3_lcb90': delta_lcb90,
        'horizon_rescuable': bool(delta_lcb90 > 0.0),
    })
  payload = {
      'atlas_version': ATLAS_VERSION,
      'commit': manifests[0]['commit'],
      'source_step': SOURCE_STEP,
      'conditions': findings,
      'rescuable_conditions': [
          row['condition_id'] for row in findings if row['horizon_rescuable']
      ],
  }
  (output_dir / 'findings.json').write_text(
      json.dumps(payload, indent=2, sort_keys=True) + '\n'
  )
  (output_dir / 'AGGREGATE_VALID').touch()
  print(json.dumps({
      'valid': True,
      'conditions': len(findings),
      'rescuable_conditions': payload['rescuable_conditions'],
      'output_dir': str(output_dir),
  }, sort_keys=True))


def _build_parser():
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest='command', required=True)
  manifest = subparsers.add_parser('write-manifest')
  manifest.add_argument('--run-dir', required=True)
  manifest.add_argument('--run-id', required=True)
  manifest.add_argument('--mode', choices=('smoke', 'full'), required=True)
  manifest.add_argument('--commit', required=True)
  manifest.add_argument('--config-hash', required=True)
  manifest.add_argument('--seed', type=int, required=True)
  manifest.add_argument('--source-run-dir', required=True)
  manifest.add_argument('--shard-index', type=int, required=True)
  manifest.add_argument('--num-shards', type=int, required=True)
  manifest.add_argument('--condition-limit', type=int)
  manifest.add_argument('--replicas', type=int, required=True)
  manifest.add_argument('--eval-steps', type=int, required=True)
  validation = subparsers.add_parser('validate')
  validation.add_argument('--run-dir', required=True)
  validation.add_argument('--allow-missing-valid-marker', action='store_true')
  aggregation = subparsers.add_parser('aggregate')
  aggregation.add_argument('--output-dir', required=True)
  aggregation.add_argument('run_dirs', nargs='+')
  return parser


def main():
  args = _build_parser().parse_args()
  if args.command == 'write-manifest':
    write_manifest(args)
  elif args.command == 'validate':
    print(json.dumps(validate(
        Path(args.run_dir),
        require_marker=not args.allow_missing_valid_marker,
    ), sort_keys=True))
  else:
    aggregate(args)


if __name__ == '__main__':
  main()
