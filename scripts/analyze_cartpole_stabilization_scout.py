#!/usr/bin/env python3
"""Estimate the learning plateau of the 50k Cartpole stabilization scout."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_eval_means(path: Path) -> list[tuple[int, float]]:
  with path.open(newline='') as handle:
    rows = csv.DictReader(handle)
    values = [
        (int(row['step']), float(row['value']))
        for row in rows
        if row['tag'] == 'eval/return_mean'
    ]
  if not values:
    raise ValueError(f'No eval/return_mean rows found in {path}')
  if any(step <= 0 or not math.isfinite(value) for step, value in values):
    raise ValueError('Evaluation rows contain invalid steps or values')
  return sorted(values)


def plateau_summary(points: list[tuple[int, float]],
                    *,
                    episode_length: int = 500,
                    final_window_points: int = 5,
                    sustained_fraction: float = 0.95,
                    sustained_points: int = 4,
                    rolling_points: int = 5,
                    max_cv: float = 0.02,
                    max_relative_slope_per_1000: float = 0.005) -> dict:
  if len(points) < max(final_window_points, sustained_points, rolling_points):
    raise ValueError('Too few evaluation points for the plateau criteria')
  steps = np.asarray([point[0] for point in points], dtype=np.float64)
  returns = np.asarray([point[1] for point in points], dtype=np.float64)
  final_level = float(np.mean(returns[-final_window_points:]))
  threshold = sustained_fraction * final_level

  sustained_step = None
  for start in range(len(points) - sustained_points + 1):
    if np.all(returns[start:start + sustained_points] >= threshold):
      sustained_step = int(steps[start])
      break

  rolling_step = None
  rolling_diagnostics = []
  for end in range(rolling_points, len(points) + 1):
    x = steps[end - rolling_points:end]
    y = returns[end - rolling_points:end]
    mean = float(np.mean(y))
    cv = float(np.std(y, ddof=0) / max(abs(mean), 1e-12))
    slope = float(np.polyfit(x / 1000.0, y, 1)[0])
    relative_slope = abs(slope) / max(abs(mean), 1e-12)
    accepted = (
        mean >= threshold and
        cv <= max_cv and
        relative_slope <= max_relative_slope_per_1000
    )
    rolling_diagnostics.append({
        'window_start_step': int(x[0]),
        'window_end_step': int(x[-1]),
        'mean': mean,
        'cv': cv,
        'relative_slope_per_1000': relative_slope,
        'accepted': accepted,
    })
    if accepted and rolling_step is None:
      rolling_step = int(x[0])

  return {
      'evaluation_points': len(points),
      'first_evaluation_step': int(steps[0]),
      'last_evaluation_step': int(steps[-1]),
      'final_reference_mean': final_level,
      'sustained_threshold': threshold,
      'sustained_plateau_step': sustained_step,
      'sustained_plateau_episode_equivalent': (
          None if sustained_step is None else sustained_step / episode_length
      ),
      'rolling_plateau_step': rolling_step,
      'rolling_plateau_episode_equivalent': (
          None if rolling_step is None else rolling_step / episode_length
      ),
      'rolling_diagnostics': rolling_diagnostics,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('run_dir', type=Path)
  parser.add_argument('--output', type=Path)
  args = parser.parse_args()
  points = read_eval_means(args.run_dir / 'metrics' / 'scalars.csv')
  summary = plateau_summary(points)
  text = json.dumps(summary, indent=2) + '\n'
  if args.output is None:
    print(text, end='')
  else:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding='utf-8')


if __name__ == '__main__':
  main()
