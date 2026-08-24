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
                    num_parallel_envs: int = 8,
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
      'sustained_plateau_episode_cycles_per_env': (
          None if sustained_step is None
          else sustained_step / (episode_length * num_parallel_envs)
      ),
      'rolling_plateau_step': rolling_step,
      'rolling_plateau_episode_equivalent': (
          None if rolling_step is None else rolling_step / episode_length
      ),
      'rolling_plateau_episode_cycles_per_env': (
          None if rolling_step is None
          else rolling_step / (episode_length * num_parallel_envs)
      ),
      'rolling_diagnostics': rolling_diagnostics,
  }


def save_plot(points: list[tuple[int, float]], summary: dict, path: Path) -> None:
  """Save a compact plot of the learning curve and frozen plateau criteria."""
  import matplotlib.pyplot as plt

  steps_k = np.asarray([step for step, _ in points], dtype=np.float64) / 1000.0
  returns = np.asarray([value for _, value in points], dtype=np.float64)
  sustained_step = summary['sustained_plateau_step']
  rolling_step = summary['rolling_plateau_step']

  fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
  ax.plot(steps_k, returns, marker='o', linewidth=2.2, color='#1769aa')
  ax.axhline(
      summary['final_reference_mean'], color='#2e7d32', linewidth=1.8,
      label=f"final-window mean = {summary['final_reference_mean']:.1f}",
  )
  ax.axhline(
      summary['sustained_threshold'], color='#2e7d32', linewidth=1.2,
      linestyle='--', label='95% of final-window mean',
  )
  if sustained_step is not None:
    sustained_k = sustained_step / 1000.0
    ax.axvline(
        sustained_k, color='#f57c00', linewidth=1.6, linestyle='--',
        label=f'sustained criterion: {sustained_k:.0f}k',
    )
  if rolling_step is not None:
    rolling_k = rolling_step / 1000.0
    ax.axvline(
        rolling_k, color='#7b1fa2', linewidth=1.8, linestyle=':',
        label=f'strict rolling plateau: {rolling_k:.0f}k',
    )
    ax.axvspan(rolling_k, steps_k[-1], color='#7b1fa2', alpha=0.08)
  ax.set(
      title='Cartpole fixed-h3 stabilization scout',
      xlabel='Collected transitions (thousands)',
      ylabel='Evaluation return (10-episode mean)',
      xlim=(0, max(52.5, steps_k[-1] + 2.5)),
  )
  ax.grid(alpha=0.22)
  ax.legend(loc='lower right', frameon=False)
  secondary = ax.secondary_xaxis(
      'top', functions=(lambda x: 2.0 * x, lambda x: x / 2.0)
  )
  secondary.set_xlabel('Aggregate 500-transition episode equivalents')
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path, dpi=180)
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('run_dir', type=Path)
  parser.add_argument('--output', type=Path)
  parser.add_argument('--plot', type=Path)
  args = parser.parse_args()
  points = read_eval_means(args.run_dir / 'metrics' / 'scalars.csv')
  summary = plateau_summary(points)
  text = json.dumps(summary, indent=2) + '\n'
  if args.output is None:
    print(text, end='')
  else:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding='utf-8')
  if args.plot is not None:
    save_plot(points, summary, args.plot)


if __name__ == '__main__':
  main()
