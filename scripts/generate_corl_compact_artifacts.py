#!/usr/bin/env python3
"""Generate tables, figures, and a base report for the compact CoRL campaign."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Curve:
  steps: np.ndarray
  values: np.ndarray


def load_goal(path: Path) -> dict[str, Any]:
  with path.open() as handle:
    return json.load(handle)


def relpath(path: str | Path) -> Path:
  path = Path(path)
  return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
  if not path.exists():
    return []
  with path.open(newline='') as handle:
    return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
      writer.writerow({field: row.get(field, '') for field in fields})


def write_latex_table(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  def esc(value: Any) -> str:
    text = '' if value is None else str(value)
    return (
        text.replace('\\', r'\textbackslash{}')
        .replace('_', r'\_')
        .replace('%', r'\%')
        .replace('&', r'\&')
    )
  lines = [
      r'\begin{tabular}{' + 'l' * len(fields) + '}',
      r'\toprule',
      ' & '.join(esc(field) for field in fields) + r' \\',
      r'\midrule',
  ]
  for row in rows:
    lines.append(' & '.join(esc(row.get(field, '')) for field in fields) + r' \\')
  lines.extend([r'\bottomrule', r'\end{tabular}', ''])
  path.write_text('\n'.join(lines))


def float_or_none(value: Any) -> float | None:
  if value in (None, ''):
    return None
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def int_or_none(value: Any) -> int | None:
  if value in (None, ''):
    return None
  try:
    return int(float(value))
  except (TypeError, ValueError):
    return None


def load_eval_curve(metrics_path: Path) -> Curve | None:
  if not metrics_path.exists():
    return None
  rows: list[tuple[int, float]] = []
  with metrics_path.open(newline='') as handle:
    for row in csv.DictReader(handle):
      if row.get('tag') != 'eval/return_mean':
        continue
      step = int_or_none(row.get('step'))
      value = float_or_none(row.get('value'))
      if step is not None and value is not None:
        rows.append((step, value))
  if not rows:
    return None
  rows.sort()
  return Curve(
      steps=np.asarray([step for step, _ in rows], dtype=np.float32),
      values=np.asarray([value for _, value in rows], dtype=np.float32),
  )


def curve_auc(curve: Curve, max_steps: int) -> float | None:
  if curve.steps.size < 2:
    return None
  mask = curve.steps <= max_steps
  steps = curve.steps[mask]
  values = curve.values[mask]
  if steps.size < 2:
    return None
  return float(np.trapz(values, steps) / float(max_steps))


def final_curve_score(curve: Curve, max_steps: int) -> tuple[int | None, float | None, float | None]:
  if curve.steps.size == 0:
    return None, None, None
  mask = curve.steps <= max_steps
  if not np.any(mask):
    return None, None, None
  steps = curve.steps[mask]
  values = curve.values[mask]
  return int(steps[-1]), float(values[-1]), float(np.max(values))


def terminal_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
  return [
      row for row in rows
      if row.get('event') in {'completed', 'partial_complete'}
      and row.get('status') in {'completed', 'passed', 'partial_complete'}
  ]


def metric_cache_path(results_dir: Path, run_id: str) -> Path:
  return results_dir / 'cache' / run_id / 'metrics' / 'scalars.csv'


def enrich_rows(goal: dict[str, Any], ledger_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
  results_dir = relpath(goal['tracking']['results_dir'])
  max_steps = int(goal['constraints']['full_run_steps'])
  enriched: list[dict[str, Any]] = []
  for row in terminal_rows(ledger_rows):
    if row.get('method') == 'checkpoint_resume_smoke':
      continue
    run_id = row['run_id']
    curve = load_eval_curve(metric_cache_path(results_dir, run_id))
    final_step = int_or_none(row.get('final_step'))
    final_score = float_or_none(row.get('final_score'))
    best_score = float_or_none(row.get('best_score'))
    auc = float_or_none(row.get('auc'))
    if curve is not None:
      curve_step, curve_final, curve_best = final_curve_score(curve, max_steps)
      final_step = curve_step if curve_step is not None else final_step
      final_score = curve_final if curve_final is not None else final_score
      best_score = curve_best if curve_best is not None else best_score
      auc = curve_auc(curve, max_steps) or auc
    enriched.append({
        **row,
        'final_step': final_step,
        'final_score': final_score,
        'best_score': best_score,
        'auc': auc,
        'curve': curve,
    })
  return enriched


def write_main_tables(goal: dict[str, Any], rows: list[dict[str, Any]], tables_dir: Path) -> None:
  final_fields = [
      'env_id', 'regime', 'method', 'seed', 'paper_horizon',
      'final_step', 'final_score', 'best_score', 'job_id', 'wall_hours',
      'checkpoint_ok', 'run_dir',
  ]
  table_rows = []
  for row in rows:
    table_rows.append({
        field: format_value(row.get(field, ''))
        for field in final_fields
    })
  write_csv(tables_dir / 'main_final_scores.csv', table_rows, final_fields)
  write_latex_table(tables_dir / 'main_final_scores.tex', table_rows, final_fields)

  auc_fields = ['env_id', 'regime', 'method', 'seed', 'auc', 'final_score', 'wall_hours']
  auc_rows = [
      {field: format_value(row.get(field, '')) for field in auc_fields}
      for row in rows
  ]
  write_csv(tables_dir / 'main_auc_scores.csv', auc_rows, auc_fields)
  write_latex_table(tables_dir / 'main_auc_scores.tex', auc_rows, auc_fields)

  compute_fields = [
      'env_id', 'regime', 'method', 'seed', 'job_id', 'wall_hours',
      'slurm_state', 'checkpoint_ok',
  ]
  compute_rows = [
      {field: format_value(row.get(field, '')) for field in compute_fields}
      for row in rows
  ]
  write_csv(tables_dir / 'compute_runtime.csv', compute_rows, compute_fields)
  write_latex_table(tables_dir / 'compute_runtime.tex', compute_rows, compute_fields)


def format_value(value: Any) -> str:
  if isinstance(value, float):
    return f'{value:.6g}'
  if value is None:
    return ''
  return str(value)


def fixed_key(row: dict[str, Any]) -> tuple[str, str, str]:
  return (row.get('env_id', ''), row.get('regime', ''), str(row.get('seed', '')))


def write_parity_table(goal: dict[str, Any], rows: list[dict[str, Any]], tables_dir: Path) -> None:
  max_steps = int(goal['constraints']['full_run_steps'])
  fixed_scores: dict[tuple[str, str, str], float] = {}
  fixed_wall: dict[tuple[str, str, str], float] = {}
  for row in rows:
    if row.get('method') == 'paper_horizon' and row.get('final_score') is not None:
      key = fixed_key(row)
      fixed_scores[key] = float(row['final_score'])
      wall = float_or_none(row.get('wall_hours'))
      if wall is not None:
        fixed_wall[key] = wall

  parity_rows = []
  paper_scores = goal.get('published_tdmpc2_scores', {})
  for row in rows:
    if row.get('method') != 'adaptive_rhs':
      continue
    key = fixed_key(row)
    curve = row.get('curve')
    fixed_score = fixed_scores.get(key)
    paper_score = paper_scores.get(row.get('env_id'))
    fixed_step = crossing_step(curve, fixed_score)
    paper_step = crossing_step(curve, paper_score)
    rhs_wall = float_or_none(row.get('wall_hours'))
    fixed_hours = fixed_wall.get(key)
    parity_rows.append({
        'env_id': row.get('env_id', ''),
        'regime': row.get('regime', ''),
        'seed': row.get('seed', ''),
        'fixed_final_score': format_value(fixed_score),
        'rhs_final_score': format_value(row.get('final_score')),
        'fixed_parity_step': fixed_step or '',
        'paper_score': format_value(paper_score),
        'paper_score_step': paper_step or '',
        'rhs_wall_hours': format_value(rhs_wall),
        'matched_fixed_wall_hours': format_value(fixed_hours),
        'max_steps': max_steps,
    })
  fields = [
      'env_id', 'regime', 'seed', 'fixed_final_score', 'rhs_final_score',
      'fixed_parity_step', 'paper_score', 'paper_score_step',
      'rhs_wall_hours', 'matched_fixed_wall_hours', 'max_steps',
  ]
  write_csv(tables_dir / 'parity_times.csv', parity_rows, fields)
  write_latex_table(tables_dir / 'parity_times.tex', parity_rows, fields)


def crossing_step(curve: Curve | None, threshold: Any) -> int | None:
  threshold_value = float_or_none(threshold)
  if curve is None or threshold_value is None:
    return None
  for step, value in zip(curve.steps, curve.values):
    if float(value) >= threshold_value:
      return int(step)
  return None


def mean_curve(curves: list[Curve]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
  curves = [curve for curve in curves if curve is not None and curve.steps.size]
  if not curves:
    return None
  grid = np.unique(np.concatenate([curve.steps for curve in curves]))
  values = []
  for curve in curves:
    values.append(np.interp(grid, curve.steps, curve.values))
  stacked = np.vstack(values)
  return grid, np.mean(stacked, axis=0), np.std(stacked, axis=0)


def plot_learning_curves(goal: dict[str, Any], rows: list[dict[str, Any]], figures_dir: Path, regime: str, filename: str) -> None:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  envs = [item['env_id'] for item in goal['matrix']['envs']]
  fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
  colors = {'paper_horizon': '#2F5D62', 'adaptive_rhs': '#B85C38'}
  labels = {'paper_horizon': 'Fixed paper horizon', 'adaptive_rhs': 'Adaptive RHS'}
  for axis, env_id in zip(axes.ravel(), envs):
    axis.set_title(env_id)
    axis.grid(alpha=0.25, linestyle='--')
    for method in ('paper_horizon', 'adaptive_rhs'):
      curves = [
          row['curve'] for row in rows
          if row.get('env_id') == env_id
          and row.get('regime') == regime
          and row.get('method') == method
          and row.get('curve') is not None
      ]
      for curve in curves:
        axis.plot(curve.steps / 1000.0, curve.values, color=colors[method], alpha=0.18, linewidth=1)
      aggregate = mean_curve(curves)
      if aggregate is not None:
        grid, mean, std = aggregate
        axis.plot(grid / 1000.0, mean, color=colors[method], linewidth=2.3, label=labels[method])
        axis.fill_between(grid / 1000.0, mean - std, mean + std, color=colors[method], alpha=0.12)
    axis.set_xlabel('Environment steps (k)')
    axis.set_ylabel('Clean eval return')
  handles, labels_out = axes.ravel()[0].get_legend_handles_labels()
  if handles:
    fig.legend(handles, labels_out, loc='upper center', ncol=2, frameon=False)
  fig.suptitle(f'{regime.title()} training: fixed horizon vs adaptive RHS', fontsize=16, fontweight='bold')
  figures_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(figures_dir / f'{filename}.png', dpi=180)
  fig.savefig(figures_dir / f'{filename}.pdf')
  plt.close(fig)


def plot_time_to_parity(tables_dir: Path, figures_dir: Path) -> None:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  rows = read_csv(tables_dir / 'parity_times.csv')
  fig, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
  plot_rows = [
      row for row in rows
      if row.get('fixed_parity_step') not in ('', None)
  ]
  if plot_rows:
    labels = [f"{row['env_id']}\n{row['regime']}\ns{row['seed']}" for row in plot_rows]
    values = [float(row['fixed_parity_step']) / 1000.0 for row in plot_rows]
    axis.bar(range(len(values)), values, color='#B85C38')
    axis.set_xticks(range(len(values)), labels, rotation=45, ha='right', fontsize=8)
    axis.set_ylabel('Steps to fixed-baseline parity (k)')
  else:
    axis.text(0.5, 0.5, 'Parity data unavailable until RHS and fixed runs complete.', ha='center', va='center')
    axis.set_xticks([])
    axis.set_yticks([])
  axis.set_title('Adaptive RHS time to fixed-baseline parity')
  axis.grid(axis='y', alpha=0.25, linestyle='--')
  figures_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(figures_dir / 'fig3_time_to_parity.png', dpi=180)
  fig.savefig(figures_dir / 'fig3_time_to_parity.pdf')
  plt.close(fig)


def plot_compute_performance(rows: list[dict[str, Any]], figures_dir: Path) -> None:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
  colors = {'paper_horizon': '#2F5D62', 'adaptive_rhs': '#B85C38'}
  labels = {'paper_horizon': 'Fixed paper horizon', 'adaptive_rhs': 'Adaptive RHS'}
  plotted = False
  for method in ('paper_horizon', 'adaptive_rhs'):
    xs = []
    ys = []
    for row in rows:
      if row.get('method') != method:
        continue
      wall = float_or_none(row.get('wall_hours'))
      score = float_or_none(row.get('final_score'))
      if wall is None or score is None:
        continue
      xs.append(wall)
      ys.append(score)
    if xs:
      plotted = True
      axis.scatter(xs, ys, s=42, alpha=0.75, color=colors[method], label=labels[method])
  if not plotted:
    axis.text(0.5, 0.5, 'Compute/performance data unavailable until runs complete.', ha='center', va='center')
  axis.set_xlabel('Wall-clock GPU-hours')
  axis.set_ylabel('Final clean eval return')
  axis.set_title('Compute-performance tradeoff')
  axis.grid(alpha=0.25, linestyle='--')
  if plotted:
    axis.legend(frameon=False)
  figures_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(figures_dir / 'fig4_compute_performance.png', dpi=180)
  fig.savefig(figures_dir / 'fig4_compute_performance.pdf')
  plt.close(fig)


def write_report(goal: dict[str, Any], rows: list[dict[str, Any]], results_dir: Path) -> None:
  report_dir = results_dir / 'report'
  report_dir.mkdir(parents=True, exist_ok=True)
  completed = len(rows)
  expected = 72
  clean_rows = [row for row in rows if row.get('regime') == 'clean']
  chaos_rows = [row for row in rows if row.get('regime') == 'chaos']
  adaptive = [row for row in rows if row.get('method') == 'adaptive_rhs']
  fixed = [row for row in rows if row.get('method') == 'paper_horizon']
  mean_adaptive = np.nanmean([float_or_none(row.get('final_score')) or np.nan for row in adaptive]) if adaptive else np.nan
  mean_fixed = np.nanmean([float_or_none(row.get('final_score')) or np.nan for row in fixed]) if fixed else np.nan
  md = f"""# Compact CoRL Adaptive-RHS Campaign Report

## Executive Summary

- Completed main profiles: {completed}/{expected}.
- Mean final fixed paper-horizon score: {format_value(mean_fixed)}.
- Mean final adaptive RHS score: {format_value(mean_adaptive)}.
- Main claim: adaptive RHS is evaluated against the standard TD-MPC2 fixed horizon without a horizon sweep.

## Method

The baseline is TD-MPC2-JAX with Dense-RHS disabled and the paper-matched fixed horizon recorded in the ledger. Adaptive RHS uses the frozen sparse high-fidelity configuration from the current repo; no architecture search or per-environment horizon tuning is part of this campaign.

## Experiment Setup

Environments: {', '.join(item['env_id'] for item in goal['matrix']['envs'])}.

Regimes: clean and chaos. Chaos enables domain randomization, observation noise, and one-step base action delay during training, while evaluation remains clean.

Seeds: {', '.join(str(seed) for seed in goal['matrix']['seeds'])}. Training budget: {goal['constraints']['full_run_steps']} environment steps.

## Main Results

See `figures/fig1_clean_learning_curves.*`, `figures/fig2_chaos_learning_curves.*`, `tables/main_final_scores.*`, and `tables/main_auc_scores.*`.

Clean completed rows: {len(clean_rows)}. Chaos completed rows: {len(chaos_rows)}.

## Robustness

The robustness comparison should use per-method chaos-minus-clean deltas after all matched task/seed cells complete.

## Compute

See `figures/fig3_time_to_parity.*`, `figures/fig4_compute_performance.*`, `tables/parity_times.*`, and `tables/compute_runtime.*`.

## Reproducibility

All rows are sourced from `experiments/corl_compact_ledger.csv`. Each row records SLURM job id, git commit, remote commit, run directory, seed, method, regime, and checkpoint status.

## Limitations

Humanoid is not included because the current MJX backend does not have a humanoid port or gate. Any blocked or failed rows must be interpreted as campaign limitations unless rerun with the same frozen method/config succeeds.
"""
  (report_dir / 'corl_compact_base_report.md').write_text(md)
  tex = md_to_minimal_tex(md)
  (report_dir / 'corl_compact_base_report.tex').write_text(tex)


def md_to_minimal_tex(markdown: str) -> str:
  lines = [
      r'\documentclass{article}',
      r'\usepackage{booktabs}',
      r'\usepackage[margin=1in]{geometry}',
      r'\begin{document}',
  ]
  for raw in markdown.splitlines():
    line = raw.strip()
    if not line:
      lines.append('')
    elif line.startswith('# '):
      lines.append(r'\section*{' + tex_escape(line[2:]) + '}')
    elif line.startswith('## '):
      lines.append(r'\subsection*{' + tex_escape(line[3:]) + '}')
    elif line.startswith('- '):
      lines.append(r'\noindent $\bullet$ ' + tex_escape(line[2:]) + r'\\')
    else:
      lines.append(tex_escape(line) + r'\\')
  lines.append(r'\end{document}')
  lines.append('')
  return '\n'.join(lines)


def tex_escape(text: str) -> str:
  return (
      text.replace('\\', r'\textbackslash{}')
      .replace('&', r'\&')
      .replace('%', r'\%')
      .replace('$', r'\$')
      .replace('#', r'\#')
      .replace('_', r'\_')
      .replace('{', r'\{')
      .replace('}', r'\}')
      .replace('~', r'\textasciitilde{}')
      .replace('^', r'\textasciicircum{}')
  )


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--goal', default=str(ROOT / 'goals' / 'dense_rhs_corl_compact.yaml'))
  args = parser.parse_args()

  goal_path = Path(args.goal)
  if not goal_path.is_absolute():
    goal_path = ROOT / goal_path
  goal = load_goal(goal_path)
  results_dir = relpath(goal['tracking']['results_dir'])
  tables_dir = results_dir / 'tables'
  figures_dir = results_dir / 'figures'
  ledger_rows = read_csv(relpath(goal['tracking']['ledger']))
  rows = enrich_rows(goal, ledger_rows)
  write_main_tables(goal, rows, tables_dir)
  write_parity_table(goal, rows, tables_dir)
  plot_learning_curves(goal, rows, figures_dir, 'clean', 'fig1_clean_learning_curves')
  plot_learning_curves(goal, rows, figures_dir, 'chaos', 'fig2_chaos_learning_curves')
  plot_time_to_parity(tables_dir, figures_dir)
  plot_compute_performance(rows, figures_dir)
  write_report(goal, rows, results_dir)
  print(f'wrote CoRL compact artifacts to {results_dir}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
