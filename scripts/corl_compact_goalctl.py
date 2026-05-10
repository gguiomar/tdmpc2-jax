#!/usr/bin/env python3
"""Ledger-driven steward for the compact CoRL Dense-RHS campaign."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOAL = ROOT / 'goals' / 'dense_rhs_corl_compact.yaml'
REMOTE_SKILL_DIR = Path.home() / '.codex' / 'skills' / 'remote-machines'
SSH_HELPER = REMOTE_SKILL_DIR / 'scripts' / 'ssh_host.sh'
NCC_GPU_STATUS = REMOTE_SKILL_DIR / 'scripts' / 'ncc_gpu_status.sh'

LEDGER_FIELDS = [
    'timestamp',
    'event',
    'run_id',
    'status',
    'job_id',
    'attempt',
    'env_id',
    'regime',
    'method',
    'seed',
    'paper_horizon',
    'run_dir',
    'launcher',
    'git_commit',
    'remote_commit',
    'slurm_state',
    'final_step',
    'final_score',
    'best_score',
    'auc',
    'wall_hours',
    'checkpoint_ok',
    'notes',
]

TERMINAL_EVENTS = {'completed', 'failed', 'blocked', 'partial_complete'}
GOOD_TERMINAL_STATUSES = {'completed', 'passed', 'partial_complete'}
RETRYABLE_STATES = {'TIMEOUT', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL'}
SLURM_TERMINAL_PREFIXES = (
    'COMPLETED',
    'FAILED',
    'CANCELLED',
    'TIMEOUT',
    'NODE_FAIL',
    'PREEMPTED',
    'OUT_OF_MEMORY',
    'BOOT_FAIL',
)
STEWARD_OWNED_PATHS = (
    'experiments/corl_compact_ledger.csv',
    'experiments/corl_compact_decisions.md',
    'goals/dense_rhs_corl_compact.yaml',
    'runs/results/corl_compact/',
)


def now_iso() -> str:
  return datetime.now(timezone.utc).isoformat(timespec='seconds')


def load_goal(path: Path) -> dict[str, Any]:
  with path.open() as handle:
    return json.load(handle)


def relpath(path: str | Path) -> Path:
  path = Path(path)
  return path if path.is_absolute() else ROOT / path


def run_local(
    argv: list[str],
    *,
    check: bool = False,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
  return subprocess.run(
      argv,
      cwd=ROOT,
      text=True,
      capture_output=True,
      timeout=timeout,
      check=check,
  )


def run_remote(
    goal: dict[str, Any],
    command: str,
    *,
    timeout: int = 90,
) -> subprocess.CompletedProcess[str]:
  host = goal['remote']['host']
  return run_local([str(SSH_HELPER), host, command], timeout=timeout)


def local_git_state() -> dict[str, Any]:
  commit = run_local(['git', 'rev-parse', 'HEAD'], timeout=15)
  status = run_local(['git', 'status', '--porcelain'], timeout=15)
  return {
      'ok': commit.returncode == 0 and status.returncode == 0,
      'commit': commit.stdout.strip() if commit.returncode == 0 else '',
      'dirty': bool(status.stdout.strip()),
      'status': status.stdout.strip(),
      'error': (commit.stderr + status.stderr).strip(),
  }


def git_dirty_paths() -> list[str]:
  status = run_local(['git', 'status', '--porcelain'], timeout=15)
  if status.returncode != 0:
    return []
  paths: list[str] = []
  for line in status.stdout.splitlines():
    if not line:
      continue
    path = line[3:].strip()
    if ' -> ' in path:
      path = path.split(' -> ', 1)[1].strip()
    paths.append(path)
  return paths


def is_steward_owned_path(path: str) -> bool:
  return any(path == prefix.rstrip('/') or path.startswith(prefix)
             for prefix in STEWARD_OWNED_PATHS)


def dirty_paths_are_steward_owned() -> bool:
  paths = git_dirty_paths()
  return bool(paths) and all(is_steward_owned_path(path) for path in paths)


def current_branch() -> str:
  result = run_local(['git', 'branch', '--show-current'], timeout=15)
  if result.returncode != 0:
    return ''
  return result.stdout.strip()


def auto_commit_steward_state(goal: dict[str, Any], reason: str) -> str:
  """Commit and sync steward-owned ledger/result updates.

  This prevents the steward from deadlocking on its own append-only ledger rows.
  It refuses to touch unrelated dirty files.
  """
  paths = git_dirty_paths()
  if not paths:
    return 'no steward state changes to commit'
  if not all(is_steward_owned_path(path) for path in paths):
    return (
        'steward auto-commit skipped; non-steward dirty paths: ' +
        ', '.join(path for path in paths if not is_steward_owned_path(path))
    )
  add_paths = [
      'experiments/corl_compact_ledger.csv',
      'experiments/corl_compact_decisions.md',
      'goals/dense_rhs_corl_compact.yaml',
      'runs/results/corl_compact',
  ]
  add = run_local(['git', 'add', *add_paths], timeout=60)
  if add.returncode != 0:
    return f'steward auto-commit git add failed: {add.stderr.strip()}'
  diff = run_local(['git', 'diff', '--cached', '--quiet'], timeout=30)
  if diff.returncode == 0:
    return 'no staged steward state changes to commit'
  message = f'Steward update: {reason}'
  commit = run_local(['git', 'commit', '-m', message], timeout=120)
  if commit.returncode != 0:
    return f'steward auto-commit failed: {commit.stderr.strip()}'
  branch = current_branch()
  if not branch:
    return 'steward auto-commit succeeded but branch was unknown; remote not synced'
  push = run_local(['git', 'push', 'gguiomar', branch], timeout=180)
  if push.returncode != 0:
    return f'steward auto-commit succeeded but push failed: {push.stderr.strip()}'
  pull = run_remote(
      goal,
      (
          f'cd {shlex.quote(goal["remote"]["path"])} && '
          f'git pull --ff-only gguiomar {shlex.quote(branch)}'
      ),
      timeout=180,
  )
  if pull.returncode != 0:
    return f'steward state pushed but remote ff-pull failed: {pull.stderr.strip()}'
  return f'steward state committed and synced: {message}'


def remote_git_state(goal: dict[str, Any]) -> dict[str, Any]:
  remote_path = shlex.quote(goal['remote']['path'])
  command = (
      f'cd {remote_path} && '
      'printf "commit=%s\\n" "$(git rev-parse HEAD 2>/dev/null || true)" && '
      'printf "status_begin\\n" && git status --porcelain 2>/dev/null || true'
  )
  result = run_remote(goal, command, timeout=60)
  state = {
      'ok': result.returncode == 0,
      'commit': '',
      'dirty': True,
      'status': '',
      'error': result.stderr.strip(),
  }
  if result.returncode != 0:
    return state
  lines = result.stdout.splitlines()
  for line in lines:
    if line.startswith('commit='):
      state['commit'] = line.split('=', 1)[1].strip()
      break
  try:
    idx = lines.index('status_begin')
    status_lines = lines[idx + 1:]
  except ValueError:
    status_lines = []
  state['status'] = '\n'.join(status_lines).strip()
  state['dirty'] = bool(state['status'])
  return state


def read_ledger(goal: dict[str, Any]) -> list[dict[str, str]]:
  ledger_path = relpath(goal['tracking']['ledger'])
  if not ledger_path.exists():
    return []
  with ledger_path.open(newline='') as handle:
    return list(csv.DictReader(handle))


def append_ledger(goal: dict[str, Any], row: dict[str, Any]) -> None:
  ledger_path = relpath(goal['tracking']['ledger'])
  ledger_path.parent.mkdir(parents=True, exist_ok=True)
  exists = ledger_path.exists()
  clean = {field: '' for field in LEDGER_FIELDS}
  clean.update({key: '' if value is None else value for key, value in row.items()})
  with ledger_path.open('a', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
    if not exists or ledger_path.stat().st_size == 0:
      writer.writeheader()
    writer.writerow(clean)


def latest_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
  latest: dict[str, dict[str, str]] = {}
  for row in rows:
    run_id = row.get('run_id', '')
    if run_id:
      latest[run_id] = row
  return latest


def rows_for_run(rows: list[dict[str, str]], run_id: str) -> list[dict[str, str]]:
  return [row for row in rows if row.get('run_id') == run_id]


def safe_slug(text: str) -> str:
  return re.sub(r'[^A-Za-z0-9_.-]+', '_', text).strip('_')


def remote_run_dir(goal: dict[str, Any], relative: str) -> str:
  return str(Path(goal['remote']['path']) / relative)


def build_setup_profile(goal: dict[str, Any]) -> dict[str, Any]:
  setup = goal['setup_smoke']
  run_dir = remote_run_dir(goal, setup['run_dir'])
  return {
      'run_id': setup['run_id'],
      'kind': 'setup_smoke',
      'env_id': setup['env_id'],
      'regime': 'clean',
      'method': 'checkpoint_resume_smoke',
      'seed': int(setup['seed']),
      'paper_horizon': int(goal['constraints']['paper_horizon']),
      'script': setup['script'],
      'run_dir': run_dir,
      'priority': 0,
      'env': {
          'RUN_ID': setup['run_id'],
          'RUN_DIR': run_dir,
          'ENV_ID': setup['env_id'],
          'SEED': str(setup['seed']),
      },
  }


def build_main_profiles(goal: dict[str, Any]) -> list[dict[str, Any]]:
  profiles: list[dict[str, Any]] = []
  defaults = goal['run_defaults']
  adaptive = goal['adaptive_rhs']
  paper_horizon = int(goal['constraints']['paper_horizon'])
  env_order = [item['env_id'] for item in goal['matrix']['envs']]
  regimes = goal['matrix']['regimes']
  priority_seeds = goal['matrix'].get('launch_seed_priority', goal['matrix']['seeds'])
  methods = goal['matrix']['methods']
  priority = 1
  for seed in priority_seeds:
    for env_id in env_order:
      for regime_name, regime_cfg in regimes.items():
        for method in methods:
          run_id = (
              f'corl500k__{safe_slug(env_id)}__{regime_name}__{method}__s{seed}'
          )
          relative_dir = (
              f'outputs/corl_pub_500k/{env_id}/{regime_name}/{method}/s{seed}'
          )
          run_dir = remote_run_dir(goal, relative_dir)
          is_adaptive = method == 'adaptive_rhs'
          script = (
              'scripts/ncc_corl_adaptive_rhs_500k.sbatch'
              if is_adaptive else
              'scripts/ncc_corl_paper_horizon_500k.sbatch'
          )
          env_vars = {
              'RUN_ID': run_id,
              'RUN_DIR': run_dir,
              'ENV_ID': env_id,
              'REGIME': regime_name,
              'SEED': str(seed),
              'PAPER_HORIZON': str(paper_horizon),
              'MAX_STEPS': str(defaults['max_steps']),
              'SAVE_INTERVAL_STEPS': str(defaults['save_interval_steps']),
              'CHECKPOINT_BUFFER': str(defaults['checkpoint_buffer']).lower(),
              'EVAL_INTERVAL_STEPS': str(defaults['eval_interval_steps']),
              'EVAL_NUM_EPISODES': str(defaults['eval_num_episodes']),
              'TRAIN_NUM_ENVS': str(defaults['train_num_envs']),
              'SEED_STEPS_OVERRIDE': str(defaults['seed_steps_override']),
              'UPDATE_CHUNK_SIZE': str(defaults['update_chunk_size']),
              'COLLECT_CHUNK_STEPS': str(defaults['collect_chunk_steps']),
              'UTD_RATIO': str(defaults['utd_ratio']),
              'RESET_POOL_SIZE': str(defaults['reset_pool_size']),
              'ENABLE_DOMAIN_RANDOMIZATION': str(
                  regime_cfg['enable_domain_randomization']
              ).lower(),
              'ENABLE_OBSERVATION_NOISE': str(
                  regime_cfg['enable_observation_noise']
              ).lower(),
              'BASE_ACTION_DELAY': str(regime_cfg['base_action_delay']),
          }
          if is_adaptive:
            budget = adaptive['candidate_budget']
            env_vars.update({
                'DENSE_QUERY_START_STEP': str(adaptive['query_start_step']),
                'DENSE_QUERY_INTERVAL_STEPS': str(adaptive['query_interval_steps']),
                'DENSE_RHS_REPLICAS': str(adaptive['num_env_eval_replicas']),
                'DENSE_RHS_EVAL_STEPS': str(adaptive['env_eval_steps']),
                'DENSE_RHS_QUERY_POPULATION': str(adaptive['query_population_size']),
                'DENSE_RHS_QUERY_POLICY_PRIOR': str(
                    adaptive['query_policy_prior_samples']
                ),
                'DENSE_RHS_QUERY_ELITES': str(adaptive['query_num_elites']),
                'DENSE_RHS_QUERY_MPPI_ITERS': str(adaptive['query_mppi_iterations']),
                'DENSE_RHS_QUERY_TEMPERATURE': str(adaptive['query_temperature']),
                'DENSE_RHS_SELECTION_RETURN_POWER': str(
                    adaptive['selection_return_power']
                ),
                'DENSE_RHS_ROUGHNESS_WEIGHT': str(adaptive['roughness_weight']),
                'DENSE_RHS_RETURN_STD_WEIGHT': str(adaptive['return_std_weight']),
                'DENSE_RHS_LOCAL_WINDOW_RADIUS': str(
                    adaptive['local_window_radius']
                ),
                'DENSE_RHS_MAX_TRANSITION_DELTA': str(
                    adaptive['max_transition_delta']
                ),
                'DENSE_RHS_INCUMBENT_MARGIN': str(
                    adaptive['incumbent_switch_margin']
                ),
                'DENSE_RHS_HORIZONS': str(adaptive['horizons']),
                'DENSE_RHS_HMAX': str(adaptive['hmax']),
                'DENSE_RHS_HORIZON_BUCKETS': str(adaptive['horizon_buckets']),
                'DENSE_RHS_CANDIDATE_BUDGET_A': str(budget['A']),
                'DENSE_RHS_CANDIDATE_BUDGET_B1': str(budget['B1']),
                'DENSE_RHS_CANDIDATE_BUDGET_B2': str(budget['B2']),
                'DENSE_RHS_CANDIDATE_BUDGET_B3': str(budget['B3']),
                'DENSE_RHS_CANDIDATE_BUDGET_B4': str(budget['B4']),
            })
          profiles.append({
              'run_id': run_id,
              'kind': 'main',
              'env_id': env_id,
              'regime': regime_name,
              'method': method,
              'seed': int(seed),
              'paper_horizon': paper_horizon,
              'script': script,
              'run_dir': run_dir,
              'priority': priority,
              'env': env_vars,
          })
          priority += 1
  return profiles


def build_profiles(goal: dict[str, Any]) -> list[dict[str, Any]]:
  return [build_setup_profile(goal)] + build_main_profiles(goal)


def validate_goal(goal: dict[str, Any]) -> tuple[bool, list[str]]:
  messages: list[str] = []
  failed = False
  main_profiles = build_main_profiles(goal)
  run_ids = [profile['run_id'] for profile in main_profiles]
  if len(main_profiles) != 72:
    messages.append(f'expected 72 main profiles, found {len(main_profiles)}')
    failed = True
  if len(run_ids) != len(set(run_ids)):
    messages.append('duplicate main profile run_ids detected')
    failed = True
  if int(goal['constraints']['paper_horizon']) != 3:
    messages.append(
        f"paper_horizon is {goal['constraints']['paper_horizon']}, expected 3"
    )
    failed = True
  for env in goal['matrix']['envs']:
    gate = env.get('gate', '')
    if gate == 'built_in':
      continue
    gate_path = relpath(gate)
    if not gate_path.exists():
      messages.append(f'missing MJX gate for {env["env_id"]}: {gate}')
      failed = True
  if not messages:
    messages.append('goal validation passed: 72 main profiles, no duplicates')
  return not failed, messages


def setup_gate_passed(goal: dict[str, Any], rows: list[dict[str, str]]) -> bool:
  latest = latest_rows(rows).get(goal['setup_smoke']['run_id'])
  return bool(latest and latest.get('event') == 'completed' and latest.get('status') == 'passed')


def profile_latest_status(
    rows: list[dict[str, str]],
    profile: dict[str, Any],
) -> dict[str, Any]:
  run_rows = rows_for_run(rows, profile['run_id'])
  latest = run_rows[-1] if run_rows else None
  attempts = sum(1 for row in run_rows if row.get('event') == 'launch')
  return {'latest': latest, 'attempts': attempts}


def is_terminal(row: dict[str, str] | None) -> bool:
  return bool(row and row.get('event') in TERMINAL_EVENTS)


def is_complete(row: dict[str, str] | None) -> bool:
  return bool(row and row.get('event') in TERMINAL_EVENTS and row.get('status') in GOOD_TERMINAL_STATUSES)


def should_retry(
    goal: dict[str, Any],
    rows: list[dict[str, str]],
    profile: dict[str, Any],
) -> bool:
  state = profile_latest_status(rows, profile)
  latest = state['latest']
  if latest is None or latest.get('event') != 'failed':
    return False
  if state['attempts'] >= int(goal['constraints']['max_retries']) + 1:
    return False
  slurm_state = latest.get('slurm_state', '').split()[0]
  return slurm_state in RETRYABLE_STATES or latest.get('status') == 'retryable'


def pending_profiles(goal: dict[str, Any], rows: list[dict[str, str]]) -> list[dict[str, Any]]:
  setup = build_setup_profile(goal)
  setup_state = profile_latest_status(rows, setup)
  if not setup_gate_passed(goal, rows):
    latest = setup_state['latest']
    if latest is None or latest.get('event') == 'failed':
      if setup_state['attempts'] <= int(goal['constraints']['max_retries']):
        return [setup]
    return []

  pending: list[dict[str, Any]] = []
  for profile in build_main_profiles(goal):
    state = profile_latest_status(rows, profile)
    latest = state['latest']
    if latest is None:
      pending.append(profile)
    elif latest.get('event') == 'launch':
      continue
    elif is_complete(latest) or latest.get('event') == 'blocked':
      continue
    elif should_retry(goal, rows, profile):
      pending.append(profile)
  return sorted(pending, key=lambda item: int(item['priority']))


def parse_gpu_status_output(
    goal: dict[str, Any],
    text: str,
) -> dict[str, Any]:
  gpu_rows: list[dict[str, Any]] = []
  slurm_jobs: list[dict[str, str]] = []
  section = None
  for raw_line in text.splitlines():
    line = raw_line.strip()
    if not line:
      continue
    if line.startswith('gpu_index,'):
      section = 'gpus'
      continue
    if line.startswith('compute_processes_'):
      section = 'processes'
      continue
    if line.startswith('all_slurm_jobs_'):
      section = 'slurm'
      continue
    if section == 'gpus' and re.match(r'^\d+,', line):
      parts = [part.strip() for part in line.split(',')]
      if len(parts) >= 6:
        try:
          gpu_rows.append({
              'index': int(parts[0]),
              'util': float(parts[3]),
              'mem_used': float(parts[4]),
              'mem_total': float(parts[5]),
          })
        except ValueError:
          pass
    elif section == 'slurm' and '|' in line:
      parts = line.split('|')
      if len(parts) >= 7:
        slurm_jobs.append({
            'job_id': parts[0],
            'user': parts[1],
            'name': parts[2],
            'state': parts[3],
            'time': parts[4],
            'gres': parts[5],
            'reason': parts[6],
        })
  util_threshold = float(goal['constraints']['physical_gpu_free_util_pct'])
  mem_threshold = float(goal['constraints']['physical_gpu_free_mem_mib'])
  free_gpus = [
      gpu for gpu in gpu_rows
      if gpu['util'] <= util_threshold and gpu['mem_used'] <= mem_threshold
  ]
  prefix = goal['remote']['steward_job_prefix']
  active_steward_jobs = [
      job for job in slurm_jobs
      if job['name'].startswith(prefix)
  ]
  return {
      'gpu_rows': gpu_rows,
      'free_gpu_count': len(free_gpus),
      'slurm_jobs': slurm_jobs,
      'active_steward_jobs': active_steward_jobs,
      'raw': text,
  }


def ncc_status(goal: dict[str, Any]) -> dict[str, Any]:
  result = run_local([str(NCC_GPU_STATUS), goal['remote']['host']], timeout=90)
  if result.returncode != 0:
    return {
        'ok': False,
        'free_gpu_count': 0,
        'active_steward_jobs': [],
        'error': result.stderr.strip(),
        'raw': result.stdout,
    }
  parsed = parse_gpu_status_output(goal, result.stdout)
  parsed['ok'] = True
  parsed['error'] = ''
  return parsed


def query_sacct(goal: dict[str, Any], job_ids: list[str]) -> dict[str, dict[str, str]]:
  if not job_ids:
    return {}
  ids = ','.join(sorted(set(job_ids)))
  command = (
      'sacct -n -P -X '
      f'-j {shlex.quote(ids)} '
      '--format=JobIDRaw,State,Elapsed,ExitCode 2>/dev/null || true'
  )
  result = run_remote(goal, command, timeout=60)
  states: dict[str, dict[str, str]] = {}
  if result.returncode != 0:
    return states
  for line in result.stdout.splitlines():
    parts = line.split('|')
    if len(parts) < 4:
      continue
    raw_id, state, elapsed, exit_code = [part.strip() for part in parts[:4]]
    if raw_id in job_ids and raw_id not in states:
      states[raw_id] = {
          'state': state,
          'elapsed': elapsed,
          'exit_code': exit_code,
      }
  return states


def elapsed_to_hours(elapsed: str) -> str:
  if not elapsed:
    return ''
  try:
    day_part = 0
    time_part = elapsed
    if '-' in elapsed:
      days, time_part = elapsed.split('-', 1)
      day_part = int(days)
    chunks = [int(float(part)) for part in time_part.split(':')]
    if len(chunks) == 3:
      hours, minutes, seconds = chunks
    elif len(chunks) == 2:
      hours, minutes, seconds = 0, chunks[0], chunks[1]
    else:
      return ''
    total = day_part * 24 + hours + minutes / 60 + seconds / 3600
    return f'{total:.4f}'
  except Exception:
    return ''


def remote_metric_summary(goal: dict[str, Any], run_dir: str, max_steps: int) -> dict[str, Any]:
  script = r'''
import csv, json, math, os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
max_steps = int(os.environ["MAX_STEPS"])
metrics = run_dir / "metrics" / "scalars.csv"
eval_rows = []
if metrics.exists():
  with metrics.open(newline="") as handle:
    for row in csv.DictReader(handle):
      if row.get("tag") != "eval/return_mean":
        continue
      try:
        eval_rows.append((int(float(row["step"])), float(row["value"])))
      except Exception:
        pass
eval_rows.sort()
usable = [(s, v) for s, v in eval_rows if s <= max_steps]
final_step = usable[-1][0] if usable else 0
final_score = usable[-1][1] if usable else None
best_score = max((v for _, v in usable), default=None)
auc = None
if len(usable) >= 2:
  area = 0.0
  for (s0, v0), (s1, v1) in zip(usable[:-1], usable[1:]):
    area += (s1 - s0) * (v0 + v1) / 2.0
  auc = area / float(max_steps)
checkpoint_dir = run_dir / "checkpoint"
checkpoint_steps = []
if checkpoint_dir.exists():
  for path in checkpoint_dir.iterdir():
    if path.is_dir() and path.name.isdigit():
      checkpoint_steps.append(int(path.name))
checkpoint_latest = max(checkpoint_steps) if checkpoint_steps else None
resume_path = run_dir / "resume_verified.json"
resume_verified = None
if resume_path.exists():
  try:
    resume_verified = json.loads(resume_path.read_text())
  except Exception:
    resume_verified = None
print(json.dumps({
  "metrics_exists": metrics.exists(),
  "final_step": final_step,
  "final_score": final_score,
  "best_score": best_score,
  "auc": auc,
  "checkpoint_latest": checkpoint_latest,
  "checkpoint_ok": checkpoint_latest is not None and checkpoint_latest >= final_step,
  "resume_verified": resume_verified,
}))
'''
  command = (
      f'cd {shlex.quote(goal["remote"]["path"])} && '
      f'RUN_DIR={shlex.quote(run_dir)} MAX_STEPS={shlex.quote(str(max_steps))} '
      f'python - <<{shlex.quote("PY")}\n{script}\nPY'
  )
  result = run_remote(goal, command, timeout=120)
  if result.returncode != 0:
    return {'metrics_exists': False, 'error': result.stderr.strip()}
  try:
    return json.loads(result.stdout.strip().splitlines()[-1])
  except Exception as exc:
    return {
        'metrics_exists': False,
        'error': f'failed to parse metric summary: {exc}; output={result.stdout[-500:]}',
    }


def slurm_state_is_terminal(state: str) -> bool:
  return any(state.startswith(prefix) for prefix in SLURM_TERMINAL_PREFIXES)


def process_terminal_jobs(goal: dict[str, Any], rows: list[dict[str, str]]) -> int:
  latest = latest_rows(rows)
  active_launches = [
      row for row in latest.values()
      if row.get('event') == 'launch' and row.get('job_id')
  ]
  sacct = query_sacct(goal, [row['job_id'] for row in active_launches])
  profiles_by_id = {profile['run_id']: profile for profile in build_profiles(goal)}
  appended = 0
  max_steps = int(goal['constraints']['full_run_steps'])
  for row in active_launches:
    job_id = row['job_id']
    job_state = sacct.get(job_id)
    if not job_state or not slurm_state_is_terminal(job_state['state']):
      continue
    profile = profiles_by_id.get(row['run_id'])
    if profile is None:
      continue
    is_setup = profile['kind'] == 'setup_smoke'
    elapsed_hours = elapsed_to_hours(job_state.get('elapsed', ''))
    summary = remote_metric_summary(
        goal,
        profile['run_dir'],
        8192 if is_setup else max_steps,
    )
    slurm_state = job_state['state'].split()[0]
    event = 'completed' if slurm_state == 'COMPLETED' else 'failed'
    status = 'completed'
    notes = ''
    if is_setup:
      resume = summary.get('resume_verified') or {}
      if slurm_state == 'COMPLETED' and resume.get('passed') is True:
        status = 'passed'
        notes = 'checkpoint resume smoke passed'
      else:
        event = 'failed'
        status = 'blocked'
        notes = f'checkpoint resume smoke failed: {resume or summary}'
    elif slurm_state != 'COMPLETED':
      status = 'retryable' if slurm_state in RETRYABLE_STATES else 'blocked'
      notes = f'SLURM state {job_state["state"]}; exit={job_state.get("exit_code", "")}'
    elif not summary.get('metrics_exists'):
      event = 'failed'
      status = 'blocked'
      notes = f'completed but metrics missing: {summary.get("error", "")}'
    else:
      final_step = int(summary.get('final_step') or 0)
      if final_step < int(max_steps * 0.9):
        event = 'failed'
        status = 'blocked'
        notes = f'completed with insufficient final_step={final_step}'
      elif final_step < max_steps:
        event = 'partial_complete'
        status = 'partial_complete'
        notes = f'partial usable metrics final_step={final_step}'
      else:
        notes = 'completed full 500k profile'
    append_ledger(goal, {
        'timestamp': now_iso(),
        'event': event,
        'run_id': profile['run_id'],
        'status': status,
        'job_id': job_id,
        'attempt': row.get('attempt', '1'),
        'env_id': profile['env_id'],
        'regime': profile['regime'],
        'method': profile['method'],
        'seed': profile['seed'],
        'paper_horizon': profile['paper_horizon'],
        'run_dir': profile['run_dir'],
        'launcher': profile['script'],
        'git_commit': row.get('git_commit', ''),
        'remote_commit': row.get('remote_commit', ''),
        'slurm_state': job_state['state'],
        'final_step': summary.get('final_step', ''),
        'final_score': summary.get('final_score', ''),
        'best_score': summary.get('best_score', ''),
        'auc': summary.get('auc', ''),
        'wall_hours': elapsed_hours,
        'checkpoint_ok': summary.get('checkpoint_ok', ''),
        'notes': notes,
    })
    appended += 1
  return appended


def launch_blockers(
    goal: dict[str, Any],
    local: dict[str, Any],
    remote: dict[str, Any],
    gpu: dict[str, Any],
) -> list[str]:
  blockers: list[str] = []
  if not local.get('ok'):
    blockers.append(f'local git unavailable: {local.get("error", "")}')
  if goal['constraints']['require_clean_local_git'] and local.get('dirty'):
    blockers.append('local git is dirty; full launches require committed/synced code')
  if not remote.get('ok'):
    blockers.append(f'remote git unavailable: {remote.get("error", "")}')
  if goal['constraints']['require_clean_remote_git'] and remote.get('dirty'):
    blockers.append('remote git is dirty')
  if goal['constraints']['require_remote_commit_match'] and local.get('commit') != remote.get('commit'):
    blockers.append(
        f'remote commit mismatch local={local.get("commit", "")[:12]} '
        f'remote={remote.get("commit", "")[:12]}'
    )
  if not gpu.get('ok'):
    blockers.append(f'NCC GPU status unavailable: {gpu.get("error", "")}')
  return blockers


def build_launch_command(
    goal: dict[str, Any],
    profile: dict[str, Any],
) -> str:
  env_prefix = ' '.join(
      f'{key}={shlex.quote(str(value))}'
      for key, value in sorted(profile['env'].items())
  )
  script = shlex.quote(profile['script'])
  job_suffix = safe_slug(profile['run_id'])[:48]
  job_name = f'{goal["remote"]["steward_job_prefix"]}-{job_suffix}'
  return (
      f'cd {shlex.quote(goal["remote"]["path"])} && '
      f'env {env_prefix} '
      f'sbatch --parsable --job-name={shlex.quote(job_name)} {script}'
  )


def launch_profile(
    goal: dict[str, Any],
    profile: dict[str, Any],
    *,
    local_commit: str,
    remote_commit: str,
    dry_run: bool,
    rows: list[dict[str, str]],
) -> str:
  state = profile_latest_status(rows, profile)
  attempt = state['attempts'] + 1
  command = build_launch_command(goal, profile)
  if dry_run:
    return f'DRY launch {profile["run_id"]}: {command}'
  result = run_remote(goal, command, timeout=60)
  if result.returncode != 0:
    raise RuntimeError(
        f'failed to launch {profile["run_id"]}: {result.stderr.strip()}'
    )
  job_id = result.stdout.strip().splitlines()[-1].split(';')[0].strip()
  if not job_id:
    raise RuntimeError(f'failed to parse sbatch job id: {result.stdout}')
  append_ledger(goal, {
      'timestamp': now_iso(),
      'event': 'launch',
      'run_id': profile['run_id'],
      'status': 'launched',
      'job_id': job_id,
      'attempt': attempt,
      'env_id': profile['env_id'],
      'regime': profile['regime'],
      'method': profile['method'],
      'seed': profile['seed'],
      'paper_horizon': profile['paper_horizon'],
      'run_dir': profile['run_dir'],
      'launcher': profile['script'],
      'git_commit': local_commit,
      'remote_commit': remote_commit,
      'notes': 'setup smoke' if profile['kind'] == 'setup_smoke' else '500k full profile',
  })
  return f'launched {profile["run_id"]} as job {job_id}'


def status_report(goal: dict[str, Any]) -> str:
  rows = read_ledger(goal)
  ok, messages = validate_goal(goal)
  local = local_git_state()
  remote = remote_git_state(goal)
  gpu = ncc_status(goal)
  main_profiles = build_main_profiles(goal)
  latest = latest_rows(rows)
  complete_main = sum(
      1 for profile in main_profiles
      if is_complete(latest.get(profile['run_id']))
  )
  blocked_main = sum(
      1 for profile in main_profiles
      if latest.get(profile['run_id'], {}).get('event') == 'blocked'
  )
  launched_open = sum(
      1 for row in latest.values()
      if row.get('event') == 'launch'
  )
  pending = pending_profiles(goal, rows)
  lines = [
      f'goal={goal["name"]}',
      f'validation={"ok" if ok else "blocked"}: {"; ".join(messages)}',
      f'local_commit={local.get("commit", "")[:12]} dirty={local.get("dirty")}',
      f'remote_commit={remote.get("commit", "")[:12]} dirty={remote.get("dirty")} ok={remote.get("ok")}',
      f'setup_checkpoint_resume_passed={setup_gate_passed(goal, rows)}',
      f'main_completed={complete_main}/72 blocked={blocked_main} open_launches={launched_open}',
      f'pending={len(pending)} next={pending[0]["run_id"] if pending else "none"}',
  ]
  if gpu.get('ok'):
    lines.append(
        f'ncc_free_physical_gpus={gpu["free_gpu_count"]} '
        f'active_steward_jobs={len(gpu["active_steward_jobs"])}'
    )
  else:
    lines.append(f'ncc_gpu_status_error={gpu.get("error", "")}')
  blockers = launch_blockers(goal, local, remote, gpu)
  if blockers:
    lines.append('launch_blockers=' + '; '.join(blockers))
  return '\n'.join(lines)


def campaign_terminal(goal: dict[str, Any], rows: list[dict[str, str]]) -> bool:
  latest = latest_rows(rows)
  for profile in build_main_profiles(goal):
    row = latest.get(profile['run_id'])
    if row is None or row.get('event') not in TERMINAL_EVENTS:
      return False
  return setup_gate_passed(goal, rows)


def tick(goal: dict[str, Any], *, goal_path: Path, dry_run: bool = False) -> str:
  rows = read_ledger(goal)
  processed = 0 if dry_run else process_terminal_jobs(goal, rows)
  sync_messages: list[str] = []
  if processed:
    sync_messages.append(auto_commit_steward_state(goal, 'record terminal job results'))
    rows = read_ledger(goal)
  ok, messages = validate_goal(goal)
  local = local_git_state()
  remote = remote_git_state(goal)
  gpu = ncc_status(goal)
  blockers = [] if ok else messages
  blockers.extend(launch_blockers(goal, local, remote, gpu))
  pending = pending_profiles(goal, rows)
  active = len(gpu.get('active_steward_jobs', []))
  free_physical = int(gpu.get('free_gpu_count', 0))
  max_active = int(goal['constraints']['max_active_gpus'])
  slots = max(0, min(max_active - active, free_physical))
  lines = [
      f'processed_terminal_jobs={processed}',
      f'pending_profiles={len(pending)}',
      f'active_steward_jobs={active}',
      f'free_physical_gpus={free_physical}',
      f'launch_slots={slots}',
  ]
  lines.extend(sync_messages)
  if blockers:
    lines.append('launch_blocked=' + '; '.join(blockers))
    return '\n'.join(lines)
  if slots == 0:
    if campaign_terminal(goal, rows):
      if dry_run:
        lines.append('campaign_terminal=true would_package_results=true')
      else:
        lines.append('campaign_terminal=true')
        lines.append(package_results(goal, goal_path=goal_path))
    lines.append('no_launch_slots_available')
    return '\n'.join(lines)
  if campaign_terminal(goal, rows):
    if dry_run:
      lines.append('campaign_terminal=true would_package_results=true')
    else:
      lines.append('campaign_terminal=true')
      lines.append(package_results(goal, goal_path=goal_path))
    return '\n'.join(lines)
  launched = 0
  for profile in pending[:slots]:
    try:
      line = launch_profile(
          goal,
          profile,
          local_commit=local['commit'],
          remote_commit=remote['commit'],
          dry_run=dry_run,
          rows=rows,
      )
      lines.append(line)
      launched += 1
    except Exception as exc:
      lines.append(f'launch_failed {profile["run_id"]}: {exc}')
      break
  if launched and not dry_run:
    lines.append(auto_commit_steward_state(goal, f'record {launched} launch(es)'))
  return '\n'.join(lines)


def cache_remote_file(
    goal: dict[str, Any],
    remote_file: str,
    local_file: Path,
) -> bool:
  command = (
      f'if test -f {shlex.quote(remote_file)}; then '
      f'(base64 -w 0 {shlex.quote(remote_file)} 2>/dev/null || base64 {shlex.quote(remote_file)}); '
      'fi'
  )
  result = run_remote(goal, command, timeout=180)
  if result.returncode != 0 or not result.stdout.strip():
    return False
  local_file.parent.mkdir(parents=True, exist_ok=True)
  payload = ''.join(result.stdout.split())
  local_file.write_bytes(base64.b64decode(payload))
  return True


def cache_completed_metrics(goal: dict[str, Any]) -> str:
  rows = read_ledger(goal)
  results_dir = relpath(goal['tracking']['results_dir'])
  cached = 0
  for row in rows:
    if row.get('event') not in {'completed', 'partial_complete'}:
      continue
    if row.get('status') not in GOOD_TERMINAL_STATUSES:
      continue
    run_id = row['run_id']
    remote_dir = row.get('run_dir', '')
    if not remote_dir:
      continue
    for rel in (
        'metrics/scalars.csv',
        'metrics/episodes.csv',
        'metrics/horizon_queries.csv',
        'resume_verified.json',
    ):
      if cache_remote_file(
          goal,
          f'{remote_dir}/{rel}',
          results_dir / 'cache' / run_id / rel,
      ):
        cached += 1
  return f'cached_remote_metric_files={cached}'


def package_results(
    goal: dict[str, Any],
    *,
    goal_path: Path,
    report_only: bool = False,
) -> str:
  lines = []
  if not report_only:
    lines.append(cache_completed_metrics(goal))
  script = ROOT / 'scripts' / 'generate_corl_compact_artifacts.py'
  result = run_local(
      [
          sys.executable,
          str(script),
          '--goal',
          str(goal_path),
      ],
      timeout=180,
  )
  if result.returncode != 0:
    raise RuntimeError(result.stderr.strip() or result.stdout.strip())
  lines.append(result.stdout.strip())
  return '\n'.join(line for line in lines if line)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--goal', default=str(DEFAULT_GOAL), help='Path to goal JSON/YAML file.')
  subparsers = parser.add_subparsers(dest='command', required=True)
  subparsers.add_parser('validate')
  subparsers.add_parser('status')
  tick_parser = subparsers.add_parser('tick')
  tick_parser.add_argument('--dry-run', action='store_true')
  subparsers.add_parser('package-results')
  subparsers.add_parser('generate-report')
  args = parser.parse_args(argv)

  goal_path = Path(args.goal)
  if not goal_path.is_absolute():
    goal_path = ROOT / goal_path
  goal = load_goal(goal_path)

  if args.command == 'validate':
    ok, messages = validate_goal(goal)
    print('\n'.join(messages))
    return 0 if ok else 2
  if args.command == 'status':
    print(status_report(goal))
    return 0
  if args.command == 'tick':
    print(tick(goal, goal_path=goal_path, dry_run=args.dry_run))
    return 0
  if args.command == 'package-results':
    print(package_results(goal, goal_path=goal_path))
    return 0
  if args.command == 'generate-report':
    print(package_results(goal, goal_path=goal_path, report_only=True))
    return 0
  parser.error(f'unknown command {args.command}')
  return 2


if __name__ == '__main__':
  raise SystemExit(main())
