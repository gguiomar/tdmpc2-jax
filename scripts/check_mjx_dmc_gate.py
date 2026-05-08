#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, List

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from tdmpc2_jax.envs.mjx_dmc import TASK_DOMAIN, make_mjx_dmc_env


def _jsonable(value: Any) -> Any:
  if isinstance(value, np.ndarray):
    return value.tolist()
  if hasattr(value, 'shape'):
    return np.asarray(value).tolist()
  if isinstance(value, (np.floating, np.integer)):
    return value.item()
  return value


def _make_env_config(task: str,
                     num_envs: int,
                     episode_length: int,
                     reset_pool_size: int,
                     chaos: bool) -> SimpleNamespace:
  mjx_dmc = SimpleNamespace(
      task=task,
      action_repeat=2,
      episode_length=episode_length,
      observation_noise_scale=0.01,
      enable_domain_randomization=chaos,
      enable_observation_noise=chaos,
      base_action_delay=1 if chaos else 0,
      desired_speed=5.0,
      action_repeat_dt=0.02,
      wind_scale=5.0,
      push_scale=25.0,
      slip_scale=0.15,
      jitter_prob=0.02,
      reset_pool_size=reset_pool_size,
  )
  return SimpleNamespace(mjx_dmc=mjx_dmc, num_envs=num_envs)


def _dmcontrol_reference(task: str, actions: np.ndarray) -> Dict[str, Any]:
  try:
    from dm_control import suite
  except Exception as exc:
    return {'available': False, 'error': f'{type(exc).__name__}: {exc}'}

  domain, task_name = TASK_DOMAIN[task]
  try:
    env = suite.load(domain, task_name, task_kwargs={'random': 0}, visualize_reward=False)
    env.reset()
    rewards = []
    spec = env.action_spec()
    for action in actions:
      clipped = np.clip(action, spec.minimum, spec.maximum)
      timestep = env.step(clipped)
      rewards.append(float(timestep.reward or 0.0))
    return {
        'available': True,
        'reward_sum': float(np.sum(rewards)),
        'reward_mean': float(np.mean(rewards)) if rewards else 0.0,
        'reward_finite': bool(np.all(np.isfinite(rewards))),
    }
  except Exception as exc:
    return {'available': False, 'error': f'{type(exc).__name__}: {exc}'}


def run_gate(task: str,
             seed: int,
             num_envs: int,
             steps: int,
             reset_pool_size: int,
             chaos: bool) -> Dict[str, Any]:
  env_config = _make_env_config(
      task=task,
      num_envs=num_envs,
      episode_length=max(steps + 2, 8),
      reset_pool_size=reset_pool_size,
      chaos=chaos,
  )
  env = make_mjx_dmc_env(env_config, seed=seed, num_envs=num_envs)
  obs, _ = env.reset(seed=seed)
  obs = jax.block_until_ready(obs)
  actions_for_reference: List[np.ndarray] = []
  rewards = []
  terminated_counts = []
  truncated_counts = []
  current_obs = obs
  for _ in range(steps):
    action = env.sample_actions()
    actions_for_reference.append(np.asarray(action[0]))
    current_obs, reward, terminated, truncated, _ = env.step(action)
    current_obs = jax.block_until_ready(current_obs)
    reward = jax.block_until_ready(reward)
    rewards.append(np.asarray(reward))
    terminated_counts.append(int(np.asarray(terminated).sum()))
    truncated_counts.append(int(np.asarray(truncated).sum()))

  rewards_arr = np.stack(rewards, axis=0) if rewards else np.zeros((0, num_envs))
  obs_arr = np.asarray(current_obs)
  action_ref = np.stack(actions_for_reference, axis=0) if actions_for_reference else np.zeros((0, env.metadata.action_dim))
  dmcontrol = _dmcontrol_reference(task, action_ref)
  reward_sum = float(np.sum(rewards_arr))
  reward_finite = bool(np.all(np.isfinite(rewards_arr)))
  obs_finite = bool(np.all(np.isfinite(obs_arr)))
  passed = (
      tuple(obs_arr.shape) == (num_envs, env.metadata.observation_dim)
      and env.single_action_space.shape == (env.metadata.action_dim,)
      and reward_finite
      and obs_finite
  )
  return {
      'task': task,
      'seed': seed,
      'num_envs': num_envs,
      'steps': steps,
      'chaos': chaos,
      'passed': bool(passed),
      'observation_shape': list(obs_arr.shape),
      'observation_dim': int(env.metadata.observation_dim),
      'action_shape': list(env.single_action_space.shape),
      'action_dim': int(env.metadata.action_dim),
      'physics_substeps_per_control': int(env.metadata.physics_substeps_per_control),
      'reward_sum': reward_sum,
      'reward_mean': float(np.mean(rewards_arr)) if rewards_arr.size else 0.0,
      'reward_min': float(np.min(rewards_arr)) if rewards_arr.size else math.nan,
      'reward_max': float(np.max(rewards_arr)) if rewards_arr.size else math.nan,
      'reward_finite': reward_finite,
      'observation_finite': obs_finite,
      'terminated_counts': terminated_counts,
      'truncated_counts': truncated_counts,
      'dmcontrol_reference': dmcontrol,
      'jax_backend': jax.default_backend(),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description='Validate TD-MPC2-JAX MJX DMC task gates.')
  parser.add_argument('--tasks', nargs='+', required=True)
  parser.add_argument('--out-dir', required=True)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--num-envs', type=int, default=4)
  parser.add_argument('--steps', type=int, default=16)
  parser.add_argument('--reset-pool-size', type=int, default=8)
  parser.add_argument('--chaos', action='store_true')
  args = parser.parse_args()

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  summaries = []
  for task in args.tasks:
    print(f'Running MJX gate for {task}...', flush=True)
    try:
      summary = run_gate(
          task=task,
          seed=args.seed,
          num_envs=args.num_envs,
          steps=args.steps,
          reset_pool_size=args.reset_pool_size,
          chaos=args.chaos,
      )
    except Exception as exc:
      summary = {
          'task': task,
          'seed': args.seed,
          'num_envs': args.num_envs,
          'steps': args.steps,
          'chaos': args.chaos,
          'passed': False,
          'error': f'{type(exc).__name__}: {exc}',
          'jax_backend': jax.default_backend(),
      }
    summaries.append(summary)
    task_dir = out_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / 'mjx_gate.json').write_text(
        json.dumps(summary, indent=2, default=_jsonable),
        encoding='utf-8',
    )
    print(json.dumps({
        'task': task,
        'passed': summary['passed'],
        'reward_sum': summary.get('reward_sum'),
        'obs': summary.get('observation_shape'),
        'action': summary.get('action_shape'),
        'error': summary.get('error'),
    }), flush=True)

  aggregate = {
      'passed': all(item['passed'] for item in summaries),
      'tasks': summaries,
  }
  (out_dir / 'mjx_gate_summary.json').write_text(
      json.dumps(aggregate, indent=2, default=_jsonable),
      encoding='utf-8',
  )
  if not aggregate['passed']:
    raise SystemExit(1)


if __name__ == '__main__':
  main()
