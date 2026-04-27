import csv
import os
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import flax.linen as nn
import gymnasium as gym
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tqdm
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from tdmpc2_jax import (
    HorizonSearchState,
    TDMPC2,
    WorldModel,
    build_dense_query_kernels,
    dense_checkpoint_eval,
    prewarm_dense_rhs_kernels,
)
from tdmpc2_jax.common.activations import mish, simnorm
from tdmpc2_jax.data import (
    SequentialReplayBuffer,
    insert_into_state,
    sample_from_state,
    sample_many_from_state,
)
from tdmpc2_jax.envs import make_dm_control_env, make_mjx_dmc_env
from tdmpc2_jax.networks import NormedLinear

try:
  from flax.metrics import tensorboard
except Exception:  # pragma: no cover - optional runtime dependency
  tensorboard = None

try:
  import tensorflow as tf  # Tensorboard helper only.
except Exception:  # pragma: no cover - optional runtime dependency
  tf = None

try:
  import wandb
except Exception:  # pragma: no cover - optional runtime dependency
  wandb = None

if tf is not None:
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class _NullSummaryWriter:
  def hparams(self, *_args, **_kwargs):
    return None

  def scalar(self, *_args, **_kwargs):
    return None

  def flush(self):
    return None

  def close(self):
    return None


class _ArtifactWriter:
  def __init__(self, output_dir: str):
    metrics_dir = Path(output_dir) / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)

    self._scalar_file = (metrics_dir / 'scalars.csv').open('w', newline='')
    self._scalar_writer = csv.DictWriter(
        self._scalar_file,
        fieldnames=['step', 'tag', 'value'],
    )
    self._scalar_writer.writeheader()

    self._episode_file = (metrics_dir / 'episodes.csv').open('w', newline='')
    self._episode_writer = csv.DictWriter(
        self._episode_file,
        fieldnames=[
            'step',
            'env_index',
            'episode_index',
            'episode_return',
            'episode_length',
            'selected_horizon',
        ],
    )
    self._episode_writer.writeheader()

    self._query_file = (metrics_dir / 'horizon_queries.csv').open('w', newline='')
    self._query_writer = csv.DictWriter(
        self._query_file,
        fieldnames=[
            'step',
            'previous_horizon',
            'selected_horizon',
            'proposed_horizon',
            'best_h',
            'phase_id',
            'phase_name',
            'num_active_horizons',
            'num_candidate_horizons',
            'entropy',
            'norm_entropy',
            'prob_best_h',
            'gauss_mean_best_h',
            'gauss_post_std_best_h',
            'best_fitness',
            'deployment_score_best',
            'incumbent_deployment_score',
            'proposed_deployment_score',
            'return_term_best',
            'roughness_term_best',
            'return_std_term_best',
            'robust_return_best',
            'query_total_s',
            'query_model_diag_s',
            'query_env_eval_s',
        ],
    )
    self._query_writer.writeheader()
    self._pending_scalars = []
    self._pending_episodes = []
    self._pending_queries = []

  def scalar(self, tag: str, value: float, step: int):
    self._pending_scalars.append(
        {'step': int(step), 'tag': str(tag), 'value': float(value)}
    )

  def episode(self,
              step: int,
              env_index: int,
              episode_index: int,
              episode_return: float,
              episode_length: int,
              selected_horizon: int):
    self._pending_episodes.append(
        {
            'step': int(step),
            'env_index': int(env_index),
            'episode_index': int(episode_index),
            'episode_return': float(episode_return),
            'episode_length': int(episode_length),
            'selected_horizon': int(selected_horizon),
        }
    )

  def horizon_query(self, row):
    self._pending_queries.append(row)

  def flush(self):
    for row in self._pending_scalars:
      self._scalar_writer.writerow(row)
    for row in self._pending_episodes:
      self._episode_writer.writerow(row)
    for row in self._pending_queries:
      self._query_writer.writerow(row)
    self._pending_scalars.clear()
    self._pending_episodes.clear()
    self._pending_queries.clear()
    self._scalar_file.flush()
    self._episode_file.flush()
    self._query_file.flush()

  def close(self):
    self._scalar_file.close()
    self._episode_file.close()
    self._query_file.close()


class _CompositeSummaryWriter:
  def __init__(self, output_dir: str, tb_writer=None, wandb_run=None):
    self._tb_writer = tb_writer
    self._wandb_run = wandb_run
    self._artifact_writer = _ArtifactWriter(output_dir)
    self._wandb_pending = defaultdict(dict)

  def hparams(self, cfg):
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    if self._tb_writer is not None:
      self._tb_writer.hparams(config_dict)
    if self._wandb_run is not None:
      self._wandb_run.config.update(config_dict, allow_val_change=True)

  def scalar(self, tag, value, step):
    scalar_value = float(np.asarray(value))
    scalar_step = int(step)
    if self._tb_writer is not None:
      self._tb_writer.scalar(tag, scalar_value, scalar_step)
    self._artifact_writer.scalar(tag, scalar_value, scalar_step)
    if self._wandb_run is not None:
      self._wandb_pending[scalar_step][str(tag)] = scalar_value

  def scalar_dict(self, values, step):
    for tag, value in values.items():
      self.scalar(tag, value, step)

  def episode(self, **kwargs):
    self._artifact_writer.episode(**kwargs)

  def horizon_query(self, **kwargs):
    self._artifact_writer.horizon_query(kwargs)

  def flush(self):
    self._artifact_writer.flush()
    if self._wandb_run is not None:
      for step in sorted(self._wandb_pending):
        self._wandb_run.log(self._wandb_pending[step], step=int(step))
      self._wandb_pending.clear()
    if self._tb_writer is not None:
      self._tb_writer.flush()

  def close(self):
    self._artifact_writer.close()
    if self._tb_writer is not None:
      self._tb_writer.close()
    if self._wandb_run is not None:
      self._wandb_run.finish()


def _make_zero_plan(agent: TDMPC2, batch_shape):
  plan_shape = tuple(batch_shape) + (
      int(agent.planning_hmax),
      int(agent.model.action_dim),
  )
  return (
      jnp.zeros(plan_shape, dtype=jnp.float32),
      jnp.full(plan_shape, agent.max_plan_std, dtype=jnp.float32),
  )


def _reset_plan_for_done(plan, done_mask, max_plan_std: float):
  expanded_done = jnp.reshape(
      jnp.asarray(done_mask, dtype=bool),
      done_mask.shape + (1, 1),
  )
  return (
      jnp.where(expanded_done, 0.0, plan[0]),
      jnp.where(expanded_done, jnp.asarray(max_plan_std, dtype=jnp.float32), plan[1]),
  )


def _horizon_buckets_from_config(dense_rhs_config) -> tuple[int, ...]:
  buckets = dense_rhs_config.get('horizon_buckets', None)
  if buckets is None:
    return ()
  return tuple(sorted({int(bucket) for bucket in buckets if int(bucket) > 0}))


def _bucket_for_horizon(horizon: int, horizon_buckets: tuple[int, ...]) -> int:
  horizon = int(horizon)
  for bucket in horizon_buckets:
    if horizon <= bucket:
      return int(bucket)
  return int(horizon_buckets[-1]) if horizon_buckets else horizon


def _make_training_horizon_agent(agent: TDMPC2,
                                 horizon: int,
                                 horizon_buckets: tuple[int, ...] = ()) -> TDMPC2:
  horizon = int(horizon)
  bucket_horizon = _bucket_for_horizon(horizon, horizon_buckets)
  return agent.replace(horizon=bucket_horizon, planning_hmax=bucket_horizon)


@partial(jax.jit, static_argnames=('num_updates', 'batch_size', 'sequence_length'))
def _run_train_chunk(agent,
                     buffer_state,
                     key,
                     *,
                     num_updates: int,
                     batch_size: int,
                     sequence_length: int,
                     train_horizon: int):
  buffer_state, batch = sample_many_from_state(
      buffer_state,
      num_updates=num_updates,
      batch_size=batch_size,
      sequence_length=sequence_length,
  )
  update_keys = jax.random.split(key, num_updates)
  agent, train_info = agent.update_many(
      observations=batch['observation'],
      actions=batch['action'],
      rewards=batch['reward'],
      next_observations=batch['next_observation'],
      terminated=batch['terminated'],
      truncated=batch['truncated'],
      keys=update_keys,
      train_horizon=jnp.asarray(train_horizon, dtype=jnp.int32),
  )
  return agent, buffer_state, train_info


def _accumulate_train_info_host(accumulator, train_info):
  for k, v in train_info.items():
    arr = np.asarray(v, dtype=np.float32)
    accumulator[k]['sum'] += float(arr.sum())
    accumulator[k]['sum_sq'] += float(np.square(arr).sum())
    accumulator[k]['count'] += int(arr.size)


def _log_chunk_episodes(writer,
                        *,
                        chunk_start_step: int,
                        num_envs: int,
                        ep_count: np.ndarray,
                        chunk_logs,
                        selected_horizon: int):
  done = np.asarray(chunk_logs['done'])
  episode_return = np.asarray(chunk_logs['episode_return'], dtype=np.float32)
  episode_length = np.asarray(chunk_logs['episode_length'], dtype=np.int32)
  for step_idx, env_idx in np.argwhere(done):
    event_step = int(chunk_start_step + step_idx * num_envs + env_idx)
    r = float(episode_return[step_idx, env_idx])
    l = int(episode_length[step_idx, env_idx])
    print(f"Episode {ep_count[env_idx]}: r = {r:.2f}, l = {l}")
    writer.episode(
        step=event_step,
        env_index=int(env_idx),
        episode_index=int(ep_count[env_idx]),
        episode_return=r,
        episode_length=l,
        selected_horizon=int(selected_horizon),
    )
    writer.scalar('episode/return', r, event_step)
    writer.scalar('episode/length', l, event_step)
    ep_count[env_idx] += 1


def build_mjx_seed_chunk_fn(env, *, chunk_steps: int):
  action_dim = int(env._metadata.action_dim)

  @jax.jit
  def _run_chunk(env_state,
                 observation,
                 buffer_state,
                 rng):
    def step_fn(carry, _):
      env_state, observation, buffer_state, rng = carry
      rng, action_key = jax.random.split(rng)
      action = jax.random.uniform(
          action_key,
          shape=observation.shape[:-1] + (action_dim,),
          minval=-1.0,
          maxval=1.0,
          dtype=jnp.float32,
      )
      (
          next_env_state,
          transition_next_observation,
          next_observation,
          reward,
          terminated,
          truncated,
          episode_return,
          episode_length,
          done,
      ) = env._step_autoreset_state(env_state, action)
      next_buffer_state = insert_into_state(
          buffer_state,
          dict(
              observation=observation,
              action=action,
              reward=reward,
              next_observation=transition_next_observation,
              terminated=terminated,
              truncated=truncated,
          ),
          mask=jnp.ones_like(done, dtype=bool),
      )
      next_carry = (
          next_env_state,
          next_observation,
          next_buffer_state,
          rng,
      )
      step_logs = {
          'done': done,
          'episode_return': episode_return,
          'episode_length': episode_length,
      }
      return next_carry, step_logs

    final_carry, chunk_logs = jax.lax.scan(
        step_fn,
        (env_state, observation, buffer_state, rng),
        xs=None,
        length=chunk_steps,
    )
    return (*final_carry, chunk_logs)

  return _run_chunk


def build_mjx_collect_step_fn(env):
  @jax.jit
  def _run_step(agent,
                env_state,
                observation,
                buffer_state,
                plan,
                rng,
                *,
                horizon: int):
    rng, action_key = jax.random.split(rng)
    action, next_plan = agent.act(
        observation,
        prev_plan=plan,
        deterministic=False,
        train=True,
        horizon=horizon,
        key=action_key,
    )
    (
        next_env_state,
        transition_next_observation,
        next_observation,
        reward,
        terminated,
        truncated,
        episode_return,
        episode_length,
        done,
    ) = env._step_autoreset_state(env_state, action)
    next_plan = _reset_plan_for_done(next_plan, done, agent.max_plan_std)
    next_buffer_state = insert_into_state(
        buffer_state,
        dict(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=transition_next_observation,
            terminated=terminated,
            truncated=truncated,
        ),
        mask=jnp.ones_like(done, dtype=bool),
    )
    step_logs = {
        'done': done,
        'episode_return': episode_return,
        'episode_length': episode_length,
    }
    return (
        agent,
        next_env_state,
        next_observation,
        next_buffer_state,
        next_plan,
        rng,
        step_logs,
    )

  return _run_step


def build_mjx_collect_chunk_fn(env, *, chunk_steps: int):
  @jax.jit
  def _run_chunk(agent,
                 env_state,
                 observation,
                 buffer_state,
                 plan,
                 rng,
                 *,
                 horizon: int):
    def step_fn(carry, _):
      env_state, observation, buffer_state, plan, rng = carry
      rng, action_key = jax.random.split(rng)
      action, next_plan = agent.act(
          observation,
          prev_plan=plan,
          deterministic=False,
          train=True,
          horizon=horizon,
          key=action_key,
      )
      (
          next_env_state,
          transition_next_observation,
          next_observation,
          reward,
          terminated,
          truncated,
          episode_return,
          episode_length,
          done,
      ) = env._step_autoreset_state(env_state, action)
      next_plan = _reset_plan_for_done(next_plan, done, agent.max_plan_std)
      next_buffer_state = insert_into_state(
          buffer_state,
          dict(
              observation=observation,
              action=action,
              reward=reward,
              next_observation=transition_next_observation,
              terminated=terminated,
              truncated=truncated,
          ),
          mask=jnp.ones_like(done, dtype=bool),
      )
      step_logs = {
          'done': done,
          'episode_return': episode_return,
          'episode_length': episode_length,
      }
      return (
          next_env_state,
          next_observation,
          next_buffer_state,
          next_plan,
          rng,
      ), step_logs

    final_carry, chunk_logs = jax.lax.scan(
        step_fn,
        (env_state, observation, buffer_state, plan, rng),
        xs=None,
        length=chunk_steps,
    )
    return (agent, *final_carry, chunk_logs)

  return _run_chunk


def _run_mjx_training_loop(cfg,
                           env_config,
                           eval_config,
                           dense_rhs_config,
                           env,
                           periodic_eval_env,
                           dense_query_eval_state,
                           agent,
                           buffer_state,
                           writer,
                           mngr,
                           global_step: int,
                           last_saved_step: int,
                           *,
                           seed_steps: int,
                           update_chunk_size: int,
                           horizon_state=None,
                           dense_query_kernels=None):
  num_envs = int(env_config.num_envs)
  collect_chunk_steps = int(cfg.get('collect_chunk_steps', 100))
  chunk_global_steps = collect_chunk_steps * num_envs
  num_updates_per_step = max(1, int(num_envs * env_config.utd_ratio))
  run_mjx_seed_chunk = build_mjx_seed_chunk_fn(
      env,
      chunk_steps=collect_chunk_steps,
  )
  run_mjx_collect_step = build_mjx_collect_step_fn(env)
  run_mjx_collect_chunk = build_mjx_collect_chunk_fn(
      env,
      chunk_steps=collect_chunk_steps,
  )

  ep_count = np.zeros(num_envs, dtype=int)
  prev_logged_step = int(global_step)
  train_info_accumulator = defaultdict(
      lambda: {'sum': 0.0, 'sum_sq': 0.0, 'count': 0}
  )
  collect_time_since_log = 0.0
  train_time_since_log = 0.0
  last_eval_step = -1
  plan = None
  seed_pretraining_done = bool(global_step > seed_steps)
  rng = jax.random.PRNGKey(cfg.seed + int(global_step))
  observation, _ = env.reset(seed=cfg.seed)
  pbar = tqdm.tqdm(initial=global_step, total=cfg.max_steps)
  dense_rhs_enabled = horizon_state is not None
  horizon_buckets = (
      _horizon_buckets_from_config(dense_rhs_config)
      if dense_rhs_enabled else ()
  )
  selected_horizon = (
      int(np.asarray(horizon_state.best_h))
      if horizon_state is not None else int(agent.horizon)
  )
  agent = _make_training_horizon_agent(agent, selected_horizon, horizon_buckets)

  def run_update_batches(total_updates: int, *, step_for_logs: int):
    nonlocal agent, buffer_state, rng, train_time_since_log, train_info_accumulator

    updates_completed = 0
    while updates_completed < total_updates:
      chunk_updates = min(update_chunk_size, total_updates - updates_completed)
      rng, update_key = jax.random.split(rng)
      train_start = time.perf_counter()
      agent, buffer_state, train_info = _run_train_chunk(
          agent,
          buffer_state,
          update_key,
          num_updates=chunk_updates,
          batch_size=int(agent.batch_size),
          sequence_length=int(agent.horizon),
          train_horizon=int(selected_horizon),
      )
      train_time_since_log += time.perf_counter() - train_start
      updates_completed += chunk_updates
      _accumulate_train_info_host(train_info_accumulator, train_info)

      if (
          step_for_logs == seed_steps and
          updates_completed % int(cfg.seed_pretrain_log_interval_updates) == 0
      ):
        writer.scalar('seed_pretrain/updates_completed', updates_completed, step_for_logs)
        writer.scalar('timing/collect_chunk_s', collect_time_since_log, step_for_logs)
        writer.scalar('timing/train_chunk_s', train_time_since_log, step_for_logs)
        writer.scalar('system/heartbeat', 1.0, step_for_logs)
        writer.flush()

  def run_dense_query(step_for_query: int):
    nonlocal agent, buffer_state, rng, plan, horizon_state, selected_horizon

    if horizon_state is None:
      return
    previous_horizon = int(selected_horizon)
    rng, query_key = jax.random.split(rng)
    buffer_state, query_batch = sample_from_state(
        buffer_state,
        batch_size=int(agent.batch_size),
        sequence_length=int(horizon_state.hmax),
    )
    query_start = time.perf_counter()
    dense_query_agent = _make_dense_rhs_query_agent(agent, dense_rhs_config)
    horizon_state, selected_horizon, dense_metrics = dense_checkpoint_eval(
        agent=dense_query_agent,
        replay_batch=query_batch,
        eval_state=dense_query_eval_state,
        horizon_state=horizon_state,
        rng=query_key,
        env_eval_steps=int(dense_rhs_config.env_eval_steps),
        query_step=int(step_for_query),
        dense_query_kernels=dense_query_kernels,
    )
    agent = _make_training_horizon_agent(
        agent,
        selected_horizon,
        horizon_buckets,
    )
    plan = None
    dense_metrics['timing/query_total_s'] = max(
        float(dense_metrics.get('timing/query_total_s', 0.0)),
        time.perf_counter() - query_start,
    )
    for metric_name, metric_value in dense_metrics.items():
      if isinstance(metric_value, str):
        continue
      writer.scalar(metric_name, float(metric_value), step_for_query)
    best_h = int(np.asarray(horizon_state.best_h))
    horizons_np = np.asarray(horizon_state.horizons)
    best_idx = int(np.where(horizons_np == best_h)[0][0])
    query_row = {
        'step': int(step_for_query),
        'previous_horizon': int(previous_horizon),
        'selected_horizon': int(selected_horizon),
        'proposed_horizon': int(dense_metrics['dense_rhs/proposed_horizon']),
        'best_h': int(best_h),
        'phase_id': int(np.asarray(horizon_state.phase_id)),
        'phase_name': horizon_state.phase_name(),
        'num_active_horizons': int(np.sum(np.asarray(horizon_state.active_mask))),
        'num_candidate_horizons': int(dense_metrics['dense_rhs/num_candidate_horizons']),
        'entropy': float(np.asarray(horizon_state.entropy)),
        'norm_entropy': float(np.asarray(horizon_state.norm_entropy)),
        'prob_best_h': float(np.asarray(horizon_state.prob)[best_idx]),
        'gauss_mean_best_h': float(np.asarray(horizon_state.gauss_mean)[best_idx]),
        'gauss_post_std_best_h': float(np.asarray(horizon_state.gauss_post_std)[best_idx]),
        'best_fitness': float(dense_metrics['dense_rhs/best_fitness']),
        'deployment_score_best': float(
            dense_metrics['dense_rhs/deployment_score_best']
        ),
        'incumbent_deployment_score': float(
            dense_metrics['dense_rhs/incumbent_deployment_score']
        ),
        'proposed_deployment_score': float(
            dense_metrics['dense_rhs/proposed_deployment_score']
        ),
        'return_term_best': float(dense_metrics['dense_rhs/return_term_best']),
        'roughness_term_best': float(
            dense_metrics['dense_rhs/roughness_term_best']
        ),
        'return_std_term_best': float(
            dense_metrics['dense_rhs/return_std_term_best']
        ),
        'robust_return_best': float(dense_metrics['dense_rhs/robust_return_best']),
        'query_total_s': float(dense_metrics['timing/query_total_s']),
        'query_model_diag_s': float(dense_metrics['timing/query_model_diag_s']),
        'query_env_eval_s': float(dense_metrics['timing/query_env_eval_s']),
    }
    writer.horizon_query(**query_row)
    writer.scalar('dense_rhs/previous_horizon', query_row['previous_horizon'], step_for_query)
    writer.scalar('dense_rhs/training_bucket_horizon', float(agent.horizon), step_for_query)
    writer.scalar('dense_rhs/best_h', query_row['best_h'], step_for_query)
    writer.scalar('dense_rhs/prob_best_h', query_row['prob_best_h'], step_for_query)
    writer.scalar('dense_rhs/gauss_mean_best_h', query_row['gauss_mean_best_h'], step_for_query)
    writer.scalar(
        'dense_rhs/gauss_post_std_best_h',
        query_row['gauss_post_std_best_h'],
        step_for_query,
    )
    writer.scalar('system/heartbeat', 1.0, step_for_query)
    writer.flush()

  while global_step < cfg.max_steps:
    if global_step >= seed_steps and not seed_pretraining_done:
      print('Pre-training on seed data...')
      run_update_batches(seed_steps, step_for_logs=global_step)
      seed_pretraining_done = True
      continue

    if (
        dense_rhs_enabled and
        global_step >= seed_steps and
        horizon_state is not None and
        horizon_state.should_query(global_step)
    ):
      run_dense_query(global_step)
      continue

    next_log_step = prev_logged_step + int(cfg['log_interval_steps'])
    next_save_step = last_saved_step + int(cfg['save_interval_steps'])
    next_eval_step = cfg.max_steps + chunk_global_steps
    if periodic_eval_env is not None and eval_config is not None and bool(eval_config.enabled):
      eval_interval = int(eval_config.interval_steps)
      next_eval_step = ((int(global_step) // eval_interval) + 1) * eval_interval
    next_query_step = cfg.max_steps + chunk_global_steps
    if dense_rhs_enabled and horizon_state is not None:
      next_query_step = max(int(np.asarray(horizon_state.next_query_step)), int(seed_steps))
    next_boundary_step = min(
        next_log_step,
        next_save_step,
        next_eval_step,
        next_query_step,
        int(cfg.max_steps),
    )

    can_run_seed_chunk = (
        global_step < seed_steps and
        global_step + chunk_global_steps <= min(next_boundary_step, seed_steps)
    )
    can_run_chunk = (
        global_step >= seed_steps and
        global_step + chunk_global_steps <= next_boundary_step and
        global_step + chunk_global_steps <= int(cfg.max_steps)
    )

    if can_run_seed_chunk:
      seed_start = time.perf_counter()
      (
          env.state,
          observation,
          buffer_state,
          rng,
          chunk_logs,
      ) = run_mjx_seed_chunk(
          env.state,
          observation,
          buffer_state,
          rng,
      )
      collect_time_since_log += time.perf_counter() - seed_start
      _log_chunk_episodes(
          writer,
          chunk_start_step=global_step,
          num_envs=num_envs,
          ep_count=ep_count,
          chunk_logs=chunk_logs,
          selected_horizon=int(selected_horizon),
      )
      global_step += chunk_global_steps
      pbar.update(chunk_global_steps)
    elif can_run_chunk:
      if plan is None:
        plan = _make_zero_plan(agent, observation.shape[:-1])
      collect_start = time.perf_counter()
      (
          _agent,
          env.state,
          observation,
          buffer_state,
          plan,
          rng,
          chunk_logs,
      ) = run_mjx_collect_chunk(
          agent,
          env.state,
          observation,
          buffer_state,
          plan,
          rng,
          horizon=int(selected_horizon),
      )
      collect_time_since_log += time.perf_counter() - collect_start
      _log_chunk_episodes(
          writer,
          chunk_start_step=global_step,
          num_envs=num_envs,
          ep_count=ep_count,
          chunk_logs=chunk_logs,
          selected_horizon=int(selected_horizon),
      )
      run_update_batches(
          collect_chunk_steps * num_updates_per_step,
          step_for_logs=global_step + chunk_global_steps,
      )
      global_step += chunk_global_steps
      pbar.update(chunk_global_steps)
    else:
      collect_start = time.perf_counter()
      if global_step < seed_steps:
        action = env.sample_actions()
      else:
        if plan is None:
          plan = _make_zero_plan(agent, observation.shape[:-1])
        rng, action_key = jax.random.split(rng)
        action, plan = agent.act(
            observation,
            prev_plan=plan,
            deterministic=False,
            train=True,
            horizon=int(selected_horizon),
            key=action_key,
        )

      (
          env.state,
          transition_next_observation,
          next_observation,
          reward,
          terminated,
          truncated,
          episode_return,
          episode_length,
          done,
      ) = env._step_autoreset_state(env.state, jnp.asarray(action, dtype=jnp.float32))
      buffer_state = insert_into_state(
          buffer_state,
          dict(
              observation=observation,
              action=action,
              reward=reward,
              next_observation=transition_next_observation,
              terminated=terminated,
              truncated=truncated,
          ),
          mask=jnp.ones_like(done, dtype=bool),
      )
      observation = next_observation
      collect_time_since_log += time.perf_counter() - collect_start
      if plan is not None:
        plan = _reset_plan_for_done(plan, done, agent.max_plan_std)
      done_np = np.asarray(done)
      if np.any(done_np):
        fallback_logs = {
            'done': done_np[None, ...],
            'episode_return': np.asarray(episode_return, dtype=np.float32)[None, ...],
            'episode_length': np.asarray(episode_length, dtype=np.int32)[None, ...],
        }
        _log_chunk_episodes(
            writer,
            chunk_start_step=global_step,
            num_envs=num_envs,
            ep_count=ep_count,
            chunk_logs=fallback_logs,
            selected_horizon=int(selected_horizon),
        )

      if global_step >= seed_steps:
        run_update_batches(
            num_updates_per_step,
            step_for_logs=global_step + num_envs,
        )

      global_step += num_envs
      pbar.update(num_envs)

    if global_step >= seed_steps and global_step >= next_log_step:
      for k, stats in train_info_accumulator.items():
        mean = stats['sum'] / max(stats['count'], 1)
        var = max(stats['sum_sq'] / max(stats['count'], 1) - mean**2, 0.0)
        writer.scalar(f'train/{k}_mean', mean, global_step)
        writer.scalar(f'train/{k}_std', np.sqrt(var), global_step)
      writer.scalar('timing/collect_chunk_s', collect_time_since_log, global_step)
      writer.scalar('timing/train_chunk_s', train_time_since_log, global_step)
      writer.scalar('dense_rhs/training_bucket_horizon', float(agent.horizon), global_step)
      writer.scalar('system/heartbeat', 1.0, global_step)
      writer.flush()
      prev_logged_step = int(global_step)
      train_info_accumulator = defaultdict(
          lambda: {'sum': 0.0, 'sum_sq': 0.0, 'count': 0}
      )
      collect_time_since_log = 0.0
      train_time_since_log = 0.0

    should_save = (
        global_step == 0 or
        global_step - last_saved_step >= int(cfg['save_interval_steps']) or
        global_step >= int(cfg.max_steps)
    )
    if should_save:
      checkpoint_start = time.perf_counter()
      save_args = {
          'agent': ocp.args.StandardSave(agent),
          'global_step': ocp.args.JsonSave(global_step),
      }
      if bool(cfg.get('checkpoint_buffer', True)):
        save_args['buffer_state'] = ocp.args.StandardSave(buffer_state)
      if horizon_state is not None:
        save_args['horizon_state'] = ocp.args.StandardSave(horizon_state)
      mngr.save(global_step, args=ocp.args.Composite(**save_args))
      mngr.wait_until_finished()
      writer.scalar(
          'timing/checkpoint_blocking_s',
          time.perf_counter() - checkpoint_start,
          global_step,
      )
      last_saved_step = int(global_step)

    should_eval = (
        periodic_eval_env is not None and
        eval_config is not None and
        bool(eval_config.enabled) and
        global_step > 0 and
        int(global_step) != int(last_eval_step) and
        (
            int(global_step) % int(eval_config.interval_steps) == 0 or
            global_step >= int(cfg.max_steps)
        )
    )
    if should_eval:
      eval_start = time.perf_counter()
      eval_metrics = _run_periodic_eval(
          periodic_eval_env,
          agent,
          horizon=int(selected_horizon),
          num_episodes=int(eval_config.num_episodes),
          steps_per_episode=int(periodic_eval_env._metadata.episode_length),
          key=jax.random.PRNGKey(global_step + 20_000 + cfg.seed),
      )
      for metric_name, metric_value in eval_metrics.items():
        writer.scalar(metric_name, metric_value, global_step)
      writer.scalar('timing/eval_s', time.perf_counter() - eval_start, global_step)
      writer.scalar('eval/selected_horizon', float(selected_horizon), global_step)
      writer.scalar('eval/training_bucket_horizon', float(agent.horizon), global_step)
      writer.scalar('system/heartbeat', 1.0, global_step)
      writer.flush()
      last_eval_step = int(global_step)

  pbar.close()
  writer.flush()
  return agent, buffer_state, global_step, last_saved_step


def _make_dense_rhs_query_agent(agent: TDMPC2, dense_rhs_config) -> TDMPC2:
  query_population_size = max(
      1,
      int(dense_rhs_config.get('query_population_size', agent.population_size)),
  )
  query_policy_prior_samples = min(
      max(0, int(dense_rhs_config.get('query_policy_prior_samples', agent.policy_prior_samples))),
      query_population_size,
  )
  query_num_elites = min(
      max(1, int(dense_rhs_config.get('query_num_elites', agent.num_elites))),
      query_population_size,
  )
  query_mppi_iterations = max(
      1,
      int(dense_rhs_config.get('query_mppi_iterations', agent.mppi_iterations)),
  )
  query_temperature = float(
      dense_rhs_config.get('query_temperature', agent.temperature)
  )
  return agent.replace(
      population_size=query_population_size,
      policy_prior_samples=query_policy_prior_samples,
      num_elites=query_num_elites,
      mppi_iterations=query_mppi_iterations,
      temperature=query_temperature,
      planning_hmax=int(dense_rhs_config.hmax),
  )


def _make_mjx_eval_env_config(env_config, *, num_envs: int, clean: bool):
  eval_env_config = OmegaConf.create(
      OmegaConf.to_container(env_config, resolve=True),
  )
  eval_env_config.num_envs = int(num_envs)
  if clean and str(eval_env_config.backend) == 'mjx_dmc':
    eval_env_config.mjx_dmc.enable_domain_randomization = False
    eval_env_config.mjx_dmc.enable_observation_noise = False
  return eval_env_config


def _run_periodic_eval(eval_env,
                       agent: TDMPC2,
                       *,
                       horizon: int,
                       num_episodes: int,
                       steps_per_episode: int,
                       key: jax.Array):
  returns = eval_env.run_eval_chunk(
      agent=agent,
      horizon=int(horizon),
      key=key,
      steps_per_episode=int(steps_per_episode),
  )
  returns_np = np.asarray(returns, dtype=np.float32).reshape(-1)[:int(num_episodes)]
  return {
      'eval/return_mean': float(np.mean(returns_np)),
      'eval/return_std': float(np.std(returns_np)),
      'eval/return_max': float(np.max(returns_np)),
      'eval/return_min': float(np.min(returns_np)),
  }


@hydra.main(config_name='config', config_path='.', version_base=None)
def train(cfg: dict):
  env_config = cfg['env']
  eval_config = cfg.get('evaluation', None)
  encoder_config = cfg['encoder']
  dense_rhs_config = cfg['dense_rhs']
  model_config = cfg['world_model']
  tdmpc_config = cfg['tdmpc2']

  ##############################
  # Logger setup
  ##############################
  output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
  tb_writer = None
  if tensorboard is not None:
    tb_writer = tensorboard.SummaryWriter(os.path.join(output_dir, 'tensorboard'))

  wandb_run = None
  wandb_cfg = cfg.get('wandb', None)
  if wandb_cfg is not None and bool(wandb_cfg.enabled):
    if wandb is None:
      raise ImportError('wandb.enabled=true but wandb is not installed in the active environment.')
    run_name = str(wandb_cfg.name) if wandb_cfg.name is not None else Path(output_dir).name
    tags = list(wandb_cfg.tags) if wandb_cfg.tags is not None else None
    wandb_run = wandb.init(
        project=str(wandb_cfg.project),
        entity=(None if wandb_cfg.entity is None else str(wandb_cfg.entity)),
        group=(None if wandb_cfg.group is None else str(wandb_cfg.group)),
        name=run_name,
        tags=tags,
        mode=str(wandb_cfg.mode),
        dir=output_dir,
    )
    if wandb_run is not None and getattr(wandb_run, 'url', None):
      print(f'W&B run: {wandb_run.url}')

  writer = _CompositeSummaryWriter(
      output_dir=output_dir,
      tb_writer=tb_writer,
      wandb_run=wandb_run,
  )
  writer.hparams(cfg)

  ##############################
  # Environment setup
  ##############################
  periodic_eval_env = None

  def make_env(env_config, seed):
    def make_gym_env(env_id, seed):
      env = gym.make(env_id)
      env = gym.wrappers.RescaleAction(env, min_action=-1, max_action=1)
      env = gym.wrappers.RecordEpisodeStatistics(env)
      env = gym.wrappers.Autoreset(env)
      env.action_space.seed(seed)
      env.observation_space.seed(seed)
      return env

    if env_config.backend == "gymnasium":
      return make_gym_env(env_config.env_id, seed)
    elif env_config.backend == "dmc":
      env = make_dm_control_env(env_config.env_id, seed, env_config.dmc.obs_type)
      env = gym.wrappers.RecordEpisodeStatistics(env)
      env = gym.wrappers.Autoreset(env)
      env.action_space.seed(seed)
      env.observation_space.seed(seed)
      return env
    elif env_config.backend == "mjx_dmc":
      return make_mjx_dmc_env(env_config, seed, num_envs=env_config.num_envs)
    else:
      raise ValueError("Environment not supported:", env_config)

  if env_config.backend == "mjx_dmc":
    env = make_env(env_config, cfg.seed)
    dense_rhs_eval_env = None
    if dense_rhs_config.enabled and bool(dense_rhs_config.env_eval_enabled):
      dense_rhs_eval_env = make_mjx_dmc_env(
          env_config,
          cfg.seed + 10_000,
          num_envs=dense_rhs_config.num_env_eval_replicas,
      )
    if eval_config is not None and bool(eval_config.enabled):
      periodic_eval_env_config = _make_mjx_eval_env_config(
          env_config,
          num_envs=int(eval_config.num_episodes),
          clean=bool(eval_config.clean),
      )
      periodic_eval_env = make_mjx_dmc_env(
          periodic_eval_env_config,
          cfg.seed + 20_000,
          num_envs=int(eval_config.num_episodes),
      )
  else:
    if env_config.asynchronous:
      vector_env_cls = gym.vector.AsyncVectorEnv
    else:
      vector_env_cls = gym.vector.SyncVectorEnv
    env = vector_env_cls(
        [
            partial(make_env, env_config, seed)
            for seed in range(cfg.seed, cfg.seed+env_config.num_envs)
        ]
    )
    dense_rhs_eval_env = None
  dense_query_eval_state = None
  if dense_rhs_config.enabled:
    dense_query_eval_state = dense_rhs_eval_env if bool(dense_rhs_config.env_eval_enabled) else None
    if dense_query_eval_state is None and bool(dense_rhs_config.env_eval_enabled):
      dense_query_eval_state = env
  np.random.seed(cfg.seed)
  rng = jax.random.PRNGKey(cfg.seed)

  ##############################
  # Agent setup
  ##############################
  dtype = jnp.dtype(model_config.dtype)
  rng, model_key, encoder_key = jax.random.split(rng, 3)
  encoder_module = nn.Sequential(
      [
          NormedLinear(
              encoder_config.encoder_dim, activation=mish, dtype=dtype
          )
          for _ in range(encoder_config.num_encoder_layers-1)
      ] + [
          NormedLinear(
              model_config.latent_dim, activation=None, dtype=dtype
          )
      ]
  )

  if encoder_config.tabulate:
    print("Encoder")
    print("--------------")
    print(
        encoder_module.tabulate(
            jax.random.key(0),
            env.observation_space.sample(),
            compute_flops=True
        )
    )

  ##############################
  # Replay buffer setup
  ##############################
  dummy_obs, _ = env.reset()
  if env_config.backend == "mjx_dmc":
    dummy_action = jnp.zeros(
        (int(env_config.num_envs),) + tuple(env.single_action_space.shape),
        dtype=jnp.float32,
    )
    dummy_next_obs = jnp.zeros_like(dummy_obs)
    dummy_reward = jnp.zeros((int(env_config.num_envs),), dtype=jnp.float32)
    dummy_term = jnp.zeros((int(env_config.num_envs),), dtype=bool)
    dummy_trunc = jnp.zeros((int(env_config.num_envs),), dtype=bool)
  else:
    if hasattr(env, 'sample_actions'):
      dummy_action = env.sample_actions()
    else:
      dummy_action = env.action_space.sample()
    dummy_next_obs, dummy_reward, dummy_term, dummy_trunc, _ = env.step(
        dummy_action
    )
  replay_buffer = SequentialReplayBuffer(
      capacity=cfg.buffer_size,
      vectorized=True,
      num_envs=env_config.num_envs,
      seed=cfg.seed,
      dummy_input=dict(
          observation=dummy_obs,
          action=dummy_action,
          reward=dummy_reward,
          next_observation=dummy_next_obs,
          terminated=dummy_term,
          truncated=dummy_trunc
      )
  )

  encoder = TrainState.create(
      apply_fn=encoder_module.apply,
      params=encoder_module.init(encoder_key, dummy_obs)['params'],
      tx=optax.chain(
          optax.zero_nans(),
          optax.adam(encoder_config.learning_rate, eps=encoder_config.adam_eps),
      )
  )

  model = WorldModel.create(
      action_dim=np.prod(getattr(env, 'single_action_space', env.action_space).shape),
      encoder=encoder,
      **model_config,
      key=model_key
  )
  initial_horizon = int(tdmpc_config.horizon)
  if dense_rhs_config.enabled:
    initial_horizon = int(dense_rhs_config.get('initial_horizon', initial_horizon))
  horizon_buckets = (
      _horizon_buckets_from_config(dense_rhs_config)
      if dense_rhs_config.enabled else ()
  )
  planning_hmax = _bucket_for_horizon(initial_horizon, horizon_buckets)
  agent = TDMPC2.create(
      world_model=model,
      planning_hmax=planning_hmax,
      grad_clip_norm=float(model_config.max_grad_norm),
      **tdmpc_config,
  )
  if model.action_dim >= 20:
    agent = agent.replace(mppi_iterations=agent.mppi_iterations + 2)
  horizon_state = None
  selected_horizon = int(tdmpc_config.horizon)
  dense_query_kernels = None
  if dense_rhs_config.enabled:
    horizon_state = HorizonSearchState.create(
        horizons=dense_rhs_config.horizons,
        hmax=int(dense_rhs_config.hmax),
        query_interval_steps=int(dense_rhs_config.query_interval_steps),
        start_query_step=dense_rhs_config.get('start_query_step', None),
        initial_horizon=initial_horizon,
        roughness_probe=str(dense_rhs_config.roughness_probe),
        robust_return=str(dense_rhs_config.robust_return),
        phase_min_samples_to_drop=int(dense_rhs_config.phase_min_samples_to_drop),
        candidate_budget=dense_rhs_config.candidate_budget,
        selection_return_power=float(dense_rhs_config.selection_return_power),
        roughness_weight=float(dense_rhs_config.roughness_weight),
        return_std_weight=float(dense_rhs_config.return_std_weight),
        local_window_radius=int(dense_rhs_config.local_window_radius),
        max_transition_delta=int(dense_rhs_config.max_transition_delta),
        incumbent_switch_margin=float(dense_rhs_config.incumbent_switch_margin),
    )
    selected_horizon = int(np.asarray(horizon_state.best_h))
    agent = _make_training_horizon_agent(agent, selected_horizon, horizon_buckets)
    dense_query_kernels = build_dense_query_kernels(
        eval_state=dense_query_eval_state,
        env_eval_steps=int(dense_rhs_config.env_eval_steps),
        candidate_budgets=horizon_state.candidate_budget,
    )
  global_step = 0

  options = ocp.CheckpointManagerOptions(
      max_to_keep=1, save_interval_steps=cfg['save_interval_steps']
  )
  checkpoint_path = os.path.join(output_dir, 'checkpoint')
  checkpoint_buffer = bool(cfg.get('checkpoint_buffer', True))
  item_names = ['agent', 'global_step']
  if checkpoint_buffer:
    item_names.append('buffer_state')
  if horizon_state is not None:
    item_names.append('horizon_state')
  with ocp.CheckpointManager(
      checkpoint_path,
      options=options,
      item_names=tuple(item_names)
  ) as mngr:
    buffer_state = replay_buffer.get_state()
    if mngr.latest_step() is not None:
      print('Checkpoint folder found, restoring from', mngr.latest_step())
      abstract_buffer_state = jax.tree.map(
          ocp.utils.to_shape_dtype_struct, buffer_state
      )
      restore_args = {
          'agent': ocp.args.StandardRestore(agent),
          'global_step': ocp.args.JsonRestore(),
      }
      if checkpoint_buffer:
        restore_args['buffer_state'] = ocp.args.StandardRestore(abstract_buffer_state)
      if horizon_state is not None:
        restore_args['horizon_state'] = ocp.args.StandardRestore(horizon_state)
      restored = mngr.restore(
          mngr.latest_step(),
          args=ocp.args.Composite(**restore_args)
      )
      agent, global_step = restored.agent, restored.global_step
      if checkpoint_buffer:
        buffer_state = restored.buffer_state
      if horizon_state is not None:
        horizon_state = restored.horizon_state
        selected_horizon = int(np.asarray(horizon_state.best_h))
        agent = _make_training_horizon_agent(agent, selected_horizon, horizon_buckets)
    else:
      print('No checkpoint folder found, starting from scratch')
      save_args = {
          'agent': ocp.args.StandardSave(agent),
          'global_step': ocp.args.JsonSave(global_step),
      }
      if checkpoint_buffer:
        save_args['buffer_state'] = ocp.args.StandardSave(buffer_state)
      if horizon_state is not None:
        save_args['horizon_state'] = ocp.args.StandardSave(horizon_state)
      mngr.save(global_step, args=ocp.args.Composite(**save_args))
      mngr.wait_until_finished()
    last_saved_step = int(global_step)

    ##############################
    # Training loop
    ##############################
    ep_count = np.zeros(env_config.num_envs, dtype=int)
    prev_logged_step = global_step
    plan = None
    observation, _ = env.reset(seed=cfg.seed)

    T = 500
    seed_steps_override = cfg.get('seed_steps_override', None)
    if seed_steps_override is None:
      seed_steps = int(
          max(5*T, 1000) * env_config.num_envs * env_config.utd_ratio
      )
    else:
      seed_steps = int(seed_steps_override)
    pbar = tqdm.tqdm(initial=global_step, total=cfg.max_steps)
    done = np.zeros(env_config.num_envs, dtype=bool)
    update_chunk_size = int(cfg.get('update_chunk_size', 128))
    last_eval_step = -1
    if horizon_state is not None and bool(dense_rhs_config.prewarm_horizons):
      print('Prewarming Dense-RHS kernels...', flush=True)
      buffer_state, prewarm_batch = sample_from_state(
          buffer_state,
          batch_size=int(agent.batch_size),
          sequence_length=int(horizon_state.hmax),
      )
      print('Sampled replay batch for Dense-RHS prewarm.', flush=True)
      prewarm_start = time.perf_counter()
      dense_query_agent = _make_dense_rhs_query_agent(agent, dense_rhs_config)
      prewarm_dense_rhs_kernels(
          agent=dense_query_agent,
          replay_batch=prewarm_batch,
          eval_state=dense_query_eval_state,
          horizon_state=horizon_state,
          rng=rng,
          env_eval_steps=int(dense_rhs_config.env_eval_steps),
          warm_all_phases=bool(dense_rhs_config.prewarm_all_phases),
      )
      writer.scalar(
          'timing/prewarm_dense_rhs_s',
          time.perf_counter() - prewarm_start,
          global_step,
      )
      writer.scalar('system/heartbeat', 1.0, global_step)
      writer.flush()
      print('Finished Dense-RHS prewarm.', flush=True)
    if periodic_eval_env is not None and bool(eval_config.prewarm):
      print('Prewarming periodic eval kernel...', flush=True)
      rng, eval_prewarm_key = jax.random.split(rng)
      eval_prewarm_start = time.perf_counter()
      _ = np.asarray(
          periodic_eval_env.run_eval_chunk(
              agent=agent,
              horizon=selected_horizon,
              key=eval_prewarm_key,
              steps_per_episode=int(periodic_eval_env._metadata.episode_length),
          )
      )
      writer.scalar(
          'timing/prewarm_eval_s',
          time.perf_counter() - eval_prewarm_start,
          global_step,
      )
      writer.scalar('system/heartbeat', 1.0, global_step)
      writer.flush()
      print('Finished periodic eval prewarm.', flush=True)
    if env_config.backend == "mjx_dmc":
      _run_mjx_training_loop(
          cfg,
          env_config,
          eval_config,
          dense_rhs_config,
          env,
          periodic_eval_env,
          dense_query_eval_state,
          agent,
          buffer_state,
          writer,
          mngr,
          global_step,
          last_saved_step,
          seed_steps=seed_steps,
          update_chunk_size=update_chunk_size,
          horizon_state=horizon_state,
          dense_query_kernels=dense_query_kernels,
      )
      writer.close()
      return
    for global_step in range(global_step, cfg.max_steps, env_config.num_envs):
      if horizon_state is not None and global_step >= seed_steps and horizon_state.should_query(global_step):
        rng, query_key = jax.random.split(rng)
        buffer_state, query_batch = sample_from_state(
            buffer_state,
            batch_size=int(agent.batch_size),
            sequence_length=int(horizon_state.hmax),
        )
        query_start = time.perf_counter()
        dense_query_agent = _make_dense_rhs_query_agent(agent, dense_rhs_config)
        horizon_state, selected_horizon, dense_metrics = dense_checkpoint_eval(
            agent=dense_query_agent,
            replay_batch=query_batch,
            eval_state=dense_query_eval_state,
            horizon_state=horizon_state,
            rng=query_key,
            env_eval_steps=int(dense_rhs_config.env_eval_steps),
            query_step=int(global_step),
            dense_query_kernels=dense_query_kernels,
        )
        agent = agent.replace(horizon=selected_horizon)
        plan = None
        dense_metrics['timing/query_total_s'] = max(
            float(dense_metrics.get('timing/query_total_s', 0.0)),
            time.perf_counter() - query_start,
        )
        for metric_name, metric_value in dense_metrics.items():
          if isinstance(metric_value, str):
            continue
          writer.scalar(metric_name, float(metric_value), global_step)
        best_h = int(np.asarray(horizon_state.best_h))
        horizons_np = np.asarray(horizon_state.horizons)
        best_idx = int(np.where(horizons_np == best_h)[0][0])
        query_row = {
            'step': int(global_step),
            'selected_horizon': int(selected_horizon),
            'best_h': int(best_h),
            'phase_id': int(np.asarray(horizon_state.phase_id)),
            'phase_name': horizon_state.phase_name(),
            'num_active_horizons': int(np.sum(np.asarray(horizon_state.active_mask))),
            'num_candidate_horizons': int(dense_metrics['dense_rhs/num_candidate_horizons']),
            'entropy': float(np.asarray(horizon_state.entropy)),
            'norm_entropy': float(np.asarray(horizon_state.norm_entropy)),
            'prob_best_h': float(np.asarray(horizon_state.prob)[best_idx]),
            'gauss_mean_best_h': float(np.asarray(horizon_state.gauss_mean)[best_idx]),
            'gauss_post_std_best_h': float(np.asarray(horizon_state.gauss_post_std)[best_idx]),
        }
        writer.horizon_query(**query_row)
        writer.scalar('dense_rhs/best_h', query_row['best_h'], global_step)
        writer.scalar('dense_rhs/prob_best_h', query_row['prob_best_h'], global_step)
        writer.scalar('dense_rhs/gauss_mean_best_h', query_row['gauss_mean_best_h'], global_step)
        writer.scalar(
            'dense_rhs/gauss_post_std_best_h',
            query_row['gauss_post_std_best_h'],
            global_step,
        )
        writer.scalar('system/heartbeat', 1.0, global_step)
        writer.flush()

      if global_step <= seed_steps:
        if hasattr(env, 'sample_actions'):
          action = env.sample_actions()
        else:
          action = env.action_space.sample()
      else:
        rng, action_key = jax.random.split(rng)
        action, plan = agent.act(
            observation,
            prev_plan=plan,
            deterministic=False,
            train=True,
            horizon=selected_horizon,
            key=action_key
        )

      next_observation, reward, terminated, truncated, info = env.step(action)

      if np.any(~done):
        buffer_state = insert_into_state(
            buffer_state,
            dict(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated,
            ),
            mask=jnp.asarray(~done),
        )
      observation = next_observation

      # Handle terminations/truncations
      done = np.logical_or(terminated, truncated)
      if np.any(done):
        if plan is not None:
          plan = (
              plan[0].at[done].set(0),
              plan[1].at[done].set(agent.max_plan_std)
          )
        for ienv in range(env_config.num_envs):
          if done[ienv]:
            r = float(np.asarray(info['episode']['r'][ienv]))
            l = int(np.asarray(info['episode']['l'][ienv]))
            print(
                f"Episode {ep_count[ienv]}: r = {r:.2f}, l = {l}"
            )
            writer.episode(
                step=global_step + ienv,
                env_index=ienv,
                episode_index=ep_count[ienv],
                episode_return=r,
                episode_length=l,
                selected_horizon=int(selected_horizon),
            )
            writer.scalar(f'episode/return', r, global_step + ienv)
            writer.scalar(f'episode/length', l, global_step + ienv)
            ep_count[ienv] += 1

      if global_step >= seed_steps:
        if global_step == seed_steps:
          print('Pre-training on seed data...')
          num_updates = seed_steps
        else:
          num_updates = max(1, int(env_config.num_envs * env_config.utd_ratio))

        log_this_step = global_step >= prev_logged_step + cfg['log_interval_steps']
        if log_this_step or global_step == seed_steps:
          all_train_info = defaultdict(lambda: {'sum': 0.0, 'sum_sq': 0.0, 'count': 0})
          prev_logged_step = global_step

        updates_completed = 0
        accumulated_train_time = 0.0
        while updates_completed < num_updates:
          chunk_updates = min(update_chunk_size, num_updates - updates_completed)
          rng, update_key = jax.random.split(rng)
          train_start = time.perf_counter()
          agent, buffer_state, train_info = _run_train_chunk(
              agent,
              buffer_state,
              update_key,
              num_updates=chunk_updates,
              batch_size=int(agent.batch_size),
              sequence_length=int(selected_horizon),
              train_horizon=int(selected_horizon),
          )
          accumulated_train_time += time.perf_counter() - train_start
          updates_completed += chunk_updates

          if log_this_step or global_step == seed_steps:
            for k, v in train_info.items():
              arr = np.asarray(v, dtype=np.float32)
              all_train_info[k]['sum'] += float(arr.sum())
              all_train_info[k]['sum_sq'] += float(np.square(arr).sum())
              all_train_info[k]['count'] += int(arr.size)

          if global_step == seed_steps and updates_completed % int(cfg.seed_pretrain_log_interval_updates) == 0:
            writer.scalar('seed_pretrain/updates_completed', updates_completed, global_step)
            writer.scalar('timing/train_chunk_s', accumulated_train_time, global_step)
            writer.scalar('system/heartbeat', 1.0, global_step)
            writer.flush()

        if log_this_step or global_step == seed_steps:
          for k, stats in all_train_info.items():
            mean = stats['sum'] / max(stats['count'], 1)
            var = max(stats['sum_sq'] / max(stats['count'], 1) - mean**2, 0.0)
            writer.scalar(f'train/{k}_mean', mean, global_step)
            writer.scalar(f'train/{k}_std', np.sqrt(var), global_step)
          writer.scalar('timing/train_chunk_s', accumulated_train_time, global_step)
          writer.scalar('system/heartbeat', 1.0, global_step)

        should_save = (
            global_step == 0 or
            global_step - last_saved_step >= int(cfg['save_interval_steps']) or
            global_step + env_config.num_envs >= cfg.max_steps
        )
        if should_save:
          save_args = {
              'agent': ocp.args.StandardSave(agent),
              'global_step': ocp.args.JsonSave(global_step),
          }
          if checkpoint_buffer:
            save_args['buffer_state'] = ocp.args.StandardSave(buffer_state)
          if horizon_state is not None:
            save_args['horizon_state'] = ocp.args.StandardSave(horizon_state)
          checkpoint_start = time.perf_counter()
          mngr.save(global_step, args=ocp.args.Composite(**save_args))
          mngr.wait_until_finished()
          writer.scalar('timing/checkpoint_blocking_s', time.perf_counter() - checkpoint_start, global_step)
          last_saved_step = int(global_step)

        should_eval = (
            periodic_eval_env is not None and
            global_step > 0 and
            int(global_step) != int(last_eval_step) and
            (
                int(global_step) % int(eval_config.interval_steps) == 0 or
                global_step + env_config.num_envs >= cfg.max_steps
            )
        )
        if should_eval:
          rng, eval_key = jax.random.split(rng)
          eval_start = time.perf_counter()
          eval_metrics = _run_periodic_eval(
              periodic_eval_env,
              agent,
              horizon=selected_horizon,
              num_episodes=int(eval_config.num_episodes),
              steps_per_episode=int(periodic_eval_env._metadata.episode_length),
              key=eval_key,
          )
          for metric_name, metric_value in eval_metrics.items():
            writer.scalar(metric_name, metric_value, global_step)
          writer.scalar('timing/eval_s', time.perf_counter() - eval_start, global_step)
          writer.scalar('eval/selected_horizon', float(selected_horizon), global_step)
          writer.scalar('system/heartbeat', 1.0, global_step)
          last_eval_step = int(global_step)
          writer.flush()

        if log_this_step or global_step == seed_steps:
          writer.flush()

      pbar.update(env_config.num_envs)
    pbar.close()
    writer.flush()
    writer.close()


if __name__ == '__main__':
  train()
