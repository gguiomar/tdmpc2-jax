#!/usr/bin/env python3
"""Render deterministic MJX delay trajectories as paired GIFs and PNGs.

The training process writes physical MJX states and actions, so this exporter
does not restore the policy or execute it again. Rendering is therefore safe to
run on a CPU login/visualization node and is reproducible if camera or overlay
styling changes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ANCHOR_STEPS = (0, 100_000, 150_000, 250_000, 350_000, 450_000, 500_000)
TRAJECTORY_REQUIRED_KEYS = {
    'qpos',
    'qvel',
    'ctrl',
    'commanded_action',
    'applied_action',
    'delayed_actions',
    'reward',
    'done',
    'effective_action_delay',
    'frame_timestamp_seconds',
}


def _parse_steps(value: str) -> tuple[int, ...]:
  if value.strip().lower() == 'all':
    return DEFAULT_ANCHOR_STEPS
  return tuple(sorted({int(part.strip()) for part in value.split(',') if part.strip()}))


def _step_from_dir(path: Path) -> int:
  match = re.fullmatch(r'step_(\d+)', path.name)
  if match is None:
    raise ValueError(f'Invalid rollout directory name: {path.name}')
  return int(match.group(1))


def _resolve_rollout_root(run_dir: Path, run_id: str | None) -> Path:
  roots = Path(run_dir) / 'artifacts' / 'rollouts'
  if run_id is not None:
    root = roots / run_id
    if not root.is_dir():
      raise FileNotFoundError(f'Rollout root does not exist: {root}')
    return root
  candidates = sorted(path for path in roots.iterdir() if path.is_dir())
  if len(candidates) != 1:
    raise ValueError(
        f'Expected exactly one run id under {roots}; found '
        f'{[path.name for path in candidates]}. Pass --run-id explicitly.'
    )
  return candidates[0]


def validate_rollout_dir(rollout_dir: Path) -> dict:
  metadata_path = rollout_dir / 'metadata.json'
  if not metadata_path.is_file():
    raise FileNotFoundError(f'Missing rollout metadata: {metadata_path}')
  with metadata_path.open() as metadata_file:
    metadata = json.load(metadata_file)
  conditions = tuple(sorted(metadata.get('trajectories', {})))
  if not conditions:
    raise ValueError(f'{metadata_path}: no trajectory conditions declared')
  for condition in conditions:
    trajectory_path = rollout_dir / f'trajectory_{condition}.npz'
    if not trajectory_path.is_file():
      raise FileNotFoundError(f'Missing trajectory: {trajectory_path}')
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
      missing = TRAJECTORY_REQUIRED_KEYS - set(trajectory.files)
      if missing:
        raise ValueError(
            f'{trajectory_path} is missing required arrays: {sorted(missing)}'
        )
      qpos = np.asarray(trajectory['qpos'])
      qvel = np.asarray(trajectory['qvel'])
      timestamps = np.asarray(trajectory['frame_timestamp_seconds'])
      if qpos.ndim != 3 or qvel.ndim != 3:
        raise ValueError(
            f'{trajectory_path}: qpos/qvel must have shape '
            '(frames, initial_states, coordinates).'
        )
      if qpos.shape[:2] != qvel.shape[:2]:
        raise ValueError(f'{trajectory_path}: qpos/qvel frame shapes differ.')
      if timestamps.shape != (qpos.shape[0],):
        raise ValueError(
            f'{trajectory_path}: frame timestamps do not match qpos frames.'
        )
  return metadata


def _load_mujoco_model(task: str):
  try:
    import mujoco
    from tdmpc2_jax.envs.mjx_dmc import _load_model, _parse_task
  except ImportError as exc:
    raise RuntimeError(
        'MuJoCo rendering dependencies are unavailable. Activate the same '
        'tdmpc2-jax environment used for training, then rerun this command.'
    ) from exc
  domain, _ = _parse_task(task)
  return mujoco, _load_model(domain)


def _frame_overlay(image: np.ndarray,
                   *,
                   metadata: dict,
                   condition: str,
                   initial_state: int,
                   frame_index: int,
                   delay: int,
                   cumulative_return: float,
                   queue: np.ndarray) -> Image.Image:
  frame = Image.fromarray(image)
  draw = ImageDraw.Draw(frame)
  queue_text = np.array2string(
      np.asarray(queue).reshape(-1),
      precision=2,
      separator=',',
      suppress_small=True,
  )
  environment = metadata.get('environment', {})
  schedule_start = int(environment.get('action_delay_schedule_start_step', 150_000))
  schedule_end = int(environment.get('action_delay_schedule_end_step', 350_000))
  schedule_value = int(environment.get('action_delay_schedule_value', 4))
  schedule_enabled = bool(environment.get('action_delay_schedule_enabled', True))
  boundaries = [
      int(value) for value in
      environment.get('action_delay_schedule_boundaries', [])
  ]
  values = [
      int(value) for value in
      environment.get('action_delay_schedule_values', [])
  ]
  if schedule_enabled and values:
    phase_delay = values[0]
    for boundary, value in zip(boundaries, values[1:]):
      if int(metadata['global_step']) >= boundary:
        phase_delay = value
    schedule_text = ' -> '.join(str(value) for value in values)
  else:
    phase_delay = (
        schedule_value
        if schedule_enabled and schedule_start <= int(metadata['global_step']) < schedule_end
        else int(environment.get('base_action_delay', 0))
    )
    schedule_text = f'0 --[{schedule_start:,}]--> {schedule_value} --[{schedule_end:,}]--> 0'
  lines = (
      f"{metadata.get('run_id', 'run')} | step {metadata['global_step']:,}",
      f"{metadata.get('controller', 'unknown')} | seed {metadata.get('training_seed', '?')}",
      f"{condition} challenge | init {initial_state} | delay {delay} | h={metadata['selected_horizon']}",
      (
          f"train schedule: {schedule_text} | phase d={phase_delay}"
      ),
      f"frame {frame_index} | return {cumulative_return:.2f}",
      f"queue {queue_text}",
  )
  padding = 5
  line_height = 14
  box_height = padding * 2 + line_height * len(lines)
  draw.rectangle((0, 0, frame.width, box_height), fill=(0, 0, 0))
  draw.multiline_text(
      (padding, padding),
      '\n'.join(lines),
      fill=(255, 255, 255),
      spacing=2,
  )
  return frame


def _render_condition(trajectory_path: Path,
                      *,
                      metadata: dict,
                      condition: str,
                      width: int,
                      height: int,
                      camera: int,
                      frame_stride: int) -> list[list[Image.Image]]:
  mujoco, model = _load_mujoco_model(metadata['environment']['task'])
  data = mujoco.MjData(model)
  renderer = mujoco.Renderer(model, height=height, width=width)
  try:
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
      qpos = np.asarray(trajectory['qpos'])
      qvel = np.asarray(trajectory['qvel'])
      ctrl = np.asarray(trajectory['ctrl'])
      rewards = np.asarray(trajectory['reward'])
      delays = np.asarray(trajectory['effective_action_delay'])
      queues = np.asarray(trajectory['delayed_actions'])
      done = np.asarray(trajectory['done'], dtype=bool)
      num_initial_states = int(qpos.shape[1])
      rendered = [[] for _ in range(num_initial_states)]
      cumulative_returns = np.concatenate(
          [
              np.zeros((1, num_initial_states), dtype=np.float64),
              np.cumsum(rewards, axis=0, dtype=np.float64),
          ],
          axis=0,
      )
      action_frame_count = done.shape[0]
      for initial_state in range(num_initial_states):
        completed = np.flatnonzero(done[:, initial_state])
        final_action_index = (
            int(completed[0]) + 1 if completed.size else action_frame_count
        )
        final_frame_index = min(final_action_index + 1, qpos.shape[0])
        for frame_index in range(0, final_frame_index, frame_stride):
          mujoco.mj_resetData(model, data)
          np.copyto(data.qpos, qpos[frame_index, initial_state])
          np.copyto(data.qvel, qvel[frame_index, initial_state])
          if model.nu:
            np.copyto(data.ctrl, ctrl[frame_index, initial_state])
          mujoco.mj_forward(model, data)
          renderer.update_scene(data, camera=camera)
          rgb = np.asarray(renderer.render()).copy()
          action_index = min(max(frame_index - 1, 0), action_frame_count - 1)
          rendered[initial_state].append(
              _frame_overlay(
                  rgb,
                  metadata=metadata,
                  condition=condition,
                  initial_state=initial_state,
                  frame_index=frame_index,
                  delay=int(delays[frame_index, initial_state]),
                  cumulative_return=float(
                      cumulative_returns[frame_index, initial_state]
                  ),
                  queue=queues[action_index, initial_state],
              )
          )
    return rendered
  finally:
    renderer.close()


def _compose_grid(condition_frames: dict[str, list[list[Image.Image]]]) -> list[Image.Image]:
  conditions = tuple(sorted(condition_frames))
  if not conditions:
    return []
  row_count = len(condition_frames[conditions[0]])
  if any(len(condition_frames[name]) != row_count for name in conditions):
    raise ValueError('Delay trajectories use different initial-state counts.')
  grid_frames = []
  frame_count = min(
      len(frames)
      for name in conditions
      for frames in condition_frames[name]
  )
  for frame_index in range(frame_count):
    width = condition_frames[conditions[0]][0][frame_index].width
    height = condition_frames[conditions[0]][0][frame_index].height
    grid = Image.new('RGB', (width * len(conditions), height * row_count))
    for column, name in enumerate(conditions):
      for row, frames in enumerate(condition_frames[name]):
        grid.paste(frames[frame_index], (column * width, row * height))
    grid_frames.append(grid)
  return grid_frames


def render_rollout_dir(rollout_dir: Path,
                       *,
                       width: int,
                       height: int,
                       camera: int,
                       frame_stride: int,
                       fps: float | None) -> Path:
  metadata = validate_rollout_dir(rollout_dir)
  conditions = {}
  for condition in sorted(metadata.get('trajectories', {})):
    conditions[condition] = _render_condition(
        rollout_dir / f'trajectory_{condition}.npz',
        metadata=metadata,
        condition=condition,
        width=width,
        height=height,
        camera=camera,
        frame_stride=frame_stride,
    )
  frames = _compose_grid(conditions)
  if not frames:
    raise ValueError(f'No frames were rendered for {rollout_dir}.')
  frame_dt = float(
      metadata['trajectories']['delay0']['frame_dt_seconds']
  )
  output_fps = float(fps) if fps is not None else 1.0 / (frame_dt * frame_stride)
  duration_ms = max(1, int(round(1000.0 / output_fps)))
  task_prefix = str(metadata['environment']['task']).split('-', maxsplit=1)[0]
  condition_label = '_vs_'.join(sorted(conditions))
  output_path = rollout_dir / f'{task_prefix}_{condition_label}.gif'
  frames[0].save(
      output_path,
      save_all=True,
      append_images=frames[1:],
      duration=duration_ms,
      loop=0,
      optimize=False,
  )
  frames[0].save(
      rollout_dir / f'{task_prefix}_{condition_label}_frame.png'
  )
  return output_path


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('run_dir', type=Path)
  parser.add_argument('--run-id')
  parser.add_argument('--steps', default='all')
  parser.add_argument('--width', type=int, default=320)
  parser.add_argument('--height', type=int, default=320)
  parser.add_argument('--camera', type=int, default=0)
  parser.add_argument('--frame-stride', type=int, default=2)
  parser.add_argument('--fps', type=float)
  parser.add_argument(
      '--mujoco-gl',
      default='osmesa',
      choices=('osmesa', 'egl', 'glfw'),
      help='Headless CPU rendering uses osmesa; use egl if that build lacks it.',
  )
  parser.add_argument(
      '--allow-missing',
      action='store_true',
      help='Render available anchors instead of failing on missing requested steps.',
  )
  args = parser.parse_args(argv)
  if args.frame_stride <= 0:
    parser.error('--frame-stride must be positive')
  os.environ.setdefault('MUJOCO_GL', args.mujoco_gl)

  try:
    rollout_root = _resolve_rollout_root(args.run_dir, args.run_id)
    requested_steps = _parse_steps(args.steps)
    available = {
        _step_from_dir(path): path
        for path in rollout_root.glob('step_*')
        if path.is_dir()
    }
    missing = sorted(set(requested_steps) - set(available))
    if missing and not args.allow_missing:
      raise FileNotFoundError(
          f'Missing requested rollout anchors under {rollout_root}: {missing}'
      )
    for step in requested_steps:
      if step not in available:
        continue
      output_path = render_rollout_dir(
          available[step],
          width=args.width,
          height=args.height,
          camera=args.camera,
          frame_stride=args.frame_stride,
          fps=args.fps,
      )
      print(output_path)
  except (FileNotFoundError, RuntimeError, ValueError) as exc:
    print(f'ERROR: {exc}', file=sys.stderr)
    return 2
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
