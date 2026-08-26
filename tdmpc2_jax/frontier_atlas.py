"""Frozen one-factor-at-a-time frontier atlas for Pendulum evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple


ATLAS_VERSION = 'pendulum-frontier-atlas-v1'
ATLAS_HORIZONS = (2, 3, 4, 5, 6, 7, 8)


@dataclass(frozen=True)
class FrontierCondition:
  index: int
  condition_id: str
  axis: str
  value: float


_AXIS_GRIDS = (
    ('actuator_strength_scale', (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3), 1.0),
    ('joint_damping_scale', (0.25, 0.5, 1.0, 2.0, 4.0), 1.0),
    ('gravity_scale', (0.5, 0.75, 1.0, 1.25, 1.5), 1.0),
    ('base_action_delay', (0.0, 1.0, 2.0, 3.0, 4.0), 0.0),
    ('observation_noise_scale', (0.0, 0.01, 0.03, 0.06, 0.1), 0.0),
)


def _value_code(value: float) -> str:
  return f'{float(value):g}'.replace('-', 'm').replace('.', 'p')


def frontier_conditions() -> Tuple[FrontierCondition, ...]:
  """Returns the canonical 24-cell, one-factor-at-a-time atlas."""
  records = [
      FrontierCondition(
          index=0,
          condition_id='nominal',
          axis='nominal',
          value=1.0,
      )
  ]
  for axis, values, neutral in _AXIS_GRIDS:
    for value in values:
      if float(value) == float(neutral):
        continue
      records.append(
          FrontierCondition(
              index=len(records),
              condition_id=f'{axis}__{_value_code(value)}',
              axis=axis,
              value=float(value),
          )
      )
  return tuple(records)


def shard_conditions(shard_index: int,
                     num_shards: int,
                     *,
                     conditions: Iterable[FrontierCondition] | None = None,
                     limit: int | None = None) -> Tuple[FrontierCondition, ...]:
  """Deterministically assigns canonical conditions across Slurm shards."""
  shard_index = int(shard_index)
  num_shards = int(num_shards)
  if num_shards <= 0 or not 0 <= shard_index < num_shards:
    raise ValueError(
        f'Invalid atlas shard {shard_index} of {num_shards}; expected '
        '0 <= shard_index < num_shards.'
    )
  source = tuple(frontier_conditions() if conditions is None else conditions)
  selected = tuple(
      condition for condition in source
      if condition.index % num_shards == shard_index
  )
  if limit is not None:
    if int(limit) <= 0:
      raise ValueError('Atlas condition limit must be positive when set.')
    selected = selected[:int(limit)]
  return selected


def condition_overrides(condition: FrontierCondition) -> Dict[str, Any]:
  """Maps a canonical condition to isolated MJX environment overrides."""
  overrides: Dict[str, Any] = {
      'actuator_strength_scale': 1.0,
      'joint_damping_scale': 1.0,
      'gravity_scale': 1.0,
      'base_action_delay': 0,
      'action_delay_schedule_enabled': False,
      'action_delay_observation_enabled': False,
      'enable_domain_randomization': False,
      'enable_observation_noise': False,
      'fixed_observation_noise_scale': None,
  }
  if condition.axis == 'nominal':
    return overrides
  if condition.axis == 'base_action_delay':
    overrides['base_action_delay'] = int(condition.value)
  elif condition.axis == 'observation_noise_scale':
    overrides['enable_observation_noise'] = True
    overrides['fixed_observation_noise_scale'] = float(condition.value)
  elif condition.axis in (
      'actuator_strength_scale',
      'joint_damping_scale',
      'gravity_scale',
  ):
    overrides[condition.axis] = float(condition.value)
  else:
    raise ValueError(f'Unsupported frontier axis {condition.axis!r}.')
  return overrides

