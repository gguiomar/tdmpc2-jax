from __future__ import annotations

import inspect
import math
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from tdmpc2_jax.common.loss import soft_crossentropy
from tdmpc2_jax.common.util import sg


PHASE_NAMES = ('A', 'B1', 'B2', 'B3', 'B4')
PHASE_TARGET_SIZES = (29, 18, 12, 9, 9)
ROUGHNESS_PROBE_COUNTS = (2, 4, 8, 16, 32, 64)
VALID_ROUGHNESS_PROBE_COUNTS = (0,) + ROUGHNESS_PROBE_COUNTS
VALID_SCORE_MODES = ('additive', 'multiplicative', 'legacy_multiplicative')
VALID_SCORE_VARIANTS = (
    'current_additive',
    'calibrated_local_roughness',
    'return_first',
    'curvature_bellman',
)
VALID_DECISION_RULES = ('legacy', 'paired_lcb')
EPS = 1e-6


def _tree_l2_sq(tree) -> jax.Array:
  leaves = jax.tree.leaves(tree)
  if not leaves:
    return jnp.array(0.0, dtype=jnp.float32)
  return sum(
      jnp.sum(jnp.square(jnp.asarray(leaf, dtype=jnp.float32)))
      for leaf in leaves
      if leaf is not None
  )


def _normalise(values: np.ndarray, inverse: bool = False) -> np.ndarray:
  values = np.asarray(values, dtype=np.float32)
  if not values.size:
    return values
  vmin = float(np.min(values))
  vmax = float(np.max(values))
  if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < EPS:
    out = np.ones_like(values, dtype=np.float32)
  else:
    out = (values - vmin) / (vmax - vmin + EPS)
  if inverse:
    out = 1.0 - out
  return np.clip(out, 0.0, 1.0)


def _tolerance_linear(x: np.ndarray,
                      lower: float,
                      margin: float,
                      value_at_margin: float = 0.0) -> np.ndarray:
  x = np.asarray(x, dtype=np.float32)
  below = x < lower
  if margin <= 0:
    return np.where(below, value_at_margin, 1.0).astype(np.float32)
  delta = (lower - x) / margin
  return np.where(
      below,
      np.clip(1.0 - (1.0 - value_at_margin) * delta, value_at_margin, 1.0),
      1.0,
  ).astype(np.float32)


@struct.dataclass
class HorizonSearchState:
  horizons: jax.Array
  active_mask: jax.Array
  best_h: jax.Array
  score_sum: jax.Array
  score_sum_sq: jax.Array
  score_count: jax.Array
  gauss_mean: jax.Array
  gauss_post_std: jax.Array
  prob: jax.Array
  entropy: jax.Array
  norm_entropy: jax.Array
  phase_id: jax.Array
  phase_sample_count_A: jax.Array
  phase_sample_count_B1: jax.Array
  phase_sample_count_B2: jax.Array
  phase_sample_count_B3: jax.Array
  phase_sample_count_B4: jax.Array
  query_interval_steps: jax.Array
  start_query_step: jax.Array
  next_query_step: jax.Array
  hmax: int = struct.field(pytree_node=False)
  roughness_probe: str = struct.field(pytree_node=False)
  num_roughness_probes: int = struct.field(pytree_node=False)
  score_mode: str = struct.field(pytree_node=False)
  score_variant: str = struct.field(pytree_node=False)
  additive_return_scale: float = struct.field(pytree_node=False)
  additive_return_std_scale: float = struct.field(pytree_node=False)
  additive_log_roughness_scale: float = struct.field(pytree_node=False)
  score_evidence_floor: float = struct.field(pytree_node=False)
  decision_rule: str = struct.field(pytree_node=False)
  confidence_z: float = struct.field(pytree_node=False)
  switch_threshold: float = struct.field(pytree_node=False)
  robust_return: str = struct.field(pytree_node=False)
  phase_pruning_enabled: bool = struct.field(pytree_node=False)
  phase_min_samples_to_drop: int = struct.field(pytree_node=False)
  candidate_budget: Tuple[int, int, int, int, int] = struct.field(pytree_node=False)
  selection_return_power: float = struct.field(pytree_node=False)
  roughness_weight: float = struct.field(pytree_node=False)
  return_std_weight: float = struct.field(pytree_node=False)
  learner_proxy_enabled: bool = struct.field(pytree_node=False)
  learner_proxy_weight: float = struct.field(pytree_node=False)
  learner_proxy_mode: str = struct.field(pytree_node=False)
  local_window_radius: int = struct.field(pytree_node=False)
  max_transition_delta: int = struct.field(pytree_node=False)
  incumbent_switch_margin: float = struct.field(pytree_node=False)
  credible_transition_enabled: bool = struct.field(pytree_node=False)
  credible_transition_rule: str = struct.field(pytree_node=False)
  credible_transition_min_prob: float = struct.field(pytree_node=False)
  transition_cost_scale: float = struct.field(pytree_node=False)
  transition_risk_weight: float = struct.field(pytree_node=False)
  transition_min_expected_net: float = struct.field(pytree_node=False)
  transition_model_weight: float = struct.field(pytree_node=False)
  transition_probe_weight: float = struct.field(pytree_node=False)
  transition_planner_weight: float = struct.field(pytree_node=False)
  transition_roughness_weight: float = struct.field(pytree_node=False)
  transition_return_std_weight: float = struct.field(pytree_node=False)
  transition_uncertainty_floor: float = struct.field(pytree_node=False)
  calibrated_return_weight: float = struct.field(pytree_node=False)
  calibrated_roughness_weight: float = struct.field(pytree_node=False)
  horizon_switch_cost: float = struct.field(pytree_node=False)
  return_first_tolerance: float = struct.field(pytree_node=False)
  roughness_discount: float = struct.field(pytree_node=False)
  curvature_risk_weight: float = struct.field(pytree_node=False)
  bellman_risk_weight: float = struct.field(pytree_node=False)
  curvature_risk_scale: float = struct.field(pytree_node=False)

  @classmethod
  def create(cls,
             horizons: Sequence[int],
             hmax: int,
             query_interval_steps: int,
             start_query_step: Optional[int] = None,
             initial_horizon: Optional[int] = None,
             roughness_probe: str = 'projected_jvp',
             num_roughness_probes: int = 2,
             score_mode: str = 'legacy_multiplicative',
             score_variant: str = 'current_additive',
             additive_return_scale: float = 1.0,
             additive_return_std_scale: float = 1.0,
             additive_log_roughness_scale: float = 1.0,
             score_evidence_floor: float = EPS,
             decision_rule: str = 'legacy',
             confidence_z: float = 1.6448536,
             switch_threshold: float = 0.0,
             robust_return: str = 'mean_minus_std',
             phase_pruning_enabled: bool = True,
             phase_min_samples_to_drop: int = 3,
             candidate_budget: Optional[Mapping[str, int]] = None,
             selection_return_power: float = 1.0,
             roughness_weight: float = 1.0,
             return_std_weight: float = 1.0,
             learner_proxy_enabled: bool = False,
             learner_proxy_weight: float = 0.0,
             learner_proxy_mode: str = 'probe_mean_loss',
             local_window_radius: int = 0,
             max_transition_delta: int = 0,
             incumbent_switch_margin: float = 0.0,
             credible_transition_enabled: bool = False,
             credible_transition_rule: str = 'probability',
             credible_transition_min_prob: float = 0.0,
             transition_cost_scale: float = 0.0,
             transition_risk_weight: float = 1.0,
             transition_min_expected_net: float = 0.0,
             transition_model_weight: float = 1.0,
             transition_probe_weight: float = 1.0,
             transition_planner_weight: float = 1.0,
             transition_roughness_weight: float = 1.0,
             transition_return_std_weight: float = 1.0,
             transition_uncertainty_floor: float = 0.05,
             calibrated_return_weight: float = 10.0,
             calibrated_roughness_weight: float = 1.0,
             horizon_switch_cost: float = 0.05,
             return_first_tolerance: float = 5.0,
             roughness_discount: float = 0.99,
             curvature_risk_weight: float = 1.0,
             bellman_risk_weight: float = 0.1,
             curvature_risk_scale: float = 0.01) -> 'HorizonSearchState':
    horizons_arr = np.asarray(tuple(horizons), dtype=np.int32)
    if horizons_arr.size == 0:
      raise ValueError('horizons must contain at least one value')
    num_roughness_probes = int(num_roughness_probes)
    if num_roughness_probes not in VALID_ROUGHNESS_PROBE_COUNTS:
      raise ValueError(
          'num_roughness_probes must be one of '
          f'{VALID_ROUGHNESS_PROBE_COUNTS}; got {num_roughness_probes}'
      )
    score_mode = str(score_mode)
    if score_mode not in VALID_SCORE_MODES:
      raise ValueError(
          f'score_mode must be one of {VALID_SCORE_MODES}; got {score_mode!r}'
      )
    score_variant = str(score_variant)
    if score_variant not in VALID_SCORE_VARIANTS:
      raise ValueError(
          f'score_variant must be one of {VALID_SCORE_VARIANTS}; '
          f'got {score_variant!r}'
      )
    if score_variant != 'current_additive' and score_mode != 'additive':
      raise ValueError(
          'non-legacy score variants require score_mode=additive so their '
          'return evidence remains in physical return units'
      )
    additive_scales = (
        float(additive_return_scale),
        float(additive_return_std_scale),
        float(additive_log_roughness_scale),
    )
    if any(not np.isfinite(scale) or scale <= 0.0 for scale in additive_scales):
      raise ValueError('all additive score scales must be finite and positive')
    score_evidence_floor = float(score_evidence_floor)
    if not np.isfinite(score_evidence_floor) or score_evidence_floor <= 0.0:
      raise ValueError('score_evidence_floor must be finite and positive')
    decision_rule = str(decision_rule)
    if decision_rule not in VALID_DECISION_RULES:
      raise ValueError(
          f'decision_rule must be one of {VALID_DECISION_RULES}; '
          f'got {decision_rule!r}'
      )
    confidence_z = float(confidence_z)
    switch_threshold = float(switch_threshold)
    if not np.isfinite(confidence_z) or confidence_z < 0.0:
      raise ValueError('confidence_z must be finite and non-negative')
    if not np.isfinite(switch_threshold):
      raise ValueError('switch_threshold must be finite')
    if decision_rule == 'paired_lcb' and score_mode == 'legacy_multiplicative':
      raise ValueError(
          'decision_rule=paired_lcb requires score_mode=additive or multiplicative'
      )
    variant_parameters = (
        float(calibrated_return_weight),
        float(calibrated_roughness_weight),
        float(horizon_switch_cost),
        float(return_first_tolerance),
        float(roughness_discount),
        float(curvature_risk_weight),
        float(bellman_risk_weight),
        float(curvature_risk_scale),
    )
    if any(not np.isfinite(value) or value < 0.0 for value in variant_parameters):
      raise ValueError('score-variant parameters must be finite and non-negative')
    if not 0.0 < float(roughness_discount) <= 1.0:
      raise ValueError('roughness_discount must lie in (0, 1]')
    if float(curvature_risk_scale) <= 0.0:
      raise ValueError('curvature_risk_scale must be positive')
    if initial_horizon is None or int(initial_horizon) not in set(horizons_arr.tolist()):
      initial_horizon = int(horizons_arr[0])
    if candidate_budget is None:
      phase_budget = PHASE_TARGET_SIZES
    else:
      phase_budget = tuple(
          int(candidate_budget[name]) for name in PHASE_NAMES
      )
    nh = horizons_arr.shape[0]
    phase_budget = tuple(min(int(budget), nh) for budget in phase_budget)
    if any(budget < 1 for budget in phase_budget):
      raise ValueError('every candidate budget must include at least one horizon')
    uniform = np.full(nh, 1.0 / max(nh, 1), dtype=np.float32)
    zeros = np.zeros(nh, dtype=np.float32)
    first_query_step = (
        int(query_interval_steps)
        if start_query_step is None
        else int(start_query_step)
    )
    return cls(
        horizons=jnp.asarray(horizons_arr),
        active_mask=jnp.ones(nh, dtype=bool),
        best_h=jnp.asarray(initial_horizon, dtype=jnp.int32),
        score_sum=jnp.asarray(zeros),
        score_sum_sq=jnp.asarray(zeros),
        score_count=jnp.asarray(zeros, dtype=jnp.int32),
        gauss_mean=jnp.asarray(zeros),
        gauss_post_std=jnp.ones(nh, dtype=jnp.float32),
        prob=jnp.asarray(uniform),
        entropy=jnp.asarray(math.log(max(nh, 1)), dtype=jnp.float32),
        norm_entropy=jnp.asarray(1.0, dtype=jnp.float32),
        phase_id=jnp.asarray(0, dtype=jnp.int32),
        phase_sample_count_A=jnp.zeros(nh, dtype=jnp.int32),
        phase_sample_count_B1=jnp.zeros(nh, dtype=jnp.int32),
        phase_sample_count_B2=jnp.zeros(nh, dtype=jnp.int32),
        phase_sample_count_B3=jnp.zeros(nh, dtype=jnp.int32),
        phase_sample_count_B4=jnp.zeros(nh, dtype=jnp.int32),
        query_interval_steps=jnp.asarray(query_interval_steps, dtype=jnp.int32),
        start_query_step=jnp.asarray(first_query_step, dtype=jnp.int32),
        next_query_step=jnp.asarray(first_query_step, dtype=jnp.int32),
        hmax=int(hmax),
        roughness_probe=roughness_probe,
        num_roughness_probes=num_roughness_probes,
        score_mode=score_mode,
        score_variant=score_variant,
        additive_return_scale=additive_scales[0],
        additive_return_std_scale=additive_scales[1],
        additive_log_roughness_scale=additive_scales[2],
        score_evidence_floor=score_evidence_floor,
        decision_rule=decision_rule,
        confidence_z=confidence_z,
        switch_threshold=switch_threshold,
        robust_return=robust_return,
        phase_pruning_enabled=bool(phase_pruning_enabled),
        phase_min_samples_to_drop=int(phase_min_samples_to_drop),
        candidate_budget=phase_budget,
        selection_return_power=float(selection_return_power),
        roughness_weight=float(roughness_weight),
        return_std_weight=float(return_std_weight),
        learner_proxy_enabled=bool(learner_proxy_enabled),
        learner_proxy_weight=float(learner_proxy_weight),
        learner_proxy_mode=str(learner_proxy_mode),
        local_window_radius=int(local_window_radius),
        max_transition_delta=int(max_transition_delta),
        incumbent_switch_margin=float(incumbent_switch_margin),
        credible_transition_enabled=bool(credible_transition_enabled),
        credible_transition_rule=str(credible_transition_rule),
        credible_transition_min_prob=float(credible_transition_min_prob),
        transition_cost_scale=float(transition_cost_scale),
        transition_risk_weight=float(transition_risk_weight),
        transition_min_expected_net=float(transition_min_expected_net),
        transition_model_weight=float(transition_model_weight),
        transition_probe_weight=float(transition_probe_weight),
        transition_planner_weight=float(transition_planner_weight),
        transition_roughness_weight=float(transition_roughness_weight),
        transition_return_std_weight=float(transition_return_std_weight),
        transition_uncertainty_floor=float(transition_uncertainty_floor),
        calibrated_return_weight=float(calibrated_return_weight),
        calibrated_roughness_weight=float(calibrated_roughness_weight),
        horizon_switch_cost=float(horizon_switch_cost),
        return_first_tolerance=float(return_first_tolerance),
        roughness_discount=float(roughness_discount),
        curvature_risk_weight=float(curvature_risk_weight),
        bellman_risk_weight=float(bellman_risk_weight),
        curvature_risk_scale=float(curvature_risk_scale),
    )

  def active_horizons(self) -> np.ndarray:
    horizons = np.asarray(self.horizons)
    return horizons[np.asarray(self.active_mask)]

  def phase_name(self) -> str:
    return PHASE_NAMES[int(np.asarray(self.phase_id))]

  def should_query(self, step: int) -> bool:
    return step >= int(np.asarray(self.next_query_step))

@struct.dataclass
class DenseQueryResult:
  horizon_state: HorizonSearchState
  selected_horizon: jax.Array
  proposed_horizon: jax.Array
  candidate_horizons: jax.Array
  candidate_mask: jax.Array
  fitness: jax.Array
  decision_score: jax.Array
  score_se: jax.Array
  score_lcb: jax.Array
  paired_confidence_available: jax.Array
  deployment_score: jax.Array
  return_term: jax.Array
  roughness_term: jax.Array
  sigma_r_term: jax.Array
  learner_proxy_term: jax.Array
  transition_cost: jax.Array
  transition_adjusted_score: jax.Array
  switch_probability: jax.Array
  expected_improvement: jax.Array
  expected_loss: jax.Array
  expected_net_benefit: jax.Array
  incumbent_deployment_score: jax.Array
  proposed_deployment_score: jax.Array
  proposed_transition_cost: jax.Array
  proposed_switch_probability: jax.Array
  proposed_expected_net_benefit: jax.Array
  robust_return: jax.Array
  prefix_objectives: jax.Array
  probe_prefixes: jax.Array
  planner_prefix_returns: jax.Array
  roughness: jax.Array
  roughness_nested: jax.Array
  roughness_nested_valid: jax.Array
  roughness_projections: jax.Array
  env_mean: jax.Array
  env_std: jax.Array
  local_roughness: jax.Array
  local_model_error: jax.Array
  bellman_residual: jax.Array
  curvature_bellman_risk: jax.Array
  horizon_switch_cost: jax.Array


@struct.dataclass
class DenseQueryModelStage:
  prefix_objectives: jax.Array
  probe_prefixes: jax.Array
  planner_prefix_returns: jax.Array
  roughness: jax.Array
  roughness_nested: jax.Array
  roughness_nested_valid: jax.Array
  roughness_projections: jax.Array
  candidate_horizons: jax.Array
  candidate_mask: jax.Array
  candidate_indices: jax.Array


@dataclass(frozen=True)
class DenseQueryKernelBundle:
  model_stage: Callable[..., DenseQueryModelStage]
  env_stage: Callable[..., Tuple[jax.Array, jax.Array, jax.Array, jax.Array]]
  finalize_stage: Callable[..., DenseQueryResult]


_DENSE_QUERY_KERNEL_CACHE: Dict[Tuple[int, int, int], DenseQueryKernelBundle] = {}


def _accepts_keyword(callable_obj: Callable[..., Any], keyword: str) -> bool:
  """Best-effort compatibility check for optional environment API fields."""
  try:
    parameters = inspect.signature(callable_obj).parameters.values()
  except (TypeError, ValueError):
    return False
  return any(
      parameter.name == keyword or parameter.kind == inspect.Parameter.VAR_KEYWORD
      for parameter in parameters
  )


def _normalise_jax(values: jax.Array, inverse: bool = False) -> jax.Array:
  values = jnp.asarray(values, dtype=jnp.float32)
  vmin = jnp.min(values)
  vmax = jnp.max(values)
  scale = jnp.maximum(vmax - vmin, EPS)
  out = jnp.where(
      jnp.logical_or(~jnp.isfinite(vmin), ~jnp.isfinite(vmax)),
      jnp.ones_like(values),
      (values - vmin) / scale,
  )
  out = jnp.where(scale <= EPS, jnp.ones_like(out), out)
  if inverse:
    out = 1.0 - out
  return jnp.clip(out, 0.0, 1.0)


def _normalise_masked_jax(values: jax.Array,
                          mask: jax.Array,
                          inverse: bool = False,
                          constant_value: float = 1.0) -> jax.Array:
  """Min-max normalises only finite entries selected by ``mask``.

  Non-evaluated entries are returned as zero so they cannot leak into a later
  score. If the selected values have no range, ``constant_value`` determines
  the neutral value: quality terms use one, while cost terms use zero.
  """
  values = jnp.asarray(values, dtype=jnp.float32)
  mask = jnp.logical_and(jnp.asarray(mask, dtype=bool), jnp.isfinite(values))
  vmin = jnp.min(jnp.where(mask, values, jnp.inf))
  vmax = jnp.max(jnp.where(mask, values, -jnp.inf))
  has_values = jnp.any(mask)
  scale = vmax - vmin
  varying = jnp.logical_and(has_values, jnp.isfinite(scale) & (scale > EPS))
  scaled = (values - vmin) / jnp.maximum(scale, EPS)
  scaled = jnp.clip(scaled, 0.0, 1.0)
  if inverse:
    scaled = 1.0 - scaled
  normalised = jnp.where(varying, scaled, jnp.full_like(values, constant_value))
  return jnp.where(mask, normalised, 0.0)


def _compose_decision_score(return_term: jax.Array,
                            roughness_term: jax.Array,
                            sigma_r_term: jax.Array,
                            learner_proxy_term: jax.Array,
                            weights: jax.Array,
                            score_mode: str) -> jax.Array:
  """Reproduces the historical multiplicative score from normalised terms."""
  terms = jnp.stack(
      (return_term, roughness_term, sigma_r_term, learner_proxy_term),
      axis=0,
  )
  weights = jnp.maximum(jnp.asarray(weights, dtype=jnp.float32), 0.0)
  if score_mode == 'legacy_multiplicative':
    return jnp.prod(
        jnp.power(jnp.clip(terms, EPS, 1.0), weights[:, None]),
        axis=0,
    )
  raise ValueError(f'Not a legacy score mode: {score_mode!r}')


def _incumbent_relative_decision_score(
    env_mean: jax.Array,
    env_std: jax.Array,
    roughness: jax.Array,
    incumbent_idx: jax.Array,
    candidate_mask: jax.Array,
    score_mode: str,
    weights: jax.Array,
    additive_scales: jax.Array,
    evidence_floor: float,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Builds the pilot's fixed-reference score and its signed components.

  Both pilot modes compare raw evidence with the incumbent and therefore make
  the incumbent exactly zero. ``multiplicative`` is represented in log space,
  which is the log ratio of positive, floored utilities.
  """
  env_mean = jnp.asarray(env_mean, dtype=jnp.float32)
  env_std = jnp.asarray(env_std, dtype=jnp.float32)
  roughness = jnp.asarray(roughness, dtype=jnp.float32)
  candidate_mask = jnp.asarray(candidate_mask, dtype=bool)
  weights = jnp.maximum(jnp.asarray(weights, dtype=jnp.float32), 0.0)
  scales = jnp.asarray(additive_scales, dtype=jnp.float32)
  floor = jnp.asarray(evidence_floor, dtype=jnp.float32)

  if score_mode == 'additive':
    return_term = (env_mean - env_mean[incumbent_idx]) / scales[0]
    sigma_term = -(env_std - env_std[incumbent_idx]) / scales[1]
    log_roughness = jnp.log(jnp.maximum(roughness, floor))
    roughness_term = -(
        log_roughness - log_roughness[incumbent_idx]
    ) / scales[2]
  elif score_mode == 'multiplicative':
    log_return = jnp.log(jnp.maximum(env_mean, floor))
    log_sigma = jnp.log(jnp.maximum(env_std, floor))
    log_roughness = jnp.log(jnp.maximum(roughness, floor))
    return_term = log_return - log_return[incumbent_idx]
    sigma_term = -(log_sigma - log_sigma[incumbent_idx])
    roughness_term = -(log_roughness - log_roughness[incumbent_idx])
  else:
    raise ValueError(f'Not an incumbent-relative score mode: {score_mode!r}')

  decision_score = (
      weights[0] * return_term +
      weights[1] * roughness_term +
      weights[2] * sigma_term
  )
  return tuple(
      jnp.where(candidate_mask, value, 0.0)
      for value in (decision_score, return_term, roughness_term, sigma_term)
  )


def _influence_standard_error(influence: jax.Array) -> jax.Array:
  """Standard error of a mean from paired influence-function rows."""
  influence = jnp.asarray(influence, dtype=jnp.float32)
  sample_count = influence.shape[-1]
  if sample_count < 2:
    return jnp.zeros(influence.shape[:-1], dtype=jnp.float32)
  centred = influence - jnp.mean(influence, axis=-1, keepdims=True)
  variance_of_mean = jnp.sum(jnp.square(centred), axis=-1) / (
      sample_count * (sample_count - 1)
  )
  return jnp.sqrt(jnp.maximum(variance_of_mean, 0.0))


def _paired_score_standard_error(
    paired_returns: jax.Array,
    env_mean: jax.Array,
    env_std: jax.Array,
    roughness_projections: jax.Array,
    roughness: jax.Array,
    incumbent_idx: jax.Array,
    candidate_mask: jax.Array,
    score_mode: str,
    weights: jax.Array,
    additive_scales: jax.Array,
    evidence_floor: float,
) -> jax.Array:
  """Delta-method SE preserving candidate/incumbent pairing and covariance."""
  paired_returns = jnp.asarray(paired_returns, dtype=jnp.float32)
  env_mean = jnp.asarray(env_mean, dtype=jnp.float32)
  env_std = jnp.asarray(env_std, dtype=jnp.float32)
  weights = jnp.maximum(jnp.asarray(weights, dtype=jnp.float32), 0.0)
  scales = jnp.asarray(additive_scales, dtype=jnp.float32)
  floor = jnp.asarray(evidence_floor, dtype=jnp.float32)

  centred_returns = paired_returns - env_mean[:, None]
  influence_mean = centred_returns
  influence_std = (
      jnp.square(centred_returns) - jnp.square(env_std[:, None])
  ) / (2.0 * jnp.maximum(env_std[:, None], floor))

  if score_mode == 'additive':
    return_influence = (
        weights[0] *
        (influence_mean - influence_mean[incumbent_idx]) / scales[0] -
        weights[2] *
        (influence_std - influence_std[incumbent_idx]) / scales[1]
    )
  elif score_mode == 'multiplicative':
    mean_denominator = jnp.maximum(env_mean, floor)[:, None]
    std_denominator = jnp.maximum(env_std, floor)[:, None]
    return_influence = (
        weights[0] * (
            influence_mean / mean_denominator -
            influence_mean[incumbent_idx] / mean_denominator[incumbent_idx]
        ) -
        weights[2] * (
            influence_std / std_denominator -
            influence_std[incumbent_idx] / std_denominator[incumbent_idx]
        )
    )
  else:
    return jnp.zeros_like(env_mean)
  return_se = _influence_standard_error(return_influence)

  num_probes = roughness_projections.shape[0]
  if num_probes < 2:
    roughness_se = jnp.zeros_like(return_se)
  else:
    projections = jnp.swapaxes(roughness_projections, 0, 1)
    squared = jnp.square(projections)
    mean_squared = jnp.mean(squared, axis=-1, keepdims=True)
    influence_rms = (squared - mean_squared) / (
        2.0 * jnp.maximum(roughness, floor)[:, None]
    )
    influence_log_roughness = influence_rms / jnp.maximum(
        roughness, floor
    )[:, None]
    roughness_influence = -weights[1] * (
        influence_log_roughness - influence_log_roughness[incumbent_idx]
    )
    if score_mode == 'additive':
      roughness_influence = roughness_influence / scales[2]
    roughness_se = _influence_standard_error(roughness_influence)

  score_se = jnp.sqrt(jnp.square(return_se) + jnp.square(roughness_se))
  score_se = score_se.at[incumbent_idx].set(0.0)
  return jnp.where(candidate_mask, score_se, 0.0)


def _paired_return_standard_error(
    paired_returns: jax.Array,
    incumbent_idx: jax.Array,
    candidate_mask: jax.Array,
) -> jax.Array:
  """SE of paired candidate-minus-incumbent return differences."""
  paired_returns = jnp.asarray(paired_returns, dtype=jnp.float32)
  paired_difference = paired_returns - paired_returns[incumbent_idx]
  standard_error = _influence_standard_error(paired_difference)
  standard_error = standard_error.at[incumbent_idx].set(0.0)
  return jnp.where(candidate_mask, standard_error, 0.0)


def _discounted_prefix_normalizer(horizons: jax.Array,
                                  discount: float) -> jax.Array:
  """Discounted effective length used to remove mechanical horizon growth."""
  horizons = jnp.maximum(jnp.asarray(horizons, dtype=jnp.float32), 1.0)
  discount_array = jnp.asarray(discount, dtype=jnp.float32)
  geometric = (
      1.0 - jnp.power(discount_array, horizons)
  ) / jnp.maximum(1.0 - discount_array, EPS)
  return jnp.where(
      jnp.abs(discount_array - 1.0) <= EPS,
      horizons,
      geometric,
  )


def _score_formulation_evidence(
    state: HorizonSearchState,
    model_stage: DenseQueryModelStage,
    env_mean: jax.Array,
    paired_returns: jax.Array,
    incumbent_idx: jax.Array,
    candidate_mask: jax.Array,
) -> Tuple[
    jax.Array, jax.Array, jax.Array, jax.Array,
    jax.Array, jax.Array, jax.Array,
]:
  """Common return, local roughness, and model-risk evidence for S1--S3."""
  normalizer = _discounted_prefix_normalizer(
      state.horizons,
      state.roughness_discount,
  )
  local_roughness = model_stage.roughness / normalizer
  local_model_error = model_stage.probe_prefixes / normalizer
  bellman_residual = jnp.maximum(
      model_stage.prefix_objectives - model_stage.probe_prefixes,
      0.0,
  ) / normalizer
  curvature_bellman_risk = (
      0.5 * local_roughness * local_model_error +
      state.bellman_risk_weight * bellman_residual
  )
  return_difference = env_mean - env_mean[incumbent_idx]
  return_standard_error = _paired_return_standard_error(
      paired_returns,
      incumbent_idx,
      candidate_mask,
  )
  switch_cost = state.horizon_switch_cost * jnp.abs(
      state.horizons.astype(jnp.float32) -
      state.horizons[incumbent_idx].astype(jnp.float32)
  )
  return tuple(
      jnp.where(candidate_mask, value, 0.0)
      for value in (
          return_difference,
          return_standard_error,
          local_roughness,
          local_model_error,
          bellman_residual,
          curvature_bellman_risk,
      )
  ) + (jnp.where(candidate_mask, switch_cost, 0.0),)


def _standard_normal_cdf(x: jax.Array) -> jax.Array:
  return 0.5 * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))


def _standard_normal_pdf(x: jax.Array) -> jax.Array:
  return jnp.exp(-0.5 * jnp.square(x)) / jnp.sqrt(2.0 * jnp.pi)


def _normalised_shift(values: jax.Array,
                      incumbent_idx: jax.Array,
                      mask: Optional[jax.Array] = None) -> jax.Array:
  values = jnp.asarray(values, dtype=jnp.float32)
  if mask is None:
    mask = jnp.ones_like(values, dtype=bool)
  else:
    mask = jnp.asarray(mask, dtype=bool)
  shift = jnp.abs(values - values[incumbent_idx])
  scale = jnp.max(jnp.where(mask, shift, 0.0))
  normalised = jnp.where(
      scale > EPS,
      jnp.clip(shift / jnp.maximum(scale, EPS), 0.0, 1.0),
      jnp.zeros_like(shift),
  )
  return jnp.where(mask, normalised, 0.0)


def _transition_cost(state: HorizonSearchState,
                     model_stage: DenseQueryModelStage,
                     env_std: jax.Array,
                     incumbent_idx: jax.Array,
                     candidate_eval_mask: jax.Array) -> jax.Array:
  weights = jnp.asarray(
      [
          state.transition_model_weight,
          state.transition_probe_weight,
          state.transition_planner_weight,
          state.transition_roughness_weight,
          state.transition_return_std_weight,
      ],
      dtype=jnp.float32,
  )
  components = jnp.stack(
      [
          _normalised_shift(
              model_stage.prefix_objectives, incumbent_idx, candidate_eval_mask
          ),
          _normalised_shift(
              model_stage.probe_prefixes, incumbent_idx, candidate_eval_mask
          ),
          _normalised_shift(
              model_stage.planner_prefix_returns, incumbent_idx, candidate_eval_mask
          ),
          _normalise_masked_jax(
              model_stage.roughness, candidate_eval_mask, constant_value=0.0
          ),
          _normalise_masked_jax(
              env_std, candidate_eval_mask, constant_value=0.0
          ),
      ],
      axis=0,
  )
  weight_sum = jnp.maximum(jnp.sum(weights), EPS)
  cost = state.transition_cost_scale * jnp.sum(components * weights[:, None], axis=0) / weight_sum
  cost = cost.at[incumbent_idx].set(0.0)
  return jnp.where(candidate_eval_mask, jnp.clip(cost, 0.0, 1.0), 0.0)


def _phase_counts_matrix(state: HorizonSearchState) -> jax.Array:
  return jnp.stack(
      (
          state.phase_sample_count_A,
          state.phase_sample_count_B1,
          state.phase_sample_count_B2,
          state.phase_sample_count_B3,
          state.phase_sample_count_B4,
      ),
      axis=0,
  )


def _phase_counts(state: HorizonSearchState) -> np.ndarray:
  return np.asarray(_phase_counts_matrix(state)[state.phase_id])


def _replace_phase_counts(state: HorizonSearchState,
                          counts: jax.Array,
                          phase_id: Optional[jax.Array] = None) -> HorizonSearchState:
  phase_index = state.phase_id if phase_id is None else jnp.asarray(phase_id, dtype=jnp.int32)
  matrix = _phase_counts_matrix(state).at[phase_index].set(
      jnp.asarray(counts, dtype=jnp.int32)
  )
  return state.replace(
      phase_sample_count_A=matrix[0],
      phase_sample_count_B1=matrix[1],
      phase_sample_count_B2=matrix[2],
      phase_sample_count_B3=matrix[3],
      phase_sample_count_B4=matrix[4],
  )


def _phase_budget(state: HorizonSearchState) -> jax.Array:
  budgets = jnp.asarray(state.candidate_budget, dtype=jnp.int32)
  return budgets[state.phase_id]


def _rank_scores(state: HorizonSearchState) -> jax.Array:
  incumbent_mask = state.horizons == state.best_h
  eligible_mask = jnp.logical_or(state.active_mask, incumbent_mask)
  return jnp.where(
      eligible_mask,
      incumbent_mask.astype(jnp.float32) * 1e6 +
      state.prob * 1e3 +
      state.gauss_mean * 1e1 -
      state.horizons.astype(jnp.float32) * 1e-3,
      -jnp.inf,
  )


def _candidate_scores(state: HorizonSearchState) -> jax.Array:
  counts = _phase_counts_matrix(state)[state.phase_id].astype(jnp.float32)
  incumbent_mask = state.horizons == state.best_h
  if state.local_window_radius > 0:
    candidate_scope = jnp.logical_and(
        state.active_mask,
        jnp.abs(state.horizons - state.best_h) <= state.local_window_radius,
    )
  else:
    candidate_scope = state.active_mask
  candidate_scope = jnp.logical_or(candidate_scope, incumbent_mask)
  under_sampled_mask = jnp.logical_and(
      candidate_scope,
      counts < float(state.phase_min_samples_to_drop),
  )
  return jnp.where(
      candidate_scope,
      incumbent_mask.astype(jnp.float32) * 1e6 +
      under_sampled_mask.astype(jnp.float32) * 1e5 +
      state.prob * 1e3 +
      state.gauss_mean * 1e1 -
      counts * 1e-2 -
      state.horizons.astype(jnp.float32) * 1e-3,
      -jnp.inf,
  )


def _select_candidate_slots(state: HorizonSearchState,
                            candidate_slots: int) -> Tuple[jax.Array, jax.Array, jax.Array]:
  scores, indices = jax.lax.top_k(_candidate_scores(state), candidate_slots)
  mask = jnp.isfinite(scores)
  return state.horizons[indices], mask, indices


@jax.jit
def _maybe_advance_phase(state: HorizonSearchState) -> HorizonSearchState:
  phase_counts = _phase_counts_matrix(state)[state.phase_id]
  ready = jnp.logical_and(
      jnp.logical_and(
          state.phase_pruning_enabled,
          state.phase_id < len(PHASE_NAMES) - 1,
      ),
      jnp.all(jnp.where(
          state.active_mask,
          phase_counts >= state.phase_min_samples_to_drop,
          True,
      )),
  )

  def advance(st: HorizonSearchState) -> HorizonSearchState:
    next_phase_id = st.phase_id + 1
    next_budget = jnp.asarray(st.candidate_budget, dtype=jnp.int32)[next_phase_id]
    _, ranked_indices = jax.lax.top_k(_rank_scores(st), st.horizons.shape[0])
    keep_values = jnp.arange(st.horizons.shape[0], dtype=jnp.int32) < next_budget
    new_active_mask = jnp.zeros_like(st.active_mask).at[ranked_indices].set(keep_values)
    st = st.replace(
        active_mask=new_active_mask,
        phase_id=next_phase_id,
    )
    return _replace_phase_counts(
        st,
        jnp.zeros_like(phase_counts, dtype=jnp.int32),
        phase_id=next_phase_id,
    )

  return jax.lax.cond(ready, advance, lambda st: st, state)


@partial(jax.jit, static_argnames=('mc_samples',))
def _update_distribution(state: HorizonSearchState,
                         key: jax.Array,
                         selected_horizon: jax.Array,
                         mc_samples: int = 512) -> HorizonSearchState:
  counts = state.score_count.astype(jnp.float32)
  mean = state.score_sum / jnp.maximum(counts, 1.0)
  var = state.score_sum_sq / jnp.maximum(counts, 1.0) - jnp.square(mean)
  var = jnp.where(state.score_count > 1, jnp.maximum(var, EPS), 1.0)
  post_std = jnp.where(
      state.score_count > 0,
      jnp.sqrt(var / jnp.maximum(counts, 1.0)),
      1.0,
  )

  samples = mean[None, :] + post_std[None, :] * jax.random.normal(
      key,
      shape=(mc_samples, mean.shape[0]),
      dtype=jnp.float32,
  )
  samples = jnp.where(state.active_mask[None, :], samples, -jnp.inf)
  winners = jnp.argmax(samples, axis=1)
  prob = jnp.zeros((mean.shape[0],), dtype=jnp.float32).at[winners].add(
      jnp.ones((mc_samples,), dtype=jnp.float32)
  )
  prob_total = jnp.sum(prob)
  active_count = jnp.maximum(jnp.sum(state.active_mask.astype(jnp.float32)), 1.0)
  uniform = state.active_mask.astype(jnp.float32) / active_count
  prob = jnp.where(prob_total > 0.0, prob / prob_total, uniform)
  prob = jnp.where(state.active_mask, prob, 0.0)
  entropy = -jnp.sum(
      jnp.where(state.active_mask, prob * jnp.log(prob + EPS), 0.0)
  )
  norm_entropy = entropy / jnp.log(jnp.maximum(active_count, 2.0))

  state = state.replace(
      gauss_mean=mean.astype(jnp.float32),
      gauss_post_std=post_std.astype(jnp.float32),
      prob=prob.astype(jnp.float32),
      entropy=entropy.astype(jnp.float32),
      norm_entropy=norm_entropy.astype(jnp.float32),
      best_h=jnp.asarray(selected_horizon, dtype=jnp.int32),
  )
  return _maybe_advance_phase(state)


@jax.jit
def _per_step_losses(agent, batch: Mapping[str, jax.Array], key: jax.Array):
  hmax = batch['action'].shape[0]
  batch_size = batch['action'].shape[1]
  keys = jax.random.split(key, 1 + 3 * hmax)
  encoder_key = keys[0]
  sample_action_keys = keys[1:1 + hmax]
  target_q_keys = keys[1 + hmax:1 + 2 * hmax]
  value_keys = keys[1 + 2 * hmax:1 + 3 * hmax]

  all_obs = jax.tree.map(
      lambda x, y: jnp.stack([x, y], axis=0),
      batch['observation'],
      batch['next_observation'],
  )
  all_zs = agent.model.encode(
      obs=all_obs,
      params=agent.model.encoder.params,
      key=encoder_key,
  )
  encoder_zs = all_zs[0]
  next_zs = all_zs[1]
  done = jnp.logical_or(batch['terminated'], batch['truncated'])
  z0 = encoder_zs[0]

  def step_fn(carry, inputs):
    z_t, finished = carry
    action_t, reward_t, terminated_t, done_t, next_z_t, sample_key, target_q_key, value_key = inputs
    next_latent = agent.model.next(
        z=z_t,
        a=action_t,
        params=agent.model.dynamics_model.params,
    )
    cons_loss = jnp.mean((next_latent - sg(next_z_t)) ** 2, axis=-1)
    reward_logits = agent.model.reward(
        z=z_t,
        a=action_t,
        params=agent.model.reward_model.params,
    )[1]
    reward_loss = soft_crossentropy(
        pred_logits=reward_logits,
        target=reward_t,
        low=agent.model.symlog_min,
        high=agent.model.symlog_max,
        num_bins=agent.model.num_bins,
    )
    next_action = agent.model.sample_actions(
        z=next_z_t,
        deterministic=False,
        params=agent.model.policy_model.params,
        key=sample_key,
    )[0]
    target_Q, _ = agent.model.Q(
        z=next_z_t,
        a=next_action,
        params=agent.model.target_value_model.params,
        key=target_q_key,
    )
    td_target = reward_t + (1 - terminated_t) * agent.discount * target_Q.mean(axis=0)
    _, value_logits = agent.model.Q(
        z=z_t,
        a=action_t,
        params=agent.model.value_model.params,
        key=value_key,
    )
    value_loss = soft_crossentropy(
        pred_logits=value_logits,
        target=sg(td_target),
        low=agent.model.symlog_min,
        high=agent.model.symlog_max,
        num_bins=agent.model.num_bins,
    ).mean(axis=0)
    total = jnp.where(
        finished,
        0.0,
        agent.consistency_loss_scale * cons_loss +
        agent.reward_loss_scale * reward_loss +
        agent.value_loss_scale * value_loss,
    )
    probe_total = jnp.where(
        finished,
        0.0,
        agent.consistency_loss_scale * cons_loss +
        agent.reward_loss_scale * reward_loss,
    )
    finished = jnp.logical_or(finished, done_t)
    return (next_latent, finished), (jnp.mean(total), jnp.mean(probe_total), next_latent)

  (_, _), (per_step_total, per_step_probe, rollout_states) = jax.lax.scan(
      step_fn,
      (z0, jnp.zeros((batch_size,), dtype=bool)),
      (
          batch['action'],
          batch['reward'],
          batch['terminated'],
          done,
          next_zs,
          sample_action_keys,
          target_q_keys,
          value_keys,
      ),
  )
  rollout_states = jnp.concatenate([z0[None, ...], rollout_states], axis=0)
  return per_step_total, per_step_probe, rollout_states, z0


@partial(jax.jit, static_argnames=('hmax', 'population_size'))
def _planner_prefix_returns(agent,
                            z0: jax.Array,
                            hmax: int,
                            key: jax.Array,
                            population_size: Optional[int] = None) -> jax.Array:
  pop = int(population_size or agent.population_size)
  if z0.ndim > 1:
    z0 = jnp.mean(z0, axis=0)
  keys = jax.random.split(key, 1 + 2 * hmax)
  action_keys = keys[1:1 + hmax]
  q_keys = keys[1 + hmax:1 + 2 * hmax]
  z_t = z0[None, :].repeat(pop, axis=0)

  def planner_step(z_carry, action_key):
    action = agent.model.sample_actions(
        z=z_carry,
        deterministic=False,
        params=agent.model.policy_model.params,
        key=action_key,
    )[0]
    reward, _ = agent.model.reward(
        z=z_carry,
        a=action,
        params=agent.model.reward_model.params,
    )
    next_z = agent.model.next(
        z=z_carry,
        a=action,
        params=agent.model.dynamics_model.params,
    )
    return next_z, (reward, next_z)

  _, (rewards, next_states) = jax.lax.scan(planner_step, z_t, action_keys)
  rewards = jnp.swapaxes(rewards, 0, 1)
  states = jnp.swapaxes(jnp.concatenate([z_t[None, ...], next_states], axis=0), 0, 1)
  discounts = jnp.power(agent.discount, jnp.arange(hmax, dtype=jnp.float32))
  discounted_rewards = rewards * discounts[None, :]
  prefix_rewards = jnp.cumsum(discounted_rewards, axis=1)

  def bootstrap_step(inputs):
    state_h, q_key, horizon_idx = inputs
    next_action = agent.model.sample_actions(
        z=state_h,
        deterministic=False,
        params=agent.model.policy_model.params,
        key=q_key,
    )[0]
    q_values, _ = agent.model.Q(
        z=state_h,
        a=next_action,
        params=agent.model.value_model.params,
        key=q_key,
    )
    return jnp.power(agent.discount, horizon_idx + 1) * q_values.mean(axis=0)

  bootstrap_values = jax.vmap(bootstrap_step)(
      (states[:, 1:, :].transpose(1, 0, 2), q_keys, jnp.arange(hmax, dtype=jnp.int32))
  ).transpose(1, 0)
  returns = prefix_rewards + bootstrap_values
  return jnp.max(returns, axis=0)


def _nested_roughness_from_projections(
    stacked: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Returns the full-stack RMS and nested-prefix RMS estimates.

  ``stacked`` has one directional derivative per row. Prefix estimates reuse
  exactly the same directions, so an M=64 query also exposes M=2,...,32
  precision diagnostics without another JVP evaluation.
  """
  stacked = jnp.asarray(stacked, dtype=jnp.float32)
  num_probes = stacked.shape[0]
  output_shape = stacked.shape[1:]
  if num_probes == 0:
    return (
        jnp.zeros(output_shape, dtype=jnp.float32),
        jnp.zeros((len(ROUGHNESS_PROBE_COUNTS),) + output_shape, dtype=jnp.float32),
        jnp.zeros((len(ROUGHNESS_PROBE_COUNTS),), dtype=bool),
    )

  squared_prefix = jnp.cumsum(jnp.square(stacked), axis=0)
  counts = jnp.asarray(ROUGHNESS_PROBE_COUNTS, dtype=jnp.int32)
  indices = jnp.clip(counts - 1, 0, num_probes - 1)
  reshape = (counts.shape[0],) + (1,) * len(output_shape)
  nested = jnp.sqrt(
      squared_prefix[indices] / counts.astype(jnp.float32).reshape(reshape) + EPS
  )
  valid = counts <= num_probes
  nested = jnp.where(valid.reshape(reshape), nested, jnp.zeros_like(nested))
  selected = jnp.sqrt(jnp.mean(jnp.square(stacked), axis=0) + EPS)
  return selected, nested, valid


@partial(jax.jit, static_argnames=('num_probes',))
def _roughness(agent,
               batch: Mapping[str, jax.Array],
               horizons: jax.Array,
               key: jax.Array,
               num_probes: int = 2,
               ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  hmax = batch['action'].shape[0]
  horizon_idx = jnp.clip(jnp.asarray(horizons, dtype=jnp.int32) - 1, 0, hmax - 1)

  def probe_prefix(params):
    dynamics_params, reward_params = params
    encoder_key = jax.random.split(key, 1)[0]
    all_obs = jax.tree.map(
        lambda x, y: jnp.stack([x, y], axis=0),
        batch['observation'],
        batch['next_observation'],
    )
    all_zs = agent.model.encode(
        obs=all_obs,
        params=agent.model.encoder.params,
        key=encoder_key,
    )
    next_zs = all_zs[1]
    done = jnp.logical_or(batch['terminated'], batch['truncated'])

    def step_fn(carry, inputs):
      z_t, finished = carry
      action_t, reward_t, done_t, next_z_t = inputs
      next_latent = agent.model.next(
          z=z_t,
          a=action_t,
          params=dynamics_params,
      )
      cons_loss = jnp.mean((next_latent - sg(next_z_t)) ** 2, axis=-1)
      reward_logits = agent.model.reward(
          z=z_t,
          a=action_t,
          params=reward_params,
      )[1]
      reward_loss = soft_crossentropy(
          pred_logits=reward_logits,
          target=reward_t,
          low=agent.model.symlog_min,
          high=agent.model.symlog_max,
          num_bins=agent.model.num_bins,
      )
      step_loss = jnp.where(
          finished,
          0.0,
          agent.consistency_loss_scale * cons_loss +
          agent.reward_loss_scale * reward_loss,
      )
      finished = jnp.logical_or(finished, done_t)
      return (next_latent, finished), jnp.mean(step_loss)

    (_, _), step_losses = jax.lax.scan(
        step_fn,
        (all_zs[0][0], jnp.zeros((batch['action'].shape[1],), dtype=bool)),
        (batch['action'], batch['reward'], done, next_zs),
    )
    return jnp.cumsum(step_losses)[horizon_idx]

  params = (agent.model.dynamics_model.params, agent.model.reward_model.params)
  if num_probes == 0:
    empty = jnp.zeros((0, horizons.shape[0]), dtype=jnp.float32)
    roughness, nested, valid = _nested_roughness_from_projections(empty)
    return roughness, nested, valid, empty
  proj_keys = jax.random.split(key, num_probes)

  def tangent_from_key(tangent_key, param_tree):
    leaves, treedef = jax.tree_util.tree_flatten(param_tree)
    tangent_keys = jax.random.split(tangent_key, len(leaves))
    tangent_leaves = [
        jax.random.normal(k, shape=leaf.shape, dtype=leaf.dtype)
        for k, leaf in zip(tangent_keys, leaves)
    ]
    return jax.tree_util.tree_unflatten(treedef, tangent_leaves)

  def project(proj_key):
    dyn_key, rew_key = jax.random.split(proj_key)
    tangent = (
        tangent_from_key(dyn_key, params[0]),
        tangent_from_key(rew_key, params[1]),
    )
    _, directional = jax.jvp(probe_prefix, (params,), (tangent,))
    return directional

  stacked = jax.vmap(project)(proj_keys)
  roughness, nested, valid = _nested_roughness_from_projections(stacked)
  return roughness, nested, valid, stacked


@jax.jit
def _state_with_score_updates(state: HorizonSearchState,
                              candidate_indices: jax.Array,
                              candidate_mask: jax.Array,
                              candidate_scores: jax.Array,
                              query_step: jax.Array,
                              selected_horizon: jax.Array,
                              key: jax.Array) -> HorizonSearchState:
  score_delta = jnp.where(candidate_mask, candidate_scores, 0.0)
  count_delta = candidate_mask.astype(state.score_count.dtype)
  score_sum = state.score_sum.at[candidate_indices].add(score_delta)
  score_sum_sq = state.score_sum_sq.at[candidate_indices].add(jnp.square(score_delta))
  score_count = state.score_count.at[candidate_indices].add(count_delta)
  phase_counts = _phase_counts_matrix(state)[state.phase_id].astype(state.score_count.dtype)
  phase_counts = phase_counts.at[candidate_indices].add(count_delta.astype(phase_counts.dtype))

  state = state.replace(
      score_sum=score_sum,
      score_sum_sq=score_sum_sq,
      score_count=score_count,
      best_h=jnp.asarray(selected_horizon, dtype=jnp.int32),
      next_query_step=jnp.asarray(
          query_step + state.query_interval_steps,
          dtype=jnp.int32,
      ),
  )
  state = _replace_phase_counts(state, phase_counts)
  return _update_distribution(state, key, selected_horizon)


def _build_dense_query_kernel(eval_state: Any,
                              candidate_slots: int,
                              env_eval_steps: Optional[int]) -> DenseQueryKernelBundle:
  supports_paired_eval = hasattr(
      eval_state, 'evaluate_candidate_horizons_dense_paired'
  )
  supports_legacy_env_eval = hasattr(
      eval_state, 'evaluate_candidate_horizons_dense'
  )
  supports_env_eval = supports_paired_eval or supports_legacy_env_eval
  env_eval_callable = (
      eval_state.evaluate_candidate_horizons_dense_paired
      if supports_paired_eval
      else (
          eval_state.evaluate_candidate_horizons_dense
          if supports_legacy_env_eval else None
      )
  )
  supports_transition_step = (
      supports_env_eval and
      _accepts_keyword(
          env_eval_callable,
          'global_transition_step',
      )
  )
  eval_steps = int(env_eval_steps) if env_eval_steps is not None else None

  @jax.jit
  def run_model_stage(agent,
                      replay_batch: Mapping[str, jax.Array],
                      horizon_state: HorizonSearchState,
                      rng: jax.Array) -> DenseQueryModelStage:
    model_key, planner_key = jax.random.split(rng, 2)
    horizon_indices = jnp.clip(
        horizon_state.horizons - 1,
        0,
        horizon_state.hmax - 1,
    )
    per_step_total, prefix_probe_steps, _, z0 = _per_step_losses(
        agent,
        replay_batch,
        model_key,
    )
    prefix_objectives = jnp.cumsum(per_step_total)[horizon_indices]
    probe_prefixes = jnp.cumsum(prefix_probe_steps)[horizon_indices]
    planner_prefix_returns = _planner_prefix_returns(
        agent,
        z0,
        horizon_state.hmax,
        planner_key,
    )[horizon_indices]
    (
        roughness,
        roughness_nested,
        roughness_nested_valid,
        roughness_projections,
    ) = _roughness(
        agent,
        replay_batch,
        horizon_state.horizons,
        model_key,
        num_probes=horizon_state.num_roughness_probes,
    )

    candidate_horizons, candidate_mask, candidate_indices = _select_candidate_slots(
        horizon_state,
        candidate_slots,
    )
    return DenseQueryModelStage(
        prefix_objectives=prefix_objectives,
        probe_prefixes=probe_prefixes,
        planner_prefix_returns=planner_prefix_returns,
        roughness=roughness,
        roughness_nested=roughness_nested,
        roughness_nested_valid=roughness_nested_valid,
        roughness_projections=roughness_projections,
        candidate_horizons=candidate_horizons,
        candidate_mask=candidate_mask,
        candidate_indices=candidate_indices,
    )

  @jax.jit
  def run_env_stage(agent,
                    model_stage: DenseQueryModelStage,
                    env_key: jax.Array,
                    global_transition_step: jax.Array = 0,
                    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    env_mean = model_stage.planner_prefix_returns
    env_std = jnp.zeros_like(model_stage.planner_prefix_returns)
    if supports_env_eval:
      eval_kwargs = dict(
          agent=agent,
          candidate_horizons=model_stage.candidate_horizons,
          candidate_mask=model_stage.candidate_mask,
          key=env_key,
          steps_per_episode=eval_steps,
      )
      if supports_transition_step:
        eval_kwargs['global_transition_step'] = global_transition_step
      if supports_paired_eval:
        candidate_env_mean, candidate_env_std, candidate_paired_returns = (
            eval_state.evaluate_candidate_horizons_dense_paired(**eval_kwargs)
        )
        if candidate_paired_returns.shape[0] != candidate_slots:
          candidate_paired_returns = jnp.swapaxes(candidate_paired_returns, 0, 1)
        paired_returns = jnp.zeros(
            (env_mean.shape[0], candidate_paired_returns.shape[1]),
            dtype=jnp.float32,
        ).at[model_stage.candidate_indices].set(
            jnp.where(
                model_stage.candidate_mask[:, None],
                candidate_paired_returns,
                0.0,
            )
        )
        paired_returns_valid = jnp.asarray(
            candidate_paired_returns.shape[1] >= 2,
            dtype=bool,
        )
      else:
        candidate_env_mean, candidate_env_std = (
            eval_state.evaluate_candidate_horizons_dense(**eval_kwargs)
        )
        paired_returns = jnp.zeros((env_mean.shape[0], 1), dtype=jnp.float32)
        paired_returns_valid = jnp.asarray(False)
    else:
      candidate_env_mean = model_stage.planner_prefix_returns[model_stage.candidate_indices]
      candidate_env_std = jnp.zeros((candidate_slots,), dtype=jnp.float32)
      paired_returns = jnp.zeros((env_mean.shape[0], 1), dtype=jnp.float32)
      paired_returns_valid = jnp.asarray(False)
    env_mean = env_mean.at[model_stage.candidate_indices].set(
        jnp.where(
            model_stage.candidate_mask,
            candidate_env_mean,
            env_mean[model_stage.candidate_indices],
        )
    )
    env_std = env_std.at[model_stage.candidate_indices].set(
        jnp.where(
            model_stage.candidate_mask,
            candidate_env_std,
            env_std[model_stage.candidate_indices],
        )
    )
    return env_mean, env_std, paired_returns, paired_returns_valid

  @jax.jit
  def run_finalize_stage(horizon_state: HorizonSearchState,
                         model_stage: DenseQueryModelStage,
                         env_mean: jax.Array,
                         env_std: jax.Array,
                         paired_returns: jax.Array,
                         paired_returns_valid: jax.Array,
                         query_step: jax.Array,
                         score_key: jax.Array) -> DenseQueryResult:
    if horizon_state.robust_return == 'mean_minus_std':
      robust_return = env_mean - env_std
    else:
      robust_return = env_mean
    candidate_eval_mask = jnp.zeros_like(horizon_state.active_mask).at[
        model_stage.candidate_indices
    ].set(model_stage.candidate_mask)
    incumbent_idx = jnp.argmin(jnp.abs(horizon_state.horizons - horizon_state.best_h))
    horizon_lengths = jnp.maximum(horizon_state.horizons.astype(jnp.float32), 1.0)
    learner_proxy_raw = (
        model_stage.prefix_objectives / horizon_lengths
        if horizon_state.learner_proxy_mode == 'total_mean_loss'
        else model_stage.probe_prefixes / horizon_lengths
    )
    learner_proxy_term = _normalise_masked_jax(
        learner_proxy_raw, candidate_eval_mask, inverse=True
    )
    learner_proxy_weight = (
        horizon_state.learner_proxy_weight
        if horizon_state.learner_proxy_enabled else 0.0
    )
    score_weights = jnp.asarray(
        [
            horizon_state.selection_return_power,
            horizon_state.roughness_weight,
            horizon_state.return_std_weight,
            learner_proxy_weight,
        ],
        dtype=jnp.float32,
    )
    additive_scales = jnp.asarray(
        [
            horizon_state.additive_return_scale,
            horizon_state.additive_return_std_scale,
            horizon_state.additive_log_roughness_scale,
        ],
        dtype=jnp.float32,
    )
    (
        paired_return_difference,
        paired_return_se,
        local_roughness,
        local_model_error,
        bellman_residual,
        curvature_bellman_risk,
        direct_switch_cost,
    ) = _score_formulation_evidence(
        horizon_state,
        model_stage,
        env_mean,
        paired_returns,
        incumbent_idx,
        candidate_eval_mask,
    )
    if horizon_state.score_mode == 'legacy_multiplicative':
      return_term = _normalise_masked_jax(robust_return, candidate_eval_mask)
      roughness_term = _normalise_masked_jax(
          model_stage.roughness, candidate_eval_mask, inverse=True
      )
      sigma_r_term = _normalise_masked_jax(
          env_std, candidate_eval_mask, inverse=True
      )
      decision_score = _compose_decision_score(
          return_term,
          roughness_term,
          sigma_r_term,
          learner_proxy_term,
          score_weights,
          horizon_state.score_mode,
      )
      score_se = jnp.zeros_like(decision_score)
      paired_confidence_available = jnp.asarray(False)
    elif horizon_state.score_variant == 'current_additive':
      learner_proxy_term = jnp.zeros_like(env_mean)
      (
          decision_score,
          return_term,
          roughness_term,
          sigma_r_term,
      ) = _incumbent_relative_decision_score(
          env_mean=env_mean,
          env_std=env_std,
          roughness=model_stage.roughness,
          incumbent_idx=incumbent_idx,
          candidate_mask=candidate_eval_mask,
          score_mode=horizon_state.score_mode,
          weights=score_weights,
          additive_scales=additive_scales,
          evidence_floor=horizon_state.score_evidence_floor,
      )
      estimated_score_se = _paired_score_standard_error(
          paired_returns=paired_returns,
          env_mean=env_mean,
          env_std=env_std,
          roughness_projections=model_stage.roughness_projections,
          roughness=model_stage.roughness,
          incumbent_idx=incumbent_idx,
          candidate_mask=candidate_eval_mask,
          score_mode=horizon_state.score_mode,
          weights=score_weights,
          additive_scales=additive_scales,
          evidence_floor=horizon_state.score_evidence_floor,
      )
      paired_confidence_available = jnp.asarray(
          paired_returns_valid, dtype=bool
      )
      score_se = jnp.where(
          paired_confidence_available,
          estimated_score_se,
          jnp.zeros_like(estimated_score_se),
      )
    else:
      learner_proxy_term = jnp.zeros_like(env_mean)
      sigma_r_term = jnp.zeros_like(env_mean)
      return_term = (
          horizon_state.calibrated_return_weight *
          paired_return_difference / horizon_state.additive_return_scale
      )
      paired_confidence_available = jnp.asarray(
          paired_returns_valid, dtype=bool
      )
      estimated_score_se = (
          horizon_state.calibrated_return_weight *
          paired_return_se / horizon_state.additive_return_scale
      )
      score_se = jnp.where(
          paired_confidence_available,
          estimated_score_se,
          jnp.zeros_like(estimated_score_se),
      )
      if horizon_state.score_variant == 'calibrated_local_roughness':
        log_local_roughness = jnp.log(jnp.maximum(
            local_roughness,
            horizon_state.score_evidence_floor,
        ))
        roughness_term = -horizon_state.calibrated_roughness_weight * (
            log_local_roughness - log_local_roughness[incumbent_idx]
        ) / horizon_state.additive_log_roughness_scale
        decision_score = return_term + roughness_term - direct_switch_cost
      elif horizon_state.score_variant == 'return_first':
        # Roughness is deliberately excluded from the primary objective. It
        # is used only to break ties inside the credible return set below.
        roughness_term = jnp.zeros_like(env_mean)
        decision_score = return_term
      elif horizon_state.score_variant == 'curvature_bellman':
        roughness_term = -horizon_state.curvature_risk_weight * (
            curvature_bellman_risk -
            curvature_bellman_risk[incumbent_idx]
        ) / horizon_state.curvature_risk_scale
        decision_score = return_term + roughness_term - direct_switch_cost
      else:  # Guarded by HorizonSearchState.create.
        raise ValueError(
            f'Unsupported score variant {horizon_state.score_variant!r}'
        )
    decision_score = jnp.where(candidate_eval_mask, decision_score, 0.0)
    score_lcb = jnp.where(
        candidate_eval_mask,
        decision_score - horizon_state.confidence_z * score_se,
        0.0,
    )
    # Keep historical metric names as aliases, but use one scalar everywhere.
    fitness = decision_score
    deployment_score = decision_score
    deploy_mask = candidate_eval_mask
    transition_cost = _transition_cost(
        horizon_state,
        model_stage,
        env_std,
        incumbent_idx,
        candidate_eval_mask,
    )
    transition_adjusted_score = decision_score - transition_cost
    incumbent_score = decision_score[incumbent_idx]
    delta_mean = decision_score - incumbent_score - transition_cost
    posterior_std = jnp.maximum(
        horizon_state.gauss_post_std,
        horizon_state.transition_uncertainty_floor,
    )
    incumbent_std = jnp.maximum(
        horizon_state.gauss_post_std[incumbent_idx],
        horizon_state.transition_uncertainty_floor,
    )
    delta_std = jnp.sqrt(jnp.square(posterior_std) + jnp.square(incumbent_std))
    z_score = delta_mean / jnp.maximum(delta_std, EPS)
    switch_probability = _standard_normal_cdf(z_score)
    # Expected-improvement transition rule:
    #   X_h = score(h) - score(h_current) - transition_cost(h)
    #   EI_h = E[max(X_h, 0)], EL_h = E[max(-X_h, 0)]
    #   net_h = EI_h - risk_weight * EL_h.
    # This keeps transition decisions Bayesian but avoids the brittle hard
    # probability threshold that rejected useful h=4 switches and accepted
    # an over-confident late h=5 switch in previous runs.
    density = _standard_normal_pdf(z_score)
    expected_improvement = delta_mean * switch_probability + delta_std * density
    expected_loss = (
        -delta_mean * _standard_normal_cdf(-z_score) + delta_std * density
    )
    expected_net_benefit = (
        expected_improvement - horizon_state.transition_risk_weight * expected_loss
    )
    if horizon_state.score_variant == 'return_first':
      credible_return = jnp.logical_and(
          candidate_eval_mask,
          jnp.logical_and(
              jnp.arange(horizon_state.horizons.shape[0]) != incumbent_idx,
              score_lcb > horizon_state.switch_threshold,
          ),
      )
      best_credible_lcb = jnp.max(jnp.where(
          credible_return,
          score_lcb,
          -jnp.inf,
      ))
      scaled_tolerance = (
          horizon_state.calibrated_return_weight *
          horizon_state.return_first_tolerance /
          horizon_state.additive_return_scale
      )
      return_competitive = jnp.logical_and(
          credible_return,
          score_lcb >= best_credible_lcb - scaled_tolerance,
      )
      tie_roughness = _normalise_masked_jax(
          local_roughness,
          return_competitive,
          constant_value=0.0,
      )
      tie_cost = (
          horizon_state.calibrated_roughness_weight * tie_roughness +
          direct_switch_cost
      )
      return_first_idx = jnp.argmin(jnp.where(
          return_competitive,
          tie_cost,
          jnp.inf,
      ))
      has_credible_return = jnp.logical_and(
          paired_confidence_available,
          jnp.any(credible_return),
      )
      proposed_idx = jnp.where(
          has_credible_return,
          return_first_idx,
          incumbent_idx,
      )
    elif horizon_state.decision_rule == 'paired_lcb':
      proposal_score = score_lcb
      proposed_idx = jnp.argmax(jnp.where(
          deploy_mask, proposal_score, -jnp.inf
      ))
    else:
      if not horizon_state.credible_transition_enabled:
        proposal_score = decision_score
      elif horizon_state.credible_transition_rule == 'expected_improvement':
        proposal_score = expected_net_benefit
      else:
        proposal_score = transition_adjusted_score
      proposed_idx = jnp.argmax(jnp.where(
          deploy_mask, proposal_score, -jnp.inf
      ))
    proposed_horizon = horizon_state.horizons[proposed_idx]
    proposed_score = decision_score[proposed_idx]
    if horizon_state.score_variant == 'return_first':
      switch = proposed_idx != incumbent_idx
    elif horizon_state.decision_rule == 'paired_lcb':
      switch = jnp.logical_and(
          paired_confidence_available,
          jnp.logical_and(
              proposed_idx != incumbent_idx,
              score_lcb[proposed_idx] > horizon_state.switch_threshold,
          ),
      )
    else:
      credible_switch = (
          expected_net_benefit[proposed_idx] > horizon_state.transition_min_expected_net
          if horizon_state.credible_transition_rule == 'expected_improvement'
          else switch_probability[proposed_idx] >= horizon_state.credible_transition_min_prob
      )
      margin_switch = (
          proposed_score > incumbent_score + horizon_state.incumbent_switch_margin
      )
      switch = jnp.where(
          horizon_state.credible_transition_enabled,
          jnp.logical_and(proposed_idx != incumbent_idx, credible_switch),
          margin_switch,
      )
    raw_selected_horizon = jnp.where(switch, proposed_horizon, horizon_state.best_h)
    if horizon_state.max_transition_delta > 0:
      delta = jnp.clip(
          raw_selected_horizon - horizon_state.best_h,
          -horizon_state.max_transition_delta,
          horizon_state.max_transition_delta,
      )
      target_horizon = horizon_state.best_h + delta
      selected_idx = jnp.argmin(jnp.abs(horizon_state.horizons - target_horizon))
    else:
      selected_idx = jnp.argmin(jnp.abs(horizon_state.horizons - raw_selected_horizon))
    selected_horizon = horizon_state.horizons[selected_idx]
    candidate_scores = jnp.where(
        model_stage.candidate_mask,
        decision_score[model_stage.candidate_indices],
        0.0,
    )
    new_state = _state_with_score_updates(
        state=horizon_state,
        candidate_indices=model_stage.candidate_indices,
        candidate_mask=model_stage.candidate_mask,
        candidate_scores=candidate_scores,
        query_step=jnp.asarray(query_step, dtype=jnp.int32),
        selected_horizon=selected_horizon,
        key=score_key,
    )
    return DenseQueryResult(
        horizon_state=new_state,
        selected_horizon=selected_horizon,
        proposed_horizon=proposed_horizon,
        candidate_horizons=model_stage.candidate_horizons,
        candidate_mask=model_stage.candidate_mask,
        fitness=fitness,
        decision_score=decision_score,
        score_se=score_se,
        score_lcb=score_lcb,
        paired_confidence_available=paired_confidence_available,
        deployment_score=deployment_score,
        return_term=return_term,
        roughness_term=roughness_term,
        sigma_r_term=sigma_r_term,
        learner_proxy_term=learner_proxy_term,
        transition_cost=transition_cost,
        transition_adjusted_score=transition_adjusted_score,
        switch_probability=switch_probability,
        expected_improvement=expected_improvement,
        expected_loss=expected_loss,
        expected_net_benefit=expected_net_benefit,
        incumbent_deployment_score=incumbent_score,
        proposed_deployment_score=proposed_score,
        proposed_transition_cost=transition_cost[proposed_idx],
        proposed_switch_probability=switch_probability[proposed_idx],
        proposed_expected_net_benefit=expected_net_benefit[proposed_idx],
        robust_return=robust_return,
        prefix_objectives=model_stage.prefix_objectives,
        probe_prefixes=model_stage.probe_prefixes,
        planner_prefix_returns=model_stage.planner_prefix_returns,
        roughness=model_stage.roughness,
        roughness_nested=model_stage.roughness_nested,
        roughness_nested_valid=model_stage.roughness_nested_valid,
        roughness_projections=model_stage.roughness_projections,
        env_mean=env_mean,
        env_std=env_std,
        local_roughness=local_roughness,
        local_model_error=local_model_error,
        bellman_residual=bellman_residual,
        curvature_bellman_risk=curvature_bellman_risk,
        horizon_switch_cost=direct_switch_cost,
    )

  return DenseQueryKernelBundle(
      model_stage=run_model_stage,
      env_stage=run_env_stage,
      finalize_stage=run_finalize_stage,
  )


def _get_dense_query_kernel(eval_state: Any,
                            candidate_slots: int,
                            env_eval_steps: Optional[int]) -> DenseQueryKernelBundle:
  cache_key = (id(eval_state), int(candidate_slots), int(env_eval_steps or -1))
  if cache_key not in _DENSE_QUERY_KERNEL_CACHE:
    _DENSE_QUERY_KERNEL_CACHE[cache_key] = _build_dense_query_kernel(
        eval_state=eval_state,
        candidate_slots=int(candidate_slots),
        env_eval_steps=env_eval_steps,
    )
  return _DENSE_QUERY_KERNEL_CACHE[cache_key]

def build_dense_query_kernels(eval_state: Any,
                              env_eval_steps: Optional[int],
                              candidate_budgets: Sequence[int]) -> Dict[int, DenseQueryKernelBundle]:
  return {
      slots: _get_dense_query_kernel(
          eval_state=eval_state,
          candidate_slots=slots,
          env_eval_steps=env_eval_steps,
      )
      for slots in sorted(set(int(slot) for slot in candidate_budgets))
  }


def prewarm_dense_rhs_kernels(agent,
                              replay_batch: Mapping[str, jax.Array],
                              eval_state: Any,
                              horizon_state: HorizonSearchState,
                              rng: jax.Array,
                              env_eval_steps: Optional[int],
                              warm_all_phases: bool = False) -> Dict[int, DenseQueryKernelBundle]:
  kernels = build_dense_query_kernels(
      eval_state=eval_state,
      env_eval_steps=env_eval_steps,
      candidate_budgets=horizon_state.candidate_budget,
  )
  slots_to_prewarm = sorted(kernels)
  if not warm_all_phases:
    current_slots = int(horizon_state.candidate_budget[int(np.asarray(horizon_state.phase_id))])
    slots_to_prewarm = [current_slots]
  for offset, slots in enumerate(slots_to_prewarm):
    print(f'Prewarming Dense-RHS kernel for {slots} candidate slots...', flush=True)
    model_rng, env_rng, finalize_rng = jax.random.split(jax.random.fold_in(rng, offset), 3)
    model_stage = kernels[slots].model_stage(
        agent,
        replay_batch,
        horizon_state,
        model_rng,
    )
    jax.tree.map(
        lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x,
        model_stage,
    )
    env_mean, env_std, paired_returns, paired_returns_valid = kernels[slots].env_stage(
        agent,
        model_stage,
        env_rng,
        jnp.asarray(0, dtype=jnp.int32),
    )
    jax.tree.map(
        lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x,
        (env_mean, env_std, paired_returns, paired_returns_valid),
    )
    result = kernels[slots].finalize_stage(
        horizon_state,
        model_stage,
        env_mean,
        env_std,
        paired_returns,
        paired_returns_valid,
        jnp.asarray(0, dtype=jnp.int32),
        finalize_rng,
    )
    jax.tree.map(
        lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x,
        result,
    )
    print(f'Finished Dense-RHS kernel prewarm for {slots} candidate slots.', flush=True)
  return kernels


def _shadow_score_formulations(
    state: HorizonSearchState,
    *,
    env_mean: np.ndarray,
    env_std: np.ndarray,
    roughness: np.ndarray,
    roughness_projections: np.ndarray,
    paired_returns: np.ndarray,
    paired_returns_available: bool,
    candidate_mask: np.ndarray,
    local_roughness: np.ndarray,
    curvature_bellman_risk: np.ndarray,
    switch_cost: np.ndarray,
) -> Dict[str, float]:
  """Replays S0--S3 from one evidence tensor without affecting deployment."""
  horizons = np.asarray(state.horizons, dtype=np.int32)
  incumbent_idx = int(np.argmin(np.abs(horizons - int(np.asarray(state.best_h)))))
  mask = np.asarray(candidate_mask, dtype=bool)
  if not np.any(mask):
    return {}

  current_weights = jnp.asarray([
      state.selection_return_power,
      state.roughness_weight,
      state.return_std_weight,
      0.0,
  ], dtype=jnp.float32)
  additive_scales = jnp.asarray([
      state.additive_return_scale,
      state.additive_return_std_scale,
      state.additive_log_roughness_scale,
  ], dtype=jnp.float32)
  s0_score = np.asarray(_incumbent_relative_decision_score(
      env_mean=jnp.asarray(env_mean),
      env_std=jnp.asarray(env_std),
      roughness=jnp.asarray(roughness),
      incumbent_idx=jnp.asarray(incumbent_idx),
      candidate_mask=jnp.asarray(mask),
      score_mode='additive',
      weights=current_weights,
      additive_scales=additive_scales,
      evidence_floor=state.score_evidence_floor,
  )[0])
  if paired_returns_available:
    s0_se = np.asarray(_paired_score_standard_error(
        paired_returns=jnp.asarray(paired_returns),
        env_mean=jnp.asarray(env_mean),
        env_std=jnp.asarray(env_std),
        roughness_projections=jnp.asarray(roughness_projections),
        roughness=jnp.asarray(roughness),
        incumbent_idx=jnp.asarray(incumbent_idx),
        candidate_mask=jnp.asarray(mask),
        score_mode='additive',
        weights=current_weights,
        additive_scales=additive_scales,
        evidence_floor=state.score_evidence_floor,
    ))
    paired_difference = paired_returns - paired_returns[incumbent_idx]
    if paired_difference.shape[1] >= 2:
      return_se = np.std(paired_difference, axis=1, ddof=1) / np.sqrt(
          paired_difference.shape[1]
      )
    else:
      return_se = np.zeros_like(env_mean)
  else:
    s0_se = np.zeros_like(env_mean)
    return_se = np.zeros_like(env_mean)

  return_delta = env_mean - env_mean[incumbent_idx]
  calibrated_return = (
      state.calibrated_return_weight * return_delta /
      state.additive_return_scale
  )
  calibrated_return_se = (
      state.calibrated_return_weight * return_se /
      state.additive_return_scale
  )
  log_local = np.log(np.maximum(local_roughness, state.score_evidence_floor))
  s1_score = (
      calibrated_return -
      state.calibrated_roughness_weight *
      (log_local - log_local[incumbent_idx]) /
      state.additive_log_roughness_scale -
      switch_cost
  )
  s2_score = calibrated_return
  s3_score = (
      calibrated_return -
      state.curvature_risk_weight *
      (curvature_bellman_risk - curvature_bellman_risk[incumbent_idx]) /
      state.curvature_risk_scale -
      switch_cost
  )
  lcbs = {
      's0': s0_score - state.confidence_z * s0_se,
      's1': s1_score - state.confidence_z * calibrated_return_se,
      's2': s2_score - state.confidence_z * calibrated_return_se,
      's3': s3_score - state.confidence_z * calibrated_return_se,
  }

  metrics: Dict[str, float] = {}
  for name in ('s0', 's1', 's3'):
    lcb = lcbs[name]
    proposal_idx = int(np.argmax(np.where(mask, lcb, -np.inf)))
    selected_idx = (
        proposal_idx
        if paired_returns_available and proposal_idx != incumbent_idx and
        lcb[proposal_idx] > state.switch_threshold
        else incumbent_idx
    )
    metrics[f'dense_rhs/shadow_{name}_proposed_horizon'] = float(
        horizons[proposal_idx]
    )
    metrics[f'dense_rhs/shadow_{name}_selected_horizon'] = float(
        horizons[selected_idx]
    )
    metrics[f'dense_rhs/shadow_{name}_best_lcb'] = float(lcb[proposal_idx])

  s2_lcb = lcbs['s2']
  credible = mask.copy()
  credible[incumbent_idx] = False
  credible &= s2_lcb > state.switch_threshold
  if paired_returns_available and np.any(credible):
    best_lcb = float(np.max(s2_lcb[credible]))
    tolerance = (
        state.calibrated_return_weight * state.return_first_tolerance /
        state.additive_return_scale
    )
    competitive = credible & (s2_lcb >= best_lcb - tolerance)
    roughness_tie = np.zeros_like(local_roughness)
    selected_values = local_roughness[competitive]
    if selected_values.size and np.ptp(selected_values) > EPS:
      roughness_tie[competitive] = (
          selected_values - np.min(selected_values)
      ) / np.ptp(selected_values)
    tie_cost = state.calibrated_roughness_weight * roughness_tie + switch_cost
    s2_idx = int(np.argmin(np.where(competitive, tie_cost, np.inf)))
  else:
    s2_idx = incumbent_idx
  metrics['dense_rhs/shadow_s2_proposed_horizon'] = float(horizons[s2_idx])
  metrics['dense_rhs/shadow_s2_selected_horizon'] = float(horizons[s2_idx])
  metrics['dense_rhs/shadow_s2_best_lcb'] = float(s2_lcb[s2_idx])
  return metrics


def dense_checkpoint_eval(agent,
                          replay_batch: Mapping[str, jax.Array],
                          eval_state: Any,
                          horizon_state: HorizonSearchState,
                          rng: jax.Array,
                          env_eval_steps: Optional[int] = None,
                          query_step: int = 0,
                          dense_query_kernels: Optional[Mapping[int, DenseQueryKernelBundle]] = None,
                          ) -> Tuple[HorizonSearchState, int, Dict[str, float]]:
  query_start = time.perf_counter()
  candidate_slots = int(horizon_state.candidate_budget[int(np.asarray(horizon_state.phase_id))])
  kernels = dense_query_kernels or build_dense_query_kernels(
      eval_state=eval_state,
      env_eval_steps=env_eval_steps,
      candidate_budgets=horizon_state.candidate_budget,
  )
  model_rng, env_rng, finalize_rng = jax.random.split(rng, 3)
  model_stage_start = time.perf_counter()
  model_stage = kernels[candidate_slots].model_stage(
      agent,
      replay_batch,
      horizon_state,
      model_rng,
  )
  jax.tree.map(
      lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x,
      model_stage,
  )
  model_stage_s = time.perf_counter() - model_stage_start
  env_stage_start = time.perf_counter()
  env_mean, env_std, paired_returns, paired_returns_valid = kernels[
      candidate_slots
  ].env_stage(
      agent,
      model_stage,
      env_rng,
      jnp.asarray(query_step, dtype=jnp.int32),
  )
  jax.tree.map(
      lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x,
      (env_mean, env_std, paired_returns, paired_returns_valid),
  )
  env_stage_s = time.perf_counter() - env_stage_start
  finalize_stage_start = time.perf_counter()
  result = kernels[candidate_slots].finalize_stage(
      horizon_state,
      model_stage,
      env_mean,
      env_std,
      paired_returns,
      paired_returns_valid,
      jnp.asarray(query_step, dtype=jnp.int32),
      finalize_rng,
  )
  jax.tree.map(
      lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x,
      result,
  )
  finalize_stage_s = time.perf_counter() - finalize_stage_start
  new_state = result.horizon_state
  selected_horizon = int(np.asarray(result.selected_horizon))
  proposed_horizon = int(np.asarray(result.proposed_horizon))
  horizons = np.asarray(horizon_state.horizons, dtype=np.int32)
  fitness = np.asarray(result.fitness, dtype=np.float32)
  decision_score = np.asarray(result.decision_score, dtype=np.float32)
  score_se = np.asarray(result.score_se, dtype=np.float32)
  score_lcb = np.asarray(result.score_lcb, dtype=np.float32)
  paired_confidence_available = bool(
      np.asarray(result.paired_confidence_available)
  )
  paired_returns_available = bool(np.asarray(paired_returns_valid))
  deployment_score = np.asarray(result.deployment_score, dtype=np.float32)
  return_term = np.asarray(result.return_term, dtype=np.float32)
  roughness_term = np.asarray(result.roughness_term, dtype=np.float32)
  sigma_r_term = np.asarray(result.sigma_r_term, dtype=np.float32)
  learner_proxy_term = np.asarray(result.learner_proxy_term, dtype=np.float32)
  transition_cost = np.asarray(result.transition_cost, dtype=np.float32)
  transition_adjusted_score = np.asarray(result.transition_adjusted_score, dtype=np.float32)
  switch_probability = np.asarray(result.switch_probability, dtype=np.float32)
  expected_improvement = np.asarray(result.expected_improvement, dtype=np.float32)
  expected_loss = np.asarray(result.expected_loss, dtype=np.float32)
  expected_net_benefit = np.asarray(result.expected_net_benefit, dtype=np.float32)
  robust_return = np.asarray(result.robust_return, dtype=np.float32)
  prefix_objectives = np.asarray(result.prefix_objectives, dtype=np.float32)
  probe_prefixes = np.asarray(result.probe_prefixes, dtype=np.float32)
  planner_prefix_returns = np.asarray(result.planner_prefix_returns, dtype=np.float32)
  roughness = np.asarray(result.roughness, dtype=np.float32)
  roughness_nested = np.asarray(result.roughness_nested, dtype=np.float32)
  roughness_nested_valid = np.asarray(result.roughness_nested_valid, dtype=bool)
  roughness_projections = np.asarray(
      result.roughness_projections, dtype=np.float32
  )
  env_mean_array = np.asarray(result.env_mean, dtype=np.float32)
  env_std_array = np.asarray(result.env_std, dtype=np.float32)
  local_roughness = np.asarray(result.local_roughness, dtype=np.float32)
  local_model_error = np.asarray(result.local_model_error, dtype=np.float32)
  bellman_residual = np.asarray(result.bellman_residual, dtype=np.float32)
  curvature_bellman_risk = np.asarray(
      result.curvature_bellman_risk, dtype=np.float32
  )
  horizon_switch_cost = np.asarray(
      result.horizon_switch_cost, dtype=np.float32
  )
  paired_returns_array = np.asarray(paired_returns, dtype=np.float32)
  candidate_horizons = np.asarray(result.candidate_horizons, dtype=np.int32)
  candidate_mask = np.asarray(result.candidate_mask, dtype=bool)
  selected_idx = int(np.where(horizons == selected_horizon)[0][0])
  metrics = {
      'dense_rhs/selected_horizon': float(selected_horizon),
      'dense_rhs/proposed_horizon': float(proposed_horizon),
      'dense_rhs/phase_id': float(np.asarray(new_state.phase_id)),
      'dense_rhs/phase_name': PHASE_NAMES[int(np.asarray(new_state.phase_id))],
      'dense_rhs/num_active_horizons': float(np.sum(np.asarray(new_state.active_mask))),
      'dense_rhs/num_candidate_horizons': float(np.sum(candidate_mask)),
      'dense_rhs/num_roughness_probes': float(horizon_state.num_roughness_probes),
      'dense_rhs/paired_confidence_available': float(paired_confidence_available),
      'dense_rhs/paired_returns_available': float(paired_returns_available),
      'dense_rhs/entropy': float(np.asarray(new_state.entropy)),
      'dense_rhs/norm_entropy': float(np.asarray(new_state.norm_entropy)),
      'dense_rhs/best_fitness': float(fitness[selected_idx]),
      'dense_rhs/decision_score_best': float(decision_score[selected_idx]),
      'dense_rhs/score_se_best': float(score_se[selected_idx]),
      'dense_rhs/score_lcb_best': float(score_lcb[selected_idx]),
      'dense_rhs/deployment_score_best': float(deployment_score[selected_idx]),
      'dense_rhs/incumbent_deployment_score': float(
          np.asarray(result.incumbent_deployment_score)
      ),
      'dense_rhs/proposed_deployment_score': float(
          np.asarray(result.proposed_deployment_score)
      ),
      'dense_rhs/proposed_score_se': float(
          score_se[int(np.where(horizons == proposed_horizon)[0][0])]
      ),
      'dense_rhs/proposed_score_lcb': float(
          score_lcb[int(np.where(horizons == proposed_horizon)[0][0])]
      ),
      'dense_rhs/proposed_transition_cost': float(
          np.asarray(result.proposed_transition_cost)
      ),
      'dense_rhs/proposed_switch_probability': float(
          np.asarray(result.proposed_switch_probability)
      ),
      'dense_rhs/proposed_expected_net_benefit': float(
          np.asarray(result.proposed_expected_net_benefit)
      ),
      'dense_rhs/transition_cost_best': float(transition_cost[selected_idx]),
      'dense_rhs/transition_adjusted_score_best': float(
          transition_adjusted_score[selected_idx]
      ),
      'dense_rhs/switch_probability_best': float(switch_probability[selected_idx]),
      'dense_rhs/expected_improvement_best': float(expected_improvement[selected_idx]),
      'dense_rhs/expected_loss_best': float(expected_loss[selected_idx]),
      'dense_rhs/expected_net_benefit_best': float(expected_net_benefit[selected_idx]),
      'dense_rhs/return_term_best': float(return_term[selected_idx]),
      'dense_rhs/roughness_term_best': float(roughness_term[selected_idx]),
      'dense_rhs/return_std_term_best': float(sigma_r_term[selected_idx]),
      'dense_rhs/learner_proxy_term_best': float(learner_proxy_term[selected_idx]),
      'dense_rhs/model_prefix_loss_best': float(prefix_objectives[selected_idx]),
      'dense_rhs/model_probe_prefix_best': float(probe_prefixes[selected_idx]),
      'dense_rhs/planner_return_best': float(planner_prefix_returns[selected_idx]),
      'dense_rhs/roughness_best': float(roughness[selected_idx]),
      'dense_rhs/robust_return_best': float(robust_return[selected_idx]),
      'dense_rhs/local_roughness_best': float(local_roughness[selected_idx]),
      'dense_rhs/local_model_error_best': float(local_model_error[selected_idx]),
      'dense_rhs/bellman_residual_best': float(bellman_residual[selected_idx]),
      'dense_rhs/curvature_bellman_risk_best': float(
          curvature_bellman_risk[selected_idx]
      ),
      'dense_rhs/horizon_switch_cost_best': float(
          horizon_switch_cost[selected_idx]
      ),
      'timing/query_model_diag_s': float(model_stage_s),
      'timing/query_model_diag_probe_count': float(
          horizon_state.num_roughness_probes
      ),
      'timing/query_env_eval_s': float(env_stage_s),
      'timing/query_finalize_s': float(finalize_stage_s),
      'timing/query_total_s': float(time.perf_counter() - query_start),
  }
  evaluated_indices = [
      int(np.where(horizons == horizon)[0][0])
      for horizon in candidate_horizons[candidate_mask].tolist()
  ]
  evaluated_mask = np.zeros_like(horizons, dtype=bool)
  evaluated_mask[evaluated_indices] = True
  metrics.update(_shadow_score_formulations(
      horizon_state,
      env_mean=env_mean_array,
      env_std=env_std_array,
      roughness=roughness,
      roughness_projections=roughness_projections,
      paired_returns=paired_returns_array,
      paired_returns_available=paired_returns_available,
      candidate_mask=evaluated_mask,
      local_roughness=local_roughness,
      curvature_bellman_risk=curvature_bellman_risk,
      switch_cost=horizon_switch_cost,
  ))
  if evaluated_indices:
    floor = float(horizon_state.score_evidence_floor)
    evaluated_mean = env_mean_array[evaluated_indices]
    evaluated_std = env_std_array[evaluated_indices]
    evaluated_roughness = roughness[evaluated_indices]
    for name, values in (
        ('return_mean', evaluated_mean),
        ('return_std', evaluated_std),
        ('roughness', evaluated_roughness),
    ):
      metrics[f'dense_rhs/evidence_{name}_min'] = float(np.min(values))
      metrics[f'dense_rhs/evidence_{name}_median'] = float(np.median(values))
      metrics[f'dense_rhs/evidence_{name}_floor_hits'] = float(
          np.sum(values <= floor)
      )
    metrics['dense_rhs/effective_return_weight_median'] = float(
        np.median(
            1.0 / np.maximum(evaluated_mean, floor)
            if horizon_state.score_mode == 'multiplicative'
            else np.full_like(evaluated_mean, 1.0 / horizon_state.additive_return_scale)
        )
    )
    metrics['dense_rhs/effective_return_std_weight_median'] = float(
        np.median(
            1.0 / np.maximum(evaluated_std, floor)
            if horizon_state.score_mode == 'multiplicative'
            else np.full_like(
                evaluated_std,
                1.0 / horizon_state.additive_return_std_scale,
            )
        )
    )
  for probe_idx, probe_count in enumerate(ROUGHNESS_PROBE_COUNTS):
    if roughness_nested_valid[probe_idx]:
      metrics[f'dense_rhs/roughness_m{probe_count}_best'] = float(
          roughness_nested[probe_idx, selected_idx]
      )
  # Retain all horizon/probe evidence, including horizons outside the current
  # local decision window. This supports unbiased nested-M calibration and
  # direction bootstrap intervals without changing the online selector.
  for horizon_idx, horizon in enumerate(horizons.tolist()):
    for probe_idx, probe_count in enumerate(ROUGHNESS_PROBE_COUNTS):
      if roughness_nested_valid[probe_idx]:
        metrics[f'dense_rhs/horizon_{horizon}_roughness_m{probe_count}'] = float(
            roughness_nested[probe_idx, horizon_idx]
        )
    for direction_idx in range(roughness_projections.shape[0]):
      metrics[
          f'dense_rhs/horizon_{horizon}_roughness_projection_{direction_idx}'
      ] = float(roughness_projections[direction_idx, horizon_idx])
  for horizon in candidate_horizons[candidate_mask].tolist():
    idx = int(np.where(horizons == horizon)[0][0])
    metrics[f'dense_rhs/candidate_{horizon}_fitness'] = float(fitness[idx])
    metrics[f'dense_rhs/candidate_{horizon}_decision_score'] = float(
        decision_score[idx]
    )
    metrics[f'dense_rhs/candidate_{horizon}_score_se'] = float(score_se[idx])
    metrics[f'dense_rhs/candidate_{horizon}_score_lcb'] = float(score_lcb[idx])
    metrics[f'dense_rhs/candidate_{horizon}_deployment_score'] = float(
        deployment_score[idx]
    )
    metrics[f'dense_rhs/candidate_{horizon}_transition_cost'] = float(
        transition_cost[idx]
    )
    metrics[f'dense_rhs/candidate_{horizon}_transition_adjusted_score'] = float(
        transition_adjusted_score[idx]
    )
    metrics[f'dense_rhs/candidate_{horizon}_switch_probability'] = float(
        switch_probability[idx]
    )
    metrics[f'dense_rhs/candidate_{horizon}_expected_net_benefit'] = float(
        expected_net_benefit[idx]
    )
    metrics[f'dense_rhs/candidate_{horizon}_return'] = float(robust_return[idx])
    metrics[f'dense_rhs/candidate_{horizon}_env_mean'] = float(env_mean_array[idx])
    metrics[f'dense_rhs/candidate_{horizon}_env_std'] = float(env_std_array[idx])
    metrics[f'dense_rhs/candidate_{horizon}_roughness'] = float(roughness[idx])
    metrics[f'dense_rhs/candidate_{horizon}_local_roughness'] = float(
        local_roughness[idx]
    )
    metrics[f'dense_rhs/candidate_{horizon}_local_model_error'] = float(
        local_model_error[idx]
    )
    metrics[f'dense_rhs/candidate_{horizon}_bellman_residual'] = float(
        bellman_residual[idx]
    )
    metrics[f'dense_rhs/candidate_{horizon}_curvature_bellman_risk'] = float(
        curvature_bellman_risk[idx]
    )
    metrics[f'dense_rhs/candidate_{horizon}_horizon_switch_cost'] = float(
        horizon_switch_cost[idx]
    )
    metrics[f'dense_rhs/candidate_{horizon}_return_term'] = float(return_term[idx])
    metrics[f'dense_rhs/candidate_{horizon}_roughness_term'] = float(roughness_term[idx])
    metrics[f'dense_rhs/candidate_{horizon}_return_std_term'] = float(sigma_r_term[idx])
    metrics[f'dense_rhs/candidate_{horizon}_learner_proxy_term'] = float(
        learner_proxy_term[idx]
    )
    for probe_idx, probe_count in enumerate(ROUGHNESS_PROBE_COUNTS):
      if roughness_nested_valid[probe_idx]:
        metrics[f'dense_rhs/candidate_{horizon}_roughness_m{probe_count}'] = float(
            roughness_nested[probe_idx, idx]
        )
    if paired_returns_available:
      for replica_idx, replica_return in enumerate(paired_returns_array[idx]):
        metrics[
            f'dense_rhs/candidate_{horizon}_return_replica_{replica_idx}'
        ] = float(replica_return)
  return new_state, selected_horizon, metrics


def benchmark_dense_model_stage_probe_counts(
    agent,
    replay_batch: Mapping[str, jax.Array],
    horizon_state: HorizonSearchState,
    key: jax.Array,
    *,
    candidate_slots: int,
    probe_counts: Sequence[int] = VALID_ROUGHNESS_PROBE_COUNTS,
    warmup_calls: int = 5,
    repetitions: int = 30,
) -> list[dict[str, float]]:
  """Benchmarks compile and steady-state model-stage cost for each probe count.

  The model stage contains the fixed prefix-loss and planner diagnostics plus
  roughness. Consequently M=0 is the requested fixed-cost reference rather
  than a near-empty roughness-only function. Every timer is synchronized on
  the returned device values and timed calls receive fresh PRNG keys.
  """
  probe_counts = tuple(int(count) for count in probe_counts)
  invalid = sorted(set(probe_counts) - set(VALID_ROUGHNESS_PROBE_COUNTS))
  if invalid:
    raise ValueError(f'Unsupported probe counts: {invalid}')
  if int(warmup_calls) < 0 or int(repetitions) < 2:
    raise ValueError(
        'warmup_calls must be non-negative and repetitions must be at least two'
    )
  kernel = _build_dense_query_kernel(
      eval_state=None,
      candidate_slots=int(candidate_slots),
      env_eval_steps=None,
  )

  def synchronize(value):
    jax.tree.map(
        lambda item: (
            item.block_until_ready()
            if hasattr(item, 'block_until_ready') else item
        ),
        value,
    )

  records: list[dict[str, float]] = []
  key_stream = key
  for probe_count in probe_counts:
    benchmark_state = horizon_state.replace(
        num_roughness_probes=int(probe_count)
    )
    key_stream, first_key = jax.random.split(key_stream)
    first_start = time.perf_counter()
    first_result = kernel.model_stage(
        agent,
        replay_batch,
        benchmark_state,
        first_key,
    )
    synchronize(first_result)
    first_call_s = time.perf_counter() - first_start

    for _ in range(int(warmup_calls)):
      key_stream, warmup_key = jax.random.split(key_stream)
      synchronize(
          kernel.model_stage(
              agent,
              replay_batch,
              benchmark_state,
              warmup_key,
          )
      )

    samples = []
    for _ in range(int(repetitions)):
      key_stream, timed_key = jax.random.split(key_stream)
      start = time.perf_counter()
      synchronize(
          kernel.model_stage(
              agent,
              replay_batch,
              benchmark_state,
              timed_key,
          )
      )
      samples.append(time.perf_counter() - start)
    samples_array = np.asarray(samples, dtype=np.float64)
    records.append({
        'probe_count': float(probe_count),
        'compile_plus_first_s': float(first_call_s),
        'warmup_calls': float(warmup_calls),
        'repetitions': float(repetitions),
        'wall_time_s': float(np.mean(samples_array)),
        'median_wall_time_s': float(np.median(samples_array)),
        'p95_wall_time_s': float(np.quantile(samples_array, 0.95)),
        'std_wall_time_s': float(np.std(samples_array, ddof=1)),
    })
  return records
