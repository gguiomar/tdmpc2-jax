import numpy as np
import pytest

import jax
import jax.numpy as jnp

from tdmpc2_jax.horizon_search import (
    EPS,
    PHASE_NAMES,
    ROUGHNESS_PROBE_COUNTS,
    DenseQueryModelStage,
    HorizonSearchState,
    _build_dense_query_kernel,
    _incumbent_relative_decision_score,
    _maybe_advance_phase,
    _nested_roughness_from_projections,
    _normalise_masked_jax,
    _paired_score_standard_error,
    _select_candidate_slots,
)


def _candidate_budget(size: int):
  return {name: size for name in PHASE_NAMES}


def test_masked_normalisation_ignores_unevaluated_values():
  values = jnp.asarray([0.0, 10.0, 1e9], dtype=jnp.float32)
  mask = jnp.asarray([True, True, False])

  np.testing.assert_allclose(
      _normalise_masked_jax(values, mask),
      np.asarray([0.0, 1.0, 0.0]),
      atol=1e-6,
  )
  np.testing.assert_allclose(
      _normalise_masked_jax(values, mask, inverse=True),
      np.asarray([1.0, 0.0, 0.0]),
      atol=1e-6,
  )


def test_constant_candidate_term_is_neutral_in_both_directions():
  values = jnp.asarray([4.0, 4.0, -100.0], dtype=jnp.float32)
  mask = jnp.asarray([True, True, False])

  np.testing.assert_allclose(
      _normalise_masked_jax(values, mask), np.asarray([1.0, 1.0, 0.0])
  )
  np.testing.assert_allclose(
      _normalise_masked_jax(values, mask, inverse=True),
      np.asarray([1.0, 1.0, 0.0]),
  )


@pytest.mark.parametrize('score_mode', ['additive', 'multiplicative'])
def test_incumbent_relative_score_has_zero_incumbent_and_correct_signs(score_mode):
  score, return_term, roughness_term, sigma_term = (
      _incumbent_relative_decision_score(
          env_mean=jnp.asarray([10.0, 12.0, 10.0, 10.0]),
          env_std=jnp.asarray([2.0, 2.0, 3.0, 2.0]),
          roughness=jnp.asarray([4.0, 4.0, 4.0, 8.0]),
          incumbent_idx=jnp.asarray(0),
          candidate_mask=jnp.asarray([True, True, True, True]),
          score_mode=score_mode,
          weights=jnp.asarray([1.0, 1.0, 1.0, 0.0]),
          additive_scales=jnp.asarray([2.0, 1.0, np.log(2.0)]),
          evidence_floor=EPS,
      )
  )

  np.testing.assert_allclose(score[0], 0.0, atol=1e-6)
  assert float(return_term[1]) > 0.0
  assert float(score[1]) > 0.0
  assert float(sigma_term[2]) < 0.0
  assert float(score[2]) < 0.0
  assert float(roughness_term[3]) < 0.0
  assert float(score[3]) < 0.0


@pytest.mark.parametrize('score_mode', ['additive', 'multiplicative'])
def test_incumbent_relative_score_ignores_unrelated_unevaluated_horizon(score_mode):
  kwargs = dict(
      env_std=jnp.asarray([999.0, 2.0, 2.0]),
      roughness=jnp.asarray([1e-9, 4.0, 4.0]),
      incumbent_idx=jnp.asarray(1),
      candidate_mask=jnp.asarray([False, True, True]),
      score_mode=score_mode,
      weights=jnp.asarray([1.0, 1.0, 1.0, 0.0]),
      additive_scales=jnp.ones((3,)),
      evidence_floor=EPS,
  )
  score_a = _incumbent_relative_decision_score(
      env_mean=jnp.asarray([1e9, 10.0, 11.0]), **kwargs
  )[0]
  score_b = _incumbent_relative_decision_score(
      env_mean=jnp.asarray([-1e9, 10.0, 11.0]), **kwargs
  )[0]

  np.testing.assert_allclose(score_a, score_b, atol=1e-6)
  np.testing.assert_allclose(score_a[0], 0.0, atol=1e-6)


@pytest.mark.parametrize('num_probes', [2, 4, 8, 16, 32, 64])
def test_nested_probe_estimates_reuse_prefixes(num_probes):
  stacked = jnp.arange(1, num_probes * 2 + 1, dtype=jnp.float32).reshape(
      num_probes, 2
  )
  selected, nested, valid = _nested_roughness_from_projections(stacked)

  expected_selected = np.sqrt(np.mean(np.square(np.asarray(stacked)), axis=0) + EPS)
  np.testing.assert_allclose(selected, expected_selected, rtol=1e-6)
  for index, count in enumerate(ROUGHNESS_PROBE_COUNTS):
    assert bool(valid[index]) is (count <= num_probes)
    if count <= num_probes:
      expected = np.sqrt(
          np.mean(np.square(np.asarray(stacked[:count])), axis=0) + EPS
      )
      np.testing.assert_allclose(nested[index], expected, rtol=1e-6)
    else:
      np.testing.assert_array_equal(nested[index], np.zeros((2,)))


def test_zero_probes_returns_zero_and_no_nested_estimates():
  selected, nested, valid = _nested_roughness_from_projections(
      jnp.zeros((0, 3), dtype=jnp.float32)
  )
  np.testing.assert_array_equal(selected, np.zeros((3,)))
  np.testing.assert_array_equal(nested, np.zeros((6, 3)))
  np.testing.assert_array_equal(valid, np.zeros((6,), dtype=bool))


def test_state_defaults_preserve_existing_campaign_settings():
  state = HorizonSearchState.create(
      horizons=[2, 3, 4],
      hmax=4,
      query_interval_steps=20,
      candidate_budget=_candidate_budget(3),
  )

  assert state.num_roughness_probes == 2
  assert state.score_mode == 'legacy_multiplicative'
  assert state.score_variant == 'current_additive'
  assert state.decision_rule == 'legacy'


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('num_roughness_probes', 3),
        ('score_mode', 'ratio'),
        ('score_variant', 'opaque_combo'),
    ],
)
def test_state_rejects_unsupported_static_scoring_settings(field, value):
  kwargs = dict(
      horizons=[2, 3],
      hmax=3,
      query_interval_steps=20,
      candidate_budget=_candidate_budget(2),
  )
  kwargs[field] = value
  with pytest.raises(ValueError):
    HorizonSearchState.create(**kwargs)


def test_candidate_selection_always_includes_inactive_incumbent():
  state = HorizonSearchState.create(
      horizons=[2, 3, 4],
      hmax=4,
      query_interval_steps=20,
      initial_horizon=3,
      candidate_budget=_candidate_budget(1),
  ).replace(
      active_mask=jnp.asarray([True, False, True]),
      prob=jnp.asarray([0.9, 0.0, 0.1]),
  )

  horizons, mask, indices = _select_candidate_slots(state, 1)

  np.testing.assert_array_equal(horizons, np.asarray([3]))
  np.testing.assert_array_equal(mask, np.asarray([True]))
  np.testing.assert_array_equal(indices, np.asarray([1]))


def test_disabled_phase_pruning_preserves_all_horizons_after_sampling_threshold():
  state = HorizonSearchState.create(
      horizons=[2, 3, 4],
      hmax=4,
      query_interval_steps=20,
      initial_horizon=3,
      candidate_budget={name: 1 for name in PHASE_NAMES},
      phase_pruning_enabled=False,
      phase_min_samples_to_drop=1,
  ).replace(
      phase_sample_count_A=jnp.ones((3,), dtype=jnp.int32),
  )

  updated = _maybe_advance_phase(state)

  assert int(updated.phase_id) == 0
  np.testing.assert_array_equal(updated.active_mask, np.ones((3,), dtype=bool))


def test_paired_score_se_preserves_candidate_incumbent_covariance():
  paired_returns = jnp.asarray(
      [
          [0.0, 0.0, 0.0, 0.0],
          [1.0, 2.0, 3.0, 4.0],
          [3.0, 4.0, 5.0, 6.0],
      ]
  )
  env_mean = jnp.mean(paired_returns, axis=1)
  env_std = jnp.std(paired_returns, axis=1)
  score_se = _paired_score_standard_error(
      paired_returns=paired_returns,
      env_mean=env_mean,
      env_std=env_std,
      roughness_projections=jnp.zeros((0, 3)),
      roughness=jnp.zeros((3,)),
      incumbent_idx=jnp.asarray(1),
      candidate_mask=jnp.asarray([False, True, True]),
      score_mode='additive',
      weights=jnp.asarray([1.0, 0.0, 0.0, 0.0]),
      additive_scales=jnp.ones((3,)),
      evidence_floor=EPS,
  )

  # Candidate 2 is a constant paired shift of the incumbent, so its mean
  # difference has exactly zero sampling uncertainty.
  np.testing.assert_allclose(score_se, np.zeros((3,)), atol=1e-6)


@pytest.mark.parametrize(
    ('switch_threshold', 'expected_horizon'),
    [(0.5, 4), (2.1, 3)],
)
def test_paired_lcb_gate_uses_one_sided_threshold(
    switch_threshold,
    expected_horizon,
):
  state = HorizonSearchState.create(
      horizons=[2, 3, 4],
      hmax=4,
      query_interval_steps=20,
      initial_horizon=3,
      candidate_budget=_candidate_budget(2),
      num_roughness_probes=0,
      score_mode='additive',
      decision_rule='paired_lcb',
      confidence_z=1.6448536,
      switch_threshold=switch_threshold,
      roughness_weight=0.0,
      return_std_weight=0.0,
  )
  model_stage = DenseQueryModelStage(
      prefix_objectives=jnp.zeros((3,)),
      probe_prefixes=jnp.zeros((3,)),
      planner_prefix_returns=jnp.asarray([0.0, 2.5, 4.5]),
      roughness=jnp.zeros((3,)),
      roughness_nested=jnp.zeros((6, 3)),
      roughness_nested_valid=jnp.zeros((6,), dtype=bool),
      roughness_projections=jnp.zeros((0, 3)),
      candidate_horizons=jnp.asarray([3, 4]),
      candidate_mask=jnp.asarray([True, True]),
      candidate_indices=jnp.asarray([1, 2]),
  )
  paired_returns = jnp.asarray(
      [
          [0.0, 0.0, 0.0, 0.0],
          [1.0, 2.0, 3.0, 4.0],
          [3.0, 4.0, 5.0, 6.0],
      ]
  )
  env_mean = jnp.mean(paired_returns, axis=1)
  env_std = jnp.std(paired_returns, axis=1)
  kernel = _build_dense_query_kernel(None, candidate_slots=2, env_eval_steps=None)

  result = kernel.finalize_stage(
      state,
      model_stage,
      env_mean,
      env_std,
      paired_returns,
      jnp.asarray(True),
      jnp.asarray(20, dtype=jnp.int32),
      jax.random.PRNGKey(0),
  )

  assert bool(result.paired_confidence_available)
  assert int(result.proposed_horizon) == 4
  assert int(result.selected_horizon) == expected_horizon
  np.testing.assert_allclose(result.score_se[2], 0.0, atol=1e-6)
  np.testing.assert_allclose(result.score_lcb[2], 2.0, atol=1e-6)


def _variant_finalize(score_variant):
  state = HorizonSearchState.create(
      horizons=[2, 3, 4],
      hmax=4,
      query_interval_steps=20,
      initial_horizon=2,
      candidate_budget=_candidate_budget(3),
      num_roughness_probes=0,
      score_mode='additive',
      score_variant=score_variant,
      decision_rule='paired_lcb',
      confidence_z=1.6448536,
      switch_threshold=0.0,
      additive_return_scale=50.0,
      additive_return_std_scale=50.0,
      additive_log_roughness_scale=1.0,
      calibrated_return_weight=10.0,
      calibrated_roughness_weight=1.0,
      horizon_switch_cost=0.05,
      return_first_tolerance=5.0,
      roughness_discount=1.0,
      roughness_weight=1.0,
      return_std_weight=0.0,
  )
  model_stage = DenseQueryModelStage(
      prefix_objectives=jnp.asarray([0.2, 0.3, 0.4]),
      probe_prefixes=jnp.asarray([0.1, 0.15, 0.2]),
      planner_prefix_returns=jnp.asarray([10.0, 11.0, 12.0]),
      roughness=jnp.asarray([1.0, 1.5, 2.0]),
      roughness_nested=jnp.zeros((6, 3)),
      roughness_nested_valid=jnp.zeros((6,), dtype=bool),
      roughness_projections=jnp.zeros((0, 3)),
      candidate_horizons=jnp.asarray([2, 3, 4]),
      candidate_mask=jnp.asarray([True, True, True]),
      candidate_indices=jnp.asarray([0, 1, 2]),
  )
  paired_returns = jnp.asarray([
      [9.0, 10.0, 11.0, 10.0],
      [10.0, 11.0, 12.0, 11.0],
      [11.0, 12.0, 13.0, 12.0],
  ])
  kernel = _build_dense_query_kernel(None, candidate_slots=3, env_eval_steps=None)
  return kernel.finalize_stage(
      state,
      model_stage,
      jnp.mean(paired_returns, axis=1),
      jnp.std(paired_returns, axis=1),
      paired_returns,
      jnp.asarray(True),
      jnp.asarray(20, dtype=jnp.int32),
      jax.random.PRNGKey(0),
  )


def test_calibrated_local_roughness_removes_mechanical_horizon_growth():
  current = _variant_finalize('current_additive')
  calibrated = _variant_finalize('calibrated_local_roughness')

  assert int(current.selected_horizon) == 2
  assert int(calibrated.selected_horizon) == 4
  np.testing.assert_allclose(
      calibrated.local_roughness,
      np.asarray([0.5, 0.5, 0.5]),
      atol=1e-6,
  )


def test_return_first_uses_roughness_only_inside_return_competitive_set():
  result = _variant_finalize('return_first')

  # Both longer horizons are credible and within five raw return units. Their
  # local roughness is tied, so the explicit switching cost selects h=3.
  assert int(result.proposed_horizon) == 3
  assert int(result.selected_horizon) == 3
  np.testing.assert_allclose(result.roughness_term, np.zeros((3,)))


def test_curvature_bellman_variant_exposes_risk_components():
  result = _variant_finalize('curvature_bellman')

  assert np.all(np.asarray(result.local_model_error) >= 0.0)
  assert np.all(np.asarray(result.bellman_residual) >= 0.0)
  assert np.all(np.asarray(result.curvature_bellman_risk) >= 0.0)


@pytest.mark.parametrize('score_mode', ['additive', 'multiplicative'])
def test_finalize_uses_one_candidate_only_score_for_selection_and_posterior(
    score_mode,
):
  state = HorizonSearchState.create(
      horizons=[2, 3, 4],
      hmax=4,
      query_interval_steps=20,
      initial_horizon=3,
      candidate_budget=_candidate_budget(2),
      num_roughness_probes=0,
      score_mode=score_mode,
      roughness_weight=0.0,
      return_std_weight=0.0,
  )
  model_stage = DenseQueryModelStage(
      prefix_objectives=jnp.asarray([0.0, 0.0, 0.0]),
      probe_prefixes=jnp.asarray([0.0, 0.0, 0.0]),
      planner_prefix_returns=jnp.asarray([1e9, 1.0, 2.0]),
      roughness=jnp.asarray([0.0, 0.0, 0.0]),
      roughness_nested=jnp.zeros((6, 3), dtype=jnp.float32),
      roughness_nested_valid=jnp.zeros((6,), dtype=bool),
      roughness_projections=jnp.zeros((0, 3), dtype=jnp.float32),
      candidate_horizons=jnp.asarray([3, 4]),
      candidate_mask=jnp.asarray([True, True]),
      candidate_indices=jnp.asarray([1, 2]),
  )
  kernel = _build_dense_query_kernel(
      eval_state=None,
      candidate_slots=2,
      env_eval_steps=None,
  )

  result = kernel.finalize_stage(
      state,
      model_stage,
      jnp.asarray([1e9, 1.0, 2.0]),
      jnp.zeros((3,), dtype=jnp.float32),
      jnp.zeros((3, 1), dtype=jnp.float32),
      jnp.asarray(False),
      jnp.asarray(20, dtype=jnp.int32),
      jax.random.PRNGKey(0),
  )

  np.testing.assert_allclose(result.fitness, result.decision_score)
  np.testing.assert_allclose(result.deployment_score, result.decision_score)
  assert int(result.proposed_horizon) == 4
  assert int(result.selected_horizon) == 4
  np.testing.assert_allclose(result.decision_score[0], 0.0)
  np.testing.assert_allclose(
      result.horizon_state.score_sum[model_stage.candidate_indices],
      result.decision_score[model_stage.candidate_indices],
  )
  np.testing.assert_array_equal(
      result.horizon_state.score_count, np.asarray([0, 1, 1])
  )
