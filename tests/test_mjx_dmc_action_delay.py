import unittest

import jax
import jax.numpy as jnp
import numpy as np

from tdmpc2_jax.envs.mjx_dmc import (
    CARTPOLE_ACTION_DELAY_MAX,
    action_delay_observation_dim,
    action_delay_queue_step,
    augment_observation_with_action_delay,
    broadcast_candidate_axis,
    cartpole_action_delay,
    make_action_delay_queue,
    make_global_transition_steps,
    mask_paired_candidate_returns,
    paired_candidate_keys,
)


class CartpoleActionDelayTest(unittest.TestCase):

  def test_schedule_boundaries(self):
    steps = jnp.asarray(
        [0, 149_999, 150_000, 349_999, 350_000, 500_000],
        dtype=jnp.int32,
    )
    actual = np.asarray(jax.jit(cartpole_action_delay)(steps))
    np.testing.assert_array_equal(actual, [0, 0, 4, 4, 0, 0])

  def test_vector_batch_receives_exact_global_transition_indices(self):
    actual = np.asarray(make_global_transition_steps(149_998, (4,), 4))
    np.testing.assert_array_equal(
        actual,
        [149_998, 149_999, 150_000, 150_001],
    )

    paired = np.asarray(make_global_transition_steps(149_998, (2, 4), 4))
    self.assertEqual(paired.shape, (2, 4))
    np.testing.assert_array_equal(paired[0], paired[1])

  def test_fifo_semantics_for_every_supported_delay(self):
    for delay in range(CARTPOLE_ACTION_DELAY_MAX + 1):
      with self.subTest(delay=delay):
        queue = make_action_delay_queue((), action_dim=1)

        @jax.jit
        def step(delay_queue, action):
          return action_delay_queue_step(
              delay_queue,
              action,
              jnp.asarray(delay, dtype=jnp.int32),
          )

        applied = []
        for issued in range(1, 9):
          action_to_apply, queue = step(
              queue,
              jnp.asarray([issued], dtype=jnp.float32),
          )
          applied.append(float(np.asarray(action_to_apply)[0]))

        expected = []
        for issued in range(1, 9):
          expected.append(0.0 if issued <= delay else float(issued - delay))
        np.testing.assert_array_equal(applied, expected)

  def test_queue_advances_while_delay_is_zero(self):
    queue = make_action_delay_queue((), action_dim=1)
    for issued in range(1, 6):
      action_to_apply, queue = action_delay_queue_step(
          queue,
          jnp.asarray([issued], dtype=jnp.float32),
          jnp.asarray(0, dtype=jnp.int32),
      )
      self.assertEqual(float(np.asarray(action_to_apply)[0]), float(issued))
    np.testing.assert_array_equal(np.asarray(queue[:, 0]), [2, 3, 4, 5])

  def test_switch_to_four_uses_preintervention_history(self):
    step = jax.jit(action_delay_queue_step)
    queue = make_action_delay_queue((), action_dim=1)
    for issued in range(1, 5):
      _, queue = step(
          queue,
          jnp.asarray([issued], dtype=jnp.float32),
          jnp.asarray(0, dtype=jnp.int32),
      )
    applied, next_queue = step(
        queue,
        jnp.asarray([5], dtype=jnp.float32),
        jnp.asarray(4, dtype=jnp.int32),
    )
    np.testing.assert_array_equal(np.asarray(applied), [1.0])
    np.testing.assert_array_equal(np.asarray(next_queue[:, 0]), [2, 3, 4, 5])

  def test_observation_contains_queue_then_normalized_delay(self):
    physical_obs = jnp.asarray([[10.0, 20.0]], dtype=jnp.float32)
    queue = jnp.asarray([[[1.0], [2.0], [3.0], [4.0]]], dtype=jnp.float32)
    observation = augment_observation_with_action_delay(
        physical_obs,
        queue,
        jnp.asarray([4], dtype=jnp.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(observation),
        [[10.0, 20.0, 1.0, 2.0, 3.0, 4.0, 1.0]],
    )

  def test_canonical_observation_can_remain_unaugmented(self):
    self.assertEqual(action_delay_observation_dim(5, 1, enabled=False), 5)
    self.assertEqual(action_delay_observation_dim(5, 1, enabled=True), 10)

  def test_empty_queue_is_fixed_shape_and_zero_filled(self):
    queue = make_action_delay_queue((3,), action_dim=2)
    self.assertEqual(queue.shape, (3, 4, 2))
    np.testing.assert_array_equal(np.asarray(queue), np.zeros((3, 4, 2)))

  def test_candidate_broadcast_preserves_replica_pairing(self):
    replicas = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    candidates = broadcast_candidate_axis(replicas, num_candidates=3)
    self.assertEqual(candidates.shape, (3, 2, 2))
    for candidate in np.asarray(candidates):
      np.testing.assert_array_equal(candidate, np.asarray(replicas))

    key = jax.random.PRNGKey(17)
    keys = paired_candidate_keys(key, num_candidates=3)
    for candidate_key in np.asarray(keys):
      np.testing.assert_array_equal(candidate_key, np.asarray(key))

  def test_masked_candidate_returns_are_neutral(self):
    returns = jnp.asarray(
        [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
        dtype=jnp.float32,
    )
    masked = mask_paired_candidate_returns(
        returns,
        jnp.asarray([True, False]),
    )
    np.testing.assert_array_equal(
        np.asarray(masked),
        [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
    )


if __name__ == '__main__':
  unittest.main()
