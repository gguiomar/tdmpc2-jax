import jax
import jax.numpy as jnp

from tdmpc2_jax.data.sequential_buffer import sample_from_state


def _state():
  capacity = 20
  num_envs = 2
  row = jnp.arange(capacity, dtype=jnp.int32)[:, None]
  data = {
      'observation': jnp.broadcast_to(row[..., None], (capacity, num_envs, 1)),
      'terminated': jnp.zeros((capacity, num_envs), dtype=bool),
      'truncated': jnp.zeros((capacity, num_envs), dtype=bool),
  }
  return {
      'current_ind': jnp.asarray([10, 10], dtype=jnp.int32),
      'size': jnp.asarray([10, 10], dtype=jnp.int32),
      'data': data,
      'rng_key': jax.random.PRNGKey(5),
  }


def test_recent_sampling_excludes_older_logical_rows():
  _, batch = sample_from_state(
      _state(),
      batch_size=64,
      sequence_length=2,
      recent_transition_steps=8,
  )
  sampled_rows = batch['observation'][..., 0]
  assert int(jnp.min(sampled_rows)) >= 6
  assert int(jnp.max(sampled_rows)) <= 9


def test_zero_recent_window_preserves_complete_history_sampling():
  _, batch = sample_from_state(
      _state(),
      batch_size=256,
      sequence_length=2,
      recent_transition_steps=0,
  )
  sampled_rows = batch['observation'][..., 0]
  assert int(jnp.min(sampled_rows)) == 0
  assert int(jnp.max(sampled_rows)) == 9
