import jax.numpy as jnp
import jax

from tdmpc2_jax.common.util import two_hot

def cross_entropy(logits: jax.Array, target: jax.Array) -> jax.Array:
  return -(jax.nn.log_softmax(logits, axis=-1) * target).sum(axis=-1)
