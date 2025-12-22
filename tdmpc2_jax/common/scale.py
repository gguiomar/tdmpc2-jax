import jax.numpy as jnp
import jax


def percentile_normalization(
    x: jax.Array,
    prev_scale: jax.Array,
    tau: float = 0.01
) -> jax.Array:
  scale = abs(jnp.percentile(x, jnp.array([5, 95]))).max()

  return tau * scale + (1 - tau) * prev_scale
