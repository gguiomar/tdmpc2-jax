import flax.linen as nn
import jax.numpy as jnp
import jax
from typing import Callable, Optional, Sequence


class BroNet(nn.Module):
  embed_dim: int
  num_blocks: int
  kernel_init: Callable = nn.initializers.truncated_normal(stddev=0.02)
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(
        self.embed_dim, kernel_init=self.kernel_init, dtype=self.dtype
    )(x)
    x = nn.LayerNorm(dtype=self.dtype)(x)
    x = nn.relu(x)

    for _ in range(self.num_blocks):
      skip = x
      x = nn.Dense(
          self.embed_dim, kernel_init=self.kernel_init, dtype=self.dtype
      )(x)
      x = nn.LayerNorm(dtype=self.dtype)(x)
      x = nn.relu(x)
      x = nn.Dense(
          self.embed_dim, kernel_init=self.kernel_init, dtype=self.dtype
      )(x)
      x = nn.LayerNorm(dtype=self.dtype)(x)
      x = x + skip

    return x
