from typing import Optional

from flax import linen as nn
from jax import numpy as jnp


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)
