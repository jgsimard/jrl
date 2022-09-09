from typing import Optional

from flax import linen as nn
from jax import numpy as jnp


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


def normalize(data: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray, eps: float = 1e-8):
    return (data - mean) / (std + eps)


def unnormalize(data: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray):
    return data * std + mean
