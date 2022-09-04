import functools
from typing import Any, Callable

import jax
import numpy as np

from common.types import Params


@functools.partial(jax.jit, static_argnames=('actor_apply_fn', 'distribution'))
def sample_actions(
        rng,
        actor_apply_fn: Callable[..., Any],
        actor_params: Params,
        observations: np.ndarray,
        temperature: float = 1.0,
        distribution: str = 'log_prob'):
    if distribution == 'det':
        return rng, actor_apply_fn(actor_params, observations, temperature)
    # else
    dist = actor_apply_fn(actor_params, observations, temperature)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)
