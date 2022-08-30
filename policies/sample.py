import functools
from typing import Any, Callable

import jax
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from common.types import Params, PRNGKey

tfd = tfp.distributions
tfb = tfp.bijectors


# TODO : replace tensorflow_probability with distrax
@functools.partial(jax.jit, static_argnames=('actor_apply_fn', 'distribution'))
def _sample_actions(
        rng: PRNGKey,
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


def sample_actions(
        rng: Any,
        actor_apply_fn: Callable[..., Any],
        actor_params: Any,
        observations: np.ndarray,
        temperature: float = 1.0,
        distribution: str = 'log_prob'):
    return _sample_actions(rng, actor_apply_fn, actor_params, observations,
                           temperature, distribution)
