from typing import Optional
import re

from flax import linen as nn
from jax import numpy as jnp
import numpy as np

def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


def normalize(data: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray, eps: float = 1e-8):
    return (data - mean) / (std + eps)


def unnormalize(data: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray):
    return data * std + mean


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            # else:
            mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
            return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)
