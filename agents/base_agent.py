from typing import  Dict

import jax
import numpy as np
from jax import numpy as jnp

from common.types import Batch


class BaseAgent:
    def __init__(self, seed: int) -> None:
        self.rng = jax.random.PRNGKey(seed)
        self.step = 0

    def update(self, batch: Batch) -> Dict[str, float]:
        raise NotImplementedError("")

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        raise NotImplementedError("")
