from typing import Optional
import flax
import numpy as np
from flax.training import train_state
import optax

# import jax_dataclasses as jdc


Params = flax.core.FrozenDict


class TrainState(train_state.TrainState):
    target_params: Optional[Params] = None

    def incremental_update_target(self, tau: float):
        return self.replace(
            target_params=optax.incremental_update(self.params, self.target_params, tau)
        )


@flax.struct.dataclass
# @jdc.pytree_dataclass
class Batch:  # type: ignore
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    masks: np.ndarray
    next_observations: np.ndarray
