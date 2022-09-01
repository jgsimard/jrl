from typing import Optional
import flax
import numpy as np
from flax.training import train_state

Params = flax.core.FrozenDict


class TrainState(train_state.TrainState):
    target_params: Optional[Params] = None


@flax.struct.dataclass
class Batch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    masks: np.ndarray
    next_observations: np.ndarray
