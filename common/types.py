from typing import Optional
import flax
from flax.training import train_state

Params = flax.core.FrozenDict


class TrainState(train_state.TrainState):
    target_params: Optional[Params] = None
