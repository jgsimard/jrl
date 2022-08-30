from typing import Any, Dict, Sequence

import flax
from flax.core import FrozenDict
from flax.training import train_state

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]


class TrainState(train_state.TrainState):
    target_params: FrozenDict
