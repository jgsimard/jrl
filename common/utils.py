import jax
from common.types import TrainState


def soft_target_update(state: TrainState, tau):
    updated_params = jax.tree_map(lambda p, tp: p * tau + tp * (1.0 - tau),
                                  state.params,
                                  state.target_params)
    return state.replace(target_params=updated_params)
